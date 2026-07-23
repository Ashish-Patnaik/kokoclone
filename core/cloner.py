import importlib.resources
import json
import os
import sys
import tempfile
import time
import types

import numpy as np
import torch

import soundfile as sf
from huggingface_hub import hf_hub_download
from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode
import kanade_tokenizer.module.fsq


# --- MONKEY PATCH KANADE FSQ FOR INTEL XPU ---
# The IPEX driver on Iris Xe lacks FP64 (float64) hardware support. The original kanade_tokenizer
# uses Python float literals and explicitly casts to float64, which causes fatal level_zero crashes.
# We patch these methods to strictly use float32 (or the input dtype) to bypass the driver crash!
def patched_bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    levels_f = torch.tensor(self.levels, dtype=z.dtype, device=z.device)
    # Ensure float operations don't implicitly promote to float64
    one_minus_eps = torch.tensor(1.0 - eps, dtype=z.dtype, device=z.device)
    half_l = (levels_f - 1.0) * one_minus_eps / 2.0
    offsets_list = [0.5 if l % 2 == 0 else 0.0 for l in self.levels]
    offset = torch.tensor(offsets_list, dtype=z.dtype, device=z.device)
    shift = (offset / half_l).tan()
    return (z + shift).tanh() * half_l - offset

def patched_quantize(self, z: torch.Tensor) -> torch.Tensor:
    from kanade_tokenizer.module.fsq import round_ste
    quantized = round_ste(self.bound(z))
    half_widths_list = [l // 2 for l in self.levels]
    half_width = torch.tensor(half_widths_list, dtype=quantized.dtype, device=quantized.device)
    return quantized / half_width

def patched_scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
    half_widths_list = [l // 2 for l in self.levels]
    half_width = torch.tensor(half_widths_list, dtype=zhat_normalized.dtype, device=zhat_normalized.device)
    return (zhat_normalized * half_width) + half_width

def patched_scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
    half_widths_list = [l // 2 for l in self.levels]
    half_width = torch.tensor(half_widths_list, dtype=zhat.dtype, device=zhat.device)
    return (zhat - half_width) / half_width

def patched_codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
    assert zhat.shape[-1] == len(self.levels)
    zhat = self._scale_and_shift(zhat)
    # Offload to CPU to bypass Intel Iris Xe level_zero driver casting crashes
    zhat_cpu = zhat.cpu()
    basis_list = [1]
    for i in range(len(self.levels) - 1):
        basis_list.append(basis_list[-1] * self.levels[i])
    basis_cpu = torch.tensor(basis_list, dtype=torch.float32, device='cpu')
    indices_cpu = (zhat_cpu.to(torch.float32) * basis_cpu).to(torch.long).sum(dim=-1)
    return indices_cpu.to(zhat.device)

def patched_indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
    indices_cpu = indices.cpu().unsqueeze(-1)
    basis_list = [1]
    for i in range(len(self.levels) - 1):
        basis_list.append(basis_list[-1] * self.levels[i])
    basis_cpu = torch.tensor(basis_list, dtype=torch.long, device='cpu')
    levels_cpu = torch.tensor(self.levels, dtype=torch.long, device='cpu')
    codes_non_centered = (indices_cpu // basis_cpu) % levels_cpu
    return self._scale_and_shift_inverse(codes_non_centered.to(indices.device))

def patched_fsq_forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
    latent = self.proj_in(z)
    quantized_latent, indices = self.fsq(latent)
    z_q = self.proj_out(quantized_latent)
    # Skip perplexity calculation (which crashes on XPU with .float() on long tensors)
    # since inference doesn't need it!
    info_dict = {
        "latent": latent,
        "quantized_latent": quantized_latent,
        "indices": indices,
        "perplexity": torch.tensor(0.0, device=z.device),
    }
    return z_q, info_dict

kanade_tokenizer.module.fsq.FSQ.bound = patched_bound
kanade_tokenizer.module.fsq.FSQ.quantize = patched_quantize
kanade_tokenizer.module.fsq.FSQ._scale_and_shift = patched_scale_and_shift
kanade_tokenizer.module.fsq.FSQ._scale_and_shift_inverse = patched_scale_and_shift_inverse
kanade_tokenizer.module.fsq.FSQ.codes_to_indices = patched_codes_to_indices
kanade_tokenizer.module.fsq.FSQ.indices_to_codes = patched_indices_to_codes
kanade_tokenizer.module.fsq.FiniteScalarQuantizer.forward = patched_fsq_forward

import vocos.heads
def patched_istft_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.out(x).transpose(1, 2)
    mag, p = x.chunk(2, dim=1)
    mag = torch.exp(mag)
    mag = torch.clip(mag, max=1e2)
    x_cos = torch.cos(p)
    y_sin = torch.sin(p)
    real = mag * x_cos
    imag = mag * y_sin
    
    # Offload complex math (1j) and ISTFT to CPU because Intel XPU lacks ComplexFloat support!
    real_cpu = real.cpu()
    imag_cpu = imag.cpu()
    S = real_cpu + 1j * imag_cpu
    
    # self.istft has window buffers, move it to CPU temporarily
    audio = self.istft.cpu()(S)
    return audio.to(x.device)

vocos.heads.ISTFTHead.forward = patched_istft_forward

import torchaudio.transforms
original_resample_forward = torchaudio.transforms.Resample.forward

def patched_resample_forward(self, waveform: torch.Tensor) -> torch.Tensor:
    if waveform.device.type in ["xpu", "cuda"]:
        device = waveform.device
        waveform_cpu = waveform.cpu()
        self_cpu = self.cpu()
        resampled = original_resample_forward(self_cpu, waveform_cpu)
        self.to(device) # put weights back
        return resampled.to(device)
    return original_resample_forward(self, waveform)

torchaudio.transforms.Resample.forward = patched_resample_forward

import kanade_tokenizer.module.ssl_extractor
original_ssl_forward = kanade_tokenizer.module.ssl_extractor.SSLFeatureExtractor.forward

def patched_ssl_forward(self, waveform: torch.Tensor, lengths: torch.Tensor | None = None, num_layers: int | None = None, return_lengths: bool = False):
    if waveform.device.type in ["xpu", "cuda"]:
        # wav2vec2 expects float32 on CPU
        device = waveform.device
        dtype = waveform.dtype
        waveform_cpu = waveform.cpu().float()
        lengths_cpu = lengths.cpu() if lengths is not None else None
        self_cpu = self.cpu().float()
        features = self_cpu(waveform_cpu, lengths_cpu)
        if isinstance(features, tuple):
            features = (features[0].to(dtype).to(device), features[1].to(dtype).to(device))
        elif isinstance(features, list):
            features = [f.to(dtype).to(device) for f in features]
        else:
            features = features.to(dtype).to(device)
        # Restore module back to original device and dtype
        self.to(dtype).to(device)
        return features
            
    return original_ssl_forward(self, waveform, lengths, num_layers, return_lengths)

kanade_tokenizer.module.ssl_extractor.SSLFeatureExtractor.forward = patched_ssl_forward

import torch
original_autocast = torch.autocast

class patched_autocast(original_autocast):
    def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
        if device_type == "cuda" and not torch.cuda.is_available() and hasattr(torch, "xpu") and torch.xpu.is_available():
            device_type = "xpu"
        if cache_enabled is not None:
            super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
        else:
            super().__init__(device_type, dtype=dtype, enabled=enabled)

torch.autocast = patched_autocast
# ---------------------------------------------

from kokoro import KPipeline
from misaki import espeak
from misaki.espeak import EspeakG2P
from core.chunked_convert import chunked_voice_conversion

class KokoClone:
    def __init__(self, kanade_model="frothywater/kanade-12.5hz", hf_repo="hexgrad/Kokoro-82M"):
        # Force CPU on unsupported Intel Iris Xe devices because XPU driver hangs the whole system.
        # Fallback to CUDA if available.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device("xpu")
        else:
            self.device = torch.device("cpu")
            
        print(f"Initializing KokoClone on: {self.device.type.upper()}")
        
        self.hf_repo = hf_repo
        
        # Load Kanade & Vocoder once, move to detected device
        print("Loading Kanade model...")
        # Force CPU execution for Kanade and Vocoder because Intel Iris Xe XPU is extremely slow/broken
        # for complex Transformer operations like SDPA.
        self.device = torch.device("cpu")
        print("Forcing KokoClone to CPU and applying PyTorch optimizations...")
        
        self.kanade = KanadeModel.from_pretrained(kanade_model).to(self.device).eval()
        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device).eval()

        # Use torch.compile to optimize the Kanade inference graph
        try:
            self.kanade = torch.compile(self.kanade)
            print("Successfully applied torch.compile to Kanade model.")
        except Exception as e:
            print(f"Warning: torch.compile failed or not supported: {e}")

        self.sample_rate = self.kanade.config.sample_rate
        
        # Cache for Kokoro
        self.kokoro_cache = {}
        
        # Cache for Kanade Reference Audio Embeddings
        self.kanade_ref_cache = {}

    def _get_vocab_config(self, lang):
        """Return a vocab config path compatible with the selected language/model."""
        # zh/ja model exports use the v1.1-zh vocabulary from hexgrad.
        if lang in {"zh", "ja"}:
            zh_vocab = os.path.join("model", "config-v1.1-zh.json")
            if not os.path.exists(zh_vocab):
                print("Downloading missing file 'config-v1.1-zh.json' from hexgrad/Kokoro-82M-v1.1-zh...")
                hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M-v1.1-zh",
                    filename="config.json",
                    local_dir=".",
                )
                downloaded = os.path.join("config.json")
                if os.path.exists(downloaded):
                    os.replace(downloaded, zh_vocab)

            if os.path.exists(zh_vocab):
                return zh_vocab

        local_config = os.path.join("model", "config.json")
        if os.path.exists(local_config):
            try:
                with open(local_config, encoding="utf-8") as fp:
                    config = json.load(fp)
                if isinstance(config, dict) and "vocab" in config:
                    return local_config
                print("Warning: model/config.json is missing 'vocab'; using packaged kokoro_onnx config instead")
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Warning: could not read model/config.json ({exc}); using packaged kokoro_onnx config instead")

        return str(importlib.resources.files("kokoro_onnx").joinpath("config.json"))



    def _ensure_file(self, folder, filename):
        """Auto-downloads missing models from your Hugging Face repo."""
        filepath = os.path.join(folder, filename)
        repo_filepath = f"{folder}/{filename}"
        
        if not os.path.exists(filepath):
            print(f"Downloading missing file '{filename}' from {self.hf_repo}...")
            hf_hub_download(
                repo_id=self.hf_repo,
                filename=repo_filepath,
                local_dir="." # Downloads securely into local ./model or ./voice
            )
        return filepath

    def _create_en_callable(self):
        """Create an English G2P callable for handling English tokens in non-English text."""
        en_g2p = EspeakG2P(language="en-us")
        def en_callable(text):
            try:
                phonemes, _ = en_g2p(text)
                return phonemes
            except Exception:
                return text
        return en_callable

    def _get_config(self, lang):
        """Routes the correct model, voice, and G2P based on language."""
        model_file = self._ensure_file("model", "kokoro.onnx")
        voices_file = self._ensure_file("voice", "voices-v1.0.bin")
        vocab = None
        g2p = None
        en_callable = None

        # Optimized routing: Only load the specific G2P engine requested
        if lang == "en":
            voice = "af_bella"
        elif lang == "hi":
            g2p = EspeakG2P(language="hi")
            voice = "hf_alpha"
        elif lang == "fr":
            g2p = EspeakG2P(language="fr-fr")
            voice = "ff_siwis"
        elif lang == "it":
            g2p = EspeakG2P(language="it")
            voice = "im_nicola"
        elif lang == "es":
            g2p = EspeakG2P(language="es")
            voice = "im_nicola"
        elif lang == "pt":
            g2p = EspeakG2P(language="pt-br")
            voice = "pf_dora"
        elif lang == "ja":
            from misaki import ja
            import unidic
            import subprocess
            
            # FIX: Auto-download the Japanese dictionary if it's missing!
            if not os.path.exists(unidic.DICDIR):
                print("Downloading missing Japanese dictionary (this takes a minute but only happens once)...")
                subprocess.run([sys.executable, "-m", "unidic", "download"], check=True)
                
            g2p = ja.JAG2P()
            voice = "jf_alpha"
            vocab = self._get_vocab_config(lang)
            # Provide English fallback for mixed Japanese-English text
            en_callable = self._create_en_callable()
        elif lang == "zh":
            from misaki import zh
            import re
            
            base_g2p = zh.ZHG2P(version="1.1")
            en_callable = self._create_en_callable()
            
            # Wrap ZHG2P to handle English tokens in mixed Chinese-English text.
            def mixed_g2p(text):
                # Split on English words/names and process them separately
                parts = re.split(r'([a-zA-Z]+)', text)
                phonemes_list = []
                for part in parts:
                    if part and part[0].isalpha() and part[0].isascii():
                        # English token: use English G2P
                        phonemes_list.append(en_callable(part))
                    else:
                        # Chinese token: use Chinese G2P
                        if part:
                            ph, _ = base_g2p(part)
                            phonemes_list.append(ph)
                result = "".join(phonemes_list)
                return result, text
            
            g2p = mixed_g2p
            voice = "zf_001"
            model_file = self._ensure_file("model", "kokoro-v1.1-zh.onnx")
            voices_file = self._ensure_file("voice", "voices-v1.1-zh.bin")
            vocab = self._get_vocab_config(lang)
        else:
            raise ValueError(f"Language '{lang}' not supported.")

        return model_file, voices_file, vocab, g2p, voice, en_callable

    def _create_kokoro(self, lang="en"):
        print(f"Loading Native PyTorch Kokoro pipeline for '{lang}'...")
        lang_code_map = {
            "en": "a", "hi": "h", "fr": "f", "it": "i", 
            "es": "e", "pt": "p", "ja": "j", "zh": "z"
        }
        code = lang_code_map.get(lang, "a")
        pipeline = KPipeline(lang_code=code)
        
        # Removed bfloat16 cast to prevent oneDNN LSTM driver crashes on Intel CPUs.
        # Kokoro (82M parameters) in float32 uses ~330MB RAM, which comfortably fits the 8GB budget.
        
        return pipeline

    def _generate_kokoro_audio(self, text, lang, voice, g2p):
        """Helper to generate Kokoro audio using PyTorch KPipeline."""
        if lang not in self.kokoro_cache:
            self.kokoro_cache[lang] = self._create_kokoro(lang)
            
        pipeline = self.kokoro_cache[lang]
        
        if g2p:
            phonemes, _ = g2p(text)
            pack = pipeline.load_voice(voice).to(pipeline.model.device)
            output = KPipeline.infer(pipeline.model, phonemes, pack, speed=1.0)
            return output.audio.cpu().numpy() if hasattr(output.audio, "cpu") else output.audio, 24000
        else:
            generator = pipeline(text, voice=voice, speed=0.9)
            all_audio = []
            for gs, ps, audio in generator:
                if audio is not None:
                    if hasattr(audio, "cpu"): audio = audio.cpu().numpy()
                    all_audio.append(audio)
            
            import numpy as np
            samples = np.concatenate(all_audio) if all_audio else np.array([])
            return samples, 24000

    def generate(self, text, lang, reference_audio, output_path="output.wav"):
        """Generates the speech and applies the target voice."""
        model_file, voices_file, vocab, g2p, voice, en_callable = self._get_config(lang)
        
        print(f"Synthesizing text ({lang.upper()})...")
        samples, sr = self._generate_kokoro_audio(text, lang, voice, g2p)

        # Use a secure temporary file for the base audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            sf.write(temp_path, samples, sr)

        # 2. Kanade Voice Conversion Phase
        try:
            print("Applying Voice Clone...")
            # Load and push to device
            source_wav = load_audio(temp_path, sample_rate=self.sample_rate)
            ref_wav = load_audio(reference_audio, sample_rate=self.sample_rate)

            if self.device.type == "cpu":
                pass
            elif self.device.type == "xpu":
                source_wav = source_wav.half()
                ref_wav = ref_wav.half()

            source_wav = source_wav.to(self.device)
            ref_wav = ref_wav.to(self.device)

            with torch.inference_mode():
                converted_wav = chunked_voice_conversion(
                    kanade=self.kanade,
                    vocoder_model=self.vocoder,
                    source_wav=source_wav,
                    ref_wav=ref_wav,
                    sample_rate=self.sample_rate
                )

            sf.write(output_path, converted_wav.cpu().numpy() if hasattr(converted_wav, "cpu") else converted_wav.numpy(), self.sample_rate)
            print(f"Success! Saved: {output_path}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path) # Clean up temp file silently

    def generate_pcm(self, text, lang, ref_wav_tensor):
        """Generates speech entirely in-memory. Returns numpy array at self.sample_rate."""
        model_file, voices_file, vocab, g2p, voice, en_callable = self._get_config(lang)

        import time
        _k_start = time.perf_counter()
        samples, sr = self._generate_kokoro_audio(text, lang, voice, g2p)
        _k_end = time.perf_counter()
        print(f"[Profiling] Kokoro TTS phase took {_k_end - _k_start:.2f}s")


        # 2. Direct tensor conversion (no disk I/O)
        source_wav = torch.from_numpy(samples).float()
        
        if self.device.type == "cpu":
            pass
        elif self.device.type == "xpu":
            if source_wav.device.type == "xpu":
                source_wav = source_wav.cpu()
            source_wav = source_wav.half()
            
            if ref_wav_tensor.device.type == "xpu":
                ref_wav_tensor = ref_wav_tensor.cpu()
            ref_wav_tensor = ref_wav_tensor.half()

        source_wav = source_wav.to(self.device)
        ref_wav_tensor = ref_wav_tensor.to(self.device)

        # Precompute or retrieve the global embedding to avoid redundant 2.5s encoding
        ref_key = id(ref_wav_tensor)
        if ref_key not in self.kanade_ref_cache:
            print("Caching Kanade reference embedding for the first time...")
            with torch.inference_mode():
                # Extract global embedding and cache it
                ref_features = self.kanade.encode(ref_wav_tensor, return_content=False, return_global=True)
                self.kanade_ref_cache[ref_key] = ref_features.global_embedding
        
        global_embedding = self.kanade_ref_cache[ref_key]

        # 3. Kanade Voice Conversion Phase
        with torch.inference_mode():
            converted_wav = chunked_voice_conversion(
                kanade=self.kanade,
                vocoder_model=self.vocoder,
                source_wav=source_wav,
                ref_wav=ref_wav_tensor,
                global_embedding=global_embedding,
                sample_rate=self.sample_rate
            )

        return converted_wav.float().numpy()

    def convert(self, source_audio, reference_audio, output_path="output.wav"):
        """Re-voices source_audio to sound like reference_audio using chunking."""
        print("Applying Voice Conversion...")
        # Load and push to device
        source_wav = load_audio(source_audio, sample_rate=self.sample_rate).to(self.device)
        ref_wav = load_audio(reference_audio, sample_rate=self.sample_rate).to(self.device)

        with torch.inference_mode():
            converted_wav = chunked_voice_conversion(
                kanade=self.kanade,
                vocoder_model=self.vocoder,
                source_wav=source_wav,
                ref_wav=ref_wav,
                sample_rate=self.sample_rate
            )

        sf.write(output_path, converted_wav.numpy(), self.sample_rate)
        print(f"Success! Saved: {output_path}")
