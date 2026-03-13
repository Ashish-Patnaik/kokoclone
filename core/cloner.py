import os
import tempfile
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode
from kokoro_onnx import Kokoro
from misaki import espeak
from misaki.espeak import EspeakG2P
from core.chunked_convert import chunked_voice_conversion

class KokoClone:
    def __init__(self, kanade_model="frothywater/kanade-12.5hz", hf_repo="PatnaikAshish/kokoclone"):
        # Auto-detect GPU (CUDA) or fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing KokoClone on: {self.device.type.upper()}")
        
        self.hf_repo = hf_repo
        
        # Load Kanade & Vocoder once, move to detected device
        print("Loading Kanade model...")
        self.kanade = KanadeModel.from_pretrained(kanade_model).to(self.device).eval()
        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device)
        self.sample_rate = self.kanade.config.sample_rate
        
        # Cache for Kokoro
        self.kokoro_cache = {}

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

    def _get_config(self, lang):
        """Routes the correct model, voice, and G2P based on language."""
        model_file = self._ensure_file("model", "kokoro.onnx")
        voices_file = self._ensure_file("voice", "voices-v1.0.bin")
        vocab = None
        g2p = None

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
                subprocess.run(["python", "-m", "unidic", "download"], check=True)
                
            g2p = ja.JAG2P()
            voice = "jf_alpha"
            vocab = self._ensure_file("model", "config.json")
        elif lang == "zh":
            from misaki import zh
            g2p = zh.ZHG2P(version="1.1")
            voice = "zf_001"
            model_file = self._ensure_file("model", "kokoro-v1.1-zh.onnx")
            voices_file = self._ensure_file("voice", "voices-v1.1-zh.bin")
            vocab = self._ensure_file("model", "config.json")
        else:
            raise ValueError(f"Language '{lang}' not supported.")

        return model_file, voices_file, vocab, g2p, voice

    def generate(self, text, lang, reference_audio, output_path="output.wav"):
        """Generates the speech and applies the target voice."""
        model_file, voices_file, vocab, g2p, voice = self._get_config(lang)
        
        # 1. Kokoro TTS Phase
        if model_file not in self.kokoro_cache:
            self.kokoro_cache[model_file] = Kokoro(model_file, voices_file, vocab_config=vocab) if vocab else Kokoro(model_file, voices_file)
        
        kokoro = self.kokoro_cache[model_file]
        
        print(f"Synthesizing text ({lang.upper()})...")
        if g2p:
            phonemes, _ = g2p(text)
            samples, sr = kokoro.create(phonemes, voice=voice, speed=1.0, is_phonemes=True)
        else:
            samples, sr = kokoro.create(text, voice=voice, speed=0.9, lang="en-us")

        # Use a secure temporary file for the base audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            sf.write(temp_path, samples, sr)

        # 2. Kanade Voice Conversion Phase
        try:
            print("Applying Voice Clone...")
            # Load and push to device
            source_wav = load_audio(temp_path, sample_rate=self.sample_rate).to(self.device)
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

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path) # Clean up temp file silently

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
