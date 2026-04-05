import os
import tempfile

import numpy as np
import torch
import soundfile as sf
from kanade_tokenizer import KanadeModel, load_audio, load_vocoder
from kokoro import KPipeline
from core.chunked_convert import chunked_voice_conversion

class KokoClone:
    def __init__(self, kanade_model="frothywater/kanade-12.5hz", kokoro_repo="hexgrad/Kokoro-82M"):
        # Auto-detect GPU (CUDA) or fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing KokoClone on: {self.device.type.upper()}")

        self.kokoro_repo = kokoro_repo

        # Load Kanade & Vocoder once, move to detected device
        print("Loading Kanade model...")
        self.kanade = KanadeModel.from_pretrained(kanade_model).to(self.device).eval()
        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device)
        self.sample_rate = self.kanade.config.sample_rate

        # Cache Kokoro pipelines by language code
        self.kokoro_pipeline_cache = {}

    def _get_config(self, lang):
        """Map public language codes to official Kokoro pipeline codes and default voices."""
        config = {
            "en": ("a", "af_bella"),
            "hi": ("h", "hf_alpha"),
            "fr": ("f", "ff_siwis"),
            "it": ("i", "im_nicola"),
            "es": ("e", "ef_dora"),
            "pt": ("p", "pf_dora"),
            "ja": ("j", "jf_alpha"),
            "zh": ("z", "zf_001"),
        }
        try:
            return config[lang]
        except KeyError as exc:
            raise ValueError(f"Language '{lang}' not supported.") from exc

    def _get_official_kokoro_pipeline(self, lang):
        """Return a cached official Kokoro pipeline for the requested language."""
        lang_code, _ = self._get_config(lang)
        if lang_code not in self.kokoro_pipeline_cache:
            self.kokoro_pipeline_cache[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id=self.kokoro_repo,
                device=self.device.type,
            )
        return self.kokoro_pipeline_cache[lang_code]

    def _synthesize_with_official_kokoro(self, pipeline, text, voice):
        """Use the upstream Kokoro pipeline for speech synthesis before voice conversion."""
        chunks = []
        for _, _, audio in pipeline(
            text,
            voice=voice,
            speed=1.0,
            split_pattern=r"\n+",
        ):
            chunks.append(np.asarray(audio, dtype=np.float32))

        if not chunks:
            raise RuntimeError("Official Kokoro pipeline produced no audio chunks.")

        return np.concatenate(chunks), 24000

    def generate(self, text, lang, reference_audio, output_path="output.wav"):
        """Generates the speech and applies the target voice."""
        _, voice = self._get_config(lang)

        # 1. Kokoro TTS Phase
        official_pipeline = self._get_official_kokoro_pipeline(lang)
        print(f"Synthesizing text ({lang.upper()}) with official Kokoro pipeline...")
        samples, sr = self._synthesize_with_official_kokoro(
            official_pipeline,
            text,
            voice,
        )
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
