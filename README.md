<p align="center">
  <img width="1050" height="450" alt="KokoClone Banner" src="https://github.com/user-attachments/assets/26fbb00c-220e-435a-8f54-431781449c76" />
</p>

<h1 align="center">🎙️ KokoClone</h1>

<p align="center">
  <a href="https://huggingface.co/spaces/PatnaikAshish/kokoclone">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue" alt="Hugging Face Space" />
  </a>
  <a href="https://huggingface.co/PatnaikAshish/kokoclone">
    <img src="https://img.shields.io/badge/🤗%20Models-Repository-orange" alt="Hugging Face Models" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%20to%203.12-3776AB.svg?logo=python&logoColor=white" alt="Python" />
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License" />
  </a>
</p>

**KokoClone** is a fast, real-time compatible multilingual voice cloning system built on top of **Kokoro-ONNX**, one of the fastest open-source neural TTS engines available today.

It allows you to:
* **Text → Clone:** Type text in multiple languages, provide a short reference audio clip, and instantly generate speech in that same voice.
* **Audio → Clone:** Re-voice an existing audio recording to sound like any reference speaker — *no transcription needed*.


## Features

### Multilingual Speech Generation
Generate native speech in English (`en`), Hindi (`hi`), French (`fr`), Japanese (`ja`), Chinese (`zh`), Italian (`it`), Portuguese (`pt`), and Spanish (`es`).

### Zero-Shot Voice Cloning
Upload a 3–10 second voice sample and KokoClone instantly transfers its vocal characteristics to the generated speech.

### Audio-to-Audio Voice Conversion
Upload any existing speech recording and re-voice it to sound like a reference speaker. The pipeline skips TTS entirely and runs purely through the Kanade voice-conversion model. Works on recordings of any length thanks to automatic VRAM-aware chunking!

### Automatic Model Handling
On the first run, the required model weights (`.onnx` and `.bin` files) are automatically downloaded from Hugging Face and placed in the correct directories.

### Real-Time Friendly
Built on Kokoro's efficient ONNX runtime pipeline, KokoClone detects your hardware and runs smoothly on both standard laptops (CPU) and workstations (GPU).


## Live Demo
Try it instantly without installing anything:  
👉 **[KokoClone on Hugging Face Spaces](https://huggingface.co/spaces/PatnaikAshish/kokoclone)**


## Installation

You can set up KokoClone using either **Conda** (Recommended) or **uv**.

### 1. Clone the Repository
```bash
git clone https://github.com/Ashish-Patnaik/kokoclone.git
cd kokoclone

```

### 2. Set Up the Environment & Install Dependencies

#### Option A: Using Conda (Recommended)

```bash
conda create -n kokoclone python=3.12.12 -y
conda activate kokoclone

```

**For CPU Users (Mac / Standard Laptops):**

```bash
pip install torch torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt

```

**For GPU Users (Nvidia GPUs):**

```bash
pip install -r requirements.txt
pip install kokoro-onnx[gpu]

```

#### Option B: Using `uv`

If you prefer [uv](https://docs.astral.sh/uv/) for fast package management:

```bash
# For CPU Users
uv sync

# For GPU Users (Nvidia)
uv sync --extra gpu

# Activate the environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

```



##  Usage

KokoClone is highly flexible and can be used via Web UI, CLI, or Python API.

### 1. Web Interface (Gradio)

Launch the interactive web app:

```bash
python app.py

```

* **Tab 1 (Text → Clone):** Enter text, pick a language, upload a reference voice, and generate.
* **Tab 2 (Audio → Clone):** Upload source audio and a reference voice, and get back re-voiced audio.

### 2. Command Line Interface (CLI)

Generate speech directly from your terminal.

**Text to cloned speech (default mode):**

```bash
python cli.py --text "Hello from KokoClone" --lang en --ref reference.wav --out output.wav

```

**Audio to re-voiced speech:**

```bash
python cli.py --mode convert --source original_speech.wav --ref target_voice.wav --out revoiced.wav

```

| Argument | Default | Description |
| --- | --- | --- |
| `--mode` | `tts` | `tts` (text → speech) or `convert` (audio → re-voiced audio) |
| `--text` | — | Text to synthesize *(required for `tts` mode)* |
| `--lang` | `en` | Language code: `en hi fr ja zh it es pt` |
| `--source` | — | Path to source audio *(required for `convert` mode)* |
| `--ref` | — | Path to reference voice audio *(always required)* |
| `--out` | `output.wav` | Output file path |

### 3. Python API

Integrate KokoClone into your own Python applications.

**Text to Cloned Speech:**

```python
from core.cloner import KokoClone

cloner = KokoClone()
cloner.generate(
    text="This voice is cloned using KokoClone.",
    lang="en",
    reference_audio="reference.wav",
    output_path="output.wav"
)

```

**Audio-to-Audio Voice Conversion:**

```python
import soundfile as sf
from kanade_tokenizer import load_audio
from core.cloner import KokoClone
from core.chunked_convert import chunked_voice_conversion

cloner = KokoClone()

# Load audio tensors
source_wav = load_audio("source_speech.wav", sample_rate=cloner.sample_rate).to(cloner.device)
ref_wav = load_audio("target_voice.wav", sample_rate=cloner.sample_rate).to(cloner.device)

# Convert using VRAM-aware chunking
converted = chunked_voice_conversion(
    kanade=cloner.kanade,
    vocoder_model=cloner.vocoder,
    source_wav=source_wav,
    ref_wav=ref_wav,
    sample_rate=cloner.sample_rate,
)

sf.write("revoiced_output.wav", converted.numpy(), cloner.sample_rate)

```


## Memory Management for Long Audio

The `chunked_voice_conversion` function in `core/chunked_convert.py` handles memory automatically when converting long audio recordings:

* **VRAM Budget:** On CUDA, chunks are sized so each forward pass uses at most 50% of total GPU memory (configurable via the `vram_fraction` parameter).
* **RoPE Ceiling:** The Kanade `mel_decoder` Transformer has positional embeddings precomputed for 1,024 mel frames. Chunk windows are hard-capped below this limit (≈ 8.9s of source audio per chunk) with a 10% safety margin to prevent recomputation and quality degradation.
* **Overlap Smoothing:** Each chunk includes a 0.5s overlap on both sides to suppress boundary artifacts.
* **Single-Pass Vocoding:** The full reassembled mel spectrogram is passed to the vocoder in one shot for clean waveform reconstruction.


## Project Structure

```text
app.py                → Gradio Web Interface (two-tab UI)
cli.py                → Command-line tool (tts and convert modes)
inference.py          → Example API usage script
core/
 ├── cloner.py        → Core TTS + voice cloning engine
 └── chunked_convert.py → VRAM-aware chunked audio conversion
model/                → Downloaded Kokoro model weights (Auto-populates)
voice/                → Downloaded Kokoro voice bins (Auto-populates)

```


## Acknowledgments

This project builds upon the incredible open-source work of:

* **[Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx)** — for fast and efficient neural speech synthesis.
* **[Kanade Tokenizer](https://github.com/frothywater/kanade-tokenizer)** — for the brilliant zero-shot voice conversion architecture.

## License

Licensed under the [Apache 2.0 License](https://www.google.com/search?q=LICENSE).

```
