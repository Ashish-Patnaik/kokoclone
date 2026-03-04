<img width="1050" height="450" alt="image" src="https://github.com/user-attachments/assets/26fbb00c-220e-435a-8f54-431781449c76" />


<h1 align="center">KokoClone</h1>

<p align="center">
  <a href="https://huggingface.co/spaces/PatnaikAshish/kokoclone">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue" alt="Hugging Face Space" />
  </a>
  <a href="https://huggingface.co/PatnaikAshish/kokoclone">
    <img src="https://img.shields.io/badge/🤗%20Models-Repository-orange" alt="Hugging Face Models" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python" />
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License" />
  </a>
</p>



##  What is KokoClone?

**KokoClone** is a fast, real-time compatible multilingual voice cloning system built on top of **Kokoro-ONNX**, one of the fastest open-source neural TTS engines available today.

It allows you to:

* Type text in multiple languages, provide a short reference audio clip, and instantly generate speech in that same voice
* Re-voice an existing audio recording to sound like any reference speaker — no transcription needed

Just text → voice → cloned output. 

Existing audio → reference voice → re-voiced output.



## Why Kokoro?

KokoClone is powered by **Kokoro-ONNX**, a highly optimized neural TTS engine designed for:

* Extremely fast inference
* Natural prosody and expressive speech
* Lightweight ONNX runtime compatibility
* Real-time deployment on CPU
* Even faster performance with GPU

Unlike many heavy TTS systems, Kokoro is lightweight and responsive — making KokoClone suitable for real-time applications, voice assistants, demos, and interactive tools.


## Features

### Multilingual Speech Generation

Generate native speech in:

* English (`en`)
* Hindi (`hi`)
* French (`fr`)
* Japanese (`ja`)
* Chinese (`zh`)
* Italian (`it`)
* Portuguese (`pt`)
* Spanish (`es`)


###  Zero-Shot Voice Cloning

Upload a short voice sample and KokoClone transfers its vocal characteristics to the generated speech.


### Audio-to-Audio Voice Conversion (Audio → Clone)

Upload any existing speech recording and re-voice it to sound like a reference speaker — **no transcription needed**. The pipeline skips TTS entirely and runs purely through the Kanade voice-conversion model.

Works on recordings of any length: audio is automatically split into VRAM-aware chunks, processed sequentially, and seamlessly reassembled.


### Real-Time Friendly

Built on Kokoro's efficient ONNX runtime pipeline, KokoClone runs smoothly on:

* Standard laptops (CPU)
* Workstations (GPU)


### Automatic Model Handling

On first run, required model files are downloaded automatically and placed in the correct directories.


### Built-in Web Interface

Includes a clean and responsive Gradio UI for quick testing and demos.

* **🎤 Text → Clone** — enter text, pick a language, upload a reference voice
* **🔁 Audio → Clone** — upload source audio and a reference voice, get back re-voiced audio
* **🎤 Text → Clone** — enter text, pick a language, upload a reference voice
* **🔁 Audio → Clone** — upload source audio and a reference voice, get back re-voiced audio


##  Live Demo

Try it instantly without installing anything:

👉 **[KokoClone on Hugging Face Spaces](https://huggingface.co/spaces/PatnaikAshish/kokoclone)**



## Installation

Recommended: Use `conda` for a clean environment.

or

with **[uv](https://docs.astral.sh/uv/)**, a fast Python package and project manager.

```bash
# macOS / Linux <- preferred method

### Prerequisites

* Python 3.13+
* [uv](https://docs.astral.sh/uv/) — install with:

```bash
# macOS / Linux <- preferred method

curl -LsSf https://astral.sh/uv/install.sh | sh
# or via pip
pip install uv
```

### Clone the Repository

```bash
git clone https://github.com/Ashish-Patnaik/kokoclone.git
cd kokoclone
```

###  Create Environment
With conda
```bash
conda create -n kokoclone python=3.12.12 -y
conda activate kokoclone
```

With uv
```bash
# Create a virtual environment and install all dependencies from pyproject.toml
uv sync

# Activate the environment (optional — uv run handles activation automatically)
source .venv/bin/activate # Linux / macOS
.venv\Scripts\activate # Windows
```



##  Install Dependencies

###  CPU Installation (Recommended for most users)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

###  GPU Installation (NVIDIA users)

```bash
pip install -r requirements.txt
pip install kokoro-onnx[gpu]
```
### GPU Support (NVIDIA)

The default install uses the CPU build of PyTorch. For GPU:

```bash
uv sync --extra gpu
```


## Usage

KokoClone can be used in three ways:



###  Web Interface

Launch the Gradio app:

```bash
python app.py
```

Then open the browser interface to:

**Tab 1 — 🎤 Text → Clone**
1. Enter the text to synthesize
2. Select a language
3. Upload or record a 3–10 second reference voice clip
4. Click **Generate Clone**

**Tab 2 — 🔁 Audio → Clone**
1. Upload or record the source audio you want to re-voice
2. Upload or record a 3–10 second reference voice clip (target speaker)
3. Click **Convert Voice**


### Command Line

**Text to cloned speech (default mode):**

```bash
python cli.py --text "Hello from KokoClone" --lang en --ref reference.wav --out output.wav
```



| Argument | Default | Description |
|---|---|---|
| `--mode` | `tts` | `tts` (text → speech) or `convert` (audio → re-voiced audio) |
| `--text` | — | Text to synthesize *(required for `tts` mode)* |
| `--lang` | `en` | Language code: `en hi fr ja zh it es pt` |
| `--source` | — | Path to source audio *(required for `convert` mode)* |
| `--ref` | — | Path to reference voice audio *(always required)* |
| `--out` | `output.wav` | Output file path |


### Python API

**Text to cloned speech:**

```python
from core.cloner import KokoClone

cloner = KokoClone()

cloner.generate(
 text=\"This voice is cloned using KokoClone.\",
 lang=\"en\",
 reference_audio=\"reference.wav\",
 output_path=\"output.wav\"
)
```

**Audio-to-audio voice conversion:**

```python
import soundfile as sf
from kanade_tokenizer import load_audio
from core.cloner import KokoClone
from core.chunked_convert import chunked_voice_conversion

##  Project Structure

```
## Project Structure

```
app.py → Gradio Web Interface (two-tab UI)
cli.py → Command-line tool (tts and convert modes)
core/
 cloner.py → Core TTS + voice cloning engine
 chunked_convert.py → VRAM-aware chunked audio-to-audio conversion
inference.py → Example usage script
model/ → Downloaded TTS model weights
voice/ → Voice embeddings
docs/
 plans/ → Design documents
```

## Memory Management for Long Audio

The `chunked_voice_conversion` function in `core/chunked_convert.py` handles memory automatically when converting long recordings:

* **VRAM budget**: on CUDA, chunks are sized so each forward pass uses at most 50 % of total GPU memory (configurable via the `vram_fraction` parameter).
* **RoPE ceiling**: the Kanade `mel_decoder` Transformer has positional embeddings precomputed for 1,024 mel frames (`hop_length = 256` at 24 kHz ≈ 10.9 s of audio per window). Chunk windows are hard-capped below this limit (≈ 8.9 s of source audio per chunk) with a 10 % safety margin to prevent recomputation and quality degradation.
* **Overlap smoothing**: each chunk includes a 0.5 s overlap on both sides to suppress boundary artefacts. Overlap mel frames (`overlap_samples // hop_length = 46 frames`) are trimmed before concatenation.
* **Single-pass vocoding**: the full reassembled mel spectrogram is passed to the vocoder in one shot for clean waveform reconstruction.

Progress and timing are logged to stdout:

```
[chunked_convert] VRAM budget: 11.76 GB (75% of 23.53 GB) → chunk size: 8.9s / 214,099 samples (RoPE ceiling: 8.9s)
[chunked_convert] Completed in 38.2s
```



## Use Cases

* Voice assistant prototypes
* Real-time TTS demos
* Multilingual narration tools
* Content creation and dubbing
* Re-voicing long recordings or podcasts
* Research experiments
* Interactive AI applications



## Acknowledgments

This project builds upon:

* **Kokoro-ONNX** — for fast and efficient neural speech synthesis
* **Kanade Tokenizer** — for voice conversion architecture


## License

Licensed under the Apache 2.0 License.
