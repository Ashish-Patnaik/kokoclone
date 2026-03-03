# 🎙️ KokoClone

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/PatnaikAshish/kokoclone)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Models-Repository-orange)](https://huggingface.co/PatnaikAshish/kokoclone)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python\&logoColor=white)]
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 🧠 What is KokoClone?

**KokoClone** is a fast, real-time compatible multilingual voice cloning system built on top of **Kokoro-ONNX**, one of the fastest open-source neural TTS engines available today.

It allows you to:

* Type text in multiple languages
* Provide a short 3–10 second reference audio clip
* Instantly generate speech in that same voice


Just text → voice → cloned output.


## ⚡ Why Kokoro?

KokoClone is powered by **Kokoro-ONNX**, a highly optimized neural TTS engine designed for:

* ⚡ Extremely fast inference
* 🎧 Natural prosody and expressive speech
* 🧩 Lightweight ONNX runtime compatibility
* 🖥️ Real-time deployment on CPU
* 🚀 Even faster performance with GPU

Unlike many heavy TTS systems, Kokoro is lightweight and responsive — making KokoClone suitable for real-time applications, voice assistants, demos, and interactive tools.


## ✨ Features

### 🌍 Multilingual Speech Generation

Generate native speech in:

* English (`en`)
* Hindi (`hi`)
* French (`fr`)
* Japanese (`ja`)
* Chinese (`zh`)
* Italian (`it`)
* Portuguese (`pt`)
* Spanish (`es`)

### 🎯 Zero-Shot Voice Cloning

Upload a short voice sample and KokoClone transfers its vocal characteristics to the generated speech.

### ⚡ Real-Time Friendly

Built on Kokoro’s efficient ONNX runtime pipeline, KokoClone runs smoothly on:

* Standard laptops (CPU)
* Workstations (GPU)

### 📦 Automatic Model Handling

On first run, required model files are downloaded automatically and placed in the correct directories.

### 🎨 Built-in Web Interface

Includes a clean and responsive Gradio UI for quick testing and demos.



## 🚀 Live Demo

Try it instantly without installing anything:

👉 **[KokoClone on Hugging Face Spaces](https://huggingface.co/spaces/PatnaikAshish/kokoclone)**



## 🛠 Installation

Recommended: Use `conda` for a clean environment.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ashish-Patnaik/kokoclone.git
cd kokoclone
```

### 2️⃣ Create Environment

```bash
conda create -n kokoclone python=3.12.12 -y
conda activate kokoclone
```



## 📦 Install Dependencies

### 🔹 CPU Installation (Recommended for most users)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 🔹 GPU Installation (NVIDIA users)

```bash
pip install -r requirements.txt
pip install kokoro-onnx[gpu]
```



## 🎮 Usage

KokoClone can be used in three ways:



### 1️⃣ Web Interface

Launch the Gradio app:

```bash
python app.py
```

Then open the browser interface to:

* Enter text
* Select language
* Upload a reference voice
* Generate cloned speech



### 2️⃣ Command Line

```bash
python cli.py --text "Hello from KokoClone" --lang en --ref reference.wav --out output.wav
```



### 3️⃣ Python API

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



## 📂 Project Structure

```
app.py              → Gradio Web Interface
cli.py              → Command-line tool
core/cloner.py      → Core inference engine
inference.py        → Example usage script
model/              → Downloaded TTS model weights
voice/              → Voice embeddings
```



## 🎯 Use Cases

* Voice assistant prototypes
* Real-time TTS demos
* Multilingual narration tools
* Content creation
* Research experiments
* Interactive AI applications



## 🙏 Acknowledgments

This project builds upon:

* **Kokoro-ONNX** — for fast and efficient neural speech synthesis
* **Kanade Tokenizer** — for voice conversion architecture


## 📄 License

Licensed under the Apache 2.0 License.
