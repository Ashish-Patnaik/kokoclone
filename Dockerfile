FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    espeak-ng \
    libespeak-ng1 \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install torch torchaudio --index-url ${TORCH_INDEX_URL} && \
    pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
