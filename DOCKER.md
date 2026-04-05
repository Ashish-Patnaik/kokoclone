# KokoClone Docker setup

This fork includes a repo-root Docker workflow so the project can be cloned and started without any outer wrapper folder.

Run these commands from the repository root (`kokoclone/`).

## Start

### CPU / widest compatibility

```powershell
docker compose up --build -d
```

### NVIDIA GPU

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Then open: <http://localhost:7860>

## Stop

```powershell
docker compose down
```

## Logs

```powershell
docker compose logs -f kokoclone
```

## Notes

- The first launch downloads the model weights from Hugging Face.
- Downloaded Kokoro and Kanade assets are cached in the `huggingface-cache` volume.
- `docker-compose.yml` is the **CPU-safe default**.
- `docker-compose.gpu.yml` adds the NVIDIA GPU override (`gpus: all`) and CUDA PyTorch build args.
- The repository root is mounted at `/workspace`; place CLI input files inside the repo (for example `./inputs/`) if you want to access them from inside the container.
