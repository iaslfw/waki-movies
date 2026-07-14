# WaKi-Movies

## Setup

Use the default PyTorch build on Mac or machines without a dedicated NVIDIA GPU:

```powershell
uv sync
```

This repository intentionally does not commit `uv.lock`. The CUDA training
machine replaces the default PyTorch package with a CUDA-specific build, and a
committed lock file would make it too easy to sync back to the wrong Torch
variant. `uv sync` may create a local ignored `uv.lock`.

Create `.env` from `.env.template`. The Hugging Face dataset/model repos and
the Mistral model are project defaults and are already filled in there.
Telegram and Mistral API tokens are personal and must be added by the person
running the bot:

```powershell
Copy-Item .env.template .env
```

```env
TELEGRAM_API_TOKEN=your_telegram_bot_token
MISTRAL_API_KEY=your_mistral_api_key
```

## Fresh clone

A fresh clone does not contain generated data or model artifacts:

- `src/data/cleaned_movie_data.csv`
- `src/data/movie_tags.json`
- `src/training/models/final_model`

If those files are missing, `main.py` recreates or downloads them from the
configured Hugging Face repos:

```env
HF_REPO_ID=iaslfw/waki-movie_raw
HF_MODEL_ID=iaslfw/waki-movie_model
```

## Train model

On the CUDA training machine, sync first, then install the CUDA 13.0 PyTorch
build and verify that PyTorch can see the GPU:

```powershell
uv sync
uv pip install torch --torch-backend=cu130 --reinstall
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use `--no-sync` for training runs on that machine. Otherwise `uv run` can sync
the environment again and replace the manually installed CUDA build with the
default build.

```powershell
uv run --no-sync python -m src.training.train_model
```

Selected training experiments can be started from PowerShell via Git Bash:

```powershell
& "C:\Program Files\Git\bin\bash.exe" "./run_selected_experiments.sh"
```

## Tests and quality checks

Install the dev tools and run the regression tests:

```powershell
uv sync --group dev
uv run pytest
```

## Run container

Build the container via

```powershell
docker build -t waki-movies .
```

Run & start the container

```powershell
docker run -d `
  --name waki-movies `
  --restart unless-stopped `
  --env-file .env `
  -v "${PWD}/src/training/models:/app/src/training/models" `
  -v "${PWD}/src/data:/app/src/data" `
  waki-movies:latest
```