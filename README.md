# WaKi-Movies

## Setup

Use the default PyTorch build on Mac or machines without a dedicated NVIDIA GPU:

```powershell
uv sync
```

Mistral is required for chat routing, query cleanup, localization, and
recommendation reply formatting. Add these values to `.env`:

```env
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-small-latest
```

On the CUDA training machine, sync first, then install the CUDA 13.0 PyTorch
build and verify that PyTorch can see the GPU:

```powershell
uv sync
uv pip install torch --torch-backend=cu130 --reinstall
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use `--no-sync` for training runs on that machine. Otherwise `uv run` can sync
from `uv.lock` again and replace the manually installed CUDA build with the CPU
build.

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

Useful focused commands:

```powershell
uv run pytest tests/test_dataset_splits.py
uv run pytest tests/test_similarity.py
uv run ruff check src tests main.py
uv run ruff format src tests main.py
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



## Dataset
[MovieLens-25m-dataset](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset/data)
