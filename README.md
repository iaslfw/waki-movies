# WaKi-Movies

## Setup

Use the default PyTorch build on Mac or machines without a dedicated NVIDIA GPU:

```powershell
uv sync
```

On the CUDA training machine, install the CUDA 13.0 PyTorch build after syncing:

```powershell
uv sync
uv pip install torch --torch-backend=cu130 --reinstall
```

## Dataset
[MovieLens-25m-dataset](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset/data)
