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


## Run container

Build the container via

```powershell
docker build -t wake-movies .
```

Run & start the container
```powershell
docker run -d \
  --name waki-movies \
  --restart unless-stopped \
  --env-file .env \
  -v "$PWD/src/training/models:/app/src/training/models" \ # Model-Folder get's mounted
  -v "$PWD/src/data:/app/src/data" \ # Data-Folder get's mounted
  waki-movies:latest
```



## Dataset
[MovieLens-25m-dataset](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset/data)
