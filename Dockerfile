FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   git \
	   curl \
	   ca-certificates \
	   libsndfile1 \
	   libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY . .

RUN uv sync --yes || uv sync --locked || true

CMD ["uv", "run", "main.py"]

