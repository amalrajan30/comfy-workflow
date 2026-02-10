FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/runpod-volume/models \
    HF_HOME=/runpod-volume/huggingface \
    PATH="/root/.local/bin:$PATH"

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential cmake curl git ca-certificates && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy lockfile first for layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy app code
COPY server.py download_models.py test_client.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]
