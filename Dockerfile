FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PLATFORM=runpod_cuda \
    COMFYUI_MODELS_DIR=/runpod-volume/comfyui-models \
    HF_HOME=/runpod-volume/huggingface \
    PATH="/root/.local/bin:$PATH"

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential cmake curl git ca-certificates && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Clone ComfyUI and install deps with CUDA PyTorch
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui && \
    cd /comfyui && \
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# Clone ComfyUI-GGUF custom node and install deps
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    cd ComfyUI-GGUF && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /app

# Copy lockfile first for layer caching
COPY pyproject.toml uv.lock ./

# Install wrapper server deps
RUN uv sync --frozen --no-dev

# Copy app code
COPY server.py ./
COPY workflows/ ./workflows/
COPY start-runpod.sh ./
RUN chmod +x start-runpod.sh

EXPOSE 8000 8188

CMD ["./start-runpod.sh"]
