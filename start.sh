#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/runpod-volume/models}"
export MODEL_DIR

echo "============================================"
echo " Z-Image Turbo + Qwen-Image-Layered GGUF"
echo " + Qwen2.5-VL GGUF Pipeline Server"
echo "============================================"
echo ""
echo "MODEL_DIR: ${MODEL_DIR}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

# ── Download models if not already present ──
download_if_missing() {
    local repo="$1"
    local file="$2"
    local dest="${MODEL_DIR}/${file}"

    if [ -f "$dest" ]; then
        echo "[OK] ${file} already exists"
    else
        echo "[DL] Downloading ${file} from ${repo} ..."
        uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='${repo}', filename='${file}', local_dir='${MODEL_DIR}')
"
        echo "[OK] ${file} downloaded"
    fi
}

mkdir -p "${MODEL_DIR}/split_files/vae"

echo ""
echo "── Checking models ──"

# Qwen2.5-VL GGUF (vision-language analysis)
download_if_missing "unsloth/Qwen2.5-VL-7B-Instruct-GGUF" "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
download_if_missing "unsloth/Qwen2.5-VL-7B-Instruct-GGUF" "mmproj-BF16.gguf"

# Qwen-Image-Layered GGUF (layer decomposition)
download_if_missing "unsloth/Qwen-Image-Layered-GGUF" "qwen-image-layered-Q4_K_M.gguf"

# Qwen-Image-Layered VAE
download_if_missing "Comfy-Org/Qwen-Image-Layered_ComfyUI" "split_files/vae/qwen_image_layered_vae.safetensors"

echo ""
echo "── All models ready ──"
echo ""

# ── Start the server ──
echo "Starting server on 0.0.0.0:8000 ..."
exec uv run python server.py
