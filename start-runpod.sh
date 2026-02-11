#!/bin/bash
set -e

COMFYUI_DIR="${COMFYUI_DIR:-/comfyui}"
COMFYUI_MODELS_DIR="${COMFYUI_MODELS_DIR:-/runpod-volume/comfyui-models}"
HF_HOME="${HF_HOME:-/runpod-volume/huggingface}"
export HF_HOME

if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN
fi

echo "============================================"
echo " ComfyUI Pipeline Server (RunPod CUDA)"
echo " Z-Image Turbo + Qwen-Image-Layered"
echo "============================================"
echo ""
echo "COMFYUI_DIR:       ${COMFYUI_DIR}"
echo "COMFYUI_MODELS_DIR: ${COMFYUI_MODELS_DIR}"
echo "HF_TOKEN:          ${HF_TOKEN:+set}"
echo ""
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo ""

# ── Download models if not already present ──
download_if_missing() {
    local repo="$1"
    local hf_filename="$2"
    local dest_dir="$3"
    local dest_name="$4"
    local dest="${dest_dir}/${dest_name}"

    if [ -f "$dest" ]; then
        echo "[OK] ${dest_name} already exists"
    else
        echo "[DL] Downloading ${dest_name} from ${repo} ..."
        mkdir -p "${dest_dir}"
        uv run python -c "
from huggingface_hub import hf_hub_download
import os, shutil
path = hf_hub_download(repo_id='${repo}', filename='${hf_filename}', token=os.environ.get('HF_TOKEN'))
os.makedirs('${dest_dir}', exist_ok=True)
shutil.copy2(path, '${dest}')
"
        echo "[OK] ${dest_name} downloaded"
    fi
}

# Create model directories
mkdir -p "${COMFYUI_MODELS_DIR}/diffusion_models"
mkdir -p "${COMFYUI_MODELS_DIR}/clip"
mkdir -p "${COMFYUI_MODELS_DIR}/vae"
mkdir -p "${HF_HOME}"

echo ""
echo "── Checking ComfyUI models (stages 1 & 2) ──"

# Stage 1: Z-Image Turbo
download_if_missing \
    "Comfy-Org/z_image_turbo" \
    "split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
    "${COMFYUI_MODELS_DIR}/diffusion_models" \
    "z_image_turbo_bf16.safetensors"

download_if_missing \
    "Comfy-Org/z_image_turbo" \
    "split_files/text_encoders/qwen_3_4b.safetensors" \
    "${COMFYUI_MODELS_DIR}/clip" \
    "qwen_3_4b.safetensors"

download_if_missing \
    "Comfy-Org/z_image_turbo" \
    "split_files/vae/ae.safetensors" \
    "${COMFYUI_MODELS_DIR}/vae" \
    "ae.safetensors"

# Stage 2: Qwen-Image-Layered
download_if_missing \
    "unsloth/Qwen-Image-Layered-GGUF" \
    "qwen-image-layered-Q4_K_M.gguf" \
    "${COMFYUI_MODELS_DIR}/diffusion_models" \
    "qwen-image-layered-Q4_K_M.gguf"

download_if_missing \
    "Comfy-Org/HunyuanVideo_1.5_repackaged" \
    "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "${COMFYUI_MODELS_DIR}/clip" \
    "qwen_2.5_vl_7b_fp8_scaled.safetensors"

download_if_missing \
    "Comfy-Org/Qwen-Image-Layered_ComfyUI" \
    "split_files/vae/qwen_image_layered_vae.safetensors" \
    "${COMFYUI_MODELS_DIR}/vae" \
    "qwen_image_layered_vae.safetensors"

echo ""
echo "── All models ready ──"

# ── Generate extra_model_paths.yaml for ComfyUI ──
cat > /tmp/extra_model_paths.yaml <<EOF
runpod_volume:
    base_path: ${COMFYUI_MODELS_DIR}
    diffusion_models: diffusion_models/
    clip: clip/
    vae: vae/
EOF

echo ""
echo "── Starting ComfyUI backend (port 8188) ──"

cd "$COMFYUI_DIR"
python main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --extra-model-paths-config /tmp/extra_model_paths.yaml \
    &
COMFYUI_PID=$!
cd /app

echo "ComfyUI started (PID: ${COMFYUI_PID})"

# Cleanup: kill ComfyUI when this script exits
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $COMFYUI_PID 2>/dev/null || true
    wait $COMFYUI_PID 2>/dev/null || true
    echo "ComfyUI stopped."
}
trap cleanup EXIT INT TERM

# Wait for ComfyUI to be ready (longer timeout for first-boot model loading)
echo "Waiting for ComfyUI to be ready..."
for i in $(seq 1 240); do
    if curl -s http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
        echo "ComfyUI is ready."
        break
    fi
    if [ "$i" -eq 240 ]; then
        echo "ERROR: ComfyUI did not start within 240 seconds."
        exit 1
    fi
    sleep 1
done

echo ""
echo "── Starting wrapper server (port 8000) ──"
echo "API endpoints:"
echo "  POST http://localhost:8000/generate          (full pipeline)"
echo "  POST http://localhost:8000/generate-image-only"
echo "  POST http://localhost:8000/decompose"
echo "  GET  http://localhost:8000/health"
echo ""

# Start the wrapper server (foreground — keeps the script alive)
exec uv run python server.py
