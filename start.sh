#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMFYUI_DIR="${COMFYUI_DIR:-${SCRIPT_DIR}/ComfyUI}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export MODEL_DIR HF_HOME

# MPS environment (must be set before Python/torch imports)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN
fi

echo "============================================"
echo " ComfyUI Pipeline Server"
echo " Z-Image Turbo + Qwen-Image-Layered"
echo " + Qwen2.5-VL GGUF"
echo "============================================"
echo ""
echo "COMFYUI_DIR: ${COMFYUI_DIR}"
echo "MODEL_DIR:   ${MODEL_DIR}"
echo "HF_TOKEN:    ${HF_TOKEN:+set}"
echo "Chip:        $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
echo "Memory:      $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB", $1/1073741824}' || echo 'unknown')"
echo ""

# ── Verify ComfyUI is installed ──
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "ERROR: ComfyUI not found at ${COMFYUI_DIR}"
    echo "       Run ./setup.sh first."
    exit 1
fi

# ── Check models exist ──
check_model() {
    local path="$1"
    local name="$2"
    if [ -f "$path" ] || [ -L "$path" ]; then
        echo "[OK] ${name}"
    else
        echo "[!!] MISSING: ${name}"
        echo "     Expected at: ${path}"
        echo "     Run: uv run python download_models.py"
        MISSING_MODELS=1
    fi
}

echo "── Checking ComfyUI models ──"
check_model "${COMFYUI_DIR}/models/diffusion_models/z_image_turbo_bf16.safetensors" "Z-Image Turbo diffusion model"
check_model "${COMFYUI_DIR}/models/clip/qwen_3_4b.safetensors" "Qwen3-4B text encoder"
check_model "${COMFYUI_DIR}/models/vae/ae.safetensors" "Flux1 VAE"
check_model "${COMFYUI_DIR}/models/diffusion_models/qwen-image-layered-Q4_K_M.gguf" "Qwen-Image-Layered GGUF"
check_model "${COMFYUI_DIR}/models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors" "Qwen2.5-VL 7B FP8 text encoder"
check_model "${COMFYUI_DIR}/models/vae/qwen_image_layered_vae.safetensors" "Qwen-Image-Layered VAE"

echo ""
echo "── Checking VL GGUF models ──"
check_model "${MODEL_DIR}/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" "Qwen2.5-VL-7B GGUF"
check_model "${MODEL_DIR}/mmproj-BF16.gguf" "Qwen2.5-VL mmproj"

if [ "${MISSING_MODELS}" = "1" ]; then
    echo ""
    echo "Some models are missing. Run: uv run python download_models.py"
    echo "Then re-run this script."
    exit 1
fi

echo ""
echo "── Starting ComfyUI backend (port 8188) ──"

# Start ComfyUI in the background
cd "$COMFYUI_DIR"
# --listen 127.0.0.1: only accept local connections (wrapper server proxies)
# --port 8188: default ComfyUI port
# --force-fp32: safer on Apple Silicon (avoids BF16/FP16 MPS issues)
uv run python main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --force-fp32 \
    &
COMFYUI_PID=$!
cd "$SCRIPT_DIR"

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

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
        echo "ComfyUI is ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: ComfyUI did not start within 60 seconds."
        exit 1
    fi
    sleep 2
done

echo ""
echo "── Starting wrapper server (port 8000) ──"
echo "API endpoints:"
echo "  POST http://localhost:8000/generate          (full pipeline)"
echo "  POST http://localhost:8000/generate-image-only"
echo "  POST http://localhost:8000/decompose"
echo "  POST http://localhost:8000/analyze"
echo "  GET  http://localhost:8000/health"
echo ""

# Start the wrapper server (foreground — keeps the script alive)
exec uv run python server.py
