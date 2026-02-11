#!/bin/bash
set -e

# ── ComfyUI Pipeline Setup Script ──
# One-time setup: installs ComfyUI and dependencies.
# Supports Apple Silicon (MPS/Metal) and Linux (CUDA).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMFYUI_DIR="${COMFYUI_DIR:-${SCRIPT_DIR}/ComfyUI}"

echo "============================================"
echo " ComfyUI Pipeline Setup"
echo " Z-Image Turbo + Qwen-Image-Layered"
echo "============================================"
echo ""

# ── Detect platform ──
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    PLATFORM="apple_silicon"
elif [ "$OS" = "Linux" ]; then
    if command -v nvidia-smi &>/dev/null; then
        PLATFORM="linux_cuda"
    else
        PLATFORM="linux_cpu"
    fi
else
    PLATFORM="other"
fi
echo "[OK] Detected platform: ${PLATFORM} (${OS} ${ARCH})"

# ── Check prerequisites ──
echo ""
echo "── Checking prerequisites ──"

if [ "$OS" = "Darwin" ]; then
    if ! xcode-select -p &>/dev/null; then
        echo "[!] Xcode Command Line Tools not found. Installing..."
        xcode-select --install
        echo "    Please re-run this script after installation completes."
        exit 1
    fi
    echo "[OK] Xcode Command Line Tools installed"
fi

if ! command -v uv &>/dev/null; then
    echo "[!] uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[OK] uv: $(uv --version)"

if ! command -v git &>/dev/null; then
    echo "ERROR: git is required."
    exit 1
fi
echo "[OK] git available"

echo ""
echo "── Installing ComfyUI ──"

if [ -d "$COMFYUI_DIR" ]; then
    echo "[OK] ComfyUI already cloned at ${COMFYUI_DIR}"
    cd "$COMFYUI_DIR" && git pull --ff-only 2>/dev/null || true
    cd "$SCRIPT_DIR"
else
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

echo ""
echo "── Setting up ComfyUI Python environment ──"

cd "$COMFYUI_DIR"

# Create venv if it doesn't exist
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    uv venv --python 3.11
fi

# Install ComfyUI dependencies
uv pip install -r requirements.txt

# Install PyTorch with appropriate backend
if [ "$PLATFORM" = "linux_cuda" ]; then
    echo "Installing PyTorch with CUDA support..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
elif [ "$PLATFORM" = "apple_silicon" ]; then
    echo "Installing PyTorch with MPS support..."
    uv pip install torch torchvision torchaudio
else
    echo "Installing PyTorch (CPU)..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "── Installing ComfyUI-GGUF custom node ──"

GGUF_NODE_DIR="${COMFYUI_DIR}/custom_nodes/ComfyUI-GGUF"
if [ -d "$GGUF_NODE_DIR" ]; then
    echo "[OK] ComfyUI-GGUF already installed"
    cd "$GGUF_NODE_DIR" && git pull --ff-only 2>/dev/null || true
    cd "$SCRIPT_DIR"
else
    echo "Cloning ComfyUI-GGUF (for loading GGUF diffusion models)..."
    git clone https://github.com/city96/ComfyUI-GGUF.git "$GGUF_NODE_DIR"
fi
# Install ComfyUI-GGUF dependencies into ComfyUI's venv
cd "$COMFYUI_DIR"
if [ -f "$GGUF_NODE_DIR/requirements.txt" ]; then
    uv pip install -r "$GGUF_NODE_DIR/requirements.txt"
fi
cd "$SCRIPT_DIR"
echo "[OK] ComfyUI-GGUF installed"

echo ""
echo "── Installing wrapper server dependencies ──"

cd "$SCRIPT_DIR"

# Install the wrapper server's deps (fastapi, uvicorn, requests, etc.)
uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

echo ""
echo "── Verifying installation ──"

# Verify ComfyUI's Python env
cd "$COMFYUI_DIR"
uv run python -c "
import torch
print(f'ComfyUI PyTorch {torch.__version__}')
if hasattr(torch.backends, 'mps'):
    print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
if not torch.cuda.is_available() and not getattr(torch.backends, 'mps', None):
    print('Running on CPU')
"
cd "$SCRIPT_DIR"

# Verify wrapper server's Python env
uv run python -c "
import requests, fastapi, uvicorn
print(f'FastAPI {fastapi.__version__}, uvicorn {uvicorn.__version__}')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Download models:"
echo "      uv run python download_models.py"
echo ""
echo "   2. (Optional) Set HF_TOKEN for faster downloads:"
echo "      export HF_TOKEN=hf_..."
echo ""
echo "   3. Start the pipeline:"
echo "      ./start.sh"
echo "============================================"
