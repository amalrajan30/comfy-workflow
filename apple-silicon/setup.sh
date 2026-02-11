#!/bin/bash
set -e

# ── Apple Silicon Setup Script ──
# One-time setup for the pipeline server on Apple Silicon Macs.
# Run this before start.sh to install dependencies.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " Apple Silicon Setup"
echo " Z-Image Turbo + Qwen Pipeline"
echo "============================================"
echo ""

# ── Check prerequisites ──
echo "── Checking prerequisites ──"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "ERROR: This setup is for Apple Silicon (arm64). Detected: ${ARCH}"
    exit 1
fi
echo "[OK] Architecture: ${ARCH}"

# Check Xcode CLI tools
if ! xcode-select -p &>/dev/null; then
    echo "[!] Xcode Command Line Tools not found. Installing..."
    xcode-select --install
    echo "    Please re-run this script after Xcode CLT installation completes."
    exit 1
fi
echo "[OK] Xcode Command Line Tools installed"

# Check uv
if ! command -v uv &>/dev/null; then
    echo "[!] uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[OK] uv: $(uv --version)"

# Check Python
PYTHON_ARCH=$(python3 -c "import platform; print(platform.machine())" 2>/dev/null || echo "unknown")
if [ "$PYTHON_ARCH" != "arm64" ]; then
    echo "WARNING: Python reports '${PYTHON_ARCH}', expected 'arm64'."
    echo "         Ensure you're using a native ARM64 Python (not Rosetta)."
fi
echo "[OK] Python: $(python3 --version) (${PYTHON_ARCH})"

echo ""
echo "── Installing Python dependencies ──"
cd "$SCRIPT_DIR"

# Install main dependencies via uv
uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

echo ""
echo "── Installing llama-cpp-python with Metal ──"

# llama-cpp-python: Metal is auto-enabled on macOS arm64 since v0.3.x,
# but we set the flags explicitly for reliability.
CMAKE_ARGS="-DGGML_METAL=on -DCMAKE_OSX_ARCHITECTURES=arm64" \
    uv pip install --no-cache-dir --force-reinstall llama-cpp-python

echo ""
echo "── Verifying installation ──"

uv run python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built:     {torch.backends.mps.is_built()}')
"

uv run python -c "
from llama_cpp import Llama
print('llama-cpp-python imported successfully (Metal support)')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. (Optional) Set HF_TOKEN for faster downloads:"
echo "      export HF_TOKEN=hf_..."
echo ""
echo "   2. Start the server:"
echo "      ./start.sh"
echo ""
echo " Models (~19.5 GB) will download on first run."
echo " Recommended: 32+ GB unified memory"
echo "============================================"
