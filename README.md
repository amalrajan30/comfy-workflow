# Pipeline Server — ComfyUI

Z-Image Turbo + Qwen-Image-Layered GGUF + Qwen2.5-VL GGUF pipeline, using ComfyUI as the image generation backend. Optimized for Apple Silicon Macs (M1/M2/M3/M4).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Wrapper Server (FastAPI, port 8000)                    │
│                                                         │
│  Stage 1: Z-Image Turbo ──────────┐                    │
│  Stage 2: Qwen-Image-Layered ─────┼──► ComfyUI :8188   │
│  Stage 3: Qwen2.5-VL GGUF ───────────► llama-cpp-python│
└─────────────────────────────────────────────────────────┘
```

- **Stages 1 & 2** are submitted as ComfyUI workflow_api.json to the ComfyUI backend
- **Stage 2** uses a GGUF model loaded via ComfyUI-GGUF custom node
- **Stage 3** runs Qwen2.5-VL GGUF directly via llama-cpp-python (Metal GPU)

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS Ventura 13.0+ (Sequoia recommended)
- 64 GB+ unified memory recommended (32 GB minimum)
- Xcode Command Line Tools (`xcode-select --install`)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# 1. One-time setup (installs ComfyUI + ComfyUI-GGUF + deps + llama-cpp-python with Metal)
chmod +x setup.sh start.sh
./setup.sh

# 2. Download models
export HF_TOKEN=hf_...  # optional, for faster downloads
uv run python download_models.py

# 3. Start the pipeline (launches ComfyUI + wrapper server)
./start.sh
```

## Models

All models are downloaded from HuggingFace by `download_models.py`:

| Model | Size | Used for |
|---|---|---|
| `qwen-image-layered-Q4_K_M.gguf` | 12 GB | ComfyUI diffusion model (via ComfyUI-GGUF) |
| `qwen_image_layered_vae.safetensors` | 242 MB | ComfyUI VAE |
| `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | 4.4 GB | Stage 3 VL analysis |
| `mmproj-BF16.gguf` | 1.3 GB | Stage 3 mmproj |
| `z_image_turbo_bf16.safetensors` | ~12 GB | Z-Image Turbo diffusion |
| `qwen_3_4b.safetensors` | ~8 GB | Z-Image text encoder |
| `ae.safetensors` | ~335 MB | Flux1 VAE |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | ~7.5 GB | Qwen-Image-Layered text encoder |

## Endpoints

- `POST /generate` — Full pipeline (text -> image -> layers -> analysis)
- `POST /generate-image-only` — Z-Image Turbo only
- `POST /decompose` — Qwen-Image-Layered only
- `POST /analyze` — Qwen2.5-VL GGUF only
- `GET /health` — Check ComfyUI + model status

## Workflow Customization

The ComfyUI workflow templates are in `workflows/`:
- `z_image_turbo_api.json` — Stage 1 (text-to-image)
- `qwen_image_layered_api.json` — Stage 2 (image-to-layers, uses UnetLoaderGGUF)

To modify:
1. Open ComfyUI GUI at `http://localhost:8188`
2. Build your workflow visually
3. Enable Dev Mode in settings
4. Export as API format (Workflow > Export API)
5. Replace the corresponding JSON file in `workflows/`

## Performance Notes

- ComfyUI with `--force-fp32` uses more memory but avoids MPS precision issues
- Z-Image Turbo: 4 steps with `res_multistep` sampler
- Qwen-Image-Layered: 20 steps default
- First run is slower due to ComfyUI model loading; subsequent runs benefit from caching
- If memory is tight, reduce `layer_resolution` from 1024 to 640
