# Pipeline Server — ComfyUI Edition

Z-Image Turbo + Qwen-Image-Layered GGUF + Qwen2.5-VL GGUF pipeline, using ComfyUI as the image generation backend. Optimized for Apple Silicon Macs (M1/M2/M3/M4).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Wrapper Server (FastAPI, port 8000)                    │
│  Same REST API as apple-silicon version                 │
│                                                         │
│  Stage 1: Z-Image Turbo ──────────┐                    │
│  Stage 2: Qwen-Image-Layered ─────┼──► ComfyUI :8188   │
│  Stage 3: Qwen2.5-VL GGUF ───────────► llama-cpp-python│
└─────────────────────────────────────────────────────────┘
```

- **Stages 1 & 2** are submitted as ComfyUI workflow_api.json to the ComfyUI backend
- **Stage 2** uses the same GGUF model as apple-silicon (loaded via ComfyUI-GGUF custom node)
- **Stage 3** runs Qwen2.5-VL GGUF directly via llama-cpp-python (Metal GPU)
- The wrapper server provides the same REST API, so `test_client.py` works unchanged

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

# 2. Download/link models (reuses ~18 GB from ../apple-silicon/models if available)
export HF_TOKEN=hf_...  # optional, for faster downloads
uv run python download_models.py

# 3. Start the pipeline (launches ComfyUI + wrapper server)
./start.sh
```

## Model Reuse from apple-silicon/

If `../apple-silicon/models` exists, `download_models.py` will **symlink** these files instead of re-downloading (~18 GB saved):

| Model | Size | Reused as |
|---|---|---|
| `qwen-image-layered-Q4_K_M.gguf` | 12 GB | ComfyUI diffusion model (via ComfyUI-GGUF) |
| `qwen_image_layered_vae.safetensors` | 242 MB | ComfyUI VAE |
| `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | 4.4 GB | Stage 3 VL analysis |
| `mmproj-BF16.gguf` | 1.3 GB | Stage 3 mmproj |

New downloads required (~28 GB):

| Model | Source | Size |
|---|---|---|
| `z_image_turbo_bf16.safetensors` | Comfy-Org/z_image_turbo | ~12 GB |
| `qwen_3_4b.safetensors` | Comfy-Org/z_image_turbo | ~8 GB |
| `ae.safetensors` | Comfy-Org/z_image_turbo | ~335 MB |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | Comfy-Org/HunyuanVideo_1.5_repackaged | ~7.5 GB |

## How It Differs from the apple-silicon Version

| Feature | apple-silicon | comfyui |
|---|---|---|
| Image gen backend | diffusers (direct) | ComfyUI workflow API |
| Layer decomposition | diffusers + GGUFQuantizationConfig | ComfyUI + ComfyUI-GGUF custom node |
| Layer GGUF model | Same `qwen-image-layered-Q4_K_M.gguf` | Same (symlinked) |
| VL analysis | llama-cpp-python | llama-cpp-python (same) |
| Model loading | Manual in server.py | ComfyUI handles loading/caching |
| Memory management | Manual gc + cache clearing | ComfyUI's built-in offloading |
| Workflow customization | Code changes required | Edit workflow JSON files |
| GPU precision | bfloat16 / float16 | FP32 (--force-fp32 for MPS safety) |

### Advantages of ComfyUI Backend

- **Visual workflow editor**: Open `http://localhost:8188` to inspect/modify workflows in the GUI
- **Built-in caching**: ComfyUI caches intermediate results — re-running with same inputs is instant
- **Model management**: ComfyUI handles model loading/unloading efficiently
- **Extensible**: Add custom nodes from the ComfyUI ecosystem without code changes
- **Community workflows**: Import workflows shared by the ComfyUI community

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
- Z-Image Turbo: 4 steps with `res_multistep` sampler (vs 9 steps in apple-silicon version)
- Qwen-Image-Layered: 20 steps default (vs 50 in apple-silicon version)
- First run is slower due to ComfyUI model loading; subsequent runs benefit from caching
- If memory is tight, reduce `layer_resolution` from 1024 to 640
