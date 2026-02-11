# Pipeline Server — Apple Silicon Edition

Z-Image Turbo + Qwen-Image-Layered GGUF + Qwen2.5-VL GGUF pipeline, optimized for Apple Silicon Macs (M1/M2/M3/M4) using the MPS backend and Metal GPU acceleration.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS Ventura 13.0+ (Sequoia recommended)
- 32 GB+ unified memory (64 GB recommended for full pipeline)
- Xcode Command Line Tools (`xcode-select --install`)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# 1. One-time setup (installs deps + llama-cpp-python with Metal)
./setup.sh

# 2. (Optional) Set HuggingFace token for faster downloads
export HF_TOKEN=hf_...

# 3. Start the server (downloads ~19.5 GB of models on first run)
./start.sh
```

## Key Differences from CUDA Version

| Feature | CUDA (RunPod) | Apple Silicon |
|---|---|---|
| GPU backend | CUDA | MPS (Metal) |
| Compute dtype | bfloat16 | float16 |
| Generator device | cuda | cpu (MPS generators produce black images) |
| CPU offload | `enable_model_cpu_offload()` | `pipe.to("mps")` + `enable_attention_slicing()` |
| llama.cpp GPU | CUDA | Metal |
| Deployment | Docker + RunPod | Native macOS |
| Memory model | Separate VRAM | Unified memory |

## Endpoints

- `POST /generate` — Full pipeline (text -> image -> layers -> analysis)
- `POST /generate-image-only` — Z-Image Turbo only
- `POST /decompose` — Qwen-Image-Layered GGUF only
- `POST /analyze` — Qwen2.5-VL GGUF only
- `GET /health` — Check model load status + device info

## Performance Notes

- Image generation is significantly slower than on CUDA GPUs (~1-3 min vs ~3 sec on RTX 4090)
- Layer decomposition is the most memory-intensive stage
- The Qwen2.5-VL analysis runs via llama.cpp Metal and is reasonably fast
- If you run into memory pressure, reduce `layer_resolution` from 1024 to 640
