"""Download models for the ComfyUI-based pipeline.

Downloads all required models from HuggingFace:
  - qwen-image-layered-Q4_K_M.gguf     -> ComfyUI/models/diffusion_models/ (via ComfyUI-GGUF)
  - qwen_image_layered_vae.safetensors  -> ComfyUI/models/vae/
  - Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf -> models/ (stage 3)
  - mmproj-BF16.gguf                    -> models/ (stage 3)
  - z_image_turbo_bf16.safetensors      (~12 GB, Z-Image Turbo diffusion model)
  - qwen_3_4b.safetensors              (~8 GB,  Z-Image text encoder)
  - ae.safetensors                      (~335 MB, Flux1 VAE)
  - qwen_2.5_vl_7b_fp8_scaled.safetensors (~7.5 GB, Qwen-Image-Layered text encoder)
"""

import os
from huggingface_hub import hf_hub_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", os.path.join(SCRIPT_DIR, "ComfyUI"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(SCRIPT_DIR, "models"))
HF_TOKEN = os.environ.get("HF_TOKEN")


def _hf_download_and_link(repo_id: str, filename: str, local_dir: str,
                           link_dst: str, label: str):
    """Download from HuggingFace and symlink into the target directory."""
    if os.path.exists(link_dst) or os.path.islink(link_dst):
        print(f"  [OK] {label} already exists")
        return

    print(f"  [DL] Downloading {label} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=HF_TOKEN,
    )
    # The file is downloaded to local_dir/filename — symlink to target
    src = os.path.join(local_dir, filename)
    if os.path.exists(src) and not os.path.exists(link_dst):
        os.symlink(os.path.abspath(src), link_dst)


def download_all():
    if HF_TOKEN:
        print("Using HF_TOKEN for authenticated downloads")
    else:
        print("No HF_TOKEN set (downloads may be rate-limited)")

    # ── ComfyUI model directories ──
    diffusion_dir = os.path.join(COMFYUI_DIR, "models", "diffusion_models")
    clip_dir = os.path.join(COMFYUI_DIR, "models", "clip")
    vae_dir = os.path.join(COMFYUI_DIR, "models", "vae")
    os.makedirs(diffusion_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Qwen-Image-Layered models
    # ──────────────────────────────────────────────────────────────

    print("[1/8] Qwen-Image-Layered GGUF diffusion model (~12 GB)")
    _hf_download_and_link(
        "unsloth/Qwen-Image-Layered-GGUF",
        "qwen-image-layered-Q4_K_M.gguf",
        diffusion_dir,
        os.path.join(diffusion_dir, "qwen-image-layered-Q4_K_M.gguf"),
        "qwen-image-layered-Q4_K_M.gguf",
    )

    print("[2/8] Qwen-Image-Layered VAE (~242 MB)")
    _hf_download_and_link(
        "Comfy-Org/Qwen-Image-Layered_ComfyUI",
        "split_files/vae/qwen_image_layered_vae.safetensors",
        COMFYUI_DIR,
        os.path.join(vae_dir, "qwen_image_layered_vae.safetensors"),
        "qwen_image_layered_vae.safetensors",
    )

    # ──────────────────────────────────────────────────────────────
    # Qwen2.5-VL GGUF models
    # ──────────────────────────────────────────────────────────────

    print("[3/8] Qwen2.5-VL-7B GGUF (~4.7 GB)")
    _hf_download_and_link(
        "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        MODEL_DIR,
        os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"),
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    )

    print("[4/8] Qwen2.5-VL mmproj (~1.35 GB)")
    _hf_download_and_link(
        "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "mmproj-BF16.gguf",
        MODEL_DIR,
        os.path.join(MODEL_DIR, "mmproj-BF16.gguf"),
        "mmproj-BF16.gguf",
    )

    # ──────────────────────────────────────────────────────────────
    # Z-Image Turbo models (ComfyUI-specific format)
    # ──────────────────────────────────────────────────────────────

    zimage_repo = "Comfy-Org/z_image_turbo"

    print("[5/8] Z-Image Turbo diffusion model (~12 GB)")
    _hf_download_and_link(
        zimage_repo,
        "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        COMFYUI_DIR,
        os.path.join(diffusion_dir, "z_image_turbo_bf16.safetensors"),
        "z_image_turbo_bf16.safetensors",
    )

    print("[6/8] Qwen3-4B text encoder (~8 GB)")
    _hf_download_and_link(
        zimage_repo,
        "split_files/text_encoders/qwen_3_4b.safetensors",
        COMFYUI_DIR,
        os.path.join(clip_dir, "qwen_3_4b.safetensors"),
        "qwen_3_4b.safetensors",
    )

    print("[7/8] Flux1 VAE (~335 MB)")
    _hf_download_and_link(
        zimage_repo,
        "split_files/vae/ae.safetensors",
        COMFYUI_DIR,
        os.path.join(vae_dir, "ae.safetensors"),
        "ae.safetensors",
    )

    # ──────────────────────────────────────────────────────────────
    # Qwen-Image-Layered text encoder
    # ──────────────────────────────────────────────────────────────

    print("[8/8] Qwen2.5-VL 7B FP8 text encoder (~7.5 GB)")
    _hf_download_and_link(
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        COMFYUI_DIR,
        os.path.join(clip_dir, "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
        "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    )

    # ── Summary ──
    print(f"\nDone! All 8 models downloaded.")
    print(f"  ComfyUI models: {COMFYUI_DIR}/models/")
    print(f"  VL GGUF models: {MODEL_DIR}/")


if __name__ == "__main__":
    download_all()
