"""Download models for the ComfyUI-based pipeline.

Reuses models already downloaded in ../apple-silicon/models where possible
(symlinks), and only downloads what's truly new.

Reusable from apple-silicon (~18 GB saved):
  - qwen-image-layered-Q4_K_M.gguf     -> ComfyUI/models/diffusion_models/ (via ComfyUI-GGUF)
  - qwen_image_layered_vae.safetensors  -> ComfyUI/models/vae/
  - Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf -> models/ (stage 3)
  - mmproj-BF16.gguf                    -> models/ (stage 3)

New downloads (~28 GB):
  - z_image_turbo_bf16.safetensors      (~12 GB, Z-Image Turbo diffusion model)
  - qwen_3_4b.safetensors              (~8 GB,  Z-Image text encoder)
  - ae.safetensors                      (~335 MB, Flux1 VAE)
  - qwen_2.5_vl_7b_fp8_scaled.safetensors (~7.5 GB, Qwen-Image-Layered text encoder)
"""

import os
from huggingface_hub import hf_hub_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APPLE_SILICON_MODELS = os.path.join(SCRIPT_DIR, "..", "apple-silicon", "models")
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", os.path.join(SCRIPT_DIR, "ComfyUI"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(SCRIPT_DIR, "models"))
HF_TOKEN = os.environ.get("HF_TOKEN")


def _symlink(src: str, dst: str, label: str) -> bool:
    """Create a symlink from src to dst. Returns True if successful."""
    src = os.path.abspath(src)
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) or os.path.islink(dst):
        print(f"  [OK] {label} already exists at {dst}")
        return True
    os.symlink(src, dst)
    print(f"  [LINK] {label}: {src} -> {dst}")
    return True


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

    apple_available = os.path.isdir(APPLE_SILICON_MODELS)
    if apple_available:
        print(f"\nFound apple-silicon models at {os.path.abspath(APPLE_SILICON_MODELS)}")
        print("Will reuse compatible models via symlinks.\n")
    else:
        print(f"\nNo apple-silicon models found at {APPLE_SILICON_MODELS}")
        print("All models will be downloaded fresh.\n")

    # ──────────────────────────────────────────────────────────────
    # Stage 2: Qwen-Image-Layered — reuse from apple-silicon
    # ──────────────────────────────────────────────────────────────

    print("[1/8] Qwen-Image-Layered GGUF diffusion model (~12 GB)")
    reused_gguf = False
    if apple_available:
        reused_gguf = _symlink(
            os.path.join(APPLE_SILICON_MODELS, "qwen-image-layered-Q4_K_M.gguf"),
            os.path.join(diffusion_dir, "qwen-image-layered-Q4_K_M.gguf"),
            "qwen-image-layered-Q4_K_M.gguf",
        )
    if not reused_gguf:
        _hf_download_and_link(
            "unsloth/Qwen-Image-Layered-GGUF",
            "qwen-image-layered-Q4_K_M.gguf",
            diffusion_dir,
            os.path.join(diffusion_dir, "qwen-image-layered-Q4_K_M.gguf"),
            "qwen-image-layered-Q4_K_M.gguf",
        )

    print("[2/8] Qwen-Image-Layered VAE (~242 MB)")
    reused_vae = False
    if apple_available:
        reused_vae = _symlink(
            os.path.join(APPLE_SILICON_MODELS, "split_files", "vae", "qwen_image_layered_vae.safetensors"),
            os.path.join(vae_dir, "qwen_image_layered_vae.safetensors"),
            "qwen_image_layered_vae.safetensors",
        )
    if not reused_vae:
        _hf_download_and_link(
            "Comfy-Org/Qwen-Image-Layered_ComfyUI",
            "split_files/vae/qwen_image_layered_vae.safetensors",
            COMFYUI_DIR,
            os.path.join(vae_dir, "qwen_image_layered_vae.safetensors"),
            "qwen_image_layered_vae.safetensors",
        )

    # ──────────────────────────────────────────────────────────────
    # Stage 3: Qwen2.5-VL GGUF — reuse from apple-silicon
    # ──────────────────────────────────────────────────────────────

    print("[3/8] Qwen2.5-VL-7B GGUF (~4.7 GB)")
    reused_vl = False
    if apple_available:
        reused_vl = _symlink(
            os.path.join(APPLE_SILICON_MODELS, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"),
            os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"),
            "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        )
    if not reused_vl:
        _hf_download_and_link(
            "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            MODEL_DIR,
            os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"),
            "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        )

    print("[4/8] Qwen2.5-VL mmproj (~1.35 GB)")
    reused_mm = False
    if apple_available:
        reused_mm = _symlink(
            os.path.join(APPLE_SILICON_MODELS, "mmproj-BF16.gguf"),
            os.path.join(MODEL_DIR, "mmproj-BF16.gguf"),
            "mmproj-BF16.gguf",
        )
    if not reused_mm:
        _hf_download_and_link(
            "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            "mmproj-BF16.gguf",
            MODEL_DIR,
            os.path.join(MODEL_DIR, "mmproj-BF16.gguf"),
            "mmproj-BF16.gguf",
        )

    # ──────────────────────────────────────────────────────────────
    # Stage 1: Z-Image Turbo — new downloads (ComfyUI-specific format)
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
    # Stage 2: Qwen-Image-Layered text encoder — new download
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
    reused = sum([reused_gguf, reused_vae, reused_vl, reused_mm])
    print(f"\nDone! Reused {reused}/4 models from apple-silicon ({reused} symlinks).")
    print(f"  ComfyUI models: {COMFYUI_DIR}/models/")
    print(f"  VL GGUF models: {MODEL_DIR}/")


if __name__ == "__main__":
    download_all()
