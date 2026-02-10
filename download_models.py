"""Download the required GGUF model files from Unsloth."""

from huggingface_hub import hf_hub_download
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def download_all():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Qwen2.5-VL-7B GGUF (vision-language, for image analysis) ──
    vl_repo = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"

    print("[1/4] Downloading Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf (~4.7 GB) ...")
    hf_hub_download(repo_id=vl_repo, filename="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", local_dir=MODEL_DIR)

    print("[2/4] Downloading mmproj-BF16.gguf (~1.35 GB) ...")
    hf_hub_download(repo_id=vl_repo, filename="mmproj-BF16.gguf", local_dir=MODEL_DIR)

    # ── Qwen-Image-Layered GGUF (diffusion, for layer decomposition) ──
    layered_repo = "unsloth/Qwen-Image-Layered-GGUF"

    print("[3/4] Downloading qwen-image-layered-Q4_K_M.gguf (~13.2 GB) ...")
    hf_hub_download(repo_id=layered_repo, filename="qwen-image-layered-Q4_K_M.gguf", local_dir=MODEL_DIR)

    # ── Qwen-Image-Layered VAE (safetensors, shared with Qwen-Image) ──
    # The VAE for Qwen-Image-Layered is a 4-channel RGBA variant
    vae_repo = "Comfy-Org/Qwen-Image-Layered_ComfyUI"

    print("[4/4] Downloading Qwen-Image-Layered VAE ...")
    hf_hub_download(repo_id=vae_repo, filename="split_files/vae/qwen_image_layered_vae.safetensors", local_dir=MODEL_DIR)

    print(f"\nAll models downloaded to {MODEL_DIR}")


if __name__ == "__main__":
    download_all()
