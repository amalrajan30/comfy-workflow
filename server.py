"""
ComfyUI-based pipeline server:
  Z-Image Turbo (via ComfyUI)
    -> Qwen-Image-Layered (via ComfyUI)
    -> Qwen2.5-VL GGUF (via llama-cpp-python)
    -> response

Architecture:
  - ComfyUI runs on port 8188 as the backend for image generation and layer decomposition.
  - This server runs on port 8000 and provides the REST API.
  - Stage 1 (Z-Image Turbo) and Stage 2 (Qwen-Image-Layered) are submitted as ComfyUI workflows.
  - Stage 3 (Qwen2.5-VL analysis) runs directly via llama-cpp-python with Metal GPU.

Models:
  ComfyUI (stages 1 & 2):
    - z_image_turbo_bf16.safetensors          (Comfy-Org/z_image_turbo)
    - qwen_3_4b.safetensors                   (text encoder for Z-Image)
    - ae.safetensors                           (Flux1 VAE for Z-Image)
    - qwen-image-layered-Q4_K_M.gguf          (unsloth GGUF, loaded via ComfyUI-GGUF)
    - qwen_2.5_vl_7b_fp8_scaled.safetensors   (text encoder for Qwen-Image-Layered)
    - qwen_image_layered_vae.safetensors       (VAE for Qwen-Image-Layered)
  Direct GGUF (stage 3):
    - Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf     (unsloth/Qwen2.5-VL-7B-Instruct-GGUF)
    - mmproj-BF16.gguf                         (vision projector)
"""

import base64
import copy
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "workflows")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))

# Qwen2.5-VL GGUF paths (stage 3 — direct llama-cpp-python)
QWEN_VL_GGUF = os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
QWEN_VL_MMPROJ = os.path.join(MODEL_DIR, "mmproj-BF16.gguf")

# ComfyUI timeout for workflow execution (seconds)
COMFYUI_TIMEOUT = int(os.environ.get("COMFYUI_TIMEOUT", "600"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

qwen_llm = None
client_id = str(uuid.uuid4())

# Cached workflow templates (loaded once at startup)
_wf_zimage: dict = {}
_wf_layered: dict = {}


# ---------------------------------------------------------------------------
# ComfyUI API helpers
# ---------------------------------------------------------------------------

def _load_workflow(name: str) -> dict:
    """Load a workflow_api.json template from the workflows/ directory."""
    path = os.path.join(WORKFLOW_DIR, name)
    with open(path) as f:
        return json.load(f)


def _queue_prompt(workflow: dict) -> str:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"ComfyUI rejected workflow: {data['error']}")
    if "node_errors" in data and data["node_errors"]:
        raise RuntimeError(f"ComfyUI node errors: {json.dumps(data['node_errors'], indent=2)}")
    return data["prompt_id"]


def _wait_for_prompt(prompt_id: str, timeout: int = COMFYUI_TIMEOUT) -> dict:
    """Poll /history until the prompt completes or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            entry = history[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                msgs = entry.get("status", {}).get("messages", [])
                raise RuntimeError(f"ComfyUI execution error: {msgs}")
            return entry
        time.sleep(1)
    raise TimeoutError(f"ComfyUI workflow timed out after {timeout}s")


def _get_comfyui_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    """Download an output image from ComfyUI."""
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    resp = requests.get(f"{COMFYUI_URL}/view", params=params, timeout=30)
    resp.raise_for_status()
    return resp.content


def _upload_image_to_comfyui(image: Image.Image, name: str = "input.png") -> str:
    """Upload a PIL image to ComfyUI's input directory and return the filename."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"image": (name, buf, "image/png")}
    data = {"subfolder": "", "type": "input", "overwrite": "true"}
    resp = requests.post(f"{COMFYUI_URL}/upload/image", files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["name"]


def _extract_images_from_history(history: dict, node_id: str) -> list[Image.Image]:
    """Extract all output images from a specific node in the ComfyUI history."""
    outputs = history.get("outputs", {})
    node_output = outputs.get(node_id, {})
    images_info = node_output.get("images", [])

    images = []
    for img_info in images_info:
        img_bytes = _get_comfyui_image(
            img_info["filename"],
            img_info.get("subfolder", ""),
            img_info.get("type", "output"),
        )
        images.append(Image.open(io.BytesIO(img_bytes)))
    return images


# ---------------------------------------------------------------------------
# Image scaling helper
# ---------------------------------------------------------------------------

def _scale_image(image: Image.Image, max_dim: int = 640) -> Image.Image:
    """Scale image so max dimension <= max_dim, with dims divisible by 16."""
    w, h = image.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w = max((int(w * scale) // 16) * 16, 16)
    new_h = max((int(h * scale) // 16) * 16, 16)
    if (new_w, new_h) != (w, h):
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_generate(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: Optional[int],
) -> tuple[Image.Image, int]:
    """Stage 1: Z-Image Turbo via ComfyUI — generate image from text."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    wf = copy.deepcopy(_wf_zimage)

    # Modify workflow inputs
    wf["5"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["width"] = width
    wf["7"]["inputs"]["height"] = height
    wf["8"]["inputs"]["seed"] = seed
    wf["8"]["inputs"]["steps"] = steps

    prompt_id = _queue_prompt(wf)
    history = _wait_for_prompt(prompt_id)
    images = _extract_images_from_history(history, "10")  # SaveImage node

    if not images:
        raise RuntimeError("Z-Image Turbo produced no output images")

    return images[0], seed


def stage_layered(
    image: Image.Image,
    num_layers: int,
    resolution: int,
    steps: int,
    seed: int,
) -> list[Image.Image]:
    """Stage 2: Qwen-Image-Layered via ComfyUI — decompose into RGBA layers."""
    # Scale image to target resolution (dims divisible by 16)
    scaled = _scale_image(image.convert("RGBA"), max_dim=resolution)
    scaled_w, scaled_h = scaled.size

    # Upload to ComfyUI
    upload_name = _upload_image_to_comfyui(scaled, f"layered_input_{seed}.png")

    wf = copy.deepcopy(_wf_layered)

    # Modify workflow inputs
    wf["4"]["inputs"]["image"] = upload_name
    wf["10"]["inputs"]["width"] = scaled_w
    wf["10"]["inputs"]["height"] = scaled_h
    wf["10"]["inputs"]["layers"] = num_layers
    wf["12"]["inputs"]["seed"] = seed
    wf["12"]["inputs"]["steps"] = steps

    prompt_id = _queue_prompt(wf)
    history = _wait_for_prompt(prompt_id)
    layers = _extract_images_from_history(history, "15")  # SaveImage node

    if not layers:
        raise RuntimeError("Qwen-Image-Layered produced no output layers")

    return layers


def stage_analyze(
    original: Image.Image,
    layers: list[Image.Image],
    user_prompt: str,
    max_tokens: int,
) -> str:
    """Stage 3: Qwen2.5-VL GGUF (Unsloth) — analyze the original image and its layers."""
    content = [
        {"type": "image_url", "image_url": {"url": _pil_to_data_uri(original)}},
        {"type": "text", "text": f"[Original image above] The image was decomposed into {len(layers)} RGBA layers:"},
    ]
    for i, layer in enumerate(layers):
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_uri(layer)}})
        content.append({"type": "text", "text": f"[Layer {i + 1} of {len(layers)}]"})

    content.append({"type": "text", "text": user_prompt})

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert image analyst. You receive an original image and its "
                "RGBA layer decomposition. Provide a detailed, structured analysis."
            ),
        },
        {"role": "user", "content": content},
    ]

    response = qwen_llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _pil_to_data_uri(img: Image.Image) -> str:
    b64 = _pil_to_base64(img)
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _wait_for_comfyui(timeout: int = 120):
    """Wait for ComfyUI to be ready."""
    log.info("Waiting for ComfyUI at %s ...", COMFYUI_URL)
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if resp.status_code == 200:
                stats = resp.json()
                log.info("ComfyUI ready: %s", json.dumps(stats.get("system", {}), indent=2))
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise RuntimeError(f"ComfyUI not reachable at {COMFYUI_URL} after {timeout}s")


def _load_qwen_vl():
    """Load Qwen2.5-VL GGUF via llama-cpp-python with Metal support."""
    global qwen_llm
    from llama_cpp import Llama

    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        handler_cls = Qwen25VLChatHandler
        log.info("Using Qwen25VLChatHandler")
    except ImportError:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        handler_cls = Llava15ChatHandler
        log.warning("Qwen25VLChatHandler not found, falling back to Llava15ChatHandler")

    if not os.path.exists(QWEN_VL_GGUF):
        raise FileNotFoundError(
            f"GGUF model not found at {QWEN_VL_GGUF}. "
            "Run `python download_models.py` first."
        )
    if not os.path.exists(QWEN_VL_MMPROJ):
        raise FileNotFoundError(
            f"mmproj file not found at {QWEN_VL_MMPROJ}. "
            "Run `python download_models.py` first."
        )

    _gpu_label = "CUDA GPU" if os.environ.get("PLATFORM") == "runpod_cuda" else "Metal GPU"
    log.info("Loading Qwen2.5-VL GGUF from %s (%s) ...", QWEN_VL_GGUF, _gpu_label)
    chat_handler = handler_cls(clip_model_path=QWEN_VL_MMPROJ)
    qwen_llm = Llama(
        model_path=QWEN_VL_GGUF,
        chat_handler=chat_handler,
        n_ctx=4096,
        n_gpu_layers=-1,
    )
    log.info("Qwen2.5-VL GGUF loaded (%s acceleration)", _gpu_label)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _wf_zimage, _wf_layered

    # Load workflow templates
    _wf_zimage = _load_workflow("z_image_turbo_api.json")
    _wf_layered = _load_workflow("qwen_image_layered_api.json")
    log.info("Workflow templates loaded")

    # Wait for ComfyUI backend
    _wait_for_comfyui()

    # Load Qwen2.5-VL for analysis (stage 3)
    _load_qwen_vl()

    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_PLATFORM = os.environ.get("PLATFORM", "apple_silicon")

app = FastAPI(
    title="Z-Image Turbo + Qwen-Image-Layered + Qwen2.5-VL GGUF Pipeline (ComfyUI)",
    description=(
        "Generate images with Z-Image Turbo (via ComfyUI), decompose into RGBA layers "
        "with Qwen-Image-Layered (via ComfyUI), then analyze with Qwen2.5-VL GGUF. "
        + (
            "Running on RunPod with NVIDIA CUDA GPU."
            if _PLATFORM == "runpod_cuda"
            else "Optimized for Apple Silicon (M1/M2/M3/M4) with MPS backend."
        )
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    analysis_prompt: str = Field(
        default=(
            "Analyze this image and its decomposed layers. Describe the composition, "
            "colors, subjects, mood, and how the layers separate the visual elements."
        ),
        description="Prompt to send to Qwen2.5-VL for analyzing the image",
    )
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    num_inference_steps: int = Field(
        default=4, ge=1, le=50,
        description="Z-Image Turbo inference steps (4 recommended for ComfyUI res_multistep sampler)",
    )
    seed: Optional[int] = Field(default=None, description="RNG seed for reproducibility")
    num_layers: int = Field(default=4, ge=1, le=10, description="Number of RGBA layers to decompose into")
    layer_resolution: int = Field(default=640, description="Max dimension for layer decomposition (640 or 1024)")
    layer_steps: int = Field(default=20, ge=10, le=100, description="Inference steps for Qwen-Image-Layered")
    max_tokens: int = Field(default=512, ge=64, le=2048, description="Max tokens for Qwen2.5-VL analysis")


class GenerateResponse(BaseModel):
    image_base64: str
    layers_base64: list[str]
    analysis: str
    generation_prompt: str
    seed_used: int
    num_layers: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Full pipeline: Z-Image Turbo -> Qwen-Image-Layered -> Qwen2.5-VL GGUF -> response."""
    try:
        # Stage 1: Generate image with Z-Image Turbo (via ComfyUI)
        log.info("Stage 1: Generating image for prompt: %s", req.prompt[:80])
        image, seed_used = stage_generate(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.num_inference_steps,
            seed=req.seed,
        )
        log.info("Stage 1 complete (seed=%d)", seed_used)

        # Stage 2: Decompose with Qwen-Image-Layered (via ComfyUI)
        log.info("Stage 2: Decomposing into %d layers ...", req.num_layers)
        layers = stage_layered(
            image=image,
            num_layers=req.num_layers,
            resolution=req.layer_resolution,
            steps=req.layer_steps,
            seed=seed_used,
        )
        log.info("Stage 2 complete (%d layers)", len(layers))

        # Stage 3: Analyze with Qwen2.5-VL GGUF
        log.info("Stage 3: Analyzing with Qwen2.5-VL GGUF ...")
        analysis = stage_analyze(image, layers, req.analysis_prompt, req.max_tokens)
        log.info("Stage 3 complete (%d chars)", len(analysis))

        return GenerateResponse(
            image_base64=_pil_to_base64(image),
            layers_base64=[_pil_to_base64(layer) for layer in layers],
            analysis=analysis,
            generation_prompt=req.prompt,
            seed_used=seed_used,
            num_layers=len(layers),
        )

    except Exception as e:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-image-only")
async def generate_image_only(req: GenerateRequest):
    """Stage 1 only: Z-Image Turbo image generation (via ComfyUI)."""
    try:
        image, seed_used = stage_generate(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.num_inference_steps,
            seed=req.seed,
        )
        return JSONResponse({
            "image_base64": _pil_to_base64(image),
            "seed_used": seed_used,
        })
    except Exception as e:
        log.exception("Image generation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decompose")
async def decompose(
    image_base64: str,
    num_layers: int = 4,
    resolution: int = 640,
    steps: int = 20,
    seed: int = 42,
):
    """Stage 2 only: Decompose an existing image into RGBA layers (via ComfyUI)."""
    try:
        img_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        layers = stage_layered(image, num_layers, resolution, steps, seed)
        return JSONResponse({
            "layers_base64": [_pil_to_base64(layer) for layer in layers],
            "num_layers": len(layers),
        })
    except Exception as e:
        log.exception("Decomposition error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze(
    image_base64: str,
    analysis_prompt: str = "Describe this image in detail.",
    max_tokens: int = 512,
):
    """Stage 3 only: Analyze an image with Qwen2.5-VL GGUF."""
    try:
        img_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        analysis = stage_analyze(image, [], analysis_prompt, max_tokens)
        return JSONResponse({"analysis": analysis})
    except Exception as e:
        log.exception("Analysis error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    comfyui_ok = False
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        comfyui_ok = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok" if comfyui_ok and qwen_llm is not None else "degraded",
        "comfyui_url": COMFYUI_URL,
        "comfyui_reachable": comfyui_ok,
        "qwen_vl_loaded": qwen_llm is not None,
        "backend": "comfyui",
        "platform": os.environ.get("PLATFORM", "apple_silicon"),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
