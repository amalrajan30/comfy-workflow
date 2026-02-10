"""
Pipeline server:
  Z-Image Turbo (image gen)
    -> Qwen-Image-Layered GGUF (layer decomposition, unsloth/Qwen-Image-Layered-GGUF)
    -> Qwen2.5-VL GGUF (analysis, unsloth/Qwen2.5-VL-7B-Instruct-GGUF)
    -> response

Models (all Unsloth where available):
  - Tongyi-MAI/Z-Image-Turbo              (diffusers ZImagePipeline)
  - unsloth/Qwen-Image-Layered-GGUF       (diffusers GGUF via from_single_file + GGUFQuantizationConfig)
  - unsloth/Qwen2.5-VL-7B-Instruct-GGUF   (llama-cpp-python)
"""

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))

# Qwen2.5-VL GGUF (vision-language analysis)
QWEN_VL_GGUF = os.path.join(MODEL_DIR, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
QWEN_VL_MMPROJ = os.path.join(MODEL_DIR, "mmproj-BF16.gguf")

# Qwen-Image-Layered GGUF (layer decomposition)
LAYERED_GGUF = os.path.join(MODEL_DIR, "qwen-image-layered-Q4_K_M.gguf")
LAYERED_VAE = os.path.join(MODEL_DIR, "split_files", "vae", "qwen_image_layered_vae.safetensors")

# Z-Image Turbo (text-to-image generation)
ZIMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# The base HF repo to pull pipeline config, text encoder, and scheduler from
LAYERED_BASE_REPO = "Qwen/Qwen-Image-Layered"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model handles (populated at startup)
# ---------------------------------------------------------------------------

zimage_pipe = None
layered_pipe = None
qwen_llm = None


def _load_zimage():
    """Load the Z-Image Turbo diffusion pipeline."""
    global zimage_pipe
    from diffusers import ZImagePipeline

    log.info("Loading Z-Image Turbo from %s ...", ZIMAGE_MODEL_ID)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    zimage_pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    zimage_pipe.to(DEVICE)
    log.info("Z-Image Turbo loaded on %s", DEVICE)


def _load_layered():
    """Load Qwen-Image-Layered with the GGUF transformer from Unsloth.

    The GGUF transformer is loaded via diffusers' from_single_file + GGUFQuantizationConfig.
    The text encoder, scheduler, and VAE come from the base HF repo / downloaded files.
    """
    global layered_pipe
    from diffusers import QwenImageLayeredPipeline, GGUFQuantizationConfig
    from diffusers.models import AutoencoderKLWan, QwenImageTransformer2DModel

    if not os.path.exists(LAYERED_GGUF):
        raise FileNotFoundError(
            f"Layered GGUF not found at {LAYERED_GGUF}. "
            "Run `uv run python download_models.py` first."
        )

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load the GGUF-quantized transformer (DiT)
    log.info("Loading Qwen-Image-Layered GGUF transformer from %s ...", LAYERED_GGUF)
    transformer = QwenImageTransformer2DModel.from_single_file(
        LAYERED_GGUF,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
    )

    # Load the full pipeline from the base repo, swapping in the GGUF transformer.
    # The text encoder (Qwen2.5-VL) and scheduler come from the HF repo.
    # If we have a locally downloaded VAE, use it; otherwise fall back to the repo.
    log.info("Loading Qwen-Image-Layered pipeline (text encoder + scheduler from %s) ...", LAYERED_BASE_REPO)

    pipe_kwargs = {
        "transformer": transformer,
        "torch_dtype": dtype,
    }

    # Use local VAE if downloaded
    if os.path.exists(LAYERED_VAE):
        log.info("Using local VAE from %s", LAYERED_VAE)
        vae = AutoencoderKLWan.from_single_file(LAYERED_VAE, torch_dtype=dtype)
        pipe_kwargs["vae"] = vae

    layered_pipe = QwenImageLayeredPipeline.from_pretrained(
        LAYERED_BASE_REPO,
        **pipe_kwargs,
    )
    layered_pipe.enable_model_cpu_offload()
    layered_pipe.set_progress_bar_config(disable=None)
    log.info("Qwen-Image-Layered (GGUF) loaded")


def _load_qwen_vl():
    """Load Qwen2.5-VL GGUF via llama-cpp-python with vision support."""
    global qwen_llm
    from llama_cpp import Llama

    # Try the dedicated Qwen2.5-VL handler first, fall back to Llava15
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
            "Run `uv run python download_models.py` first."
        )
    if not os.path.exists(QWEN_VL_MMPROJ):
        raise FileNotFoundError(
            f"mmproj file not found at {QWEN_VL_MMPROJ}. "
            "Run `uv run python download_models.py` first."
        )

    log.info("Loading Qwen2.5-VL GGUF from %s ...", QWEN_VL_GGUF)
    chat_handler = handler_cls(clip_model_path=QWEN_VL_MMPROJ)
    n_gpu = -1 if DEVICE in ("cuda", "mps") else 0
    qwen_llm = Llama(
        model_path=QWEN_VL_GGUF,
        chat_handler=chat_handler,
        n_ctx=4096,
        n_gpu_layers=n_gpu,
    )
    log.info("Qwen2.5-VL GGUF loaded (unsloth/Qwen2.5-VL-7B-Instruct-GGUF)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_zimage()
    _load_layered()
    _load_qwen_vl()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Z-Image Turbo + Qwen-Image-Layered GGUF + Qwen2.5-VL GGUF Pipeline",
    description=(
        "Generate images with Z-Image Turbo, decompose into RGBA layers "
        "with Qwen-Image-Layered (Unsloth GGUF), then analyze with Qwen2.5-VL (Unsloth GGUF)."
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
        default=9, ge=1, le=50,
        description="Z-Image Turbo inference steps (9 = 8 DiT forwards)",
    )
    seed: Optional[int] = Field(default=None, description="RNG seed for reproducibility")
    num_layers: int = Field(default=4, ge=1, le=10, description="Number of RGBA layers to decompose into")
    layer_resolution: int = Field(default=640, description="Resolution bucket for layer decomposition (640 or 1024)")
    layer_steps: int = Field(default=50, ge=10, le=100, description="Inference steps for Qwen-Image-Layered")
    max_tokens: int = Field(default=512, ge=64, le=2048, description="Max tokens for Qwen2.5-VL analysis")


class GenerateResponse(BaseModel):
    image_base64: str
    layers_base64: list[str]
    analysis: str
    generation_prompt: str
    seed_used: int
    num_layers: int


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
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_generate(prompt: str, width: int, height: int, steps: int, seed: Optional[int]) -> tuple[Image.Image, int]:
    """Stage 1: Z-Image Turbo — generate image from text."""
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    generator = torch.Generator(DEVICE).manual_seed(seed)
    result = zimage_pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=0.0,  # must be 0 for Turbo distilled model
        generator=generator,
    )
    return result.images[0], seed


def stage_layered(image: Image.Image, num_layers: int, resolution: int, steps: int, seed: int) -> list[Image.Image]:
    """Stage 2: Qwen-Image-Layered (Unsloth GGUF) — decompose into RGBA layers."""
    rgba_image = image.convert("RGBA")
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = layered_pipe(
        image=rgba_image,
        generator=generator,
        true_cfg_scale=4.0,
        negative_prompt=" ",
        num_inference_steps=steps,
        num_images_per_prompt=1,
        layers=num_layers,
        resolution=resolution,
    )
    return result.images


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
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Full pipeline: Z-Image Turbo -> Qwen-Image-Layered GGUF -> Qwen2.5-VL GGUF -> response."""
    try:
        # Stage 1: Generate image with Z-Image Turbo
        log.info("Stage 1: Generating image for prompt: %s", req.prompt[:80])
        image, seed_used = stage_generate(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.num_inference_steps,
            seed=req.seed,
        )
        log.info("Stage 1 complete (seed=%d)", seed_used)

        # Stage 2: Decompose with Qwen-Image-Layered GGUF (Unsloth)
        log.info("Stage 2: Decomposing into %d layers (GGUF) ...", req.num_layers)
        layers = stage_layered(
            image=image,
            num_layers=req.num_layers,
            resolution=req.layer_resolution,
            steps=req.layer_steps,
            seed=seed_used,
        )
        log.info("Stage 2 complete (%d layers)", len(layers))

        # Stage 3: Analyze with Qwen2.5-VL GGUF (Unsloth)
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
    """Stage 1 only: Z-Image Turbo image generation."""
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
    steps: int = 50,
    seed: int = 42,
):
    """Stage 2 only: Decompose an existing image into RGBA layers (GGUF)."""
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
    return {
        "status": "ok",
        "zimage_loaded": zimage_pipe is not None,
        "layered_loaded": layered_pipe is not None,
        "qwen_vl_loaded": qwen_llm is not None,
        "device": DEVICE,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
