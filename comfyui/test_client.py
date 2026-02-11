"""Quick test client for the ComfyUI pipeline server."""

import base64
import json
import sys
from urllib.request import Request, urlopen

SERVER = "http://localhost:8000"


def test_full_pipeline(prompt: str = "A photorealistic cat sitting on a wooden desk next to a coffee mug"):
    payload = json.dumps({
        "prompt": prompt,
        "analysis_prompt": (
            "Analyze this image and its decomposed layers. Describe the composition, "
            "colors, subjects, mood, and how the layers separate the visual elements."
        ),
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 4,
        "num_layers": 4,
        "layer_resolution": 640,
        "layer_steps": 20,
        "max_tokens": 512,
    }).encode()

    req = Request(f"{SERVER}/generate", data=payload, headers={"Content-Type": "application/json"})
    print(f"Sending prompt: {prompt}")
    print("Running 3-stage pipeline: Z-Image Turbo -> Qwen-Image-Layered -> Qwen2.5-VL (via ComfyUI) ...")

    with urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())

    # Save the original image
    img_bytes = base64.b64decode(data["image_base64"])
    with open("output.png", "wb") as f:
        f.write(img_bytes)
    print(f"Original image saved to output.png (seed={data['seed_used']})")

    # Save each layer
    for i, layer_b64 in enumerate(data["layers_base64"]):
        layer_bytes = base64.b64decode(layer_b64)
        filename = f"layer_{i + 1}.png"
        with open(filename, "wb") as f:
            f.write(layer_bytes)
        print(f"Layer {i + 1} saved to {filename}")

    # Print the analysis
    print(f"\n--- Qwen2.5-VL Analysis ({data['num_layers']} layers) ---")
    print(data["analysis"])


def test_health():
    with urlopen(f"{SERVER}/health") as resp:
        data = json.loads(resp.read())
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        test_health()
    else:
        prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
        if prompt:
            test_full_pipeline(prompt)
        else:
            test_full_pipeline()
