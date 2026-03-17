"""
RunPod Serverless handler for SEVA (Stable Virtual Camera) novel view synthesis.

INPUT:
{
    "image_url": "https://...",
    "camera_preset": "orbit",
    "seed": 42,
    "num_frames": 1
}

OUTPUT:
{
    "image_url": "https://snapshots.kaleidoscopeai.org/cache/<hash>.png",
    "cached": true/false,
    "inference_time_ms": 1234
}
"""

import hashlib
import io
import os
import time
import tempfile
import traceback

import requests
import runpod
import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# B2 / CDN helpers
# ---------------------------------------------------------------------------

B2_KEY_ID = os.environ.get("B2_KEY_ID", "")
B2_APP_KEY = os.environ.get("B2_APP_KEY", "")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME", "")
B2_CDN_BASE_URL = os.environ.get("B2_CDN_BASE_URL", "").rstrip("/")

_b2_bucket = None


def _get_b2_bucket():
    """Lazy-init B2 bucket handle."""
    global _b2_bucket
    if _b2_bucket is not None:
        return _b2_bucket
    from b2sdk.v2 import B2Api, InMemoryAccountInfo
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
    _b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
    return _b2_bucket


def b2_file_exists(file_name: str) -> bool:
    """Check if a file exists in the B2 bucket."""
    try:
        bucket = _get_b2_bucket()
        bucket.get_file_info_by_name(file_name)
        return True
    except Exception:
        return False


def b2_upload_png(file_name: str, png_bytes: bytes) -> str:
    """Upload PNG bytes to B2 and return the CDN URL."""
    bucket = _get_b2_bucket()
    bucket.upload_bytes(
        data_bytes=png_bytes,
        file_name=file_name,
        content_type="image/png",
    )
    return f"{B2_CDN_BASE_URL}/{file_name}"


def make_cache_key(image_url: str, camera_preset: str, seed: int) -> str:
    """SHA256 hash of inputs for deterministic cache key."""
    raw = f"{image_url}|{camera_preset}|{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# SEVA model (loaded once at cold start)
# ---------------------------------------------------------------------------

PRESET_MAP = {
    "orbit": "orbit",
    "zoom_in": "zoom-in",
    "zoom_out": "zoom-out",
    "pan_left": "move-left",
    "pan_right": "move-right",
    "pan_up": "move-up",
    "pan_down": "move-down",
    "move_forward": "move-forward",
    "move_backward": "move-backward",
    "dolly_zoom_in": "dolly zoom-in",
    "dolly_zoom_out": "dolly zoom-out",
    "spiral": "spiral",
    "roll": "roll",
    "lemniscate": "lemniscate",
}

# Weights location — RunPod Network Volume
WEIGHTS_DIR = None
for candidate in [
    "/workspace/seva-weights",
    "/runpod-volume/seva-weights",
    os.path.expanduser("~/.cache/huggingface/hub"),
]:
    if os.path.isdir(candidate):
        WEIGHTS_DIR = candidate
        break

_model = None
_ae = None
_conditioner = None
_denoiser = None


def load_models():
    """Load SEVA model components once. Called at cold-start."""
    global _model, _ae, _conditioner, _denoiser

    if _model is not None:
        return

    from seva.utils import load_model, seed_everything
    from seva.model import SGMWrapper
    from seva.modules.autoencoder import AutoEncoder
    from seva.modules.conditioner import CLIPConditioner
    from seva.sampling import DiscreteDenoiser

    print("[SEVA] Loading model v1.1 ...")
    t0 = time.time()

    raw_model = load_model(model_version=1.1, device="cpu", verbose=True).eval()
    _model = SGMWrapper(raw_model).to("cuda")
    _ae = AutoEncoder(chunk_size=1).to("cuda")
    _conditioner = CLIPConditioner().to("cuda")
    _denoiser = DiscreteDenoiser(num_idx=1000, device="cuda")

    print(f"[SEVA] Models loaded in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    source_image: Image.Image,
    camera_preset: str,
    seed: int,
    num_frames: int,
) -> list[Image.Image]:
    """
    Run SEVA inference. Returns a list of PIL Images (target frames only).
    """
    from seva.utils import seed_everything
    from seva.eval import (
        create_samplers,
        do_sample,
        get_value_dict,
        load_img_and_K,
        transform_img_and_K,
    )
    from seva.geometry import get_preset_pose_fov, get_default_intrinsics

    load_models()
    seed_everything(seed)

    device = "cuda"
    H, W = 576, 576
    C, F = 4, 8
    T = num_frames + 1  # 1 input + N targets

    # Map preset name
    seva_preset = PRESET_MAP.get(camera_preset)
    if seva_preset is None:
        raise ValueError(f"Unknown camera_preset: {camera_preset}")

    # Save source image to a temp file for load_img_and_K
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        source_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    try:
        # Camera trajectory
        c2ws, fovs = get_preset_pose_fov(
            option=seva_preset,
            num_frames=T,
            start_w2c=torch.eye(4),
            look_at=torch.tensor([0.0, 0.0, 10.0]),
        )

        img_W, img_H = source_image.size
        aspect_ratio = img_W / img_H
        Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)
        Ks[:, :2] *= torch.tensor([img_W, img_H]).reshape(1, -1, 1).repeat(
            Ks.shape[0], 1, 1
        )
        K_list = Ks.float()

        # Prepare frames
        imgs = []
        for i in range(T):
            if i == 0:
                img, K = load_img_and_K(tmp_path, None, K=K_list[i], device="cpu")
            else:
                img, K = load_img_and_K(
                    img.shape[-2:], None, K=K_list[i], device="cpu"
                )
            img, K = transform_img_and_K(img, (W, H), K=K[None])
            K = K[0]
            K[0] /= W
            K[1] /= H
            K_list[i] = K
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)
        c2ws_th = torch.tensor(c2ws[:, :3]).float()

        # Value dict
        value_dict = get_value_dict(
            curr_imgs=imgs.to(device),
            curr_input_frame_indices=[0],
            curr_c2ws=c2ws_th,
            curr_Ks=K_list,
            curr_input_camera_indices=list(range(T)),
            all_c2ws=c2ws_th,
            camera_scale=2.0,
        )

        # Sampler
        samplers = create_samplers(
            guider_types=1,
            discretization=_denoiser.discretization,
            num_frames=[T],
            num_steps=50,
            cfg_min=1.2,
            device=device,
        )

        # Sample
        samples = do_sample(
            model=_model,
            ae=_ae,
            conditioner=_conditioner,
            denoiser=_denoiser,
            sampler=samplers[0],
            value_dict=value_dict,
            H=H, W=W, C=C, F=F, T=T,
            cfg=2.0,
            encoding_t=1,
            decoding_t=1,
        )

        # Convert to PIL — skip frame 0 (input), return only targets
        output_images = []
        for i in range(1, samples.shape[0]):
            frame = samples[i].detach().cpu()
            frame = ((frame + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
            frame = frame.permute(1, 2, 0).numpy()
            output_images.append(Image.fromarray(frame))

        return output_images

    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """RunPod serverless handler entry point."""
    try:
        inp = job["input"]
        image_url = inp["image_url"]
        camera_preset = inp.get("camera_preset", "orbit")
        seed = int(inp.get("seed", 42))
        num_frames = int(inp.get("num_frames", 1))

        # Validate preset
        if camera_preset not in PRESET_MAP:
            return {
                "error": f"Invalid camera_preset '{camera_preset}'. "
                         f"Valid: {list(PRESET_MAP.keys())}"
            }

        # Cache key
        cache_key = make_cache_key(image_url, camera_preset, seed)
        file_name = f"{cache_key}.png"
        cdn_url = f"{B2_CDN_BASE_URL}/{file_name}"

        # Check cache (only for single-frame requests)
        if num_frames == 1 and B2_KEY_ID and b2_file_exists(file_name):
            return {
                "image_url": cdn_url,
                "cached": True,
                "inference_time_ms": 0,
            }

        # Download source image
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        source_image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        # Run inference
        t0 = time.time()
        output_images = run_inference(source_image, camera_preset, seed, num_frames)
        inference_ms = int((time.time() - t0) * 1000)

        # For single frame — upload to B2 and return URL
        if num_frames == 1 and len(output_images) >= 1:
            buf = io.BytesIO()
            output_images[0].save(buf, format="PNG")
            png_bytes = buf.getvalue()

            if B2_KEY_ID:
                cdn_url = b2_upload_png(file_name, png_bytes)

            return {
                "image_url": cdn_url,
                "cached": False,
                "inference_time_ms": inference_ms,
            }

        # Multi-frame — upload all frames
        urls = []
        for i, img in enumerate(output_images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            frame_name = f"{cache_key}_frame{i:03d}.png"
            if B2_KEY_ID:
                url = b2_upload_png(frame_name, buf.getvalue())
            else:
                url = f"{B2_CDN_BASE_URL}/{frame_name}"
            urls.append(url)

        return {
            "image_urls": urls,
            "cached": False,
            "inference_time_ms": inference_ms,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[SEVA Worker] Pre-loading models ...")
    load_models()
    print("[SEVA Worker] Starting RunPod handler ...")
    runpod.serverless.start({"handler": handler})
