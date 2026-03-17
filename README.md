# SEVA RunPod Serverless Worker

RunPod Serverless worker for [SEVA (Stable Virtual Camera)](https://github.com/Stability-AI/stable-virtual-camera) novel view synthesis. Receives a source image + camera preset, generates a new view, caches the result to Backblaze B2, and returns a CDN URL.

## Required Environment Variables

### RunPod Secrets (set in RunPod dashboard):

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token (must have access to `stabilityai/stable-virtual-camera`) |
| `B2_KEY_ID` | Backblaze B2 application key ID |
| `B2_APP_KEY` | Backblaze B2 application key |
| `B2_BUCKET_NAME` | B2 bucket name (e.g. `seva-snapshots`) |
| `B2_CDN_BASE_URL` | CDN base URL (e.g. `https://snapshots.kaleidoscopeai.org/cache`) |

### For local testing (optional):

| Variable | Description |
|----------|-------------|
| `RUNPOD_ENDPOINT_ID` | RunPod endpoint ID for remote testing |
| `RUNPOD_API_KEY` | RunPod API key |

## Build & Push Docker Image

```bash
# Build (downloads weights at build time if HF_TOKEN is provided)
docker build --build-arg HF_TOKEN=hf_your_token_here -t seva-worker .

# Tag and push to Docker Hub (or any registry RunPod can access)
docker tag seva-worker your-dockerhub/seva-worker:latest
docker push your-dockerhub/seva-worker:latest
```

**Alternative: Network Volume** — If you don't want to bake weights into the image (it will be ~15 GB), skip `--build-arg HF_TOKEN` and instead:
1. Create a RunPod Network Volume
2. Download SEVA v1.1 weights to `/runpod-volume/seva-weights/` on the volume
3. Attach the volume to your serverless endpoint

## Deploy on RunPod Serverless

1. Go to [RunPod Console](https://www.runpod.io/console/serverless) → **New Endpoint**
2. Set **Container Image** to `your-dockerhub/seva-worker:latest`
3. Select a GPU tier (recommended: **RTX A5000 24GB** or **A100 40GB**)
4. Add environment variables (`B2_KEY_ID`, `B2_APP_KEY`, `B2_BUCKET_NAME`, `B2_CDN_BASE_URL`)
5. (Optional) Attach Network Volume if weights are stored there
6. Set **Max Workers** based on expected load
7. Deploy

## Backblaze B2 + Cloudflare CDN Setup

### 1. Create B2 Bucket
- Go to [Backblaze B2](https://www.backblaze.com/b2/) → **Create Bucket**
- Bucket name: e.g. `seva-snapshots`
- Set to **Public**
- Create an application key with read/write access

### 2. Configure Cloudflare CDN
- Add your domain to Cloudflare (e.g. `kaleidoscopeai.org`)
- Create a **CNAME record**: `snapshots` → `f000.backblazeb2.com` (your B2 endpoint)
- Add a **Transform Rule** to rewrite the URL path:
  - Match: `snapshots.kaleidoscopeai.org/cache/*`
  - Rewrite to: `/file/seva-snapshots/cache/*` (adjust bucket name)
- Enable caching in Cloudflare (Cache Everything page rule)

The resulting `B2_CDN_BASE_URL` would be: `https://snapshots.kaleidoscopeai.org/cache`

## API Usage

### Request
```json
{
  "image_url": "https://example.com/photo.jpg",
  "camera_preset": "orbit",
  "seed": 42,
  "num_frames": 1
}
```

### Camera Presets
`orbit`, `zoom_in`, `zoom_out`, `pan_left`, `pan_right`, `pan_up`, `pan_down`, `move_forward`, `move_backward`, `dolly_zoom_in`, `dolly_zoom_out`, `spiral`, `roll`, `lemniscate`

### Response
```json
{
  "image_url": "https://snapshots.kaleidoscopeai.org/cache/abc123...def.png",
  "cached": false,
  "inference_time_ms": 4521
}
```

## Local Testing

```bash
# Remote test (against deployed endpoint)
RUNPOD_ENDPOINT_ID=xyz RUNPOD_API_KEY=rp_xxx python test_local.py

# Local test (requires GPU + SEVA installed)
python test_local.py --camera_preset zoom_in --seed 123
```
