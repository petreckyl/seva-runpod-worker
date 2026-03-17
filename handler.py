"""Minimal debug handler to test if RunPod worker starts correctly."""

import os
import sys
import runpod

print("[DEBUG] Handler starting...", flush=True)
print(f"[DEBUG] Python: {sys.version}", flush=True)
print(f"[DEBUG] Working dir: {os.getcwd()}", flush=True)
print(f"[DEBUG] ENV keys: {list(os.environ.keys())}", flush=True)

try:
    import torch
    print(f"[DEBUG] PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"[DEBUG] PyTorch error: {e}", flush=True)

try:
    import seva
    print(f"[DEBUG] SEVA imported OK", flush=True)
except Exception as e:
    print(f"[DEBUG] SEVA import error: {e}", flush=True)


def handler(job):
    """Simple echo handler for testing."""
    print(f"[DEBUG] Got job: {job}", flush=True)
    return {"status": "ok", "message": "Debug handler works!", "input": job.get("input", {})}


if __name__ == "__main__":
    print("[DEBUG] Starting RunPod handler...", flush=True)
    runpod.serverless.start({"handler": handler})
