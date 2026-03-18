"""Minimal debug handler — just proves the worker starts and responds."""
import os
import sys
import runpod

print(f"[DEBUG] Python: {sys.version}", flush=True)
print(f"[DEBUG] CWD: {os.getcwd()}", flush=True)

try:
    import torch
    print(f"[DEBUG] PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"[DEBUG] torch error: {e}", flush=True)


def handler(job):
    return {"status": "ok", "input": job.get("input", {})}


print("[DEBUG] Starting runpod handler...", flush=True)
runpod.serverless.start({"handler": handler})
