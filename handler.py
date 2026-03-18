"""Absolute minimum handler — no torch, no deps."""
import os
import sys
import runpod

print(f"[DEBUG] Python: {sys.version}", flush=True)
print(f"[DEBUG] CWD: {os.getcwd()}", flush=True)
print(f"[DEBUG] Disk: checking...", flush=True)
os.system("df -h / | tail -1")


def handler(job):
    return {"status": "ok", "msg": "hello from seva-worker"}


print("[DEBUG] Starting handler...", flush=True)
runpod.serverless.start({"handler": handler})
