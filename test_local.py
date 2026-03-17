"""
Local test script for the SEVA RunPod worker.

Usage:
    python test_local.py
    python test_local.py --image_url "https://example.com/photo.jpg"
    python test_local.py --camera_preset zoom_in --seed 123

Requires RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY env vars for remote testing,
or falls back to simulating a local handler call.
"""

import json
import os
import sys
import time
import argparse


def test_remote(endpoint_id: str, api_key: str, payload: dict):
    """Test against a deployed RunPod endpoint."""
    import requests

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"[TEST] Sending request to RunPod endpoint {endpoint_id} ...")
    print(f"[TEST] Payload: {json.dumps(payload, indent=2)}")

    t0 = time.time()
    resp = requests.post(url, json={"input": payload}, headers=headers, timeout=300)
    elapsed = time.time() - t0

    print(f"\n[TEST] Status: {resp.status_code}")
    print(f"[TEST] Round-trip time: {elapsed:.1f}s")
    print(f"[TEST] Response:\n{json.dumps(resp.json(), indent=2)}")


def test_local(payload: dict):
    """Simulate a local handler call (requires GPU + SEVA installed)."""
    print("[TEST] Running local handler simulation ...")
    print(f"[TEST] Payload: {json.dumps(payload, indent=2)}")

    try:
        from handler import handler

        job = {"id": "test-local-001", "input": payload}

        t0 = time.time()
        result = handler(job)
        elapsed = time.time() - t0

        print(f"\n[TEST] Handler time: {elapsed:.1f}s")
        print(f"[TEST] Result:\n{json.dumps(result, indent=2)}")
    except ImportError:
        print("[TEST] Cannot import handler locally (missing SEVA/GPU?).")
        print("[TEST] Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY for remote testing.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test SEVA RunPod worker")
    parser.add_argument(
        "--image_url",
        default="https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800",
        help="Source image URL",
    )
    parser.add_argument("--camera_preset", default="orbit", help="Camera preset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of output frames")
    args = parser.parse_args()

    payload = {
        "image_url": args.image_url,
        "camera_preset": args.camera_preset,
        "seed": args.seed,
        "num_frames": args.num_frames,
    }

    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")

    if endpoint_id and api_key:
        test_remote(endpoint_id, api_key, payload)
    else:
        test_local(payload)


if __name__ == "__main__":
    main()
