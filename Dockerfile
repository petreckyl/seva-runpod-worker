# ---- Stage 1: Build (compile flash-attn and install everything) ----
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps needed for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone SEVA and install
RUN git clone --recursive https://github.com/Stability-AI/stable-virtual-camera /app/seva-repo \
    && cd /app/seva-repo \
    && rm -rf .git */.git \
    && pip install --no-cache-dir -e .

# Python deps (handler-specific)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Flash attention (needs CUDA headers at build time)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Clean up build artifacts to reduce copy size
RUN pip cache purge 2>/dev/null || true \
    && find /usr/local/lib/python3.10 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.10 -name "*.pyc" -delete 2>/dev/null || true \
    && rm -rf /root/.cache /tmp/*

# ---- Stage 2: Runtime (slim, no compilers) ----
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy SEVA repo (needed for -e install)
COPY --from=builder /app/seva-repo /app/seva-repo

# Copy handler
COPY handler.py /app/handler.py

# Weights are expected on RunPod Network Volume at /workspace/seva-weights
CMD ["python", "/app/handler.py"]
