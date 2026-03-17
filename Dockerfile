FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone SEVA and install
RUN git clone --recursive https://github.com/Stability-AI/stable-virtual-camera /app/seva-repo \
    && cd /app/seva-repo \
    && pip install --no-cache-dir -e .

# Python deps (handler-specific)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Flash attention (needs CUDA at build time)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Copy handler
COPY handler.py /app/handler.py

# Weights are expected on RunPod Network Volume at /workspace/seva-weights
CMD ["python", "/app/handler.py"]
