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

# Clone SEVA and install (remove .git to save space)
RUN git clone --recursive https://github.com/Stability-AI/stable-virtual-camera /app/seva-repo \
    && cd /app/seva-repo \
    && pip install --no-cache-dir -e . \
    && rm -rf .git */.git

# Python deps (handler-specific)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Skip flash-attn compilation — it takes 20+ min and exceeds RunPod build limit.
# SEVA falls back to standard PyTorch attention (slightly slower but works fine).

# Clean up to reduce image size
RUN pip cache purge 2>/dev/null || true \
    && rm -rf /root/.cache /tmp/* \
    && find /usr/local/lib/python3.10 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.10 -name "*.pyc" -delete 2>/dev/null || true \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

# Copy handler
COPY handler.py /app/handler.py

# Weights are expected on RunPod Network Volume at /workspace/seva-weights
CMD ["python", "/app/handler.py"]
