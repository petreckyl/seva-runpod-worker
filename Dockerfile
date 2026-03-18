FROM runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204

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

# Install numpy first (SEVA needs it, but old numpy fails on Python 3.12+)
RUN pip install --no-cache-dir "numpy<2" setuptools

# Clone SEVA and install — patch numpy requirement to avoid rebuild
RUN git clone --recursive https://github.com/Stability-AI/stable-virtual-camera /app/seva-repo \
    && cd /app/seva-repo \
    && sed -i 's/numpy[^"]*"/numpy<2"/g' pyproject.toml setup.cfg setup.py 2>/dev/null || true \
    && pip install --no-cache-dir --no-build-isolation -e . \
    && rm -rf .git */.git

# Python deps (handler-specific)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Skip flash-attn compilation — it takes 20+ min and exceeds RunPod build limit.
# SEVA falls back to standard PyTorch attention (slightly slower but works fine).

# Clean up to reduce image size (safe — no package removal)
RUN rm -rf /root/.cache /tmp/*

# Copy handlers
COPY handler.py /app/handler.py
COPY handler_full.py /app/handler_full.py

CMD ["python", "/app/handler_full.py"]
