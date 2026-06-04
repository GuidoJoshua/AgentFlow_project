FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    curl \
    ffmpeg \
    git \
    jq \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    parallel \
    tmux \
    && rm -rf /var/lib/apt/lists/*