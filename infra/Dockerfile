# ─────────────────────────────────────────────────────────────────────────────
# bonsainet / astra training image
#
# Provides: PyTorch 2.x + cu128, NVIDIA DALI 2.x, FFCV, timm, Hydra, W&B
#
# Build:
#   docker build -t bonsainet:latest -f infra/Dockerfile .
#
# Run (single-node, 2 GPUs):
#   docker run --gpus all --ipc=host --ulimit memlock=-1 \
#     -v /media/data/datasets:/data \
#     -v $(pwd):/workspace \
#     bonsainet:latest \
#     torchrun --nproc_per_node=2 scripts/train_imagenet.py
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── system packages ──────────────────────────────────────────────────────────
# libopencv-dev + pkg-config: required to build ffcv from source
# libjpeg-turbo8-dev: provides JPEG-12 symbols that avoid the LD_PRELOAD hack
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        libopencv-dev pkg-config \
        libjpeg-turbo8-dev \
        git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

# ── PyTorch (cu128) ──────────────────────────────────────────────────────────
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ── NVIDIA DALI ──────────────────────────────────────────────────────────────
RUN pip install nvidia-dali-cuda120

# ── FFCV ─────────────────────────────────────────────────────────────────────
# pkg-config finds system opencv4, so no PKG_CONFIG_PATH tweak needed here
RUN pip install ffcv

# ── ML / training deps ───────────────────────────────────────────────────────
RUN pip install \
    timm \
    hydra-core omegaconf \
    "wandb>=0.23,<0.24" \
    numpy pillow tqdm

# ── astra (local package) ────────────────────────────────────────────────────
# Copy only what pip needs so layer cache stays valid on source changes
WORKDIR /workspace
COPY pyproject.toml ./
COPY astra/ ./astra/
RUN pip install -e . --no-deps   # deps already installed above

# Default working dir at runtime — mount the repo here
WORKDIR /workspace
