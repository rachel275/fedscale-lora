FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV FEDSCALE_HOME=/opt/FedScale
ENV PYTHONPATH=/opt/FedScale
ENV PYTHONUNBUFFERED=1

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    unzip \
    nano \
    openssh-client \
    procps \
    iproute2 \
    libglib2.0-0 \
    libgl1 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch stack.
# Python 3.10 is intentionally retained because it is already known to work
# with this FedScale environment. PyTorch is upgraded because current
# bitsandbytes releases require PyTorch >= 2.4.
RUN python -m pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu

# FedScale/runtime dependencies.
RUN python -m pip install \
    "numpy<2" \
    pandas \
    scipy \
    scikit-learn \
    pyyaml \
    overrides \
    grpcio \
    protobuf \
    psutil \
    wandb \
    tensorboard \
    tensorboardX \
    redis \
    boto3 \
    h5py \
    pillow

# Transformer / LoRA / QLoRA stack.
RUN python -m pip install \
    "transformers==4.45.0" \
    "peft==0.13.2" \
    "accelerate==0.34.2" \
    "bitsandbytes==0.49.2" \
    "safetensors>=0.4.3" \
    "huggingface-hub>=0.24.0,<1.0" \
    "tokenizers>=0.20,<0.21" \
    sentencepiece

# Fail the image build early if the core LoRA/QLoRA stack cannot import.
RUN python - <<'PY'
import torch
import torchvision
import transformers
import peft
import accelerate
import bitsandbytes as bnb

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
print("bitsandbytes:", bnb.__version__)
print("cuda available:", torch.cuda.is_available())
PY

WORKDIR /opt/FedScale

CMD ["/bin/bash"]
