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

# Install a matched CPU-only PyTorch stack.
RUN python -m pip install \
    torch==1.13.1+cpu \
    torchvision==0.14.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

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

# Install these later when we move to transformer/LoRA experiments.
# Keeping them out initially makes the FedScale baseline easier to debug.
# The version are pinned to work with PyTorch 1.13.1
#

RUN python -m pip install \
    transformers==4.33.3 \
    peft==0.5.0 \
    accelerate==0.23.0 \
    safetensors==0.3.3 \
    huggingface-hub==0.17.3 \
    tokenizers==0.13.3 \
    sentencepiece

WORKDIR /opt/FedScale

CMD ["/bin/bash"]
