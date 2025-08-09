FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    ROOT=/stable-diffusion-webui \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates tini jq procps \
    python3 python3-venv python3-pip \
    libgl1 libglib2.0-0 libgoogle-perftools-dev libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Clone AUTOMATIC1111 (pinned)
ARG A1111_RELEASE="v1.9.3"
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "$ROOT" \
 && cd "$ROOT" && git checkout -q "${A1111_RELEASE}"

# Torch/cuDNN for CUDA 12.1 + xformers (fallback to SDPA enabled too)
RUN python3 -m pip install --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir xformers==0.0.24.post1 \
 && pip install --no-cache-dir -r "$ROOT/requirements_versions.txt"

# Extensions (pinned)
WORKDIR $ROOT/extensions
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git \
 && cd sd-webui-controlnet && git checkout -q 8a5af37 || true
RUN git clone https://github.com/Bing-su/adetailer.git \
 && cd adetailer && git checkout -q 26f1b6a || true
WORKDIR $ROOT

# Model folders
RUN mkdir -p \
  "$ROOT/models/Stable-diffusion" \
  "$ROOT/models/VAE" \
  "$ROOT/models/Lora" \
  "$ROOT/models/ESRGAN" \
  "$ROOT/models/adetailer" \
  "$ROOT/models/ControlNet"

# API-only, perf flags
ENV COMMANDLINE_ARGS="--api --nowebui --listen --enable-insecure-extension-access --xformers --opt-sdp-attention"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=10 \
  CMD curl -fsSL http://127.0.0.1:7860/sdapi/v1/sd-models >/dev/null || exit 1

# Start script
COPY start.sh /start.sh
RUN chmod +x /start.sh
WORKDIR $ROOT
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/start.sh"]
