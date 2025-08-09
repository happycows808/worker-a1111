# ------------------------------------------------------------------------------
# Stage 0 — Download artifacts (checkpoint, LoRAs, detectors, upscalers)
# ------------------------------------------------------------------------------

FROM alpine:3.20 AS assets
RUN apk add --no-cache curl ca-certificates bash coreutils jq

# ====== Build args you can override at build time ======
# Civitai files you provided (leave as-is)
ARG CKPT_URL="https://civitai.com/api/download/models/2071650?type=Model&format=SafeTensor&size=pruned&fp=fp16"
ARG LORA_ALLINONE_URL="https://civitai.com/api/download/models/160240?type=Model&format=SafeTensor"
ARG LORA_PONY_AMATEUR_URL="https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor"

# Optional: your Civitai API token (some downloads require it). If set, we'll add an Authorization header.
ARG CIVITAI_TOKEN=""

# ADetailer face detector: you asked for yolov9c; provide a URL if you have one.
# Fallback (works very well): yolov8n face model maintained in ADetailer repos.
ARG ADETAILER_FACE_URL="https://raw.githubusercontent.com/Bing-su/adetailer/main/other/face_yolov8n.pt"
# Optional hand detector
ARG ADETAILER_HAND_URL="https://raw.githubusercontent.com/Bing-su/adetailer/main/other/hand_yolov8n.pt"

# Optional ESRGAN upscaler (4x-UltraSharp). If you provide a URL, we'll bake it.
ARG UPSCALER_URL=""

# Helper to download with optional Bearer token
SHELL ["/bin/sh", "-c"]
RUN set -eux; \
    dl() { \
      URL="$1"; OUT="$2"; \
      if [ -n "$CIVITAI_TOKEN" ]; then \
        curl -fsSL --retry 5 -H "Authorization: Bearer $CIVITAI_TOKEN" -o "$OUT" "$URL"; \
      else \
        curl -fsSL --retry 5 -o "$OUT" "$URL"; \
      fi; \
    }; \
    mkdir -p /artifacts/ckpt /artifacts/lora /artifacts/adetailer /artifacts/esrgan; \
    # Checkpoint
    dl "$CKPT_URL" "/artifacts/ckpt/model.safetensors"; \
    # LoRAs
    dl "$LORA_ALLINONE_URL" "/artifacts/lora/nsfw_all_in_one.safetensors"; \
    dl "$LORA_PONY_AMATEUR_URL" "/artifacts/lora/pony_amateur.safetensors"; \
    # ADetailer detectors
    dl "$ADETAILER_FACE_URL" "/artifacts/adetailer/face.pt" || true; \
    dl "$ADETAILER_HAND_URL" "/artifacts/adetailer/hand.pt" || true; \
    # Optional ESRGAN upscaler
    if [ -n "$UPSCALER_URL" ]; then dl "$UPSCALER_URL" "/artifacts/esrgan/4x-UltraSharp.pth"; fi

# ------------------------------------------------------------------------------
# Stage 1 — Final runtime image with CUDA, A1111, extensions, and assets
# ------------------------------------------------------------------------------

# CUDA runtime w/ cuDNN (works well with Torch + xformers). RunPod GPUs align here.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS final

ENV DEBIAN_FRONTEND=noninteractive \
    ROOT=/stable-diffusion-webui \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Base OS deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git curl wget aria2 ca-certificates tini procps jq \
      libgl1 libglib2.0-0 libgoogle-perftools-dev libtcmalloc-minimal4 \
      fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

# Python
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Clone A1111 (pin to a stable release tag)
ARG A1111_RELEASE="v1.9.3"
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "$ROOT" && \
    cd "$ROOT" && git checkout -q "${A1111_RELEASE}"

# Pre-install torch/torchvision/torchaudio (CUDA 12.1 builds) + xformers
# If xformers mismatches your GPU, we'll fall back to SDPA at runtime.
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir xformers==0.0.24.post1 && \
    pip install --no-cache-dir -r "$ROOT/requirements_versions.txt"

# ----------------- Extensions (pinned) -----------------
WORKDIR $ROOT/extensions
# ControlNet
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git && \
    cd sd-webui-controlnet && git checkout -q 8a5af37 || true
# ADetailer
RUN git clone https://github.com/Bing-su/adetailer.git && \
    cd adetailer && git checkout -q 26f1b6a || true
WORKDIR $ROOT

# ----------------- Models & assets -----------------
# Create folders
RUN mkdir -p \
    "$ROOT/models/Stable-diffusion" \
    "$ROOT/models/VAE" \
    "$ROOT/models/Lora" \
    "$ROOT/models/ESRGAN" \
    "$ROOT/models/adetailer" \
    "$ROOT/models/ControlNet"

# Copy artifacts from Stage 0
COPY --from=assets /artifacts/ckpt/model.safetensors         $ROOT/models/Stable-diffusion/primary_model.safetensors
COPY --from=assets /artifacts/lora/nsfw_all_in_one.safetensors $ROOT/models/Lora/nsfw_all_in_one.safetensors
COPY --from=assets /artifacts/lora/pony_amateur.safetensors    $ROOT/models/Lora/pony_amateur.safetensors
# Detectors (best-effort)
COPY --from=assets /artifacts/adetailer/face.pt              $ROOT/models/adetailer/face_yolo.pt
COPY --from=assets /artifacts/adetailer/hand.pt              $ROOT/models/adetailer/hand_yolo.pt
# Optional upscaler
COPY --from=assets /artifacts/esrgan/4x-UltraSharp.pth       $ROOT/models/ESRGAN/4x-UltraSharp.pth

# ----------------- Launch configuration -----------------
# A1111 flags: API-only, insecure extensions for script access, performance flags
ENV COMMANDLINE_ARGS="--xformers --api --nowebui --listen --enable-insecure-extension-access --opt-sdp-attention"
# Default model & common overrides will be set by API calls; we still warm-start.
ENV PYTHONPATH="$ROOT"

# Healthcheck path
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=10 \
  CMD curl -fsSL http://127.0.0.1:7860/sdapi/v1/sd-models >/dev/null || exit 1

# Start script: verifies assets, starts webui, then warm-starts model
COPY start.sh /start.sh
RUN chmod +x /start.sh

WORKDIR $ROOT
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/start.sh"]
