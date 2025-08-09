FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    ROOT=/stable-diffusion-webui \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# --- Your Civitai token (hard-coded by request) ---
ENV CIVITAI_TOKEN="cbc8f589f04ba1a0299b10473a792b42"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Base deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates tini jq procps \
    python3 python3-venv python3-pip \
    libgl1 libglib2.0-0 libgoogle-perftools-dev libtcmalloc-minimal4 \
 && rm -rf /var/lib/apt/lists/*

# Clone AUTOMATIC1111 (pinned)
ARG A1111_RELEASE="v1.9.3"
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "$ROOT" \
 && cd "$ROOT" && git checkout -q "${A1111_RELEASE}"

# Torch CUDA 12.1 + xformers (SDPA enabled via launch flags)
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

# Model dirs
RUN mkdir -p \
  "$ROOT/models/Stable-diffusion" \
  "$ROOT/models/Lora" \
  "$ROOT/models/VAE" \
  "$ROOT/models/ESRGAN" \
  "$ROOT/models/adetailer" \
  "$ROOT/models/ControlNet"

# -------------------------
# Bake ALL assets at build
# -------------------------
# URLs you provided
ENV CKPT_URL="https://civitai.com/api/download/models/2071650?type=Model&format=SafeTensor&size=pruned&fp=fp16" \
    LORA_ALL_URL="https://civitai.com/api/download/models/160240?type=Model&format=SafeTensor" \
    LORA_PONY_URL="https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor" \
    ADETAILER_FACE_URL="https://raw.githubusercontent.com/Bing-su/adetailer/main/other/face_yolov8n.pt" \
    ADETAILER_HAND_URL="https://raw.githubusercontent.com/Bing-su/adetailer/main/other/hand_yolov8n.pt" \
    ULTRASHARP_URL="https://huggingface.co/ClashSAN/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth?download=true" \
    CN_OPENPOSE_URL="https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth?download=true" \
    CN_DEPTH_URL="https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth?download=true" \
    CN_LINEART_URL="https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth?download=true"

# Required: checkpoint + LoRAs (use token)
RUN set -euo pipefail; \
  curl -fsSL --retry 5 -H "Authorization: Bearer $CIVITAI_TOKEN" \
    -o "$ROOT/models/Stable-diffusion/primary_model.safetensors" "$CKPT_URL"; \
  curl -fsSL --retry 5 -H "Authorization: Bearer $CIVITAI_TOKEN" \
    -o "$ROOT/models/Lora/nsfw_all_in_one.safetensors" "$LORA_ALL_URL"; \
  curl -fsSL --retry 5 -H "Authorization: Bearer $CIVITAI_TOKEN" \
    -o "$ROOT/models/Lora/pony_amateur.safetensors" "$LORA_PONY_URL";

# Optional assets (no token needed)
RUN set -euo pipefail; \
  curl -fsSL --retry 5 -o "$ROOT/models/adetailer/face_yolo.pt" "$ADETAILER_FACE_URL"; \
  curl -fsSL --retry 5 -o "$ROOT/models/adetailer/hand_yolo.pt" "$ADETAILER_HAND_URL"; \
  curl -fsSL --retry 5 -o "$ROOT/models/ESRGAN/4x-UltraSharp.pth" "$ULTRASHARP_URL"; \
  curl -fsSL --retry 5 -o "$ROOT/models/ControlNet/control_v11p_sd15_openpose.pth" "$CN_OPENPOSE_URL"; \
  curl -fsSL --retry 5 -o "$ROOT/models/ControlNet/control_v11f1p_sd15_depth.pth" "$CN_DEPTH_URL"; \
  curl -fsSL --retry 5 -o "$ROOT/models/ControlNet/control_v11p_sd15_lineart.pth" "$CN_LINEART_URL" || true

# Preconfigure defaults: select checkpoint + Clip Skip = 2
# (A1111 reads these from config.json on first run)
RUN jq -n \
  --arg ckpt "primary_model.safetensors" \
  --argjson clip 2 \
  '{ "sd_model_checkpoint": $ckpt, "CLIP_stop_at_last_layers": $clip }' \
  > "$ROOT/config.json"

# Launch flags: API only, performance
ENV COMMANDLINE_ARGS="--api --nowebui --listen --enable-insecure-extension-access --xformers --opt-sdp-attention"

# Healthcheck (gives time on first boot)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=20 \
  CMD curl -fsSL http://127.0.0.1:7860/sdapi/v1/sd-models >/dev/null || exit 1

# Run A1111 directly (no start.sh)
WORKDIR $ROOT
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","webui.sh","${COMMANDLINE_ARGS}"]
