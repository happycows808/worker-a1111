FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    ROOT=/stable-diffusion-webui \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Your Civitai token (hard-coded as requested)
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

# Torch CUDA 12.1 + xformers
RUN python3 -m pip install --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir xformers==0.0.24.post1

# Install A1111 requirements
RUN pip install --no-cache-dir -r "$ROOT/requirements_versions.txt"

# Extensions (pinned)
WORKDIR $ROOT/extensions
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git \
 && cd sd-webui-controlnet && git checkout -q 8a5af37 || true
RUN git clone https://github.com/Bing-su/adetailer.git \
 && cd adetailer && git checkout -q 26f1b6a || true
WORKDIR $ROOT

# PRE-INSTALL ALL EXTENSION DEPENDENCIES (this is the key part!)
# Install A1111 base dependencies
RUN pip install --no-cache-dir \
    clip \
    open-clip-torch \
    httpcore==0.15 \
    fastapi==0.94.0 \
    gradio==3.41.2

# Install ADetailer dependencies specifically
RUN pip install --no-cache-dir \
    ultralytics>=8.0.75 \
    mediapipe \
    rich \
    pydantic

# Install ControlNet dependencies
RUN pip install --no-cache-dir \
    opencv-contrib-python \
    scikit-image

# Pre-create all necessary directories
RUN mkdir -p \
  "$ROOT/models/Stable-diffusion" \
  "$ROOT/models/Lora" \
  "$ROOT/models/VAE" \
  "$ROOT/models/ESRGAN" \
  "$ROOT/models/adetailer" \
  "$ROOT/models/ControlNet" \
  "$ROOT/repositories" \
  "$ROOT/embeddings" \
  "$ROOT/textual_inversion"

# Clone required repositories that A1111 needs
WORKDIR $ROOT/repositories
RUN git clone https://github.com/Stability-AI/stablediffusion.git stable-diffusion-stability-ai \
 && git clone https://github.com/Stability-AI/generative-models.git generative-models \
 && git clone https://github.com/crowsonkb/k-diffusion.git k-diffusion \
 && git clone https://github.com/salesforce/BLIP.git BLIP \
 && git clone https://github.com/sczhou/CodeFormer.git CodeFormer
WORKDIR $ROOT

# Download models with build-time checks
RUN echo "Downloading checkpoint..." \
 && curl -fsSL --retry 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "$ROOT/models/Stable-diffusion/primary_model.safetensors" \
    "https://civitai.com/api/download/models/2071650?type=Model&format=SafeTensor&size=pruned&fp=fp16" \
 && echo "Checkpoint downloaded successfully"

RUN echo "Downloading nsfw_all_in_one LoRA..." \
 && curl -fsSL --retry 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "$ROOT/models/Lora/nsfw_all_in_one.safetensors" \
    "https://civitai.com/api/download/models/160240?type=Model&format=SafeTensor" \
 && echo "nsfw_all_in_one downloaded successfully"

RUN echo "Downloading pony_amateur LoRA..." \
 && curl -fsSL --retry 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "$ROOT/models/Lora/pony_amateur.safetensors" \
    "https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor" \
 && echo "pony_amateur downloaded successfully"

# Download ADetailer models
RUN echo "Downloading ADetailer face model..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/adetailer/face_yolov8n.pt" \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
 && echo "Face model downloaded"

RUN echo "Downloading ADetailer hand model..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/adetailer/hand_yolov8n.pt" \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt" \
 && echo "Hand model downloaded"

# Download 4x-UltraSharp upscaler
RUN echo "Downloading 4x-UltraSharp..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/ESRGAN/4x-UltraSharp.pth" \
    "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" \
 && echo "4x-UltraSharp downloaded"

# Download ControlNet models (with progress)
RUN echo "Downloading ControlNet OpenPose..." \
 && curl -L --retry 5 \
    -o "$ROOT/models/ControlNet/control_v11p_sd15_openpose.pth" \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
 && echo "OpenPose downloaded"

RUN echo "Downloading ControlNet Depth..." \
 && curl -L --retry 5 \
    -o "$ROOT/models/ControlNet/control_v11f1p_sd15_depth.pth" \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" \
 && echo "Depth downloaded"

# Set default config
RUN echo '{"sd_model_checkpoint": "primary_model.safetensors", "CLIP_stop_at_last_layers": 2}' > "$ROOT/config.json"

# Pre-compile Python modules to speed up startup
RUN python3 -m compileall "$ROOT" || true

# Create a startup script that bypasses dependency checks
RUN echo '#!/bin/bash\n\
export COMMANDLINE_ARGS="--api --nowebui --listen --enable-insecure-extension-access --xformers --opt-sdp-attention --skip-install"\n\
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"\n\
cd /stable-diffusion-webui\n\
python3 launch.py --skip-torch-cuda-test --skip-python-version-check --skip-install\n\
' > /start.sh && chmod +x /start.sh

# Launch flags
ENV COMMANDLINE_ARGS="--api --nowebui --listen --enable-insecure-extension-access --xformers --opt-sdp-attention --skip-install"

# Healthcheck with longer start period for big models
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=20 \
  CMD curl -fsSL http://127.0.0.1:7860/sdapi/v1/sd-models >/dev/null || exit 1

# Set tcmalloc for performance
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

WORKDIR $ROOT
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/start.sh"]
