FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    ROOT=/stable-diffusion-webui \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Your Civitai token
ENV CIVITAI_TOKEN="cbc8f589f04ba1a0299b10473a792b42"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates tini jq procps \
    python3 python3-venv python3-pip \
    libgl1 libglib2.0-0 libgoogle-perftools-dev libtcmalloc-minimal4 \
 && rm -rf /var/lib/apt/lists/*

# Clone AUTOMATIC1111
ARG A1111_RELEASE="v1.9.3"
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "$ROOT" \
 && cd "$ROOT" && git checkout -q "${A1111_RELEASE}"

# Install PyTorch and xformers for CUDA 12.1
RUN python3 -m pip install --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir xformers==0.0.25.post1 \
      --index-url https://download.pytorch.org/whl/cu121

# Install A1111 requirements
RUN pip install --no-cache-dir -r "$ROOT/requirements_versions.txt"

# Install extensions
WORKDIR $ROOT/extensions
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git \
 && cd sd-webui-controlnet && git checkout -q 8a5af37 || true
RUN git clone https://github.com/Bing-su/adetailer.git \
 && cd adetailer && git checkout -q 26f1b6a || true
WORKDIR $ROOT

# Install all extension dependencies
RUN pip install --no-cache-dir \
    clip \
    open-clip-torch \
    httpcore==0.15 \
    fastapi==0.94.0 \
    gradio==3.41.2 \
    ultralytics>=8.0.75 \
    mediapipe \
    rich \
    pydantic \
    opencv-contrib-python \
    scikit-image

# Create all required directories
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

# Clone required repositories
WORKDIR $ROOT/repositories
RUN git clone https://github.com/Stability-AI/stablediffusion.git stable-diffusion-stability-ai \
 && git clone https://github.com/Stability-AI/generative-models.git generative-models \
 && git clone https://github.com/crowsonkb/k-diffusion.git k-diffusion \
 && git clone https://github.com/salesforce/BLIP.git BLIP \
 && git clone https://github.com/sczhou/CodeFormer.git CodeFormer
WORKDIR $ROOT

# Download main checkpoint
RUN echo "Downloading primary checkpoint..." \
 && curl -fsSL --retry 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "$ROOT/models/Stable-diffusion/primary_model.safetensors" \
    "https://civitai.com/api/download/models/2071650?type=Model&format=SafeTensor&size=pruned&fp=fp16" \
 && ls -lh "$ROOT/models/Stable-diffusion/primary_model.safetensors"

# Download LoRA models
RUN echo "Downloading nsfw_all_in_one LoRA..." \
 && curl -fsSL --retry 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "$ROOT/models/Lora/nsfw_all_in_one.safetensors" \
    "https://civitai.com/api/download/models/160240?type=Model&format=SafeTensor" \
 && ls -lh "$ROOT/models/Lora/nsfw_all_in_one.safetensors"

RUN echo "Downloading pony_amateur LoRA..." \
 && curl -fsSL --retry 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "$ROOT/models/Lora/pony_amateur.safetensors" \
    "https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor" \
 && ls -lh "$ROOT/models/Lora/pony_amateur.safetensors"

# Download ADetailer models
RUN echo "Downloading ADetailer face model..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/adetailer/face_yolov8n.pt" \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
 && ls -lh "$ROOT/models/adetailer/face_yolov8n.pt"

RUN echo "Downloading ADetailer hand model..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/adetailer/hand_yolov8n.pt" \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt" \
 && ls -lh "$ROOT/models/adetailer/hand_yolov8n.pt"

# Download upscaler
RUN echo "Downloading 4x-UltraSharp upscaler..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/ESRGAN/4x-UltraSharp.pth" \
    "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" \
 && ls -lh "$ROOT/models/ESRGAN/4x-UltraSharp.pth"

# Download ControlNet models
RUN echo "Downloading ControlNet OpenPose..." \
 && curl -L --retry 5 \
    -o "$ROOT/models/ControlNet/control_v11p_sd15_openpose.pth" \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
 && ls -lh "$ROOT/models/ControlNet/control_v11p_sd15_openpose.pth"

RUN echo "Downloading ControlNet Depth..." \
 && curl -L --retry 5 \
    -o "$ROOT/models/ControlNet/control_v11f1p_sd15_depth.pth" \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" \
 && ls -lh "$ROOT/models/ControlNet/control_v11f1p_sd15_depth.pth"

# Set default config
RUN echo '{"sd_model_checkpoint": "primary_model.safetensors", "CLIP_stop_at_last_layers": 2}' > "$ROOT/config.json"

# Pre-compile Python modules
RUN python3 -m compileall "$ROOT" || true

# Create the startup script using heredoc
RUN cat > /start.sh << 'EOF'
#!/bin/bash
set -e
echo "Starting Stable Diffusion WebUI API..."
export COMMANDLINE_ARGS="--api --nowebui --listen --enable-insecure-extension-access --xformers --opt-sdp-attention --skip-install"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
cd /stable-diffusion-webui
exec python3 launch.py --skip-torch-cuda-test --skip-python-version-check --skip-install
EOF

# Make script executable and verify it exists
RUN chmod +x /start.sh \
 && echo "Verifying start.sh:" \
 && ls -la /start.sh \
 && echo "Content of start.sh:" \
 && cat /start.sh

# Set environment variables
ENV COMMANDLINE_ARGS="--api --nowebui --listen --enable-insecure-extension-access --xformers --opt-sdp-attention --skip-install"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=20 \
  CMD curl -fsSL http://127.0.0.1:7860/sdapi/v1/sd-models >/dev/null || exit 1

WORKDIR $ROOT
EXPOSE 7860

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/start.sh"]
