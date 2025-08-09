FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies (jq included!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl wget ca-certificates tini jq procps \
    python3 python3-venv python3-pip python-is-python3 \
    libgl1 libglib2.0-0 libgoogle-perftools-dev libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Clone A1111 (using master branch which exists)
WORKDIR /
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /stable-diffusion-webui

WORKDIR /stable-diffusion-webui

# Pre-install Python deps to speed up first boot
RUN python -m pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
RUN pip install xformers==0.0.24

# Install extensions
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git /stable-diffusion-webui/extensions/sd-webui-controlnet && \
    git clone https://github.com/Bing-su/adetailer.git /stable-diffusion-webui/extensions/adetailer

# Create model directories
RUN mkdir -p models/Stable-diffusion models/Lora models/VAE models/ESRGAN && \
    mkdir -p /stable-diffusion-webui/extensions/adetailer/models

# Download ADetailer face model (using yolov8n which definitely exists)
RUN curl -L -o /stable-diffusion-webui/extensions/adetailer/models/face_yolov8n.pt \
    https://github.com/Bing-su/adetailer/raw/main/adetailer/model_yolov8n.pt

# Your Civitai token
ENV CIVITAI_TOKEN=cbc8f589f04ba1a0299b10473a792b42

# Download your models with token
RUN curl -H "Authorization: Bearer ${CIVITAI_TOKEN}" -L -o models/Stable-diffusion/main.safetensors \
    "https://civitai.com/api/download/models/2071650?type=Model&format=SafeTensor&size=pruned&fp=fp16"

RUN curl -H "Authorization: Bearer ${CIVITAI_TOKEN}" -L -o models/Lora/nsfw_all_in_one.safetensors \
    "https://civitai.com/api/download/models/160240?type=Model&format=SafeTensor"

RUN curl -H "Authorization: Bearer ${CIVITAI_TOKEN}" -L -o models/Lora/pony_amateur.safetensors \
    "https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor"

# Download 4x-UltraSharp upscaler
RUN curl -L -o models/ESRGAN/4x-UltraSharp.pth \
    "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"

# Set default config (checkpoint + clip skip)
RUN echo '{"sd_model_checkpoint": "main.safetensors", "CLIP_stop_at_last_layers": 2}' > config.json

# A1111 launch arguments
ENV COMMANDLINE_ARGS="--listen --port 7860 --api --xformers --enable-insecure-extension-access --no-half-vae --no-download-sd-model"

EXPOSE 7860

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "launch.py"]
