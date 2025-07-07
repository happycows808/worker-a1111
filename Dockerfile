# ---------------------------------------------------------------------------- #
#                         Stage 1: Download the models                         #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.43.0 AS download

# Install dependencies
RUN apk add --no-cache wget curl bash

# Create directories
RUN mkdir -p /stable-diffusion-webui/models/Lora

# Download main model - MUST succeed for build to continue
RUN echo "Downloading main model (REQUIRED)..." && \
    curl -L -H "Authorization: Bearer 31daa44aec2ea7c87e3bf582fd4640a9" \
        -o /model.safetensors \
        -f --retry 3 --retry-delay 5 \
        "https://civitai.com/api/download/models/915814?type=Model&format=SafeTensor&size=pruned&fp=fp16" || \
    (echo "ERROR: Failed to download main model - this is required!" && exit 1)

# Download LoRA models - failures are non-critical
# Using curl with auth for all models since civitai.green redirects require authentication

RUN echo "Downloading feet_pose_realistic (optional)..." && \
    curl -L -H "Authorization: Bearer 31daa44aec2ea7c87e3bf582fd4640a9" \
        -o /stable-diffusion-webui/models/Lora/feet_pose_realistic.safetensors \
        -f --retry 3 --retry-delay 5 \
        "https://civitai.com/api/download/models/19130?type=Model&format=SafeTensor&size=full&fp=fp16" || \
    echo "Warning: Failed to download feet_pose_realistic, continuing without it..."

RUN echo "Downloading feet_fetish_pony (optional)..." && \
    curl -L -H "Authorization: Bearer 31daa44aec2ea7c87e3bf582fd4640a9" \
        -o /stable-diffusion-webui/models/Lora/feet_fetish_pony.safetensors \
        -f --retry 3 --retry-delay 5 \
        "https://civitai.com/api/download/models/1442192?type=Model&format=SafeTensor" || \
    echo "Warning: Failed to download feet_fetish_pony, continuing without it..."

RUN echo "Downloading innies_better_vulva (optional)..." && \
    curl -L -H "Authorization: Bearer 31daa44aec2ea7c87e3bf582fd4640a9" \
        -o /stable-diffusion-webui/models/Lora/innies_better_vulva.safetensors \
        -f --retry 3 --retry-delay 5 \
        "https://civitai.com/api/download/models/12873?type=Model&format=SafeTensor&size=full&fp=fp16" || \
    echo "Warning: Failed to download innies_better_vulva, continuing without it..."

RUN echo "Downloading pony_amateur (optional)..." && \
    curl -L -H "Authorization: Bearer 31daa44aec2ea7c87e3bf582fd4640a9" \
        -o /stable-diffusion-webui/models/Lora/pony_amateur.safetensors \
        -f --retry 3 --retry-delay 5 \
        "https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor" || \
    echo "Warning: Failed to download pony_amateur, continuing without it..."

# Verify downloads and clean up any empty files
RUN echo "Verifying downloads..." && \
    echo "Main model:" && ls -la /model.safetensors && \
    echo "LoRA models:" && ls -la /stable-diffusion-webui/models/Lora/ && \
    find /stable-diffusion-webui/models/Lora/ -type f -size 0 -delete && \
    echo "Cleaned up any empty files"

# ---------------------------------------------------------------------------- #
#                        Stage 2: Build the final image                        #
# ---------------------------------------------------------------------------- #
FROM python:3.10.14-slim AS build_final_image

ARG A1111_RELEASE=v1.9.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev \
    libtcmalloc-minimal4 procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copy models from download stage
COPY --from=download /stable-diffusion-webui/models/Lora /stable-diffusion-webui/models/Lora
COPY --from=download /model.safetensors /model.safetensors

# Install additional dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy test input and source files
COPY test_input.json .
ADD src .

# Make start script executable
RUN chmod +x /start.sh

CMD ["/start.sh"]
