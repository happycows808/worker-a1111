# ---------------------------------------------------------------------------- #
#                         Stage 1: Download the models                         #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.43.0 AS download

# Install dependencies and set up error handling
RUN apk add --no-cache wget curl bash

# Create directories
RUN mkdir -p /stable-diffusion-webui/models/Lora

# Download main model with proper error handling
RUN echo "Downloading main model..." && \
    curl -L -H "Authorization: Bearer 31daa44aec2ea7c87e3bf582fd4640a9" \
        -o /model.safetensors \
        -f --retry 3 --retry-delay 5 \
        "https://civitai.com/api/download/models/1920896?type=Model&format=SafeTensor&size=full&fp=fp16" || \
    (echo "Failed to download main model" && exit 1)

# Download LoRA models one by one with error handling
RUN echo "Downloading feet_pose_realistic..." && \
    wget --tries=3 --timeout=300 \
        -O /stable-diffusion-webui/models/Lora/feet_pose_realistic.safetensors \
        "https://civitai.green/api/download/models/19130?type=Model&format=SafeTensor&size=full&fp=fp16" || \
    (echo "Failed to download feet_pose_realistic" && exit 1)

RUN echo "Downloading feet_fetish_pony..." && \
    wget --tries=3 --timeout=300 \
        -O /stable-diffusion-webui/models/Lora/feet_fetish_pony.safetensors \
        "https://civitai.green/api/download/models/1442192?type=Model&format=SafeTensor" || \
    (echo "Failed to download feet_fetish_pony" && exit 1)

RUN echo "Downloading innies_better_vulva..." && \
    wget --tries=3 --timeout=300 \
        -O /stable-diffusion-webui/models/Lora/innies_better_vulva.safetensors \
        "https://civitai.green/api/download/models/12873?type=Model&format=SafeTensor&size=full&fp=fp16" || \
    (echo "Failed to download innies_better_vulva" && exit 1)

RUN echo "Downloading pony_amateur..." && \
    curl -L -H "Authorization: Bearer 31daa44aec2ea7c87e3bf582fd4640a9" \
        -o /stable-diffusion-webui/models/Lora/pony_amateur.safetensors \
        -f --retry 3 --retry-delay 5 \
        "https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor" || \
    (echo "Failed to download pony_amateur" && exit 1)

# Verify all files were downloaded
RUN echo "Verifying downloads..." && \
    ls -la /model.safetensors && \
    ls -la /stable-diffusion-webui/models/Lora/

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
