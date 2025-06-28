# ---------------------------------------------------------------------------- #
#                         Stage 1: Download the models                         #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.43.0 AS download

RUN apk add --no-cache wget curl && \
    mkdir -p /stable-diffusion-webui/models/Lora && \
    # Main model - Pony Realism (using curl for auth)
    curl -L -H "Authorization: Bearer 31daa44aec2ea7c87e3bf582fd4640a9" \
        -o /model.safetensors \
        "https://civitai.com/api/download/models/1920896?type=Model&format=SafeTensor&size=full&fp=fp16" && \
    # LoRA 1 - Feet Pose (public, using wget)
    wget -q -O /stable-diffusion-webui/models/Lora/feet_pose_realistic.safetensors \
        "https://civitai.green/api/download/models/19130?type=Model&format=SafeTensor&size=full&fp=fp16" && \
    # LoRA 2 - Feet Fetish Pony (public, using wget)
    wget -q -O /stable-diffusion-webui/models/Lora/feet_fetish_pony.safetensors \
        "https://civitai.green/api/download/models/1442192?type=Model&format=SafeTensor" && \
    # LoRA 3 - Innies (public, using wget)
    wget -q -O /stable-diffusion-webui/models/Lora/innies_better_vulva.safetensors \
        "https://civitai.green/api/download/models/12873?type=Model&format=SafeTensor&size=full&fp=fp16" && \
    # LoRA 4 - Pony Amateur (using curl for auth)
    curl -L -H "Authorization: Bearer 31daa44aec2ea7c87e3bf582fd4640a9" \
        -o /stable-diffusion-webui/models/Lora/pony_amateur.safetensors \
        "https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor"

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
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

COPY --from=download /stable-diffusion-webui/models/Lora /stable-diffusion-webui/models/Lora

COPY --from=download /model.safetensors /model.safetensors

# install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY test_input.json .

ADD src .

RUN chmod +x /start.sh
CMD /start.sh
