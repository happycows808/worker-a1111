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

# IMPORTANT: Install RunPod SDK for serverless
RUN pip install --no-cache-dir runpod

# Install extensions
WORKDIR $ROOT/extensions
RUN git clone https://github.com/Mikubill/sd-webui-controlnet.git \
 && cd sd-webui-controlnet && git checkout -q 8a5af37 || true
RUN git clone https://github.com/Bing-su/adetailer.git \
 && cd adetailer && git checkout -q 26f1b6a || true
WORKDIR $ROOT

# Install all extension dependencies INCLUDING controlnet_aux fix
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
    scikit-image \
    controlnet_aux

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

# CREATE RUNPOD HANDLER with dynamic port detection
RUN cat > /handler.py << 'EOF'
import runpod
import requests
import time
import json
import os

# Get port from environment variable - this ensures consistency
API_PORT = os.getenv("API_PORT", "7860")
API_URL = f"http://localhost:{API_PORT}"

def wait_for_service(timeout=120):
    """Wait for SD WebUI to be ready"""
    start = time.time()
    url = f"{API_URL}/sdapi/v1/options"
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ SD WebUI is ready at {url}")
                return True
        except Exception as e:
            print(f"⏳ Waiting for SD WebUI on port {API_PORT}... ({int(time.time() - start)}s)")
            pass
        time.sleep(3)
    return False

def handler(job):
    """Handler function for RunPod serverless"""
    try:
        job_input = job["input"]
        
        # Log the incoming request
        print(f"📥 Received job: {job['id']}")
        print(f"🔧 Using API endpoint: {API_URL}")
        
        # Wait for SD WebUI to be ready
        if not wait_for_service():
            return {"error": f"SD WebUI failed to start on port {API_PORT} after 120 seconds"}
        
        # Set default values if not provided
        default_params = {
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7,
            "sampler_name": "Euler a",
            "batch_size": 1,
            "n_iter": 1,
            "enable_hr": False,
            "denoising_strength": 0.7,
            "hr_scale": 2,
            "hr_upscaler": "4x-UltraSharp"
        }
        
        # Merge defaults with input
        for key, value in default_params.items():
            if key not in job_input:
                job_input[key] = value
        
        # Handle LoRA if specified
        if "lora" in job_input:
            lora_name = job_input.pop("lora")
            lora_map = {
                "nsfw": "nsfw_all_in_one",
                "all": "nsfw_all_in_one",
                "pony": "pony_amateur",
                "amateur": "pony_amateur"
            }
            if lora_name in lora_map:
                # Add LoRA to prompt
                lora_file = lora_map[lora_name]
                job_input["prompt"] = f"<lora:{lora_file}:1> {job_input.get('prompt', '')}"
        
        # Forward the request to SD WebUI API
        print(f"📤 Sending request to {API_URL}/sdapi/v1/txt2img")
        response = requests.post(
            f"{API_URL}/sdapi/v1/txt2img",
            json=job_input,
            timeout=300
        )
        
        if response.status_code != 200:
            return {"error": f"SD API error: {response.status_code}", "details": response.text}
        
        result = response.json()
        print(f"✅ Job {job['id']} completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error processing job {job.get('id', 'unknown')}: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    print(f"🚀 Starting RunPod handler (API port: {API_PORT})...")
    runpod.serverless.start({"handler": handler})
EOF

# Create startup script with guaranteed port consistency
RUN cat > /start.sh << 'EOF'
#!/bin/bash
set -e

# SINGLE SOURCE OF TRUTH FOR PORT
export API_PORT=7860

echo "🚀 Starting Stable Diffusion WebUI API on port $API_PORT..."
export COMMANDLINE_ARGS="--api --nowebui --port $API_PORT --listen --enable-insecure-extension-access --xformers --opt-sdp-attention --skip-install"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

cd /stable-diffusion-webui

# Start SD WebUI in background
echo "📦 Launching SD WebUI on port $API_PORT..."
python3 launch.py --skip-torch-cuda-test --skip-python-version-check --skip-install &
SD_PID=$!

# Wait a bit for SD to start initializing
sleep 10

# Start RunPod handler (it will use the same API_PORT env var)
echo "🔌 Starting RunPod handler (connecting to port $API_PORT)..."
python3 /handler.py &
HANDLER_PID=$!

# Monitor both processes
echo "👀 Monitoring processes..."
wait $SD_PID $HANDLER_PID
EOF

# Make scripts executable
RUN chmod +x /start.sh /handler.py

# Verify everything is in place
RUN echo "=== Verification ===" \
 && echo "Scripts:" && ls -la /start.sh /handler.py \
 && echo "Models:" && ls -la "$ROOT/models/Stable-diffusion/" \
 && echo "LoRAs:" && ls -la "$ROOT/models/Lora/" \
 && echo "Upscalers:" && ls -la "$ROOT/models/ESRGAN/" \
 && echo "==================="

# Set environment variables
ENV COMMANDLINE_ARGS="--api --nowebui --port 7860 --listen --enable-insecure-extension-access --xformers --opt-sdp-attention --skip-install"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

WORKDIR $ROOT
EXPOSE 7860

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/start.sh"]
