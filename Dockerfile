FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    ROOT=/stable-diffusion-webui \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024 \
    API_PORT=7860

# Your Civitai token (kept as requested)
ENV CIVITAI_TOKEN="cbc8f589f04ba1a0299b10473a792b42"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl wget ca-certificates tini jq procps \
    python3 python3-venv python3-pip \
    libgl1 libglib2.0-0 libgoogle-perftools-dev libtcmalloc-minimal4 \
 && rm -rf /var/lib/apt/lists/*

# Clone AUTOMATIC1111 - using latest version for better SDXL support
ARG A1111_RELEASE="v1.9.3"
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "$ROOT" \
 && cd "$ROOT" && git checkout -q "${A1111_RELEASE}"

# Install PyTorch and xformers for CUDA 12.1 - SDXL compatible versions
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

# Clone required repositories for SDXL support
WORKDIR $ROOT/repositories
RUN git clone https://github.com/Stability-AI/stablediffusion.git stable-diffusion-stability-ai \
 && git clone https://github.com/Stability-AI/generative-models.git generative-models \
 && git clone https://github.com/crowsonkb/k-diffusion.git k-diffusion \
 && git clone https://github.com/salesforce/BLIP.git BLIP \
 && git clone https://github.com/sczhou/CodeFormer.git CodeFormer
WORKDIR $ROOT

# Download CyberRealistic Pony SDXL model (keeping your original)
RUN echo "Downloading CyberRealistic Pony SDXL model..." \
 && curl -fsSL --retry 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "$ROOT/models/Stable-diffusion/cyberrealistic_pony.safetensors" \
    "https://civitai.com/api/download/models/2071650?type=Model&format=SafeTensor&size=pruned&fp=fp16" \
 && [ -f "$ROOT/models/Stable-diffusion/cyberrealistic_pony.safetensors" ] \
 && [ -s "$ROOT/models/Stable-diffusion/cyberrealistic_pony.safetensors" ] \
 || (echo "Failed to download primary model" && exit 1) \
 && ls -lh "$ROOT/models/Stable-diffusion/cyberrealistic_pony.safetensors"

# Download SDXL VAE (NOT SD1.5 VAE!)
RUN echo "Downloading SDXL VAE..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/VAE/sdxl_vae.safetensors" \
    "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors" \
 && [ -f "$ROOT/models/VAE/sdxl_vae.safetensors" ] \
 && [ -s "$ROOT/models/VAE/sdxl_vae.safetensors" ] \
 || (echo "Failed to download SDXL VAE" && exit 1)

# Download LoRA models (keeping your NSFW ones)
RUN echo "Downloading nsfw_all_in_one LoRA..." \
 && curl -fsSL --retry 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "$ROOT/models/Lora/nsfw_all_in_one.safetensors" \
    "https://civitai.com/api/download/models/160240?type=Model&format=SafeTensor" \
 && [ -f "$ROOT/models/Lora/nsfw_all_in_one.safetensors" ] \
 && [ -s "$ROOT/models/Lora/nsfw_all_in_one.safetensors" ] \
 || (echo "Failed to download nsfw_all_in_one LoRA" && exit 1) \
 && ls -lh "$ROOT/models/Lora/nsfw_all_in_one.safetensors"

RUN echo "Downloading pony_amateur LoRA..." \
 && curl -fsSL --retry 5 -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "$ROOT/models/Lora/pony_amateur.safetensors" \
    "https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor" \
 && [ -f "$ROOT/models/Lora/pony_amateur.safetensors" ] \
 && [ -s "$ROOT/models/Lora/pony_amateur.safetensors" ] \
 || (echo "Failed to download pony_amateur LoRA" && exit 1) \
 && ls -lh "$ROOT/models/Lora/pony_amateur.safetensors"

# Download ADetailer models
RUN echo "Downloading ADetailer face model..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/adetailer/face_yolov8n.pt" \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
 && [ -f "$ROOT/models/adetailer/face_yolov8n.pt" ] \
 && [ -s "$ROOT/models/adetailer/face_yolov8n.pt" ] \
 || (echo "Failed to download ADetailer face model" && exit 1) \
 && ls -lh "$ROOT/models/adetailer/face_yolov8n.pt"

RUN echo "Downloading ADetailer hand model..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/adetailer/hand_yolov8n.pt" \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt" \
 && [ -f "$ROOT/models/adetailer/hand_yolov8n.pt" ] \
 && [ -s "$ROOT/models/adetailer/hand_yolov8n.pt" ] \
 || (echo "Failed to download ADetailer hand model" && exit 1) \
 && ls -lh "$ROOT/models/adetailer/hand_yolov8n.pt"

# Download upscaler
RUN echo "Downloading 4x-UltraSharp upscaler..." \
 && curl -fsSL --retry 5 \
    -o "$ROOT/models/ESRGAN/4x-UltraSharp.pth" \
    "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" \
 && [ -f "$ROOT/models/ESRGAN/4x-UltraSharp.pth" ] \
 && [ -s "$ROOT/models/ESRGAN/4x-UltraSharp.pth" ] \
 || (echo "Failed to download 4x-UltraSharp upscaler" && exit 1) \
 && ls -lh "$ROOT/models/ESRGAN/4x-UltraSharp.pth"

# Skip ControlNet models for now (SD1.5 ControlNet won't work with SDXL)

# Set default config for SDXL
RUN echo '{"sd_model_checkpoint": "cyberrealistic_pony.safetensors", "sd_vae": "sdxl_vae.safetensors", "CLIP_stop_at_last_layers": 2}' > "$ROOT/config.json"

# Pre-compile Python modules
RUN python3 -m compileall "$ROOT" || true

# Copy UPDATED handler script for SDXL/Pony
COPY <<'HANDLER_EOF' /handler.py
import runpod
import requests
import time
import json
import os
import base64
from typing import Dict, Any, Optional

# Get port from environment variable
API_PORT = os.getenv("API_PORT", "7860")
API_URL = f"http://localhost:{API_PORT}"

def wait_for_service(timeout=180):
    """Wait for SD WebUI to be fully ready - SDXL needs more time"""
    start = time.time()
    urls_to_check = [
        f"{API_URL}/sdapi/v1/options",
        f"{API_URL}/sdapi/v1/sd-models",
        f"{API_URL}/sdapi/v1/samplers"
    ]
    
    while time.time() - start < timeout:
        try:
            all_ready = True
            for url in urls_to_check:
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    all_ready = False
                    break
            
            if all_ready:
                print(f"✅ SD WebUI is fully ready")
                # Get available samplers for debugging
                try:
                    samplers_response = requests.get(f"{API_URL}/sdapi/v1/samplers", timeout=5)
                    if samplers_response.status_code == 200:
                        samplers = samplers_response.json()
                        print(f"Available samplers: {[s['name'] for s in samplers]}")
                except:
                    pass
                return True
        except Exception as e:
            print(f"⏳ Waiting for SD WebUI on port {API_PORT}... ({int(time.time() - start)}s)")
            pass
        time.sleep(5)
    return False

def handle_txt2img(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle text-to-image generation for SDXL/Pony"""
    
    # Add Pony score tags if not present
    prompt = params.get("prompt", "")
    if not prompt.startswith("score_"):
        prompt = "score_9, score_8_up, score_7_up, " + prompt
        params["prompt"] = prompt
    
    # SDXL/Pony-specific defaults
    default_params = {
        "width": 896,  # SDXL resolution
        "height": 1152,  # SDXL resolution
        "steps": 30,  # Pony needs 30+ steps
        "cfg_scale": 5,  # Pony uses CFG 5
        "sampler_name": "DPM++ 2M Karras",  # Try this first
        "batch_size": 1,
        "n_iter": 1,
        "enable_hr": False,
        "denoising_strength": 0.35,
        "hr_scale": 1.5,
        "hr_upscaler": "4x-UltraSharp",
        "negative_prompt": "score_6, score_5, score_4, (worst quality:1.2), (low quality:1.2), (normal quality:1.2), lowres, bad anatomy, bad hands, signature, watermarks, ugly, imperfect eyes, skewed eyes, unnatural face, unnatural body, error, extra limb, missing limbs",
        "override_settings": {
            "sd_model_checkpoint": "cyberrealistic_pony.safetensors",
            "sd_vae": "sdxl_vae.safetensors",
            "CLIP_stop_at_last_layers": 2
        }
    }
    
    # Try multiple sampler names in case one fails
    sampler_fallbacks = [
        "DPM++ 2M Karras",
        "DPM++ SDE Karras", 
        "Euler a",
        "Euler",
        "DPM++ 2M",
        "DDIM"
    ]
    
    # Merge defaults with input
    for key, value in default_params.items():
        if key not in params:
            params[key] = value
    
    # Handle LoRA if specified
    if "lora" in params:
        lora_name = params.pop("lora")
        lora_map = {
            "nsfw": "nsfw_all_in_one",
            "all": "nsfw_all_in_one",
            "pony": "pony_amateur",
            "amateur": "pony_amateur"
        }
        if lora_name in lora_map:
            lora_file = lora_map[lora_name]
            params["prompt"] = f"<lora:{lora_file}:0.8> {params.get('prompt', '')}"
    
    # Configure ADetailer if enabled
    if params.get("enable_adetailer", False):  # Disabled by default for now
        params["alwayson_scripts"] = {
            "ADetailer": {
                "args": [
                    True,
                    False,
                    {
                        "ad_model": "face_yolov8n.pt",
                        "ad_confidence": 0.3,
                        "ad_dilate_erode": 4,
                        "ad_mask_blur": 4,
                        "ad_denoising_strength": 0.3,
                        "ad_inpaint_only_masked": True,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_use_inpaint_width_height": False,
                        "ad_use_steps": False,
                        "ad_use_cfg_scale": False
                    }
                ]
            }
        }
    
    # Try different samplers if one fails
    last_error = None
    for sampler in sampler_fallbacks:
        try:
            params["sampler_name"] = sampler
            print(f"Trying sampler: {sampler}")
            response = requests.post(f"{API_URL}/sdapi/v1/txt2img", json=params, timeout=300)
            if response.status_code == 200:
                print(f"Successfully used sampler: {sampler}")
                return response.json()
            elif "Sampler not found" in response.text:
                last_error = f"Sampler {sampler} not found"
                continue
            else:
                raise Exception(f"SD API error: {response.status_code} - {response.text}")
        except Exception as e:
            if "Sampler" in str(e):
                last_error = str(e)
                continue
            raise
    
    # If all samplers failed, raise the last error
    raise Exception(f"All samplers failed. Last error: {last_error}")

def handle_img2img(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle image-to-image generation for SDXL"""
    if "init_images" not in params:
        raise ValueError("init_images is required for img2img")
    
    default_params = {
        "resize_mode": 0,
        "denoising_strength": 0.5,
        "width": 896,
        "height": 1152,
        "cfg_scale": 5,
        "steps": 30,
        "sampler_name": "DPM++ 2M Karras",
        "override_settings": {
            "sd_model_checkpoint": "cyberrealistic_pony.safetensors",
            "sd_vae": "sdxl_vae.safetensors",
            "CLIP_stop_at_last_layers": 2
        }
    }
    
    for key, value in default_params.items():
        if key not in params:
            params[key] = value
    
    response = requests.post(f"{API_URL}/sdapi/v1/img2img", json=params, timeout=300)
    if response.status_code != 200:
        raise Exception(f"SD API error: {response.status_code} - {response.text}")
    
    return response.json()

def handle_upscale(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle image upscaling"""
    if "image" not in params:
        raise ValueError("image is required for upscaling")
    
    upscale_params = {
        "resize_mode": 0,
        "show_extras_results": True,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscaling_resize": params.get("upscale_factor", 2),  # Reduced from 4 for SDXL
        "upscaler_1": params.get("upscaler", "4x-UltraSharp"),
        "upscaler_2": "None",
        "extras_upscaler_2_visibility": 0,
        "upscale_first": False,
        "image": params["image"]
    }
    
    response = requests.post(f"{API_URL}/sdapi/v1/extra-single-image", json=upscale_params, timeout=300)
    if response.status_code != 200:
        raise Exception(f"SD API error: {response.status_code} - {response.text}")
    
    return response.json()

def handler(job):
    """Enhanced handler function for RunPod serverless with SDXL support"""
    try:
        job_input = job["input"]
        
        # Log the incoming request
        print(f"📥 Received job: {job['id']}")
        print(f"🔧 Using API endpoint: {API_URL}")
        print(f"📝 Input parameters: {json.dumps(job_input, indent=2)}")
        
        # Wait for SD WebUI to be ready (longer timeout for SDXL)
        if not wait_for_service(timeout=180):
            return {"error": f"SD WebUI failed to start on port {API_PORT} after 180 seconds"}
        
        # Determine the operation type
        operation = job_input.get("operation", "txt2img")
        
        if operation == "txt2img":
            result = handle_txt2img(job_input)
        elif operation == "img2img":
            result = handle_img2img(job_input)
        elif operation == "upscale":
            result = handle_upscale(job_input)
        else:
            # Default to txt2img for backward compatibility
            result = handle_txt2img(job_input)
        
        print(f"✅ Job {job['id']} completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error processing job {job.get('id', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print(f"🚀 Starting RunPod handler for SDXL/Pony (API port: {API_PORT})...")
    runpod.serverless.start({"handler": handler})
HANDLER_EOF

# Copy startup script with SDXL optimizations
COPY <<'STARTUP_EOF' /start.sh
#!/bin/bash
set -e

# SINGLE SOURCE OF TRUTH FOR PORT
export API_PORT=7860

echo "🚀 Starting Stable Diffusion WebUI API (SDXL/Pony) on port $API_PORT..."

# SDXL-optimized command line arguments
export COMMANDLINE_ARGS="--api --nowebui --port $API_PORT --listen --enable-insecure-extension-access --xformers --opt-sdp-attention --skip-install --no-half-vae --medvram-sdxl"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

cd /stable-diffusion-webui

# Start SD WebUI in background
echo "📦 Launching SD WebUI with SDXL support on port $API_PORT..."
python3 launch.py --skip-torch-cuda-test --skip-python-version-check --skip-install &
SD_PID=$!

# Wait longer for SDXL to initialize
sleep 20

# Start RunPod handler
echo "🔌 Starting RunPod handler (connecting to port $API_PORT)..."
python3 /handler.py &
HANDLER_PID=$!

# Monitor both processes
echo "👀 Monitoring processes..."
wait $SD_PID $HANDLER_PID
STARTUP_EOF

# Make scripts executable
RUN chmod +x /start.sh /handler.py

# Verify everything is in place
RUN echo "=== Verification ===" \
 && echo "Scripts:" && ls -la /start.sh /handler.py \
 && echo "Models:" && ls -la "$ROOT/models/Stable-diffusion/" \
 && echo "VAE:" && ls -la "$ROOT/models/VAE/" \
 && echo "LoRAs:" && ls -la "$ROOT/models/Lora/" \
 && echo "Upscalers:" && ls -la "$ROOT/models/ESRGAN/" \
 && echo "ADetailer:" && ls -la "$ROOT/models/adetailer/" \
 && echo "==================="

# Set environment variables for SDXL
ENV COMMANDLINE_ARGS="--api --nowebui --port 7860 --listen --enable-insecure-extension-access --xformers --opt-sdp-attention --skip-install --no-half-vae --medvram-sdxl"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

WORKDIR $ROOT
EXPOSE 7860

# Health check with longer start period for SDXL
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:7860/sdapi/v1/options || exit 1

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/start.sh"]
