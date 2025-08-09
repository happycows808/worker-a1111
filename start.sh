#!/usr/bin/env bash
set -euo pipefail

ROOT="/stable-diffusion-webui"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

# ========= HARD-CODED TOKEN & REQUIRED ASSETS =========
TOKEN="cbc8f589f04ba1a0299b10473a792b42"

# Your checkpoint + LoRAs (REQUIRED; we block on these)
CKPT_URL="https://civitai.com/api/download/models/2071650?type=Model&format=SafeTensor&size=pruned&fp=fp16"
LORA_ALL_URL="https://civitai.com/api/download/models/160240?type=Model&format=SafeTensor"
LORA_PONY_URL="https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor"

# Optional assets (downloaded in background after API is up)
# ADetailer detectors (face_yolov9c public link is not universal; use reliable yolov8n/hand v8n)
ADETAILER_FACE_URL="https://raw.githubusercontent.com/Bing-su/adetailer/main/other/face_yolov8n.pt"
ADETAILER_HAND_URL="https://raw.githubusercontent.com/Bing-su/adetailer/main/other/hand_yolov8n.pt"

# ESRGAN upscaler
ULTRASHARP_URL="https://huggingface.co/ClashSAN/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth?download=true"

# ControlNet weights (SD1.5 trio; optional)
CN_OPENPOSE_URL="https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth?download=true"
CN_DEPTH_URL="https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth?download=true"
CN_LINEART_URL="https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth?download=true"

# ========= HELPERS =========
authget () {  # $1=url $2=out
  local url="$1" out="$2"
  echo "[dl] $out"
  curl -fsSL --retry 5 -H "Authorization: Bearer $TOKEN" -o "$out" "$url"
}

get_opt () {  # $1=url $2=out  (optional; do not fail boot)
  local url="$1" out="$2"
  echo "[dl-opt] $out"
  curl -fsSL --retry 5 -o "$out" "$url" || echo "[warn] optional asset failed: $out"
}

# ========= REQUIRED: download checkpoint + LoRAs (block) =========
mkdir -p "$ROOT/models/Stable-diffusion" "$ROOT/models/Lora" "$ROOT/models/ESRGAN" "$ROOT/models/adetailer" "$ROOT/models/ControlNet"

authget "$CKPT_URL"    "$ROOT/models/Stable-diffusion/primary_model.safetensors"
authget "$LORA_ALL_URL" "$ROOT/models/Lora/nsfw_all_in_one.safetensors"
authget "$LORA_PONY_URL" "$ROOT/models/Lora/pony_amateur.safetensors"

echo "[ok] required assets present."

# ========= START A1111 ASAP =========
cd "$ROOT"
echo "[start] webui.sh ${COMMANDLINE_ARGS}"
bash webui.sh ${COMMANDLINE_ARGS} &

# Wait for API
echo "[wait] waiting for A1111 API..."
for i in {1..240}; do
  if curl -fsS "http://127.0.0.1:7860/sdapi/v1/sd-models" >/dev/null; then
    echo "[wait] API up."
    break
  fi
  sleep 1
done

# Warm-up: set model + Clip Skip 2, tiny generation to prime VRAM
echo "[warmup] set options + tiny gen"
curl -fsS -X POST "http://127.0.0.1:7860/sdapi/v1/options" \
  -H "Content-Type: application/json" \
  -d '{"sd_model_checkpoint":"primary_model.safetensors","CLIP_stop_at_last_layers":2}' >/dev/null || true

curl -fsS -X POST "http://127.0.0.1:7860/sdapi/v1/txt2img" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"warmup","width":64,"height":64,"steps":8,"sampler_name":"Euler a","batch_size":1,"seed":12345}' >/dev/null || true

# ========= OPTIONAL ASSETS IN BACKGROUND (non-blocking) =========
nohup bash -lc '
  get_opt "'"$ADETAILER_FACE_URL"'" "$ROOT/models/adetailer/face_yolo.pt"
  get_opt "'"$ADETAILER_HAND_URL"'" "$ROOT/models/adetailer/hand_yolo.pt"
  get_opt "'"$ULTRASHARP_URL"'" "$ROOT/models/ESRGAN/4x-UltraSharp.pth"
  get_opt "'"$CN_OPENPOSE_URL"'" "$ROOT/models/ControlNet/control_v11p_sd15_openpose.pth"
  get_opt "'"$CN_DEPTH_URL"'"    "$ROOT/models/ControlNet/control_v11f1p_sd15_depth.pth"
  get_opt "'"$CN_LINEART_URL"'"  "$ROOT/models/ControlNet/control_v11p_sd15_lineart.pth"
  echo "[bg] optional assets fetched."
' >/tmp/bg_fetch.log 2>&1 &

echo "[ready] worker ready."
# Keep process in foreground (wait for webui)
wait -n
