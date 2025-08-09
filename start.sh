#!/usr/bin/env bash
set -euo pipefail

ROOT="/stable-diffusion-webui"

# ---- Hard-coded token and assets you provided ----
TOKEN="cbc8f589f04ba1a0299b10473a792b42"

CKPT_URL="https://civitai.com/api/download/models/2071650?type=Model&format=SafeTensor&size=pruned&fp=fp16"
LORA_ALL_URL="https://civitai.com/api/download/models/160240?type=Model&format=SafeTensor"
LORA_PONY_URL="https://civitai.com/api/download/models/717403?type=Model&format=SafeTensor"

# ADetailer detectors:
# You asked for face_yolov9c.pt; if you share a direct URL later I’ll switch it.
# For guaranteed first-boot success, we use the widely-available yolov8n models:
ADETAILER_FACE_URL="https://raw.githubusercontent.com/Bing-su/adetailer/main/other/face_yolov8n.pt"
ADETAILER_HAND_URL="https://raw.githubusercontent.com/Bing-su/adetailer/main/other/hand_yolov8n.pt"

authcurl () {
  local url="$1" out="$2"
  curl -fsSL --retry 5 -H "Authorization: Bearer $TOKEN" -o "$out" "$url"
}

dl_opt () {
  local url="$1" out="$2"
  curl -fsSL --retry 5 -o "$out" "$url" || true
}

echo "[init] Downloading models and assets..."

# Checkpoint
if [[ ! -f "$ROOT/models/Stable-diffusion/primary_model.safetensors" ]]; then
  echo "[download] checkpoint..."
  authcurl "$CKPT_URL" "$ROOT/models/Stable-diffusion/primary_model.safetensors"
fi

# LoRAs
[[ -f "$ROOT/models/Lora/nsfw_all_in_one.safetensors" ]] || authcurl "$LORA_ALL_URL" "$ROOT/models/Lora/nsfw_all_in_one.safetensors"
[[ -f "$ROOT/models/Lora/pony_amateur.safetensors"   ]] || authcurl "$LORA_PONY_URL" "$ROOT/models/Lora/pony_amateur.safetensors"

# ADetailer detectors (face + hand)
[[ -f "$ROOT/models/adetailer/face_yolo.pt" ]] || dl_opt "$ADETAILER_FACE_URL" "$ROOT/models/adetailer/face_yolo.pt"
[[ -f "$ROOT/models/adetailer/hand_yolo.pt" ]] || dl_opt "$ADETAILER_HAND_URL" "$ROOT/models/adetailer/hand_yolo.pt"

# Start A1111
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
cd "$ROOT"
bash webui.sh ${COMMANDLINE_ARGS} &

# Wait for API to be ready
echo "[wait] Waiting for API..."
for i in {1..120}; do
  if curl -fsS "http://127.0.0.1:7860/sdapi/v1/sd-models" >/dev/null; then
    echo "[wait] API is up."
    break
  fi
  sleep 1
done

# Warm-up: load model, set Clip Skip 2
echo "[warmup] Loading model..."
curl -fsS -X POST "http://127.0.0.1:7860/sdapi/v1/options" \
  -H "Content-Type: application/json" \
  -d '{"sd_model_checkpoint":"primary_model.safetensors","CLIP_stop_at_last_layers":2}' >/dev/null

# Tiny dummy gen to populate caches (keeps serverless snappy)
curl -fsS -X POST "http://127.0.0.1:7860/sdapi/v1/txt2img" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"warmup","width":64,"height":64,"steps":8,"sampler_name":"Euler a"}' >/dev/null

echo "[ready] Worker ready."
wait -n
