#!/usr/bin/env bash
set -euo pipefail

ROOT="/stable-diffusion-webui"

echo "[init] Verifying required assets..."
ls -lh "$ROOT/models/Stable-diffusion" || true
ls -lh "$ROOT/models/Lora" || true
ls -lh "$ROOT/models/ESRGAN" || true
ls -lh "$ROOT/models/adetailer" || true

# Run with tcmalloc for perf (optional)
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

# Prefer xformers; if it fails at runtime we still have SDPA enabled
export HF_HOME="/root/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_CUDA_ARCH_LIST="8.6+PTX;8.9;9.0"

echo "[init] Launching A1111 with: ${COMMANDLINE_ARGS}"
# Start A1111 in background
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

# Warm-start: load the baked checkpoint and run a tiny generation to populate GPU caches
echo "[warmup] Loading model and running a tiny warm-up job..."
set +e
curl -fsS -X POST "http://127.0.0.1:7860/sdapi/v1/options" \
  -H "Content-Type: application/json" \
  -d '{
    "sd_model_checkpoint": "primary_model.safetensors",
    "CLIP_stop_at_last_layers": 2
  }' >/dev/null

curl -fsS -X POST "http://127.0.0.1:7860/sdapi/v1/txt2img" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"warmup",
    "width":64, "height":64,
    "steps":8, "sampler_name":"Euler a",
    "batch_size":1, "n_iter":1,
    "seed": 12345
  }' >/dev/null
set -e

echo "[ready] Worker is warmed up and ready."
wait -n
