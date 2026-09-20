#!/usr/bin/env bash
# One-off A/B test: run the MikeCaldera/SergiioB guide's EXACT pinned docker
# stack on GPU1 (47:00.0 = lz:1 = renderD131) to compare against our venv vLLM.
#
# Usage:
#   bash run-guide-docker-gpu1.sh start    # launch container (qw38speed)
#   bash run-guide-docker-gpu1.sh logs     # tail container logs
#   bash run-guide-docker-gpu1.sh health   # curl health + models
#   bash run-guide-docker-gpu1.sh stop     # stop + remove container
#
# Env overrides:
#   GUIDE_HOST_PORT (default 11436)  host-side port
#   GUIDE_SERVED_NAME (default qwen38) served model name
#   GUIDE_CONTAINER_NAME (default qw38speed) container name
#   DRAFT_INT4 (default 0) 1 = add S+M1 draft-INT4 overlay (cookbook section 11)
#   B70_RENDER_NODE (default /dev/dri/renderD131 = GPU1)
#   B70_AFFINITY_MASK (default 1 = GPU1; use 0 for GPU0)
#   GUIDE_MAX_MODEL_LEN (default 131072)
#   GUIDE_MAX_NUM_SEQS (default 64)
#   GUIDE_GPU_MEM_UTIL (default 0.88)
#   B70_VISION (default 0 = --language-model-only; 1 = vision via --limit-mm-per-prompt)
#   GUIDE_GEN_CONFIG (optional JSON -> --override-generation-config)
#   GUIDE_API_KEY (optional -> --api-key)
#   Use GUIDE_HOST_PORT=8000 GUIDE_SERVED_NAME=qwen-256k to back the WebUI.
#
# Prereqs (done separately):
#   - image pulled: vllm/vllm-openai-xpu@sha256:f01e24f6...
#   - model at /mnt/data/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 (pinned rev)
#   - cookbook at ~/intel-arc-pro-b70-inference-cookbook (patches verified)
#   - our vLLM on GPU1 stopped (port 8000 freed)
set -euo pipefail

IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL_DIR="${B70_MODEL_DIR:-/mnt/data/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
COOKBOOK="$HOME/intel-arc-pro-b70-inference-cookbook"
# GPU selection via env overrides. Defaults = GPU1 (47:00.0 = lz:1 = renderD131).
# For GPU0: B70_RENDER_NODE=/dev/dri/renderD130 B70_AFFINITY_MASK=0
B70_RENDER_NODE="${B70_RENDER_NODE:-/dev/dri/renderD131}"
B70_AFFINITY_MASK="${B70_AFFINITY_MASK:-1}"
RENDER_GID="$(stat -c '%g' "$B70_RENDER_NODE")"
HOST_PORT="${GUIDE_HOST_PORT:-11436}"
SERVED_NAME="${GUIDE_SERVED_NAME:-qwen38}"
CONTAINER="${GUIDE_CONTAINER_NAME:-qw38speed}"
DRAFT_INT4="${DRAFT_INT4:-0}"
MAX_MODEL_LEN="${GUIDE_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${GUIDE_MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${GUIDE_GPU_MEM_UTIL:-0.88}"
VISION="${B70_VISION:-0}"
SPEC_TOKENS="${B70_SPEC_TOKENS:-4}"
# B70_PREFIX_CACHE: 0 = --no-enable-prefix-caching (cookbook default; avoids the
# GDN/mamba state-copy path that corrupts output on this older image), 1 = on.
PREFIX_CACHE="${B70_PREFIX_CACHE:-0}"
GEN_CONFIG="${GUIDE_GEN_CONFIG:-}"
API_KEY="${GUIDE_API_KEY:-}"

cmd="${1:-start}"

case "$cmd" in
start)
  echo "render node: $B70_RENDER_NODE (gid $RENDER_GID)"
  echo "model dir:   $MODEL_DIR"
  echo "host port:   $HOST_PORT  served name: $SERVED_NAME"
  [ -d "$MODEL_DIR" ] || { echo "MISSING model dir"; exit 1; }
  ls "$MODEL_DIR" | head -5

  docker rm -f "$CONTAINER" 2>/dev/null || true

  EXTRA_ENV=()
  if [ "$DRAFT_INT4" = "1" ]; then
    EXTRA_ENV+=(-e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1)
  fi

  docker run -d \
    --name "$CONTAINER" \
    -p ${HOST_PORT}:8000 \
    --device /dev/dri:/dev/dri \
    -v /dev/dri:/dev/dri:ro \
    --group-add "$RENDER_GID" \
    -v "$MODEL_DIR:/model:ro" \
    -v "$COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
    -v "$COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
    -v "$COOKBOOK/patches/patch_gdn_mixed_split_v5.py:/patch_v5.py:ro" \
    -v "$COOKBOOK/patches/patch_draft_lmhead_int4.py:/patch_s.py:ro" \
    -v "$COOKBOOK/patches/patch_draft_mtp_int4.py:/patch_m1.py:ro" \
    "${EXTRA_ENV[@]}" \
    -e VLLM_TARGET_DEVICE=xpu \
    -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
    -e ZE_AFFINITY_MASK="$B70_AFFINITY_MASK" \
    -e B70_MTP_BF16_DRAFT="${B70_MTP_BF16_DRAFT:-1}" \
    -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e SERVED_NAME="$SERVED_NAME" \
    -e DRAFT_INT4="$DRAFT_INT4" \
    -e B70_SPEC="${B70_SPEC:-1}" \
    -e SPEC_TOKENS="$SPEC_TOKENS" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    -e VISION="$VISION" \
    -e PREFIX_CACHE="$PREFIX_CACHE" \
    -e GEN_CONFIG="$GEN_CONFIG" \
    -e API_KEY="$API_KEY" \
    --entrypoint bash \
    "$IMAGE" \
    -lc 'set -e
python /patch_mtp.py
python /patch_boundary.py
python /patch_v5.py
if [ "$DRAFT_INT4" = "1" ]; then python /patch_s.py; python /patch_m1.py; fi
SPEC_ARG=""
if [ "$B70_SPEC" != "0" ]; then SPEC_ARG="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}"; fi
# Pinned vLLM parses --limit-mm-per-prompt via json.loads; bare "image=4" fails
# argparse. Use compact JSON, same trick as SPEC_ARG.
if [ "$VISION" = "1" ]; then MM_ARG="--limit-mm-per-prompt {\"image\":4}"; else MM_ARG="--language-model-only"; fi
if [ "$PREFIX_CACHE" = "1" ]; then PC_ARG="--enable-prefix-caching"; else PC_ARG="--no-enable-prefix-caching"; fi
if [ -n "$GEN_CONFIG" ]; then GENCFG_ARG="--override-generation-config $GEN_CONFIG"; else GENCFG_ARG=""; fi
if [ -n "$API_KEY" ]; then APIKEY_ARG="--api-key $API_KEY"; else APIKEY_ARG=""; fi
exec vllm serve /model \
  --quantization gptq \
  --dtype float16 \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  --kv-cache-dtype fp8 \
  --port 8000 \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens 8192 \
  $PC_ARG \
  --served-model-name $SERVED_NAME \
  $MM_ARG \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  $GENCFG_ARG \
  $APIKEY_ARG \
  $SPEC_ARG'
  echo "launched $CONTAINER on :$HOST_PORT (served as $SERVED_NAME, draft-int4=$DRAFT_INT4)"
  ;;
logs)
  docker logs --tail "${2:-40}" "$CONTAINER" 2>&1
  ;;
health)
  curl -fsS "http://127.0.0.1:$HOST_PORT/health" && echo " HEALTH OK"
  curl -fsS "http://127.0.0.1:$HOST_PORT/v1/models" | python3 -c "import json,sys;d=json.load(sys.stdin);print('models:',[m['id'] for m in d.get('data',[])])"
  ;;
stop)
  docker rm -f "$CONTAINER" 2>/dev/null || true
  echo "stopped"
  ;;
*)
  echo "usage: $0 {start|logs|health|stop}"
  ;;
esac
