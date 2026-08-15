#!/usr/bin/env bash
# =============================================================================
# vllm-launch.sh — Interactive vLLM launcher (any model, any GPU subset)
#
# Interactively asks for:
#   1. Which GPUs to use (and how many)  -> sets ONEAPI_DEVICE_SELECTOR,
#      ZE_AFFINITY_MASK, and --tensor-parallel-size
#   2. Which model to serve              -> picked from the model library
#   3. Optimization options              -> context length, KV-cache dtype,
#      prefix caching, GPU memory utilization, port
#
# It auto-detects the model's quantization method from config.json and adds the
# right --quantization flag, then launches the vLLM OpenAI server.
#
# Usage:
#   bash vllm-launch.sh            # fully interactive
#   bash vllm-launch.sh --dry-run  # print the command, don't launch
#
# This SUPERSEDES the static set-env-*.sh files: the XPU environment is generated
# here from the chosen GPU set, so no separate set-env script is needed.
# =============================================================================
set -uo pipefail

VENV="/home/dc/electric-sheep/vllm/.venv"
MODELS_DIR="$HOME/electric-sheep/models"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# --- helpers -----------------------------------------------------------------
ask() { # ask <prompt> <default>  -> reads one line, falls back to default
  local prompt="$1" def="${2:-}" ans
  if [[ -n "$def" ]]; then read -rp "$prompt [$def]: " ans; else read -rp "$prompt: " ans; fi
  ans="${ans:-$def}"
  printf '%s' "$ans"
}

# Detect Level-Zero GPUs once (cached) — sycl-ls can misbehave under command
# substitution with piped stdin, so capture it a single time up front.
SYCL_OUT="$(sycl-ls 2>/dev/null)"
GPU_IDS=""
for i in 0 1 2 3 4 5 6 7; do
  if grep -q "level_zero:$i" <<<"$SYCL_OUT"; then GPU_IDS+="$i "; fi
done
GPU_IDS="${GPU_IDS:-0 1 2 3}"   # fallback if detection failed

# --- 1. GPU selection --------------------------------------------------------
echo "=== GPU selection ==="
echo "Detected GPUs: $GPU_IDS"
read -rp "GPU preset (1=all, 2=0,1, 3=2,3, 4=0, 5=1, 6=2, 7=3, 0=custom): " gp
case "$gp" in
  1) GPU_SET="0,1,2,3" ;;
  2) GPU_SET="0,1" ;;
  3) GPU_SET="2,3" ;;
  4) GPU_SET="0" ;;
  5) GPU_SET="1" ;;
  6) GPU_SET="2" ;;
  7) GPU_SET="3" ;;
  0) GPU_SET="$(ask "Enter GPU indices (comma-separated, e.g. 0,2,3)" "0,1,2,3")" ;;
  *) GPU_SET="0,1,2,3" ;;
esac
# Tensor parallelism = number of selected GPUs
TP_SIZE="$(echo "$GPU_SET" | awk -F, '{print NF}')"
echo "Using GPUs: $GPU_SET  (TP=$TP_SIZE)"

# --- 2. Model selection ------------------------------------------------------
echo
echo "=== Model selection ==="
# Trailing slash: $MODELS_DIR is a symlink to /mnt/data/models; find won't
# follow a symlink starting point without it.
mapfile -t MODELS < <(find "$MODELS_DIR/" -maxdepth 2 -name config.json -exec dirname {} \; 2>/dev/null | sort)
if [[ ${#MODELS[@]} -gt 0 ]]; then
  echo "Models in $MODELS_DIR:"
  for i in "${!MODELS[@]}"; do printf "  %2d) %s\n" "$((i+1))" "${MODELS[$i]#$MODELS_DIR/}"; done
  read -rp "Model number (or 'p' for a custom path): " mn
  if [[ "$mn" == "p" || "$mn" == "P" ]]; then
    MODEL_PATH="$(ask "Custom model path" "")"
  else
    MODEL_PATH="${MODELS[$((mn-1))]:-}"
  fi
else
  MODEL_PATH="$(ask "Model path" "")"
fi
[[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]] && { echo "ERROR: no valid model selected."; exit 1; }
echo "Model: $MODEL_PATH"

# --- 3. Optimization options -------------------------------------------------
echo
echo "=== Optimization options ==="
read -rp "Context length (1=32768, 2=65536, 3=131072, 4=196608, 5=262144, 0=custom): " cl
case "$cl" in
  1) MAX_LEN=32768 ;; 2) MAX_LEN=65536 ;; 3) MAX_LEN=131072 ;;
  4) MAX_LEN=196608 ;; 5) MAX_LEN=262144 ;;
  0) MAX_LEN="$(ask "Custom max-model-len" "262144")" ;;
  *) MAX_LEN=262144 ;;
esac
read -rp "KV cache dtype (1=fp8, 2=auto): " kv;   KV_DTYPE=$([[ "$kv" == "2" ]] && echo auto || echo fp8)
read -rp "Prefix caching? (y/n): " pc;            PREFIX=$([[ "${pc,,}" == "n" ]] && echo "--no-enable-prefix-caching" || echo "--enable-prefix-caching")
read -rp "GPU memory utilization: " gpu_util;     GPU_UTIL="${gpu_util:-0.85}"
read -rp "Port: " port;                           PORT="${port:-8000}"
read -rp "Served model name: " served;            SERVED="${served:-$(basename "$MODEL_PATH")}"

# Auto-detect quantization method from config.json
QUANT_FLAG=""
QM="$(python3 -c "import json;c=json.load(open('$MODEL_PATH/config.json'));print((c.get('quantization_config') or {}).get('quant_method',''))" 2>/dev/null)"
case "$QM" in
  quark) QUANT_FLAG="--quantization quark" ;;
  auto-round|auto_round) QUANT_FLAG="--quantization auto-round" ;;
  gptq) QUANT_FLAG="--quantization gptq" ;;
  fp8) QUANT_FLAG="--quantization fp8" ;;
  "") : ;; # unquantized (BF16/FP16) — no flag
  *) echo "NOTE: unrecognized quant_method '$QM'; no --quantization flag set." ;;
esac

# --- 4. Generate XPU environment (supersedes set-env-*.sh) -------------------
export ONEAPI_DEVICE_SELECTOR="level_zero:$GPU_SET"
export ZE_AFFINITY_MASK="$GPU_SET"
export UR_L0_SYNC_MODE=BLOCKING
export TORCH_LLM_ALLREDUCE=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export UVICORN_KEEP_ALIVE_TIMEOUT=300
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_TARGET_DEVICE=xpu
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

# --- 5. Build + run ----------------------------------------------------------
CMD=(python3 -m vllm.entrypoints.openai.api_server
  --model "$MODEL_PATH"
  --served-model-name "$SERVED"
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_LEN"
  --kv-cache-dtype "$KV_DTYPE"
  "$PREFIX"
  --trust-remote-code
  --gpu-memory-utilization "$GPU_UTIL"
)
[[ -n "$QUANT_FLAG" ]] && CMD+=($QUANT_FLAG)

echo
echo "=== Launch command ==="
printf '%q ' "${CMD[@]}"; echo
echo "GPUs=$GPU_SET  TP=$TP_SIZE  ctx=$MAX_LEN  kv=$KV_DTYPE  util=$GPU_UTIL  quant=${QM:-none}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry-run: not launching)"
  exit 0
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd /home/dc/electric-sheep/vllm
exec "${CMD[@]}"
