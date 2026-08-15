#!/usr/bin/env bash
# =============================================================================
# start-qwen.sh — Interactive launcher for ANY Qwen model in the library
#
# Interactively asks for:
#   1. Which Qwen model to serve (from the model library)
#   2. How many GPUs to use, and specifically which ones
#   3. Optimization options (context length, KV dtype, prefix caching, MTP,
#      GPU memory utilization, port)
#
# The served model name is set automatically to  qwen-<context window size>
# (e.g. a 262144-token context -> "qwen-256k", 196608 -> "qwen-192k"),
# matching the convention of the existing start-qwen3.6-*.sh scripts.
#
# Qwen-specific flags are added automatically:
#   --reasoning-parser qwen3  --tool-call-parser qwen3_coder
#   --enable-auto-tool-choice --generation-config vllm
#   (+ optional MTP speculative decoding)
#
# Usage:
#   bash start-qwen.sh            # fully interactive
#   bash start-qwen.sh --dry-run  # print the command, don't launch
# =============================================================================
set -uo pipefail

VENV="/home/dc/electric-sheep/vllm/.venv"
MODELS_DIR="$HOME/electric-sheep/models"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

ask() { local prompt="$1" def="${2:-}" ans; if [[ -n "$def" ]]; then read -rp "$prompt [$def]: " ans; else read -rp "$prompt: " ans; fi; ans="${ans:-$def}"; printf '%s' "$ans"; }

# Detect Level-Zero GPUs once (cached) — sycl-ls can misbehave under command
# substitution with piped stdin, so capture it a single time up front.
SYCL_OUT="$(sycl-ls 2>/dev/null)"
GPU_IDS=""
for i in 0 1 2 3 4 5 6 7; do
  if grep -q "level_zero:$i" <<<"$SYCL_OUT"; then GPU_IDS+="$i "; fi
done
GPU_IDS="${GPU_IDS:-0 1 2 3}"   # fallback if detection failed

# --- 1. Qwen model selection -------------------------------------------------
echo "=== Qwen model selection ==="
# Trailing slash: $MODELS_DIR is a symlink to /mnt/data/models; find won't
# follow a symlink starting point without it.
mapfile -t QWEN < <(find "$MODELS_DIR/" -maxdepth 2 -name config.json -exec dirname {} \; 2>/dev/null | grep -iE "qwen" | sort)
if [[ ${#QWEN[@]} -eq 0 ]]; then echo "ERROR: no Qwen models found in $MODELS_DIR"; exit 1; fi
for i in "${!QWEN[@]}"; do
  # show name + context window + size
  ctx="$(python3 -c "import json;c=json.load(open('${QWEN[$i]}/config.json'));tc=c.get('text_config',c);print(tc.get('max_position_embeddings','?'))" 2>/dev/null)"
  size="$(du -sh "${QWEN[$i]}" 2>/dev/null | cut -f1)"
  printf "  %2d) %-55s ctx=%-7s %s\n" "$((i+1))" "${QWEN[$i]#$MODELS_DIR/}" "$ctx" "$size"
done
read -rp "Qwen model number: " qn
MODEL_PATH="${QWEN[$((qn-1))]:-}"
[[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]] && { echo "ERROR: bad selection."; exit 1; }
echo "Model: $MODEL_PATH"

# --- 2. GPU selection --------------------------------------------------------
echo
echo "=== GPU selection ==="
echo "Detected GPUs: $GPU_IDS"
read -rp "How many GPUs to use (1-4): " n
n="${n:-4}"
read -rp "Which GPUs (comma-separated, e.g. 0,1,2,3): " gsel
GPU_SET="${gsel:-$(seq -s, 0 $((n-1)))}"
TP_SIZE="$(echo "$GPU_SET" | awk -F, '{print NF}')"
echo "Using GPUs: $GPU_SET  (TP=$TP_SIZE)"

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
read -rp "Enable MTP speculative decoding? (y/n): " mtp
read -rp "GPU memory utilization: " gpu_util;     GPU_UTIL="${gpu_util:-0.85}"
read -rp "Port: " port;                           PORT="${port:-8000}"

# Served name = qwen-<context window size in k>
SERVED="qwen-$((MAX_LEN/1024))k"
read -rp "Served model name: " served;            SERVED="${served:-$SERVED}"

# Auto-detect quantization method
QUANT_FLAG=""
QM="$(python3 -c "import json;c=json.load(open('$MODEL_PATH/config.json'));print((c.get('quantization_config') or {}).get('quant_method',''))" 2>/dev/null)"
case "$QM" in
  quark) QUANT_FLAG="--quantization quark" ;;
  auto-round|auto_round) QUANT_FLAG="--quantization auto-round" ;;
  gptq) QUANT_FLAG="--quantization gptq" ;;
  fp8) QUANT_FLAG="--quantization fp8" ;;
  "") : ;;
  *) echo "NOTE: unrecognized quant_method '$QM'." ;;
esac

# --- 4. XPU environment (supersedes set-env-*.sh) ----------------------------
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

# MTP-specific env (only when enabled)
if [[ "${mtp,,}" == "y" ]]; then
  export QWEN36_27B_ENABLE_MTP=1
  export NUM_SPECULATIVE_TOKENS=3
  export VLLM_XPU_GDN_PROMOTE_ACCEPTED_SPEC_STATE=1
  export VLLM_XPU_GDN_NONSPEC_POSTPROCESS_ACCEPTED_STATE=0
  export VLLM_XPU_LM_HEAD_INT8=1
  export VLLM_XPU_LM_HEAD_INT8_SCALE_DTYPE=bf16
  export COMPILATION_CONFIG='{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":8}'
fi

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
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --generation-config vllm
)
[[ -n "$QUANT_FLAG" ]] && CMD+=($QUANT_FLAG)
if [[ "${mtp,,}" == "y" ]]; then
  CMD+=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
fi

echo
echo "=== Launch command ==="
printf '%q ' "${CMD[@]}"; echo
echo "GPUs=$GPU_SET  TP=$TP_SIZE  ctx=$MAX_LEN  served=$SERVED  kv=$KV_DTYPE  util=$GPU_UTIL  quant=${QM:-none}  mtp=${mtp:-n}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry-run: not launching)"
  exit 0
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd /home/dc/electric-sheep/vllm
exec "${CMD[@]}"
