#!/usr/bin/env bash
# =============================================================================
# start-all.sh — Provision all 4 GPUs for the Windows host workloads
#
#   GPU 0  ComfyUI instance A  ->  http://<host>:8188   (image rendering)
#   GPU 1  ComfyUI instance B  ->  http://<host>:8189   (image rendering)
#   GPU 2  vLLM Qwen3.8-27B--ara-int4  ->  http://<host>:8088  (book writing)
#   GPU 3  vLLM gemma-4-ortenzya-31b--int4 ->  http://<host>:8089 (VN writing)
#
# On start it first kills any existing ComfyUI python processes (same pattern
# as comfyui/run.sh), then launches all four services in the background.
#
# Usage:
#   bash start-all.sh           # kill old comfyui, start all 4 services
#   bash start-all.sh stop      # stop everything started by this script
#   bash start-all.sh status    # check ports + pids
#
# Env overrides: COMFY_HOST, COMFY_PORT_A, COMFY_PORT_B, COMFY_GPU_A,
#                COMFY_GPU_B, QWEN_PORT, QWEN_GPU, QWEN_MAX_LEN,
#                GEMMA_PORT, GEMMA_GPU, GEMMA_MAX_LEN, GEMMA_KV_DTYPE,
#                GEMMA_GPU_MEM_UTIL, GEMMA_CPU_OFFLOAD_GB
#
# NOTE: gemma-4-ortenzya-31b- is served as INT4 AutoRound (~18 GB),
# which fits fully on one 32 GB B70 with no CPU offload.
#
# KV cache: fp8 is the ONLY working kv-cache-dtype for Gemma4 on this vLLM/XPU
# build. turboquant_* and int4_per_token_head both crash at engine init (Gemma4
# is multimodal with mismatched head_dims across its attention groups). At
# --gpu-memory-utilization 0.92 the KV pool is ~52.7K tokens, so a single
# request can use up to ~52K context. Default GEMMA_MAX_LEN is 32K (1.6x
# headroom); raise it toward 48K for max single-request context if you do not
# need concurrent requests.
# =============================================================================
set -uo pipefail

COMFY_DIR="$HOME/comfyui"
VENV="$HOME/electric-sheep/vllm/.venv"
MODELS_DIR="$HOME/electric-sheep/models"
LOG_DIR="$HOME/electric-sheep/vllm/launch/logs"
PIDS_FILE="$LOG_DIR/start-all.pids"
mkdir -p "$LOG_DIR"

# --- Configuration -----------------------------------------------------------
COMFY_HOST="${COMFY_HOST:-0.0.0.0}"
COMFY_PORT_A="${COMFY_PORT_A:-8188}"
COMFY_PORT_B="${COMFY_PORT_B:-8189}"
COMFY_GPU_A="${COMFY_GPU_A:-0}"
COMFY_GPU_B="${COMFY_GPU_B:-1}"

QWEN_PORT="${QWEN_PORT:-8088}"
QWEN_GPU="${QWEN_GPU:-2}"
QWEN_MODEL="$MODELS_DIR/Qwen3.8-27B--ara-int4-AutoRound/Qwen3.8-27B--ara-w4g128"
QWEN_NAME="qwen-256k"
QWEN_MAX_LEN="${QWEN_MAX_LEN:-262144}"

GEMMA_PORT="${GEMMA_PORT:-8089}"
GEMMA_GPU="${GEMMA_GPU:-3}"
GEMMA_MODEL="$MODELS_DIR/gemma-4-ortenzya-31b--int4-AutoRound/gemma-4-ortenzya-31b--w4g128"
GEMMA_NAME="gemma-31b"
GEMMA_MAX_LEN="${GEMMA_MAX_LEN:-32768}"
GEMMA_KV_DTYPE="${GEMMA_KV_DTYPE:-fp8}"
GEMMA_GPU_MEM_UTIL="${GEMMA_GPU_MEM_UTIL:-0.92}"
GEMMA_CPU_OFFLOAD_GB="${GEMMA_CPU_OFFLOAD_GB:-0}"

# --- Helpers -----------------------------------------------------------------
record_pid() { echo "$1" >> "$PIDS_FILE"; }

port_open() { curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/"; }

wait_ready() { # wait_ready <name> <port> <timeout_s>
  local name="$1" port="$2" timeout="${3:-600}" i=0
  while (( i < timeout )); do
    if port_open "$port"; then echo "  [${name}] ready on port ${port}"; return 0; fi
    sleep 5; (( i += 5 ))
  done
  echo "  [${name}] NOT ready after ${timeout}s — check $LOG_DIR/"
  return 1
}

# --- Stop --------------------------------------------------------------------
stop_all() {
  local killed=0
  if [[ -f "$PIDS_FILE" ]]; then
    while read -r pid; do
      if kill "$pid" 2>/dev/null; then echo "  stopped pid ${pid}"; killed=1; fi
    done < "$PIDS_FILE"
    rm -f "$PIDS_FILE"
  fi
  # stragglers: comfyui main.py + vllm api_server
  pkill -f "comfyui/venv/bin/python main.py" 2>/dev/null && killed=1
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && killed=1
  [[ "$killed" == "1" ]] && echo "all stopped" || echo "nothing was running"
}

# --- Status ------------------------------------------------------------------
status() {
  for port in "$COMFY_PORT_A" "$COMFY_PORT_B" "$QWEN_PORT" "$GEMMA_PORT"; do
    if port_open "$port"; then echo "  port ${port}: responding"
    else echo "  port ${port}: not responding"; fi
  done
  if [[ -f "$PIDS_FILE" ]]; then
    while read -r pid; do
      if kill -0 "$pid" 2>/dev/null; then echo "  pid ${pid}: RUNNING"
      else echo "  pid ${pid}: dead"; fi
    done < "$PIDS_FILE"
  else
    echo "  no pids file (nothing started via start-all.sh)"
  fi
}

# --- Kill existing ComfyUI (per run.sh) --------------------------------------
kill_comfyui() {
  local killed=0
  if [[ -f "$COMFY_DIR/logs/pids" ]]; then
    while read -r pid; do
      if kill "$pid" 2>/dev/null; then echo "  stopped old comfyui pid ${pid}"; killed=1; fi
    done < "$COMFY_DIR/logs/pids"
    rm -f "$COMFY_DIR/logs/pids"
  fi
  if pkill -f "comfyui/venv/bin/python main.py" 2>/dev/null; then
    echo "  killed stray comfyui main.py processes"; killed=1
  fi
  [[ "$killed" == "1" ]] && sleep 3
  echo "existing comfyui processes: ${killed:+cleared}${killed:-none found}"
}

# --- ComfyUI instances ---------------------------------------------------------
start_comfyui() { # start_comfyui <name> <gpu> <port> <userdir> <outdir> <tempdir>
  local name="$1" gpu="$2" port="$3" userdir="$4" outdir="$5" tempdir="$6"
  mkdir -p "$userdir" "$outdir" "$tempdir"
  (
    cd "$COMFY_DIR"
    ONEAPI_DEVICE_SELECTOR="level_zero:${gpu}" nohup ./venv/bin/python main.py \
      --listen "$COMFY_HOST" \
      --port "$port" \
      --preview-method auto \
      --user-directory "$userdir" \
      --output-directory "$outdir" \
      --temp-directory "$tempdir" \
      --enable-manager \
      > "$LOG_DIR/comfyui_${port}.log" 2>&1 &
    echo $! > "$LOG_DIR/.lastpid"
  )
  local pid; pid="$(cat "$LOG_DIR/.lastpid")"
  record_pid "$pid"
  echo "  [comfyui-${name}] GPU ${gpu}  port ${port}  pid ${pid}  (log: logs/comfyui_${port}.log)"
}

# --- vLLM instances ------------------------------------------------------------
# XPU env for a single-GPU vLLM (from set-env-*.sh / start-qwen.sh).
# ZE_AFFINITY_MASK renumbers the selected physical GPU to 0, so the
# ONEAPI_DEVICE_SELECTOR must be level_zero:0.
vllm_env() { # vllm_env <physical_gpu>
  export ZE_AFFINITY_MASK="$1"
  export ONEAPI_DEVICE_SELECTOR="level_zero:0"
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
}

start_vllm() { # start_vllm <name> <gpu> <port> <model> <served> <max_len> <offload_gb> <kv_dtype> <mem_util>
  local name="$1" gpu="$2" port="$3" model="$4" served="$5" max_len="$6" offload="$7" kv_dtype="$8" mem_util="$9"
  [[ -d "$model" ]] || { echo "  ERROR: model dir not found: $model"; return 1; }
  local cmd=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "$model"
    --served-model-name "$served"
    --host 0.0.0.0
    --port "$port"
    --tensor-parallel-size 1
    --max-model-len "$max_len"
    --kv-cache-dtype "$kv_dtype"
    --enable-prefix-caching
    --trust-remote-code
    --gpu-memory-utilization "$mem_util"
    --block-size 32
    --max-num-seqs 8
  )
  [[ "$offload" != "0" ]] && cmd+=(--cpu-offload-gb "$offload")
  # Qwen3.8-specific parsers (harmless to add only for the qwen model)
  case "$model" in *Qwen*)
    cmd+=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --generation-config vllm)
    ;;
  esac
  (
    vllm_env "$gpu"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    nohup "${cmd[@]}" > "$LOG_DIR/vllm_${port}.log" 2>&1 &
    echo $! > "$LOG_DIR/.lastpid"
  )
  local pid; pid="$(cat "$LOG_DIR/.lastpid")"
  record_pid "$pid"
  echo "  [${name}] GPU ${gpu}  port ${port}  pid ${pid}  ctx=${max_len}  kv=${kv_dtype}  util=${mem_util}  offload=${offload}GB  (log: logs/vllm_${port}.log)"
}

# --- Commands ------------------------------------------------------------------
case "${1:-start}" in
  stop)
    stop_all
    ;;
  status)
    status
    ;;
  start|*)
    echo "=== killing existing comfyui ==="
    kill_comfyui

    echo "=== starting services ==="
    : > "$PIDS_FILE"
    start_comfyui "A" "$COMFY_GPU_A" "$COMFY_PORT_A" "$COMFY_DIR/user"   "$COMFY_DIR/output" "$COMFY_DIR/temp"
    start_comfyui "B" "$COMFY_GPU_B" "$COMFY_PORT_B" "$COMFY_DIR/user2"  "$COMFY_DIR/output2" "$COMFY_DIR/temp2"
    start_vllm "qwen-256k"  "$QWEN_GPU"  "$QWEN_PORT"  "$QWEN_MODEL"  "$QWEN_NAME"  "$QWEN_MAX_LEN"  0  "fp8"  "0.85"
    start_vllm "gemma-31b"  "$GEMMA_GPU" "$GEMMA_PORT" "$GEMMA_MODEL" "$GEMMA_NAME" "$GEMMA_MAX_LEN" "$GEMMA_CPU_OFFLOAD_GB" "$GEMMA_KV_DTYPE" "$GEMMA_GPU_MEM_UTIL"

    echo
    echo "=== waiting for vLLM servers (first load can take minutes) ==="
    wait_ready "qwen-256k" "$QWEN_PORT" 900
    wait_ready "gemma-31b" "$GEMMA_PORT" 900

    echo
    echo "  comfyui A:  http://${COMFY_HOST}:${COMFY_PORT_A}   (GPU ${COMFY_GPU_A})"
    echo "  comfyui B:  http://${COMFY_HOST}:${COMFY_PORT_B}   (GPU ${COMFY_GPU_B})"
    echo "  qwen:       http://${COMFY_HOST}:${QWEN_PORT}  model '${QWEN_NAME}'"
    echo "  gemma:      http://${COMFY_HOST}:${GEMMA_PORT}  model '${GEMMA_NAME}'"
    echo
    echo "  stop with:  bash $0 stop"
    echo "  status:     bash $0 status"
    ;;
esac
