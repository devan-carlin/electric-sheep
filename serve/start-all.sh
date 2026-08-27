#!/usr/bin/env bash
# =============================================================================
# start-all.sh — Provision all 4 GPUs for the Windows host workloads
#
#   GPU 0  ComfyUI instance A  ->  http://<host>:8188   (image rendering)
#   GPU 1  ComfyUI instance B  ->  http://<host>:8189   (image rendering)
#   GPU 2  llama.cpp Qwen (currently Qwen3.6-35B-A3B Aggressive) ->  http://<host>:8088 (book writing)
#   GPU 3  llama.cpp Gemma (currently Gemma4 26B-A4B Balanced) ->  http://<host>:8089 (VN writing)
#
# On start it first kills any existing ComfyUI python processes (same pattern
# as comfyui/run.sh), then launches all four services in the background.
#
# Usage:
#   bash start-all.sh           # kill old comfyui, start all 4 services
#   bash start-all.sh stop      # stop everything started by this script
#   bash start-all.sh status    # check ports + pids
#   bash start-all.sh restart-gemma  # (re)start only the gemma llama.cpp instance
#   bash start-all.sh restart-qwen   # (re)start only the qwen llama.cpp instance
#
# Env overrides: COMFY_HOST, COMFY_PORT_A, COMFY_PORT_B, COMFY_GPU_A,
#                COMFY_GPU_B, QWEN_PORT, QWEN_GPU, QWEN_LLAMA_CTX,
#                GEMMA_PORT, GEMMA_GPU, GEMMA_LLAMA_CTX,
#                QWEN_LLAMA_REASONING, QWEN_LLAMA_REASONING_FORMAT,
#                QWEN_LLAMA_REASONING_BUDGET, GEMMA_LLAMA_REASONING,
#                GEMMA_LLAMA_REASONING_FORMAT, GEMMA_LLAMA_REASONING_BUDGET
#
# NOTE: the gemma slot serves Gemma4 26B-A4B ( Balanced, Q4_K_P GGUF)
# via llama.cpp SYCL on GPU 3, launched by start-gemma-llama.sh. It won a blind
# prose A/B over the 31B QAT (4/5) and decodes ~2.5-3x faster. 256K context
# (model native max) fits on one 32 GB B70 with a small CPU KV spill.
# The vLLM 31B (llmfan46-gemma-4-31b-qat-int4) remains a manual fallback for
# >256K context or MTP speculative decoding.
#
# NOTE: the qwen slot switched from vLLM (-ara int4, 128K cap) to
# llama.cpp on 2026-08-25 (256K fully on-GPU vs vLLM's 128K cap on one card).
# Served names are version-agnostic ("qwen", "gemma") so model swaps don't
# touch WebUI/clients. Current qwen model:  Qwen3.6-35B-A3B
# Aggressive (MoE, Q4_K_P) - chosen for MoE decode throughput over the dense
# Qwen3.8-27B. Launched by start-qwen-llama.sh (model pinned there).
# The vLLM -ara instance remains a manual fallback (start-qwen.sh) for
# prefix caching / tool-call parsing / multi-user concurrency.
# =============================================================================
set -uo pipefail

COMFY_DIR="$HOME/comfyui"
VENV="$HOME/electric-sheep/vllm/.venv"
MODELS_DIR="$HOME/electric-sheep/models"
LOG_DIR="$HOME/electric-sheep/serve/logs"
PIDS_FILE="$LOG_DIR/start-all.pids"
mkdir -p "$LOG_DIR"

# --- Configuration -----------------------------------------------------------
COMFY_HOST="${COMFY_HOST:-0.0.0.0}"
COMFY_PORT_A="${COMFY_PORT_A:-8188}"
COMFY_PORT_B="${COMFY_PORT_B:-8189}"
COMFY_GPU_A="${COMFY_GPU_A:-0}"
COMFY_GPU_B="${COMFY_GPU_B:-1}"

# Qwen slot serves the current Qwen model ( Qwen3.6-35B-A3B
# Aggressive) via llama.cpp SYCL, launched by start-qwen-llama.sh (model
# pinned there; served name is the version-agnostic "qwen").
QWEN_PORT="${QWEN_PORT:-8088}"
QWEN_GPU="${QWEN_GPU:-2}"
QWEN_NAME="qwen"
QWEN_LLAMA_LAUNCHER="$HOME/electric-sheep/serve/start-qwen-llama.sh"

# Gemma slot serves the 26B-A4B ( Balanced) via llama.cpp SYCL,
# launched by start-gemma-llama.sh (locked in 2026-08-25 after a blind A/B).
# The vLLM 31B (llmfan46-gemma-4-31b-qat-int4) remains a manual fallback.
GEMMA_PORT="${GEMMA_PORT:-8089}"
GEMMA_GPU="${GEMMA_GPU:-3}"
GEMMA_NAME="gemma"
GEMMA_LLAMA_LAUNCHER="$HOME/electric-sheep/serve/start-gemma-llama.sh"

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
  # stragglers: comfyui main.py + vllm api_server + llama-server (qwen/gemma slots)
  # NOTE: comfyui is launched as `./venv/bin/python main.py` (relative, after cd),
  # so the pattern must NOT include the `comfyui/` prefix - it never matched and
  # old instances survived restarts (port-conflict bug, fixed 2026-08-25).
  pkill -f "venv/bin/python main.py.*--port ${COMFY_PORT_A}" 2>/dev/null && killed=1
  pkill -f "venv/bin/python main.py.*--port ${COMFY_PORT_B}" 2>/dev/null && killed=1
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && killed=1
  pkill -f "llama-server.*--port ${QWEN_PORT}" 2>/dev/null && killed=1
  pkill -f "llama-server.*--port ${GEMMA_PORT}" 2>/dev/null && killed=1
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
  # Run BOTH pkills unconditionally: `pkill A || pkill B` short-circuits when A
  # succeeds, silently skipping B (left the 8189 instance alive on 2026-08-25).
  pkill -f "venv/bin/python main.py.*--port ${COMFY_PORT_A}" 2>/dev/null && killed=1
  pkill -f "venv/bin/python main.py.*--port ${COMFY_PORT_B}" 2>/dev/null && killed=1
  # Wait for the processes to actually exit (ComfyUI shutdown can take a while;
  # starting a replacement before the port is released -> "port in use" death).
  if [[ "$killed" == "1" ]]; then
    local i=0
    while (( i < 60 )) && pgrep -f "venv/bin/python main.py" >/dev/null; do
      sleep 2; (( i += 2 ))
    done
    if pgrep -f "venv/bin/python main.py" >/dev/null; then
      echo "  comfyui still alive after 120s, force-killing"
      pkill -9 -f "venv/bin/python main.py.*--port ${COMFY_PORT_A}" 2>/dev/null
      pkill -9 -f "venv/bin/python main.py.*--port ${COMFY_PORT_B}" 2>/dev/null
      sleep 2
    fi
  fi
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
  # Model-specific parsers / tool-calling support
  case "$model" in
    *Qwen*)
      cmd+=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --generation-config vllm)
      ;;
    *gemma*)
      # Gemma4 needs a tool parser so clients (e.g. Open WebUI) can send
      # tools + tool_choice:"auto" without a 400.
      cmd+=(--enable-auto-tool-choice --tool-call-parser gemma4)
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

# --- Restart a single instance ------------------------------------------------
restart_gemma_only() {
  echo "=== restarting gemma (llama.cpp) on port ${GEMMA_PORT} ==="
  GEMMA_LLAMA_GPU="$GEMMA_GPU" GEMMA_LLAMA_PORT="$GEMMA_PORT" \
    GEMMA_LLAMA_REASONING="${GEMMA_LLAMA_REASONING:-}" \
    GEMMA_LLAMA_REASONING_FORMAT="${GEMMA_LLAMA_REASONING_FORMAT:-}" \
    GEMMA_LLAMA_REASONING_BUDGET="${GEMMA_LLAMA_REASONING_BUDGET:-}" \
    bash "$GEMMA_LLAMA_LAUNCHER" start
  wait_ready "gemma" "$GEMMA_PORT" 900
}

restart_qwen_only() {
  echo "=== restarting qwen (llama.cpp) on port ${QWEN_PORT} ==="
  QWEN_LLAMA_GPU="$QWEN_GPU" QWEN_LLAMA_PORT="$QWEN_PORT" \
    QWEN_LLAMA_REASONING="${QWEN_LLAMA_REASONING:-}" \
    QWEN_LLAMA_REASONING_FORMAT="${QWEN_LLAMA_REASONING_FORMAT:-}" \
    QWEN_LLAMA_REASONING_BUDGET="${QWEN_LLAMA_REASONING_BUDGET:-}" \
    bash "$QWEN_LLAMA_LAUNCHER" start
  wait_ready "qwen" "$QWEN_PORT" 900
}

# --- Commands ------------------------------------------------------------------
case "${1:-start}" in
  stop)
    stop_all
    ;;
  status)
    status
    ;;
  restart-gemma)
    restart_gemma_only
    ;;
  restart-qwen)
    restart_qwen_only
    ;;
  start|*)
    echo "=== killing existing comfyui ==="
    kill_comfyui

    echo "=== starting services ==="
    : > "$PIDS_FILE"
    start_comfyui "A" "$COMFY_GPU_A" "$COMFY_PORT_A" "$COMFY_DIR/user"   "$COMFY_DIR/output" "$COMFY_DIR/temp"
    start_comfyui "B" "$COMFY_GPU_B" "$COMFY_PORT_B" "$COMFY_DIR/user2"  "$COMFY_DIR/output2" "$COMFY_DIR/temp2"
    QWEN_LLAMA_GPU="$QWEN_GPU" QWEN_LLAMA_PORT="$QWEN_PORT" \
      QWEN_LLAMA_REASONING="${QWEN_LLAMA_REASONING:-}" \
      QWEN_LLAMA_REASONING_FORMAT="${QWEN_LLAMA_REASONING_FORMAT:-}" \
      QWEN_LLAMA_REASONING_BUDGET="${QWEN_LLAMA_REASONING_BUDGET:-}" \
      bash "$QWEN_LLAMA_LAUNCHER" start
    GEMMA_LLAMA_GPU="$GEMMA_GPU" GEMMA_LLAMA_PORT="$GEMMA_PORT" \
      GEMMA_LLAMA_REASONING="${GEMMA_LLAMA_REASONING:-}" \
      GEMMA_LLAMA_REASONING_FORMAT="${GEMMA_LLAMA_REASONING_FORMAT:-}" \
      GEMMA_LLAMA_REASONING_BUDGET="${GEMMA_LLAMA_REASONING_BUDGET:-}" \
      bash "$GEMMA_LLAMA_LAUNCHER" start

    echo
    echo "=== waiting for llama.cpp servers (first load can take minutes) ==="
    wait_ready "qwen" "$QWEN_PORT" 900
    wait_ready "gemma" "$GEMMA_PORT" 900

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
