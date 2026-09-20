#!/usr/bin/env bash
# =============================================================================
# start-qwen3.8-27b-gptq.sh — Qwen3.8-27B GPTQ-INT4 (sym, G128, MTP) on Intel Arc B70
#
# Recipe: SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 (Hugging Face)
#   - GPTQ INT4 symmetric, group_size=128, desc_act=false
#   - MTP draft head preserved unquantized (BF16) -> native MTP spec decode
#   - fp8 KV cache, fp16 activations, XPU graph mode
#   - This is the GPTQ-INT4 route. Do NOT mix with the FP8 W8A16 or AutoRound
#     int4 routes (different quantization + patch sets).
#
# The venv (~/vllm-fresh-venv) is pre-patched for the qwen3_5 GDN/MTP path
# (patch_mtp_nightly + patch_mtp_boundary are already applied in the venv).
# No runtime patch step is needed here.
#
# Usage:
#   bash start-qwen3.8-27b-gptq.sh start      # launch (background, logs)
#   bash start-qwen3.8-27b-gptq.sh stop       # stop
#   bash start-qwen3.8-27b-gptq.sh status     # health + log tail
#   bash start-qwen3.8-27b-gptq.sh smoke      # one completion
#   bash start-qwen3.8-27b-gptq.sh logs [n]   # tail the log
#
# Env overrides (defaults in parens):
#   QWEN_GPTQ_MODEL   model dir (/mnt/data/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
#   QWEN_GPTQ_VENV    venv path (~/vllm-fresh-venv)
#   QWEN_GPTQ_GPU     GPU set (0,1)  [TP2: the 128K ctx needs 2 cards]
#   QWEN_GPTQ_PORT    port (8000)
#   QWEN_GPTQ_CTX     max-model-len (131072)
#   QWEN_GPTQ_UTIL    gpu-memory-utilization (0.88)
#   QWEN_GPTQ_KV      kv-cache-dtype (fp8)
#   QWEN_GPTQ_SERVED  served name (qwen38)
#   QWEN_GPTQ_MTP     MTP draft tokens (4); 0 = no spec decode
#   QWEN_GPTQ_VISION  1 = enable vision tower (adds --limit-mm-per-prompt)
#
# NOTE: this binds :8000, the same port as the Flash-Next backend. Stop the
# Flash-Next first (or override QWEN_GPTQ_PORT) before starting this one.
# =============================================================================
set -euo pipefail

VENV="${QWEN_GPTQ_VENV:-$HOME/vllm-fresh-venv}"
MODEL_DIR="${QWEN_GPTQ_MODEL:-/mnt/data/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
GPU_SET="${QWEN_GPTQ_GPU:-0,1}"
PORT="${QWEN_GPTQ_PORT:-8000}"
MAX_LEN="${QWEN_GPTQ_CTX:-131072}"
GPU_UTIL="${QWEN_GPTQ_UTIL:-0.88}"
KV_DTYPE="${QWEN_GPTQ_KV:-fp8}"
SERVED="${QWEN_GPTQ_SERVED:-qwen38}"
MTP_N="${QWEN_GPTQ_MTP:-4}"
VISION="${QWEN_GPTQ_VISION:-0}"
LOG="$HOME/electric-sheep/serve/logs/vllm_${PORT}.log"
PIDFILE="$HOME/electric-sheep/serve/logs/vllm_${PORT}.pid"

TP_SIZE="$(echo "$GPU_SET" | awk -F, '{print NF}')"
# ZE_AFFINITY_MASK picks PHYSICAL GPUs and renumbers them 0..N-1; the selector
# must then use 0..N-1 (NOT physical ids) or torch sees 0 devices.
SEL_IDX="$(seq -s, 0 $((TP_SIZE-1)))"

GEN_CONFIG="{\"temperature\": ${QWEN_TEMP:-0.7}, \"top_p\": ${QWEN_TOP_P:-0.80}, \"top_k\": ${QWEN_TOP_K:-20}, \"min_p\": ${QWEN_MIN_P:-0.0}, \"presence_penalty\": ${QWEN_PRESENCE_PENALTY:-1.5}, \"repetition_penalty\": ${QWEN_REPETITION_PENALTY:-1.0}}"

build_cmd() {
  CMD=(python3 -m vllm.entrypoints.openai.api_server
    --model "$MODEL_DIR"
    --served-model-name "$SERVED"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_LEN"
    --dtype float16
    --quantization gptq
    --kv-cache-dtype "$KV_DTYPE"
    --no-enable-prefix-caching
    --trust-remote-code
    --gpu-memory-utilization "$GPU_UTIL"
    --max-num-seqs 64
    --max-num-batched-tokens 8192
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --generation-config vllm
    --language-model-only
    --override-generation-config "$GEN_CONFIG"
  )
  if [ "$VISION" = "1" ]; then
    CMD+=(--limit-mm-per-prompt '{"image":4}')
  fi
  if [ "$MTP_N" != "0" ]; then
    CMD+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_N}}")
  fi
}

export_env() {
  export ZE_AFFINITY_MASK="$GPU_SET"
  export ONEAPI_DEVICE_SELECTOR="level_zero:$SEL_IDX"
  export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
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
  export B70_MTP_BF16_DRAFT=1
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT="${QWEN_GPTQ_MASTER_PORT:-29531}"
}

cmd="${1:-start}"
case "$cmd" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    [ -d "$MODEL_DIR" ] || { echo "MISSING model dir: $MODEL_DIR"; exit 1; }
    mkdir -p "$(dirname "$LOG")"
    build_cmd
    echo "=== Launch (GPTQ-INT4) ==="
    printf '%q ' "${CMD[@]}"; echo
    echo "GPUs=$GPU_SET (selector level_zero:$SEL_IDX)  TP=$TP_SIZE  ctx=$MAX_LEN  kv=$KV_DTYPE  mtp=$MTP_N  vision=$VISION  served=$SERVED  port=$PORT"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export_env
    cd "$HOME/electric-sheep/vllm"
    nohup "${CMD[@]}" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "launched pid=$(cat "$PIDFILE"); log=$LOG"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      PID="$(cat "$PIDFILE")"
      if kill -0 "$PID" 2>/dev/null; then
        echo "stopping pid=$PID"; kill "$PID"
        for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
        kill -0 "$PID" 2>/dev/null && { echo "force kill"; kill -9 "$PID"; }
      else
        echo "not running (stale pidfile)"
      fi
      rm -f "$PIDFILE"
    else
      pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null && echo "pkilled by cmdline" || echo "nothing to stop"
    fi
    ;;
  status)
    echo "=== process ==="
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      ps -p "$(cat "$PIDFILE")" -o pid,stat,%cpu,etime --no-headers
    else
      echo "NOT RUNNING"
    fi
    echo "=== http health (backend :$PORT) ==="
    curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && echo "health: OK" || echo "health: DOWN"
    curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print('models:',[m['id'] for m in d.get('data',[])])" 2>/dev/null || true
    echo "=== log tail ==="
    tail -n 15 "$LOG" 2>/dev/null
    ;;
  smoke)
    echo "=== one completion via :$PORT (greedy, short) ==="
    curl -s -m 60 "http://127.0.0.1:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":12,\"temperature\":0,\"top_k\":1}" \
      | python3 -c "import json,sys;d=json.load(sys.stdin);c=d.get('choices',[{}])[0];print('text:',repr(c.get('text','')));print('tokens:',[t.get('token_ids') for t in c.get('tokens',[])] if isinstance(c.get('tokens'),list) else 'n/a')" 2>/dev/null || echo "(smoke failed or server not ready)"
    ;;
  logs)
    tail -n "${2:-40}" "$LOG" 2>/dev/null
    ;;
  *)
    echo "usage: $0 {start|stop|status|smoke|logs [n]}"
    exit 1
    ;;
esac
