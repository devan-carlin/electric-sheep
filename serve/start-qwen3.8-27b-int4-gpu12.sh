#!/usr/bin/env bash
# =============================================================================
# start-qwen3.8-27b-int4-gpu12.sh — vLLM Qwen3.8-27B int4 (AutoRound) on GPUs 1,2
#
# Purpose: sustained-load GPU-health stress test on the two previously-faulty
# PCIe lanes (level_zero:1 = 47:00.0, level_zero:2 = 63:00.0), wired into the
# Open WebUI via the shared :8000 endpoint (WebUI OPENAI_API_BASE_URL already
# points at host.docker.internal:8000/v1).
#
# Model: devan-carlin-Qwen3.8-27B-int4-AutoRound (w4g128, ~19 GB, fits 2x32GB)
# Served name: qwen-256k   Port: 8000   Context: 128K   KV: fp8   TP=2
#
# Usage:
#   bash start-qwen3.8-27b-int4-gpu12.sh start     # launch (background, logs)
#   bash start-qwen3.8-27b-int4-gpu12.sh stop       # stop
#   bash start-qwen3.8-27b-int4-gpu12.sh status     # health + AER + log tail
#   bash start-qwen3.8-27b-int4-gpu12.sh smoke      # one completion via WebUI backend
#   bash start-qwen3.8-27b-int4-gpu12.sh logs       # tail the log
#
# Env overrides: QWEN12_GPU (default "1,2"), QWEN12_PORT (8000), QWEN12_CTX
#   (131072), QWEN12_UTIL (0.90), QWEN12_KV (fp8), QWEN12_SERVED (qwen-256k).
# =============================================================================
set -uo pipefail

VENV="${QWEN12_VENV:-$HOME/vllm-fresh-venv}"
MODEL_DIR="$HOME/electric-sheep/models/devan-carlin-Qwen3.8-27B-int4-AutoRound/Qwen3.8-27B-w4g128"
GPU_SET="${QWEN12_GPU:-1,2}"
PORT="${QWEN12_PORT:-8000}"
MAX_LEN="${QWEN12_CTX:-131072}"
GPU_UTIL="${QWEN12_UTIL:-0.90}"
KV_DTYPE="${QWEN12_KV:-fp8}"
SERVED="${QWEN12_SERVED:-qwen-256k}"
# QWEN12_MTP: 0 = off (default); N = enable MTP speculative decoding with N
# draft tokens (model ships mtp.layers.* weights; use 4 per the B70 cookbook).
MTP_N="${QWEN12_MTP:-0}"
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
    --dtype bfloat16
    --kv-cache-dtype "$KV_DTYPE"
    --enable-prefix-caching
    --trust-remote-code
    --gpu-memory-utilization "$GPU_UTIL"
    --block-size 32
    --max-num-batched-tokens 16384
    --max-num-seqs 2
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --generation-config vllm
    --quantization auto-round
    --language-model-only
    --override-generation-config "$GEN_CONFIG"
  )
  if [ "${MTP_N}" != "0" ]; then
    CMD+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_N}}")
  fi
}

export_env() {
  export ZE_AFFINITY_MASK="$GPU_SET"
  export ONEAPI_DEVICE_SELECTOR="level_zero:$SEL_IDX"
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
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT="${QWEN12_MASTER_PORT:-29530}"
}

cmd="${1:-start}"
case "$cmd" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    mkdir -p "$(dirname "$LOG")"
    build_cmd
    echo "=== Launch ==="
    printf '%q ' "${CMD[@]}"; echo
    echo "GPUs=$GPU_SET (selector level_zero:$SEL_IDX)  TP=$TP_SIZE  ctx=$MAX_LEN  kv=$KV_DTYPE  util=$GPU_UTIL  mtp=${MTP_N}  served=$SERVED  port=$PORT"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export_env
    cd /home/dc/electric-sheep/vllm
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
    echo "=== AER timeouts since boot (GPU fabric health) ==="
    sudo dmesg 2>/dev/null | grep -E "\[12\] Timeout" | awk '{print $NF}' | sort | uniq -c | sort -rn
    echo "(total: $(sudo dmesg 2>/dev/null | grep -cE '\[12\] Timeout'))"
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
    echo "usage: $0 {start|stop|status|smoke|logs [n]}"; exit 1
    ;;
esac
