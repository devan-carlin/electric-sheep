#!/usr/bin/env bash
# =============================================================================
# start-qwen-256k-intel-autoround-vllm.sh
# Serve Intel/Qwen3.8-Flash-Next-W4A16-AutoRound via vLLM on 4 GPUs.
#
# Variant of start-qwen-256k-vllm.sh. Same base model (Qwen3.8-Flash-Next,
# 125B MoE / 6B active), same serving shape: port 8000, alias qwen-256k,
# 256K context, fp8 KV, TP4 + expert parallel, PLE n-gram table memory-mapped
# from host RAM. The only delta is the checkpoint: Intel's official AutoRound
# W4A16 quant (vs the devancarlin W4A16-BF16src base).
#
# Model: Intel-Qwen3.8-Flash-Next-W4A16-AutoRound (W4A16, AutoRound tuning).
# Override with QWEN256K_MODEL to serve another checkpoint.
# Requires the qwen4exp port in the vLLM tree (site-packages qwen4_exp.py)
# and the PLE table at PLE_TABLE_PATH (built by phase_b_ple_table_prep.py).
#
# NOTE: XPU graph mode (VLLM_XPU_ENABLE_XPU_GRAPH=1, no --enforce-eager).
# PLE does host-side (CPU) n-gram hashing with GPU->CPU syncs; vLLM's
# piecewise graph capture (FULL_AND_PIECEWISE) breaks the graph only at PLE
# and captures the other layers.
#
# Usage:
#   bash start-qwen-256k-intel-autoround-vllm.sh          # start (kills :8000 first)
#   bash start-qwen-256k-intel-autoround-vllm.sh stop     # stop
#   bash start-qwen-256k-intel-autoround-vllm.sh status   # check
#
# Env overrides: QWEN256K_PORT, QWEN256K_ALIAS, QWEN256K_GPUS, QWEN256K_MODEL,
#                QWEN256K_CTX, QWEN256K_MEM_UTIL, QWEN256K_TEMP, PLE_TABLE_PATH
# =============================================================================
set -uo pipefail

PORT="${QWEN256K_PORT:-8000}"
ALIAS="${QWEN256K_ALIAS:-qwen-256k}"
GPUS="${QWEN256K_GPUS:-0,1,2,3}"
MODEL="${QWEN256K_MODEL:-/mnt/data/models/Intel-Qwen3.8-Flash-Next-W4A16-AutoRound}"
CTX="${QWEN256K_CTX:-262144}"
MEM_UTIL="${QWEN256K_MEM_UTIL:-0.85}"
# Sampling defaults. The model's generation_config.json ships temperature 1.0,
# which degrades fast on this model (repetition / incoherent tails). Pin the
# vendor-recommended Instruct-mode values below; overridden onto the model
# config via --override-generation-config. Each is env-overridable.
TEMP="${QWEN256K_TEMP:-0.7}"
TOP_P="${QWEN256K_TOP_P:-0.80}"
TOP_K="${QWEN256K_TOP_K:-20}"
MIN_P="${QWEN256K_MIN_P:-0.0}"
PRESENCE_PENALTY="${QWEN256K_PRESENCE_PENALTY:-1.5}"
REPETITION_PENALTY="${QWEN256K_REPETITION_PENALTY:-1.0}"
GEN_CONFIG="{\"temperature\": $TEMP, \"top_p\": $TOP_P, \"top_k\": $TOP_K, \"min_p\": $MIN_P, \"presence_penalty\": $PRESENCE_PENALTY, \"repetition_penalty\": $REPETITION_PENALTY}"
# Per-GPU GiB of weights to UVA-offload to pinned host RAM (0 = none).
# W4A16 fits without it; raise only if VRAM is tight.
CPU_OFFLOAD_GB="${QWEN256K_CPU_OFFLOAD_GB:-0}"
# Distinct log so this variant and the base launcher don't clobber each other.
LOG="$HOME/electric-sheep/serve/logs/vllm_${PORT}-intel-autoround.log"
# Verified production venv: clean upstream main + the qwen4exp port, built from
# scratch. Same venv as the base start-qwen-256k-vllm.sh.
VENV="${QWEN256K_VENV:-$HOME/vllm-fresh-venv}"
# Optional API key. Only passed when the key file exists, so the launcher
# still works keyless. Same source of truth as the base launcher.
API_KEY_FILE="$HOME/electric-sheep/serve/.api_key"
API_KEY=""
[ -f "$API_KEY_FILE" ] && API_KEY="$(cat "$API_KEY_FILE")"

stop() {
  pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null && sleep 5
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null
  pgrep -af "vllm.entrypoints.openai.api_server.*--port $PORT" >/dev/null && echo "still running" || echo "stopped"
}

status() {
  if pgrep -af "vllm.entrypoints.openai.api_server.*--port $PORT" >/dev/null; then
    echo "RUNNING: $(pgrep -af "vllm.entrypoints.openai.api_server.*--port $PORT" | head -1)"
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" | head -c 200; echo
  else
    echo "not running"
  fi
}

case "${1:-start}" in
  stop)   stop ;;
  status) status ;;
  start)
    stop
    # cwd must NOT be inside site-packages/vllm: `import tokenizers` would
    # resolve to vllm/tokenizers/ (shadowing the real package) -> circular import.
    cd "$HOME/electric-sheep/serve"
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
    source "$VENV/bin/activate"
    export ZE_AFFINITY_MASK="$GPUS"
    export ONEAPI_DEVICE_SELECTOR="level_zero:$GPUS"
    export UR_L0_SYNC_MODE=BLOCKING
    export VLLM_TARGET_DEVICE=xpu
    export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
    export TRITON_CACHE_DIR="$HOME/.cache/triton"
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29530
    # Overridable: set VLLM_XPU_ENABLE_XPU_GRAPH=0 if graph capture fails
    # (e.g. with UVA-offloaded weights).
    export VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}"
    # XPU: Triton kernels reject CPU pointers, so UVA zero-copy offload breaks
    # (fused RMSNorm got a CPU ptr). Disable UVA -> offloader falls back to
    # on-demand to(device) per forward (correct, slower).
    # The on-demand fallback also breaks graph replay (pointers change per
    # call), so graph capture is disabled too when offloading.
    if [ "$CPU_OFFLOAD_GB" != "0" ]; then
      export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
      export VLLM_XPU_ENABLE_XPU_GRAPH=0
    fi
    export PLE_TABLE_PATH="${PLE_TABLE_PATH:-/mnt/data/ple_cache/ple_table_qwen4exp.pt}"
    CMD=(python -m vllm.entrypoints.openai.api_server
      --model "$MODEL"
      --served-model-name "$ALIAS"
      --host 0.0.0.0
      --port "$PORT"
      --tensor-parallel-size 4
      --enable-expert-parallel
      --dtype bfloat16
      --max-model-len "$CTX"
      --max-num-seqs 4
      --gpu-memory-utilization "$MEM_UTIL"
      --kv-cache-dtype fp8
      --reasoning-parser qwen3
      --enable-auto-tool-choice
      --tool-call-parser qwen3_xml
      --generation-config vllm
      --override-generation-config "$GEN_CONFIG"
    )
    if [ "$CPU_OFFLOAD_GB" != "0" ]; then
      CMD+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
    fi
    # Optional extra flags (e.g. speculative decoding). Space-separated.
    #   QWEN256K_SPEC_FLAGS='--speculative-config {"method":"mtp","num_speculative_tokens":1}'
    if [ -n "${QWEN256K_SPEC_FLAGS:-}" ]; then
      # shellcheck disable=SC2206
      CMD+=($QWEN256K_SPEC_FLAGS)
    fi
    # Optional API key (only when the key file is present).
    if [ -n "$API_KEY" ]; then
      CMD+=(--api-key "$API_KEY")
    fi
    echo "=== Launch command ==="
    printf '%q ' "${CMD[@]}"; echo
    nohup "${CMD[@]}" > "$LOG" 2>&1 &
    echo "started pid $! (log: $LOG)"
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac