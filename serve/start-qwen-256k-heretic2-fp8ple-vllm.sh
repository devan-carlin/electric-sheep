#!/usr/bin/env bash
# =============================================================================
# start-qwen-256k-heretic2-fp8ple-vllm.sh
# Serve densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE via vLLM on 4 GPUs.
#
# Abliterated (heretic-2) sibling of the Intel AutoRound checkpoint, same
# AutoRound W4A16 target packing (qweight I32 + scales F16 + BF16 norms), so it
# rides the exact same serving shape as start-qwen-256k-intel-autoround-vllm.sh:
# port 8000, alias qwen-256k, 256K ctx, fp8 KV, TP4 + expert parallel, PLE n-gram
# table memory-mapped from host RAM.
#
# WHY IT LOADS ON THE XPU FORK (verified 2026-09-26):
#   - Target shards are AutoRound W4A16, byte-for-byte the same packing the Intel
#     run already serves. No new quant path.
#   - The "FP8PLE" part is the n-gram/PLE table shipped as FP8 E4M3 inside
#     model-plefp8-*.safetensors. The fork's qwen4_exp.load_weights runs
#     skip_substrs=[".ngram_embedding.", ".indexer."], so those tensors are
#     SKIPPED and PLE is read from the mmap'd BF16 PLE_TABLE_PATH file instead.
#     The n-gram table is target-independent, so the shared BF16 table is valid.
#   - The FP8 PLE shards are NOT downloaded (target-only download). The loader
#     globs *.safetensors on disk (default_loader.py), not the index weight_map,
#     so the absent plefp8 files are simply never opened.
#   - config.json is a strict superset of the Intel config; the only extra keys
#     are the 128 skipped ngram_embedding entries + a vision model_type the text
#     path ignores. No config patch needed.
#
# The FP8 PLE saves host RAM (102 GB BF16 -> ~51 GB FP8) for a 128 GB-RAM CUDA
# rig. This box has 247 GB RAM + 255 GB swap, so the BF16 table fits with room to
# spare and the FP8 table buys nothing here. That is why we skip it.
#
# Usage:
#   bash start-qwen-256k-heretic2-fp8ple-vllm.sh          # start (kills :8000 first)
#   bash start-qwen-256k-heretic2-fp8ple-vllm.sh stop      # stop
#   bash start-qwen-256k-heretic2-fp8ple-vllm.sh status    # check
#
# Env overrides: QWEN256K_PORT, QWEN256K_ALIAS, QWEN256K_GPUS, QWEN256K_MODEL,
#                QWEN256K_CTX, QWEN256K_MEM_UTIL, QWEN256K_TEMP, PLE_TABLE_PATH
# =============================================================================
set -uo pipefail

PORT="${QWEN256K_PORT:-8000}"
ALIAS="${QWEN256K_ALIAS:-qwen-256k}"
GPUS="${QWEN256K_GPUS:-0,1,2,3}"
MODEL="${QWEN256K_MODEL:-/mnt/data/models/densohax-Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE}"
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
# Shared BF16 n-gram PLE table (target-independent). Bound here, not at the
# export site below, because the start-case guard checks it under set -u.
PLE_TABLE_PATH="${PLE_TABLE_PATH:-/mnt/data/ple_cache/ple_table_qwen4exp.pt}"
# Distinct log so this variant and the other launchers don't clobber each other.
LOG="$HOME/electric-sheep/serve/logs/vllm_${PORT}-heretic2-fp8ple.log"
# Verified production venv: clean upstream main + the qwen4exp port, built from
# scratch. Same venv as the Intel AutoRound launcher.
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
    # Guard: the target-only download must be complete before launch. The
    # loader globs *.safetensors on disk, so a partial download loads a
    # partial model. 49 target shards + index + configs.
    NSHARD=$(find "$MODEL" -maxdepth 1 -name 'model-*-of-00049.safetensors' 2>/dev/null | wc -l)
    if [ "$NSHARD" -lt 49 ]; then
      echo "ERROR: only $NSHARD/49 target shards present in $MODEL." >&2
      echo "       Download incomplete. Re-run the hf download (see header)." >&2
      exit 1
    fi
    if [ ! -f "$PLE_TABLE_PATH" ]; then
      echo "ERROR: PLE table not found at $PLE_TABLE_PATH." >&2
      echo "       Build it with phase_b_ple_table_prep.py (target-independent)." >&2
      exit 1
    fi
    # cwd must NOT be inside site-packages/vllm: `import tokenizers` would
    # resolve to vllm/tokenizers/ (shadowing the real package) -> circular import.
    cd "$HOME/electric-sheep/serve"
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
    source "$VENV/bin/activate"
    # Device selection: process-wide mask lists all four cards.
    # NOTE: do NOT set ONEAPI_DEVICE_SELECTOR here. In this torch build any
    # selector makes torch._C._xpu_init() report 0 devices once workers run with
    # per-worker ZE_AFFINITY_MASK values (verified 2026-09-05: sel+mask -> "No XPU
    # devices"; mask alone works). The mask alone filters correctly for sycl+torch.
    export ZE_AFFINITY_MASK="$GPUS"
    export UR_L0_SYNC_MODE=BLOCKING
    export VLLM_TARGET_DEVICE=xpu
    export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
    export TRITON_CACHE_DIR="$HOME/.cache/triton"
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29530
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    # oneCCL large-message thresholds (Intel-documented values, cookbook DUAL-B70-TP2).
    # Keep the simple (host-staged) path for big collectives; avoids peer-memory IPC
    # handles that Arc desktop PCIe cannot open.
    export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
    export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
    export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
    export CCL_SYCL_ALLTOALL_TMP_BUF=1
    # Worker-affinity patch (venv fork): at spawn time each worker gets
    # ZE_AFFINITY_MASK=<its own card>, so every rank initializes exactly one device
    # (no ~GB-scale host-RAM duplication, no cross-device IPC handles). Validated
    # on this host 2026-09-05 (TP2 test + TP4 ladder). Kill switch: set
    # B70_WORKER_AFFINITY=0 to revert to unpatched behavior.
    export B70_WORKER_AFFINITY="${B70_WORKER_AFFINITY:-1}"
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
    # Shared BF16 n-gram PLE table (target-independent). The FP8 PLE shards in
    # the checkpoint are skipped by the loader; this mmap'd file is the real table.
    export PLE_TABLE_PATH
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
