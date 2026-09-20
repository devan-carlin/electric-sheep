#!/usr/bin/env bash
# Offline MTP re-fit data generation launcher.
#
# Loads the Flash-Next W4A16 target on all 4 GPUs (TP4, spec OFF, eager, no
# prefix caching, max_num_seqs=1) and generates diverse completions while the
# B70_MTP_DATAGEN_HOOK tracer captures per-token wide hyper-connection states
# to $PWD/trace.bin.
#
# PREREQUISITE: the :8000 serving instance must be STOPPED first (this uses
# all 4 GPUs). Stop it with:
#   pkill -f "vllm.entrypoints.openai.api_server.*--port 8000"
#
# Env overrides: DATAGEN_PROMPTS, DATAGEN_TRACE, DATAGEN_MAX_TOKENS,
#                DATAGEN_COUNT, DATAGEN_OUT
set -uo pipefail
cd "$(dirname "$0")"

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
source "$HOME/vllm-fresh-venv/bin/activate"

GPUS="0,1,2,3"
# Device selection: process-wide mask lists all four cards. Do NOT set
# ONEAPI_DEVICE_SELECTOR (torch _xpu_init sees 0 devices when combined with
# per-worker masks; see repo memory qwen-serving.md).
export ZE_AFFINITY_MASK="$GPUS"
export UR_L0_SYNC_MODE=BLOCKING
export VLLM_TARGET_DEVICE=xpu
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29531
export VLLM_XPU_ENABLE_XPU_GRAPH=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PLE_TABLE_PATH="${PLE_TABLE_PATH:-/mnt/data/ple_cache/ple_table_qwen4exp.pt}"

# oneCCL large-message thresholds (Intel-documented, cookbook DUAL-B70-TP2).
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLTOALL_TMP_BUF=1

# Tracer: capture wide states to ./trace.bin (rank 0 only).
export B70_MTP_DATAGEN=1
export B70_MTP_DATAGEN_TRACE="${DATAGEN_TRACE:-$PWD/trace.bin}"

echo "[run_datagen] trace -> $B70_MTP_DATAGEN_TRACE"
echo "[run_datagen] prompts <- ${DATAGEN_PROMPTS:-prompts.jsonl}"
echo "[run_datagen] max_tokens=${DATAGEN_MAX_TOKENS:-2048} count=${DATAGEN_COUNT:-all}"

if ss -tlnp 2>/dev/null | grep -q ":8000 "; then
  echo "[run_datagen] WARNING: port 8000 is in use. Stop the serving instance first." >&2
  exit 1
fi

exec python gen_data.py
