#!/usr/bin/env bash
# One-off infra test: Qwen3.8-27B GPTQ on GPU2+GPU3 (TP2) via the fresh venv.
# Validates the multi-GPU contract (worker-affinity patch + oneCCL vars) on
# this host WITHOUT touching the live fleet on GPU0/GPU1.
#
# Success criteria: clean startup, [B70_WORKER_AFFINITY_SPAWN] rank=0 mask=2 /
# rank=1 mask=3, [B70_WORKER_AFFINITY] visible=1 on both workers, /health OK,
# coherent completion. Speed is NOT the goal (expect below single-card 81-83).
#
# Run:   bash test-27b-tp2.sh            (foreground, logs to serve/logs/)
# Stop:  pkill -f "vllm.entrypoints.openai.api_server.*--port 8002"
set -uo pipefail
PORT=8002
ALIAS=qwen38-tp2-test
GPUS="2,3"
MODEL="$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
CTX=131072
MAX_SEQS=64
MEM_UTIL=0.88
LOG="$HOME/electric-sheep/serve/logs/vllm_${PORT}.log"
VENV="$HOME/vllm-fresh-venv"

cd "$HOME/electric-sheep/serve"
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
source "$VENV/bin/activate"

# Device selection: only GPU2+GPU3 visible to this process.
# NOTE: do NOT set ONEAPI_DEVICE_SELECTOR here. In this torch build the
# selector makes torch._C._xpu_init() report 0 devices even when it matches
# ZE_AFFINITY_MASK (verified: sel=level_zero:2 + mask=2 -> "No XPU devices";
# mask alone -> OK). The mask alone filters correctly for both sycl and torch.
export ZE_AFFINITY_MASK="$GPUS"
export UR_L0_SYNC_MODE=BLOCKING
export VLLM_TARGET_DEVICE=xpu
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29531   # avoid 29530 (Flash-Next launcher)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_XPU_ENABLE_XPU_GRAPH="${TEST_TP2_XPU_GRAPH:-1}"

# Worker-affinity patch: spawn-time per-worker ZE_AFFINITY_MASK=<rank>.
# Kill switch: set B70_WORKER_AFFINITY=0 to run unpatched behavior.
export B70_WORKER_AFFINITY="${B70_WORKER_AFFINITY:-1}"

# oneCCL large-message thresholds (Intel-documented values, cookbook DUAL-B70-TP2).
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLTOALL_TMP_BUF=1

exec python -m vllm.entrypoints.openai.api_server > "$LOG" 2>&1 \
  --model "$MODEL" \
  --served-model-name "$ALIAS" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len "$CTX" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
