#!/usr/bin/env bash
# Phase A / A8: launch the 4-rank XPU weight-load test for Qwen4-Exp.
# Mirrors the XPU env from serve/fallback/start-qwen.sh.
set -uo pipefail

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
source /home/dc/electric-sheep/vllm/.venv/bin/activate

GPU_SET="${GPU_SET:-0,1,2,3}"
TP_SIZE="$(echo "$GPU_SET" | awk -F, '{print NF}')"

export ZE_AFFINITY_MASK="$GPU_SET"
export ONEAPI_DEVICE_SELECTOR="level_zero:$(seq -s, 0 $((TP_SIZE-1)))"
export UR_L0_SYNC_MODE=BLOCKING
export TORCH_LLM_ALLREDUCE=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TARGET_DEVICE=xpu
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29512
export WORLD_SIZE="$TP_SIZE"

cd /home/dc/electric-sheep/vllm

pids=()
for r in $(seq 0 $((TP_SIZE-1))); do
  RANK=$r WORLD_SIZE=$TP_SIZE python phase_a_xpu_load_test.py \
    > "/tmp/phase_a_xpu_rank${r}.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

echo "=== rank 0 summary ==="
tail -25 /tmp/phase_a_xpu_rank0.log
exit $fail