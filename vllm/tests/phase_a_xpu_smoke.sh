#!/usr/bin/env bash
# Phase A / A9: launch the XPU generation smoke test for Qwen4-Exp.
# vllm.LLM spawns the 4 TP workers itself, so this is a single process.
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
export MASTER_PORT=29513
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300

cd /home/dc/electric-sheep/vllm

python phase_a_xpu_smoke_test.py > /tmp/phase_a_xpu_smoke.log 2>&1
rc=$?

echo "=== smoke test exit=$rc ==="
tail -40 /tmp/phase_a_xpu_smoke.log
exit $rc