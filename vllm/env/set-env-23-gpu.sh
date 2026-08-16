#!/usr/bin/env bash
# Device Bindings: GPUs 2,3
export ONEAPI_DEVICE_SELECTOR=level_zero:2,3
export UR_L0_SYNC_MODE=BLOCKING
export TORCH_LLM_ALLREDUCE=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

# GPU Affinity (binds GPUs to CPU cores for NUMA locality)
export ZE_AFFINITY_MASK=2,3

# Graph capture with communication ops (improves multi-GPU performance)
export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1

# vLLM Execution Parameters
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export UVICORN_KEEP_ALIVE_TIMEOUT=300
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_TARGET_DEVICE=xpu

# Tensor Parallelism (2 GPUs)
export TP_SIZE=2
