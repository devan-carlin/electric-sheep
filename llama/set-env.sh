#!/usr/bin/env bash
# ============================================
# llama.cpp Environment Configuration
# ============================================
# Load oneAPI and configure SYCL for 4x Arc B70
#
# Usage:
#   source set-env.sh              # All 4 GPUs
#   source set-env.sh 0,1          # GPUs 0 and 1 only
#   source set-env.sh 0            # GPU 0 only
# ============================================

# Load oneAPI (idempotent)
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

# Device selector (override with argument)
export ONEAPI_DEVICE_SELECTOR="level_zero:${1:-0,1,2,3}"
export ZES_ENABLE_SYSMAN=1

# SYCL performance tuning
export GGML_SYCL_ENABLE_FLASH_ATTN=1
export GGML_SYCL_ENABLE_OPT=1
export GGML_SYCL_ENABLE_DNN=1
export GGML_SYCL_ENABLE_MKL_FA=1

# Long-context stability (prevents watchdog resets).
# 65536: oneDNN flash-attention fast path stays active up to 64K KV length
# (above this the backend falls back to the slower path).
export GGML_SYCL_FA_ONEDNN_MAX_KV=65536

# Allow large allocations
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

echo "✓ oneAPI loaded, SYCL devices: $ONEAPI_DEVICE_SELECTOR"
