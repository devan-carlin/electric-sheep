#!/usr/bin/env bash
# Smoke test: load the lued INT8 W8A16 model on the B70 XPU stack and run a
# short generation. Uses the int8 test venv (patched kernels + vllm).
set -uo pipefail

VENV="/home/dc/electric-sheep/vllm/.venv-int8-test"
MODEL="/mnt/data/models/lued-Qwen3.8-27B-INT8-W8A16-MTP"

source "$VENV/bin/activate"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

export CUDA_VISIBLE_DEVICES=""
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1   # GPUs 0,1 (TP=2)

# Must be a real file: vllm's spawn multiprocessing re-imports __main__ by
# path, which fails for `python - <<heredoc` (no file for <stdin>).
python "$(dirname "$0")/test-int8-model.py"
