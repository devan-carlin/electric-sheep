#!/usr/bin/env bash
# Build vllm-xpu-kernels (baseline, then patched) into the int8 test venv.
# Usage: bash build-kernels.sh [baseline|patched]
set -uo pipefail

MODE="${1:-baseline}"
VENV="/home/dc/electric-sheep/vllm/.venv-int8-test"
KDIR="/home/dc/electric-sheep/vllm/vllm-src-int8-test/vllm-xpu-kernels"

source "$VENV/bin/activate"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

echo "=== icpx: $(command -v icpx && icpx --version 2>&1 | head -1) ==="
echo "=== mode: $MODE ==="

cd "$KDIR"
export VLLM_XPU_KERNELS_MAX_JOBS=4
pip install -q setuptools setuptools-scm cmake ninja packaging psutil
pip install . --no-build-isolation 2>&1 | tail -25

echo "=== build exit: $? ==="
python - <<'EOF' 2>&1 | tail -5
import torch
import vllm_xpu_kernels._xpu_C  # registers ops
ops = sorted({str(s) for s in torch._C._jit_get_all_schemas() if s.name.startswith("_xpu_C::") and ("gemm" in s.name or "int8" in s.name)})
print("gemm/int8 ops:")
for o in ops:
    print(" ", o)
EOF
