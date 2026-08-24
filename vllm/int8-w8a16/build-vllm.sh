#!/usr/bin/env bash
# Build vLLM (XPU) into the int8 test venv, using the locally-built
# vllm-xpu-kernels (with the int8 W8A16 patch).
# Prereq: bash build-kernels.sh patched  (kernels already installed)
set -uo pipefail

VENV="/home/dc/electric-sheep/vllm/.venv-int8-test"
VLLM_DIR="/home/dc/electric-sheep/vllm/vllm-src-int8-test/vllm"
KDIR="/home/dc/electric-sheep/vllm/vllm-src-int8-test/vllm-xpu-kernels"

source "$VENV/bin/activate"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

echo "=== python: $(command -v python) ==="
python -c "import torch; print('torch', torch.__version__, 'xpu', torch.xpu.is_available())"

# packaging tools
pip install -q "setuptools>=77.0.3,<81.0.0" setuptools-scm cmake ninja packaging psutil setuptools-rust

# vLLM's requirements/xpu.txt pins vllm_xpu_kernels==0.1.13.2, which overwrites
# the local (patched) build during `pip install .`. Back up the installed
# patched package now so we can restore the .so files without a rebuild.
KPKG="$(python -c 'import vllm_xpu_kernels, os; print(os.path.dirname(vllm_xpu_kernels.__file__))')"
BACKUP="/tmp/vllm_xpu_kernels_patched_backup"
rm -rf "$BACKUP"
cp -a "$KPKG" "$BACKUP"
echo "=== backed up patched kernels -> $BACKUP ($(ls "$BACKUP"/*.so 2>/dev/null | wc -l) .so files) ==="

# build vllm
cd "$VLLM_DIR"
export VLLM_TARGET_DEVICE="xpu"
# vllm's requirements/xpu.txt pins triton==3.7.2+xpu, which only exists on
# vLLM's own wheel index. setup.py strips the --extra-index-url line from
# xpu.txt, so pass it explicitly here.
VLLM_XPU_INDEX="https://wheels.vllm.ai/xpu/"
pip install -r requirements-build.txt 2>/dev/null || pip install -r requirements/build.txt 2>/dev/null || true
pip install . --no-build-isolation --extra-index-url "$VLLM_XPU_INDEX" 2>&1 | tail -30
echo "=== vllm build exit: $? ==="

# Restore the patched kernels (the vllm install replaced them with the PyPI
# 0.1.13.2 wheel). Copy the entire backed-up package back over the installed
# one so the exact patched build (main @ 13013c5 + int8 W8A16) is in place.
KPKG="$(python -c 'import vllm_xpu_kernels, os; print(os.path.dirname(vllm_xpu_kernels.__file__))')"
cp -a "$BACKUP"/. "$KPKG"/
echo "=== kernels restore exit: $? ==="

# enforce triton-xpu override
pip install triton-xpu --index-url https://download.pytorch.org/whl/xpu --force-reinstall --no-deps 2>&1 | tail -3

# torchcodec: vllm pulls in the CUDA build, which fails to load on XPU
# (needs libc10_cuda.so / libnvrtc.so.13). vllm's import guard at
# multimodal/video.py catches ImportError and substitutes a placeholder, so
# uninstalling it is the clean path for a text-only model.
pip uninstall -y torchcodec 2>&1 | tail -1

# gguf plugin (not needed for this model, but harmless)
pip install --no-deps vllm-gguf-plugin 2>&1 | tail -2 || true

echo "=== verify ==="
# run from a neutral dir so `import vllm` resolves to the installed package,
# not the source tree in cwd
cd /tmp
python - <<'EOF'
import torch, vllm, vllm_xpu_kernels
print('vllm', vllm.__version__)
print('xpu devices', torch.xpu.device_count())
import vllm_xpu_kernels._xpu_C  # registers ops
ops = sorted({str(s) for s in torch._C._jit_get_all_schemas() if s.name.startswith("_xpu_C::") and ("gemm" in s.name or "int8" in s.name)})
print("gemm/int8 ops:")
for o in ops:
    print(" ", o)
EOF
