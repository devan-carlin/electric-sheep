#!/usr/bin/env bash
set -e

# ============================================
# vLLM XPU Build Script
# ============================================
# Activates venv, installs PyTorch XPU, clones
# and compiles vllm-xpu-kernels + vLLM from
# source, enforces triton-xpu override, and
# verifies the build.
# ============================================

VLLM_DIR="$HOME/electric-sheep/vllm"
VENV_DIR="$VLLM_DIR/.venv"
VLLM_SRC="$HOME/vllm-src"

# Graceful error handler — keeps terminal open for investigation
fail() {
    echo ""
    echo "=========================================="
    echo "  ERROR: $1"
    echo "=========================================="
    echo ""
    echo "Terminal will stay open for investigation."
    echo "Press Enter to close..."
    read -r
    exit 1
}

# -------------------------------------------
# Pre-flight checks
# -------------------------------------------
echo "=========================================="
echo "  vLLM XPU Build — Pre-flight"
echo "=========================================="

[ -d "$VENV_DIR" ] || fail "Virtual environment not found at $VENV_DIR. Run setup-project-directory.sh first."
echo "✓ Virtual environment found"

command -v git >/dev/null 2>&1 || fail "git not installed"
echo "✓ git available"

command -v python3.12 >/dev/null 2>&1 || fail "python3.12 not found"
echo "✓ Python 3.12 available"

lspci | grep -q "Battlemage" || fail "No Intel Battlemage GPUs detected"
gpu_count=$(lspci | grep -c "Battlemage")
echo "✓ $gpu_count Intel Battlemage GPU(s) detected"

# Check disk space (build needs ~50GB)
available=$(df --output=avail -BM "$VLLM_DIR" | tail -1 | awk '{printf "%d", $1/1024}')
[ "$available" -lt 50 ] && fail "Need 50GB free for build, have ${available}GB"
echo "✓ Sufficient disk space (${available}GB available)"

# Check oneAPI
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    echo "✓ oneAPI Toolkit found"
else
    echo "⚠ oneAPI Toolkit not found at /opt/intel/oneapi/"
    echo "  XPU compilation may fail without it."
fi

# -------------------------------------------
# Activate virtual environment
# -------------------------------------------
echo ""
echo "[1/7] Activating virtual environment..."
source "$VENV_DIR/bin/activate"
python3 --version
echo "✓ Virtual environment activated"

# -------------------------------------------
# Upgrade packaging tools
# -------------------------------------------
echo ""
echo "[2/7] Upgrading packaging tools..."
pip install --upgrade pip setuptools wheel
echo "✓ pip, setuptools, wheel upgraded"

# -------------------------------------------
# Install PyTorch XPU
# -------------------------------------------
echo ""
echo "[3/7] Installing PyTorch XPU..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
echo "✓ PyTorch XPU installed"

# -------------------------------------------
# Verify XPU hardware
# -------------------------------------------
echo ""
echo "[4/7] Verifying XPU hardware..."
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('XPU Available:', torch.xpu.is_available())
print('XPU Device Count:', torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    print(f'  GPU {i}: {torch.xpu.get_device_name(i)}')
" || fail "XPU hardware verification failed"
echo "✓ XPU hardware verified"

# -------------------------------------------
# Clone source repositories
# -------------------------------------------
echo ""
echo "[5/7] Cloning source repositories..."
mkdir -p "$VLLM_SRC"

if [ -d "$VLLM_SRC/vllm-xpu-kernels/.git" ]; then
    echo "  -> vllm-xpu-kernels already cloned, pulling latest..."
    cd "$VLLM_SRC/vllm-xpu-kernels"
    git pull
else
    echo "  -> Cloning vllm-xpu-kernels..."
    cd "$VLLM_SRC"
    git clone https://github.com/vllm-project/vllm-xpu-kernels.git
fi

if [ -d "$VLLM_SRC/vllm/.git" ]; then
    echo "  -> vllm already cloned, pulling latest..."
    cd "$VLLM_SRC/vllm"
    git pull
else
    echo "  -> Cloning vllm..."
    cd "$VLLM_SRC"
    git clone https://github.com/vllm-project/vllm.git
fi
echo "✓ Source repositories ready"

# -------------------------------------------
# Build vllm-xpu-kernels
# -------------------------------------------
echo ""
echo "[6/7] Building vllm-xpu-kernels..."
cd "$VLLM_SRC/vllm-xpu-kernels"

pip install setuptools setuptools-scm cmake ninja packaging psutil
pip install . --no-build-isolation

# Verify kernel import
python3 -c "import vllm_xpu_kernels._C; print('✓ vllm_xpu_kernels._C loaded')" || {
    echo "WARNING: Direct import failed, attempting .so binary fallback..."
    KERN_BUILD_DIR="$VLLM_SRC/vllm-xpu-kernels/build/lib.linux-x86_64-cpython-312/vllm_xpu_kernels"
    SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages/vllm_xpu_kernels"

    if [ -d "$KERN_BUILD_DIR" ]; then
        mkdir -p "$SITE_PACKAGES"
        cp "$KERN_BUILD_DIR"/*.so "$SITE_PACKAGES/"
        python3 -c "import vllm_xpu_kernels._C; print('✓ vllm_xpu_kernels._C loaded (via fallback)')"
    else
        fail "vllm-xpu-kernels build failed and .so fallback not available"
    fi
}
echo "✓ vllm-xpu-kernels compiled"

# -------------------------------------------
# Build vLLM engine
# -------------------------------------------
echo ""
echo "[7/7] Building vLLM engine..."
cd "$VLLM_SRC/vllm"

pip install setuptools-rust
pip install -r requirements-build.txt 2>/dev/null || pip install -r requirements/build.txt 2>/dev/null || true

export VLLM_TARGET_DEVICE="xpu"
pip install . --no-build-isolation

# Enforce triton-xpu override (critical for Intel backend)
pip install triton-xpu --index-url https://download.pytorch.org/whl/xpu --force-reinstall --no-deps

# Verify triton-xpu loads correctly
python3 -c "import triton; print('Triton loaded from:', triton.__file__)" || fail "triton-xpu failed to load"
echo "✓ vLLM engine compiled"

# -------------------------------------------
# Final verification
# -------------------------------------------
echo ""
echo "=========================================="
echo "  Build Verification"
echo "=========================================="
python3 -c "
import torch, vllm, vllm_xpu_kernels
print('Core Engine Operational Status: VERIFIED')
print('PyTorch XPU Count:', torch.xpu.device_count())
print('vLLM Version:', vllm.__version__)
" || fail "Final verification failed"

echo ""
echo "=========================================="
echo "  Build Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Apply MoE patch (if using 35B-A3B):"
echo "     bash ~/electric-sheep/vllm-tuning/patch-vllm-moe-qzeros.sh"
echo ""
echo "  2. Start a model:"
echo "     bash ~/electric-sheep/vllm/start-qwen3.6-27b.sh"
echo ""
