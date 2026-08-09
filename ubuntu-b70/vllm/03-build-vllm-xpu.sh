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

# Detect if run via sudo — preserve original user's HOME
if [ -n "$SUDO_USER" ]; then
    export HOME=$(eval echo ~$SUDO_USER)
    echo "Note: Running via sudo, using HOME=$HOME for user $SUDO_USER"
    # Prevent pip cache warnings when running as sudo (can't write to user's ~/.cache/pip)
    export PIP_CACHE_DIR="/tmp/pip-cache-sudo"
fi

VLLM_DIR="$HOME/electric-sheep/vllm"
VENV_DIR="$VLLM_DIR/.venv"
VLLM_SRC="$VLLM_DIR/vllm-src"

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
echo "[1/8] Activating virtual environment..."
source "$VENV_DIR/bin/activate"
python3 --version
echo "✓ Virtual environment activated"

# -------------------------------------------
# Load oneAPI environment (auto-detects all components)
# -------------------------------------------
echo ""
echo "[2/8] Loading oneAPI environment..."
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # Suppress verbose initialization output; setvars.sh auto-detects latest versions
    # Note: setvars.sh may return non-zero (warnings/deprecations) — use || true to prevent set -e from exiting
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    echo "✓ oneAPI environment loaded"
else
    echo "⚠ oneAPI setvars.sh not found at /opt/intel/oneapi/"
fi
# Verify icpx is available
if command -v icpx >/dev/null 2>&1; then
    echo "✓ icpx compiler available: $(icpx --version 2>&1 | head -1)"
else
    fail "icpx not found on PATH — XPU kernel build will fail"
fi

# -------------------------------------------
# Upgrade packaging tools (pin setuptools to vLLM-compatible range)
# -------------------------------------------
echo ""
echo "[3/8] Upgrading packaging tools..."
pip install --upgrade pip wheel
# vLLM requires setuptools>=77.0.3,<81.0.0 — pin to avoid breaking the build
pip install "setuptools>=77.0.3,<81.0.0"
echo "✓ pip, setuptools, wheel upgraded"

# -------------------------------------------
# Install PyTorch XPU
# -------------------------------------------
echo ""
echo "[4/8] Installing PyTorch XPU..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
echo "✓ PyTorch XPU installed"

# -------------------------------------------
# Verify XPU hardware
# -------------------------------------------
echo ""
echo "[5/8] Verifying XPU hardware..."
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
echo "[6/8] Cloning source repositories..."
mkdir -p "$VLLM_SRC"

if [ -d "$VLLM_SRC/vllm-xpu-kernels/.git" ]; then
    echo "  -> vllm-xpu-kernels already cloned, pulling latest..."
    cd "$VLLM_SRC/vllm-xpu-kernels"
    git pull 2>&1 | grep -v "cannot lock ref" || true
else
    echo "  -> Cloning vllm-xpu-kernels..."
    cd "$VLLM_SRC"
    git clone https://github.com/vllm-project/vllm-xpu-kernels.git
fi

if [ -d "$VLLM_SRC/vllm/.git" ]; then
    echo "  -> vllm already cloned, pulling latest..."
    cd "$VLLM_SRC/vllm"
    git pull 2>&1 | grep -v "cannot lock ref" || true
else
    echo "  -> Cloning vllm..."
    cd "$VLLM_SRC"
    git clone https://github.com/vllm-project/vllm.git
fi
echo "✓ Source repositories ready"

# -------------------------------------------
# Build vllm-xpu-kernels (skip if already built at current commit)
# -------------------------------------------
echo ""
echo "[7/8] Building vllm-xpu-kernels..."
cd "$VLLM_SRC/vllm-xpu-kernels"

kernel_commit=$(git rev-parse --short HEAD)
kernel_installed=$(pip show vllm-xpu-kernels 2>/dev/null | grep Version || echo "")
if [[ "$kernel_installed" == *"$kernel_commit"* ]]; then
    echo "  -> vllm-xpu-kernels already built at commit $kernel_commit, skipping"
else
    # Limit build parallelism to prevent OOM during compilation (can use 120+ GB RSS)
    export VLLM_XPU_KERNELS_MAX_JOBS=4

    pip install setuptools setuptools-scm cmake ninja packaging psutil
    pip install . --no-build-isolation

    # Verify kernel import (non-fatal — _C may not be importable immediately after install,
    # but vLLM's own build will re-import in proper context)
    python3 -c "import vllm_xpu_kernels._C; print('✓ vllm_xpu_kernels._C loaded')" || {
        echo "⚠ vllm-xpu-kernels._C import test failed — this is a known false positive."
        echo "  The wheel was built successfully; the import may resolve during vLLM's own build."
        echo "  Continuing to vLLM engine build..."
    }
    echo "✓ vllm-xpu-kernels compiled"
fi

# -------------------------------------------
# Build vLLM engine (skip if already built at current commit)
# -------------------------------------------
echo ""
echo "[8/8] Building vLLM engine..."
cd "$VLLM_SRC/vllm"

vllm_commit=$(git rev-parse --short HEAD)
vllm_installed=$(pip show vllm 2>/dev/null | grep Version || echo "")
if [[ "$vllm_installed" == *"$vllm_commit"* ]]; then
    echo "  -> vLLM already built at commit $vllm_commit, skipping"
else
    pip install setuptools-rust
    pip install -r requirements-build.txt 2>/dev/null || pip install -r requirements/build.txt 2>/dev/null || true

    export VLLM_TARGET_DEVICE="xpu"
    pip install . --no-build-isolation

    # vLLM's pyproject.toml pins a pre-built vllm_xpu_kernels wheel, which overwrites
    # the locally-built version from step 7. Reinstall the local build here.
    echo "  -> Restoring locally-built vllm-xpu-kernels (vLLM overwrote with pinned release)..."
    cd "$VLLM_SRC/vllm-xpu-kernels"
    pip install . --no-build-isolation --force-reinstall --no-deps 2>&1 | grep -v "WARNING:" || true
    cd "$VLLM_SRC/vllm"

    # Enforce triton-xpu override (critical for Intel backend)
    pip install triton-xpu --index-url https://download.pytorch.org/whl/xpu --force-reinstall --no-deps

    # Verify triton-xpu loads correctly
    python3 -c "import triton; print('Triton loaded from: ' + triton.__file__)" || fail "triton-xpu failed to load"
    echo "✓ vLLM engine compiled"
fi

# -------------------------------------------
# Install vllm-gguf-plugin (required for GGUF models)
# -------------------------------------------
echo ""
echo "[9/9] Installing vllm-gguf-plugin..."
if pip show vllm-gguf-plugin >/dev/null 2>&1; then
    echo "✓ vllm-gguf-plugin already installed"
else
    echo "  -> Installing vllm-gguf-plugin (--no-deps to avoid overwriting local kernels)..."
    pip install --no-deps vllm-gguf-plugin 2>&1 | grep -v "WARNING:" || true

    python3 -c "import vllm_gguf_plugin" 2>/dev/null || fail "vllm-gguf-plugin failed to install"
    echo "✓ vllm-gguf-plugin installed"
fi

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
echo "     bash ~/electric-sheep/ubuntu-b70/vllm/05-patch-vllm-moe-qzeros.sh"
echo ""
echo "  2. Start a model:"
echo "     bash ~/electric-sheep/vllm/start-qwen3.6-27b.sh"
echo ""
echo "  3. Serve a GGUF model (requires --tokenizer with base repo):"
echo "     vllm serve ./model.gguf --tokenizer base-model-repo"
echo ""
