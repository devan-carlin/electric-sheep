#!/usr/bin/env bash
set -e

# ============================================
# llama.cpp Project Setup Script
# ============================================
# Creates the llama.cpp project structure and
# clones the source repository.
# ============================================

# Detect if run via sudo — preserve original user's HOME
if [ -n "$SUDO_USER" ]; then
    export HOME=$(eval echo ~$SUDO_USER)
    echo "Note: Running via sudo, using HOME=$HOME for user $SUDO_USER"
fi

LLAMA_DIR="$HOME/electric-sheep/llama"
LLAMA_SRC="$LLAMA_DIR/llama.cpp"
MODELS_DIR="$HOME/electric-sheep/models"

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
echo "  llama.cpp Setup — Pre-flight"
echo "=========================================="

# Check for Intel GPUs
lspci | grep -q "Battlemage" || fail "No Intel Battlemage GPUs detected"
gpu_count=$(lspci | grep -c "Battlemage")
echo "✓ $gpu_count Intel Battlemage GPU(s) detected"

# Check for oneAPI
if [ ! -f /opt/intel/oneapi/setvars.sh ]; then
    fail "oneAPI Toolkit not found. Install intel-oneapi-toolkit first."
fi
echo "✓ oneAPI Toolkit found"

# Check disk space
available=$(df --output=avail -BM "$HOME" | tail -1 | awk '{printf "%d", $1/1024}')
[ "$available" -lt 20 ] && fail "Need 20GB free for build, have ${available}GB"
echo "✓ Sufficient disk space (${available}GB available)"

# -------------------------------------------
# Setup
# -------------------------------------------
echo ""
echo "=========================================="
echo "  Setting up llama.cpp Project"
echo "=========================================="

# [1/4] Create directory structure
echo ""
echo "[1/4] Creating directory structure..."
mkdir -p "$LLAMA_DIR"
mkdir -p "$MODELS_DIR"
echo "✓ Created $LLAMA_DIR"
echo "✓ Created $MODELS_DIR (shared with vLLM)"

# [2/4] Clone or update llama.cpp source
echo ""
echo "[2/4] Cloning llama.cpp source..."
if [ -d "$LLAMA_SRC/.git" ]; then
    echo "  -> llama.cpp already cloned at $LLAMA_SRC"
    read -p "  Pull latest? (Y/n): " confirm
    if [[ ! "$confirm" =~ ^[Nn]$ ]]; then
        cd "$LLAMA_SRC" && git pull
        echo "✓ Updated llama.cpp"
    else
        echo "  Keeping existing source."
    fi
else
    cd "$LLAMA_DIR"
    git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
    echo "✓ Cloned llama.cpp to $LLAMA_SRC"
fi

# [3/4] Create environment config
echo ""
echo "[3/4] Creating environment config..."
cat > "$LLAMA_DIR/set-env.sh" << 'EOF'
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

# Long-context stability (prevents watchdog resets)
export GGML_SYCL_FA_ONEDNN_MAX_KV=24576

# Allow large allocations
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

echo "✓ oneAPI loaded, SYCL devices: $ONEAPI_DEVICE_SELECTOR"
EOF

chmod +x "$LLAMA_DIR/set-env.sh"
echo "✓ Created set-env.sh"

# [4/4] Create models symlink (for backward compat)
echo ""
echo "[4/4] Creating models directory..."
if [ ! -d "$MODELS_DIR" ]; then
    mkdir -p "$MODELS_DIR"
    echo "✓ Created $MODELS_DIR"
else
    echo "✓ $MODELS_DIR already exists"
fi

echo ""
echo "=========================================="
echo "  Project Setup Complete!"
echo "=========================================="
echo ""
echo "Structure:"
echo "  ~/electric-sheep/llama/"
echo "  ├── llama.cpp/          (source code)"
echo "  ├── set-env.sh          (environment config)"
echo "  └── deepseek/           (model-specific scripts)"
echo ""
echo "  ~/electric-sheep/models/  (shared with vLLM)"
echo ""
echo "Next steps:"
echo "  1. Build:  bash ~/electric-sheep/build/ubuntu/05-build-llama-cpp.sh"
echo "  2. Download models:"
echo "     hf download <repo> --local-dir ~/electric-sheep/models/"
echo ""
