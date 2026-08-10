#!/usr/bin/env bash
set -e

# ============================================
# Project Directory Setup Script
# ============================================
# Creates the vLLM project structure and virtual
# environment. Does NOT download models or create
# startup scripts (those are in 04-download-models.sh).
# ============================================

# Detect if run via sudo — preserve original user's HOME
if [ -n "$SUDO_USER" ]; then
    export HOME=$(eval echo ~$SUDO_USER)
    echo "Note: Running via sudo, using HOME=$HOME for user $SUDO_USER"
    # Ensure pipx-installed binaries (hf CLI) are on PATH when run via sudo
    export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
else
    # When run directly, pipx installs to ~/.local/bin
    export PATH="$HOME/.local/bin:$PATH"
fi

VLLM_DIR="$HOME/electric-sheep/vllm"
VENV_DIR="$VLLM_DIR/.venv"
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

# Pre-flight checks
echo "=========================================="
echo "  Pre-flight Checks"
echo "=========================================="
command -v python3.12 >/dev/null 2>&1 || fail "python3.12 not found"
echo "✓ Python 3.12 available"

# Check for Intel Battlemage GPUs
lspci | grep -q "Battlemage" || fail "No Intel Battlemage GPUs detected"
gpu_count=$(lspci | grep -c "Battlemage")
echo "✓ $gpu_count Intel Battlemage GPU(s) detected"

# Check disk space (need ~50GB for build + venv, models need more)
available=$(df --output=avail -BM "$HOME" | tail -1 | awk '{printf "%d", $1/1024}')
[ "$available" -lt 50 ] && fail "Need 50GB free for build, have ${available}GB"
echo "✓ Sufficient disk space (${available}GB available)"

echo ""
echo "=========================================="
echo "  Setting up vLLM Project Directory"
echo "=========================================="

# 1. Create directory structure
echo "[1/2] Creating directory structure..."
mkdir -p "$VLLM_DIR"
mkdir -p "$MODELS_DIR"
echo "✓ Created $VLLM_DIR"
echo "✓ Created $MODELS_DIR (shared with llama.cpp)"

# 2. Create Python 3.12 virtual environment
echo "[2/2] Creating Python 3.12 virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "  -> Virtual environment already exists at $VENV_DIR"
    read -p "  Recreate? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "  Keeping existing venv."
    else
        rm -rf "$VENV_DIR"
        python3.12 -m venv "$VENV_DIR"
        echo "✓ Virtual environment recreated"
    fi
else
    python3.12 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
fi

echo ""
echo "=========================================="
echo "  Directory Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Build vLLM:  bash 03-build-vllm-xpu.sh"
echo "  2. Download models + create configs:  bash 04-download-models.sh"
echo ""
