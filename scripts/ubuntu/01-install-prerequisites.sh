#!/usr/bin/env bash
set -e

# ============================================
# Prerequisites Installation Script
# ============================================
# Installs system packages, Python 3.12 via
# deadsnakes PPA, hf CLI, and pip.
# Requires sudo for system package installs.
# ============================================

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
echo "  Prerequisites — Pre-flight"
echo "=========================================="

# Note: sudo is used inline for apt commands (will prompt for password when needed)
echo "✓ Sudo will be prompted when needed"

# Check OS
[ -f /etc/os-release ] || fail "/etc/os-release not found"
source /etc/os-release
echo "✓ OS: $PRETTY_NAME"

# Check python3 (needed for hf CLI version check)
command -v python3 >/dev/null 2>&1 || fail "python3 not found (needed for hf CLI)"
echo "✓ python3 available"

# Check pip (needed for hf_xet install)
command -v pip >/dev/null 2>&1 || fail "pip not found (needed for hf_xet install)"
echo "✓ pip available"

# -------------------------------------------
# System packages
# -------------------------------------------
echo ""
echo "[1/5] Updating package cache..."
sudo apt-get update
echo "✓ Package cache updated"

echo ""
echo "[2/5] Installing system packages..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    clinfo \
    libdrm-dev \
    software-properties-common \
    intel-opencl-icd \
    libze-intel-gpu1 \
    libze1 \
    libze-dev \
    python3 \
    python3-pip
echo "✓ System packages installed"

# -------------------------------------------
# oneAPI Toolkit
# -------------------------------------------
echo ""
echo "[3/6] Checking oneAPI Toolkit..."

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    echo "✓ oneAPI Toolkit found at /opt/intel/oneapi/"
else
    echo "⚠ oneAPI Toolkit not found at /opt/intel/oneapi/"
    echo "  Install with: sudo apt-get install -y intel-oneapi-toolkit"
    echo "  Or download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkit-overview.html"
fi

# -------------------------------------------
# Python 3.12 (deadsnakes PPA)
# -------------------------------------------
echo ""
echo "[4/6] Installing Python 3.12..."

if command -v python3.12 >/dev/null 2>&1; then
    echo "✓ Python 3.12 already installed ($(python3.12 --version))"
else
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    echo "✓ Python 3.12 installed ($(python3.12 --version))"
fi

# -------------------------------------------
# HuggingFace CLI (global install via pipx)
# -------------------------------------------
echo ""
echo "[5/6] Installing HuggingFace CLI..."

# Ensure pipx is available (PEP 668 on Ubuntu 24.04+ blocks system pip)
if ! command -v pipx >/dev/null 2>&1; then
    echo "  Installing pipx..."
    sudo apt-get install -y pipx
fi

# Ensure pipx binaries are on PATH for this session and future logins
pipx ensurepath >/dev/null 2>&1 || true
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"

if command -v hf >/dev/null 2>&1; then
    # Check version — 1.26+ recommended for hf_xet transport
    current_ver=$(python3 -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null || echo "0")
    major=$(echo "$current_ver" | cut -d. -f1)
    minor=$(echo "$current_ver" | cut -d. -f2)
    if [ "$major" -ge 1 ] 2>/dev/null && [ "$minor" -ge 26 ] 2>/dev/null; then
        echo "✓ HuggingFace CLI installed (v$current_ver)"
    else
        echo "⚠ huggingface_hub v$current_ver is outdated (1.26+ recommended for hf_xet)"
        echo "  Updating via pipx..."
        pipx install --force huggingface-hub[hf_xet]
        echo "✓ HuggingFace CLI updated"
    fi
else
    pipx install huggingface-hub[hf_xet]
    echo "✓ HuggingFace CLI installed"
fi

# Check swap space
swap_total=$(free -m | awk '/Swap:/{print $2}')
echo ""
echo "[6/6] Swap space check..."
if [ "$swap_total" -ge 64000 ] 2>/dev/null; then
    echo "✓ Swap space: ${swap_total}MB (adequate, no action needed)"
else
    echo "⚠ Low swap space: ${swap_total}MB (recommended: 64GB+ for large model loading)"
    echo ""
    echo "Would you like to create a 64GB swap file now? [y/N]"
    read -r answer
    if [[ "$answer" =~ ^[Yy] ]]; then
        echo "Creating 64GB swap file..."
        # Use dd as fallocate may fail on Btrfs
        if sudo fallocate -l 64G /swap_b70.img 2>/dev/null; then
            echo "  ✓ fallocate succeeded"
        else
            echo "  ⚠ fallocate failed (Btrfs?), falling back to dd..."
            sudo dd if=/dev/zero of=/swap_b70.img bs=1M count=65536 status=progress
        fi
        sudo chmod 600 /swap_b70.img
        sudo mkswap /swap_b70.img
        sudo swapon /swap_b70.img
        # Persist across reboots
        echo '/swap_b70.img none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
        echo "✓ Swap file created, activated, and added to /etc/fstab"
        echo "  New swap: $(free -m | awk '/Swap:/{print $2}')MB"
    else
        echo "⚠ Skipping swap creation. Large model loading may fail with low swap."
    fi
fi

# -------------------------------------------
# Verification
# -------------------------------------------
echo ""
echo "=========================================="
echo "  Prerequisites — Verification"
echo "=========================================="

# Python 3.12
python3.12 --version 2>&1 | awk '{print "✓", $0, "installed"}'

# Python 3.12 venv (required for setup-project-directory.sh)
python3.12 -m venv --help >/dev/null 2>&1 && echo "✓ python3.12-venv available" || echo "✗ python3.12-venv missing"

# python3 (needed for hf CLI version check)
python3 --version 2>&1 | awk '{print "✓", $0, "available"}'

# pip (needed for hf_xet install)
pip --version 2>&1 | awk '{print "✓", $0}'

# HuggingFace CLI
if command -v hf >/dev/null 2>&1; then
    echo "✓ HuggingFace CLI installed"
else
    echo "✗ HuggingFace CLI missing"
fi

# System packages
for pkg in build-essential cmake git git-lfs clinfo libdrm-dev software-properties-common python3-pip python3.12-venv python3.12-dev; do
    if dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        echo "✓ $pkg installed"
    else
        echo "✗ $pkg missing"
    fi
done

# Intel GPU drivers
for pkg in intel-opencl-icd libze-intel-gpu1 libze1 libze-dev; do
    if dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        echo "✓ $pkg installed"
    else
        echo "✗ $pkg missing"
    fi
done

# oneAPI Toolkit
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    oneapi_version=$(grep 'VERSION_ID' /opt/intel/oneapi/compiler/latest/etc/version.txt 2>/dev/null | sed -n 's/.*VERSION_ID="\([^"]*\)".*/\1/p' || echo "latest")
    [ -z "$oneapi_version" ] && oneapi_version="latest"
    echo "✓ oneAPI Toolkit found ($oneapi_version)"
else
    echo "✗ oneAPI Toolkit not found"
fi

# GPU detection
if lspci | grep -q "Battlemage"; then
    gpu_count=$(lspci | grep -c "Battlemage")
    echo "✓ $gpu_count Intel Battlemage GPU(s) detected"
else
    echo "✗ No Battlemage GPUs detected"
fi

# Swap space
swap_total=$(free -m | awk '/Swap:/{print $2}')
if [ "$swap_total" -ge 64000 ] 2>/dev/null; then
    echo "✓ Swap space: ${swap_total}MB (adequate)"
elif [ "$swap_total" -gt 0 ] 2>/dev/null; then
    echo "⚠ Swap space: ${swap_total}MB (recommended: 64GB+)"
else
    echo "✗ No swap space configured"
fi

echo ""
echo "=========================================="
echo "  Prerequisites Complete!"
echo "=========================================="
echo ""
echo "The prerequisites are installed. Next steps:"
echo ""
echo "  1. Setup project directory (creates ~/electric-sheep/vllm/):"
echo "     bash 02-setup-project-directory.sh"
echo ""
echo "  2. Build vLLM from source:"
echo "     bash 03-build-vllm-xpu.sh"
echo ""
