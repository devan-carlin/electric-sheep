#!/usr/bin/env bash
set -e

# ============================================
# Prerequisites Installation Script
# ============================================
# Installs system packages, Python 3.12 via
# deadsnakes PPA, huggingface-cli, and pip.
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

# Check for sudo access
sudo -n true 2>/dev/null || fail "This script requires sudo access. Run: sudo bash install-prerequisites.sh"
echo "✓ Sudo access confirmed"

# Check OS
[ -f /etc/os-release ] || fail "/etc/os-release not found"
source /etc/os-release
echo "✓ OS: $PRETTY_NAME"

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
    python3-pip
echo "✓ System packages installed"

# -------------------------------------------
# oneAPI Toolkit
# -------------------------------------------
echo ""
echo "[3/5] Checking oneAPI Toolkit..."

if command -v source >/dev/null 2>&1 && [ -f /opt/intel/oneapi/setvars.sh ]; then
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
echo "[4/5] Installing Python 3.12..."

if command -v python3.12 >/dev/null 2>&1; then
    echo "✓ Python 3.12 already installed ($(python3.12 --version))"
else
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    echo "✓ Python 3.12 installed ($(python3.12 --version))"
fi

# -------------------------------------------
# HuggingFace CLI & pip
# -------------------------------------------
echo ""
echo "[5/5] Installing HuggingFace CLI & pip..."

if command -v hf >/dev/null 2>&1; then
    echo "✓ HuggingFace CLI already installed"
else
    python3.12 -m pip install --upgrade pip huggingface_hub[hf_xet]
    echo "✓ HuggingFace CLI installed"
fi

# Check swap space
swap_total=$(free -m | awk '/Swap:/{print $2}')
if [ "$swap_total" -lt 64000 ] 2>/dev/null; then
    echo "⚠ Low swap space: ${swap_total}MB (recommended: 64GB+ for large model loading)"
    echo ""
    echo "Would you like to create a 64GB swap file now? [y/N]"
    read -r answer
    if [[ "$answer" =~ ^[Yy] ]]; then
        echo "Creating 64GB swap file..."
        sudo fallocate -l 64G /swap_b70.img
        sudo chmod 600 /swap_b70.img
        sudo mkswap /swap_b70.img
        sudo swapon /swap_b70.img
        echo "✓ Swap file created and activated"
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

# HuggingFace CLI
if command -v hf >/dev/null 2>&1; then
    echo "✓ HuggingFace CLI installed"
else
    echo "✗ HuggingFace CLI missing"
fi

# System packages
for pkg in build-essential cmake git git-lfs clinfo libdrm-dev software-properties-common python3-pip; do
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
    oneapi_version=$(grep -oP 'VERSION_ID="\K[^"]+' /opt/intel/oneapi/compiler/latest/etc/version.txt 2>/dev/null || echo "latest")
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
echo "     bash setup-project-directory.sh"
echo ""
echo "  2. Build vLLM from source:"
echo "     bash build-vllm-xpu.sh"
echo ""
