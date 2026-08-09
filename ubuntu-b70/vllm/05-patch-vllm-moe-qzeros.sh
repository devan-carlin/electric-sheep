#!/usr/bin/env bash
set -e

# ============================================
# vLLM MoE qzeros Patch Script
# ============================================
# Fixes the RuntimeError when loading Intel's 35B-A3B-int4-mixed-AutoRound
# MoE model, which uses asymmetric quantization on attention layers and
# symmetric quantization (empty qzeros) on expert layers.
#
# Problem: vLLM's Intel Extension layer (inc_wna16_linear.py) assumed all
# layers had matching qzeros shapes, causing torch.copy_() to throw a
# RuntimeError (0 vs 2 dimensions).
#
# Solution: Add guarded existence, element-count (numel() > 0), and shape
# checks prior to copying qzeros. If the source tensor is empty or
# mismatched, safely fall back to setting ark_linear.qzeros = None.
# ============================================

# Detect if run via sudo — preserve original user's HOME
if [ -n "$SUDO_USER" ]; then
    export HOME=$(eval echo ~$SUDO_USER)
    echo "Note: Running via sudo, using HOME=$HOME for user $SUDO_USER"
fi

VENV_DIR="$HOME/electric-sheep/vllm/.venv"
TARGET_FILE="$VENV_DIR/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py"

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

echo "=========================================="
echo "  Patching vLLM for 35B-A3B MoE Support"
echo "=========================================="

# Pre-flight checks
[ -d "$VENV_DIR" ] || fail "Virtual environment not found at $VENV_DIR. Run setup-project-directory.sh first."
echo "✓ Virtual environment found"

command -v python3.12 >/dev/null 2>&1 || fail "python3.12 not found"
echo "✓ Python 3.12 available"

lspci | grep -q "Battlemage" || fail "No Intel Battlemage GPUs detected"
echo "✓ Intel Battlemage GPU(s) detected"

if [ ! -f "$TARGET_FILE" ]; then
    fail "Target file not found: $TARGET_FILE\n       Have you compiled vLLM in the virtual environment yet?"
fi

echo "✓ Target file found: $TARGET_FILE"

# Activate venv
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"

echo ""
echo "Patching: $TARGET_FILE"

python3 - "$TARGET_FILE" << 'PYTHON_SCRIPT'
import sys

target_file = sys.argv[1]

with open(target_file, 'r') as f:
    content = f.read()

old_block = '''    ark_linear.qweight.copy_(qweight_src)
    if hasattr(layer, "qzeros") and layer.qzeros is not None:
        ark_linear.qzeros.copy_(layer.qzeros.detach())
    else:
        ark_linear.qzeros = None
    ark_linear.scales.copy_(layer.scales.detach())'''

new_block = '''    ark_linear.qweight.copy_(qweight_src)

    # Safely handle qzeros to prevent empty/mismatched tensor copy crashes on symmetric MoE layers
    if (
        hasattr(layer, "qzeros")
        and layer.qzeros is not None
        and layer.qzeros.numel() > 0
        and hasattr(ark_linear, "qzeros")
        and ark_linear.qzeros is not None
        and ark_linear.qzeros.numel() > 0
        and layer.qzeros.shape == ark_linear.qzeros.shape
    ):
        ark_linear.qzeros.copy_(layer.qzeros.detach())
    else:
        ark_linear.qzeros = None

    ark_linear.scales.copy_(layer.scales.detach())'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(target_file, 'w') as f:
        f.write(content)
    print("SUCCESS: vLLM inc_wna16_linear.py patched successfully.")
elif new_block in content:
    print("SKIPPED: Patch already applied. No changes needed.")
else:
    print("WARNING: Neither the original nor patched code was found.")
    print("         The vLLM version may have changed. Manual review recommended.")
    sys.exit(1)
PYTHON_SCRIPT

# Verify patch was applied
if grep -q "Safely handle qzeros" "$TARGET_FILE"; then
    echo "✓ Patch verified successfully"
else
    fail "Patch verification failed — target file may not have been modified"
fi

echo ""
echo "=========================================="
echo "  Patch Complete!"
echo "=========================================="
echo ""
echo "The 35B-A3B MoE model can now be loaded without qzeros crashes."
echo ""
