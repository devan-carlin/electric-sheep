#!/usr/bin/env bash
# ============================================
# vLLM Virtual Environment Comparison Tool
# ============================================
# Compares two vLLM virtual environments side-by-side
# to identify version differences, package drift,
# and GPU detection status.
#
# Usage:
#   ./compare-venvs.sh <venv1> <venv2>
#   ./compare-venvs.sh ~/electric-sheep/vllm/.venv ~/other-venv
# ============================================

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <venv1> <venv2>"
    echo ""
    echo "Example:"
    echo "  $0 ~/electric-sheep/vllm/.venv ~/optimized-venv"
    exit 1
fi

VENV1="$1"
VENV2="$2"

# Resolve to absolute paths
VENV1=$(cd "$VENV1" && pwd)
VENV2=$(cd "$VENV2" && pwd)

# Validate both are real venvs
[ -f "$VENV1/bin/activate" ] || { echo "ERROR: $VENV1 is not a valid venv"; exit 1; }
[ -f "$VENV2/bin/activate" ] || { echo "ERROR: $VENV2 is not a valid venv"; exit 1; }

# Short labels
LABEL1=$(basename $(dirname "$VENV1"))
LABEL2=$(basename $(dirname "$VENV2"))

echo "=========================================="
echo "  vLLM Environment Comparison"
echo "=========================================="
echo ""
echo "  ENV 1: $VENV1"
echo "  ENV 2: $VENV2"
echo ""

# Helper: run a command in a venv and capture output
run_in_venv() {
    local venv="$1"
    shift
    source "$venv/bin/activate"
    "$@" 2>&1
    deactivate
}

# --- Python Version ---
echo "--- Python Version ---"
PY1=$(run_in_venv "$VENV1" python3 --version)
PY2=$(run_in_venv "$VENV2" python3 --version)
echo "  $LABEL1: $PY1"
echo "  $LABEL2: $PY2"
[ "$PY1" = "$PY2" ] && echo "  ✓ Match" || echo "  ⚠ Differ"
echo ""

# --- Core Package Versions ---
echo "--- Core Package Versions ---"
printf "  %-25s %-40s %-40s\n" "Package" "$LABEL1" "$LABEL2"
printf "  %-25s %-40s %-40s\n" "-------" "------" "------"

for pkg in vllm vllm-xpu-kernels torch torchvision torchaudio triton-xpu setuptools pip wheel; do
    VER1=$(run_in_venv "$VENV1" pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "N/A")
    VER2=$(run_in_venv "$VENV2" pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "N/A")
    MATCH=""
    [ "$VER1" = "$VER2" ] && MATCH="✓" || MATCH="⚠"
    printf "  %-25s %-40s %-40s %s\n" "$pkg" "$VER1" "$VER2" "$MATCH"
done
echo ""

# --- GPU Detection ---
echo "--- GPU Detection ---"
echo "  $LABEL1:"
run_in_venv "$VENV1" python3 -c "
import torch
print(f'    PyTorch: {torch.__version__}')
print(f'    XPU Available: {torch.xpu.is_available()}')
print(f'    XPU Count: {torch.xpu.device_count()}')
for i in range(torch.xpu.device_count()):
    props = torch.xpu.get_device_properties(i)
    print(f'    GPU {i}: {torch.xpu.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)')
"
echo ""
echo "  $LABEL2:"
run_in_venv "$VENV2" python3 -c "
import torch
print(f'    PyTorch: {torch.__version__}')
print(f'    XPU Available: {torch.xpu.is_available()}')
print(f'    XPU Count: {torch.xpu.device_count()}')
for i in range(torch.xpu.device_count()):
    props = torch.xpu.get_device_properties(i)
    print(f'    GPU {i}: {torch.xpu.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)')
"
echo ""

# --- vLLM Import Test ---
echo "--- vLLM Import Test ---"
for label_venv in "$LABEL1:$VENV1" "$LABEL2:$VENV2"; do
    label="${label_venv%%:*}"
    venv="${label_venv#*:}"
    echo "  $label:"
    run_in_venv "$venv" python3 -c "
import vllm
print(f'    vLLM version: {vllm.__version__}')
print(f'    Location: {vllm.__file__}')
" 2>&1 | sed 's/^/    /'
done
echo ""

# --- Kernel Import Test ---
echo "--- vllm_xpu_kernels._C Import ---"
for label_venv in "$LABEL1:$VENV1" "$LABEL2:$VENV2"; do
    label="${label_venv%%:*}"
    venv="${label_venv#*:}"
    echo "  $label:"
    run_in_venv "$venv" python3 -c "
try:
    import vllm_xpu_kernels._C
    print('    ✓ _C module loaded successfully')
except ImportError as e:
    print(f'    ✗ _C module failed: {e}')
" 2>&1 | sed 's/^/    /'
done
echo ""

# --- Relevant Environment Variables ---
echo "--- Environment Variables (VLLM/ONEAPI/ZE/UR/TORCH/CCL/TRITON) ---"
echo "  Current session:"
env | grep -E "^(VLLM|ONEAPI|ZE_|UR_|TORCH|CCL_|TRITON)" | sort | sed 's/^/    /' || echo "    (none set)"
echo ""

# --- Installed Package Count ---
echo "--- Installed Package Summary ---"
COUNT1=$(run_in_venv "$VENV1" pip list --format=columns 2>/dev/null | tail -n +3 | wc -l)
COUNT2=$(run_in_venv "$VENV2" pip list --format=columns 2>/dev/null | tail -n +3 | wc -l)
echo "  $LABEL1: $COUNT1 packages"
echo "  $LABEL2: $COUNT2 packages"
echo ""

# --- Dependency Conflicts ---
echo "--- Dependency Conflicts ---"
echo "  $LABEL1:"
run_in_venv "$VENV1" pip check 2>&1 | head -20 | sed 's/^/    /' || echo "    (none)"
echo ""
echo "  $LABEL2:"
run_in_venv "$VENV2" pip check 2>&1 | head -20 | sed 's/^/    /' || echo "    (none)"
echo ""

echo "=========================================="
echo "  Comparison Complete"
echo "=========================================="
