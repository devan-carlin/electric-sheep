#!/usr/bin/env bash
# apply-balanced-split-mode.sh
# Applies the --split-mode balanced patch to llama.cpp
# This enables quantization-aware balanced layer assignment across multiple GPUs
# for MoE models with mixed quantization (MXFP4, IQ2_XXS, Q8_0, etc.)
#
# Usage:
#   cd /path/to/llama.cpp
#   bash /path/to/apply-balanced-split-mode.sh
#   cd build && cmake --build . --target llama-server -j $(nproc)
#
# Tested on: llama.cpp b10331
# Author: electric-sheep project
# Date: 2026-08-10

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Verify we're in a llama.cpp directory
if [[ ! -f "include/llama.h" ]]; then
    error "Not in a llama.cpp directory (include/llama.h not found)"
    exit 1
fi

if [[ ! -f "src/llama-model.cpp" ]]; then
    error "src/llama-model.cpp not found"
    exit 1
fi

if [[ ! -f "common/arg.cpp" ]]; then
    error "common/arg.cpp not found"
    exit 1
fi

if [[ ! -f "common/fit.cpp" ]]; then
    error "common/fit.cpp not found"
    exit 1
fi

info "llama.cpp source directory detected"

# Check if already patched
if grep -q "LLAMA_SPLIT_MODE_BALANCED" include/llama.h 2>/dev/null; then
    warn "Patch already applied (LLAMA_SPLIT_MODE_BALANCED found in llama.h)"
    warn "Skipping..."
    exit 0
fi

# Create backups
info "Creating backups..."
BACKUP_SUFFIX=".balanced-patch-$(date +%Y%m%d%H%M%S)"

cp include/llama.h "include/llama.h${BACKUP_SUFFIX}"
cp common/arg.cpp "common/arg.cpp${BACKUP_SUFFIX}"
cp common/fit.cpp "common/fit.cpp${BACKUP_SUFFIX}"
cp src/llama-model.cpp "src/llama-model.cpp${BACKUP_SUFFIX}"

info "Backups created with suffix: ${BACKUP_SUFFIX}"

# ============================================================
# Patch 1: include/llama.h - Add LLAMA_SPLIT_MODE_BALANCED enum
# ============================================================
info "Patching include/llama.h..."

if grep -q "LLAMA_SPLIT_MODE_TENSOR" include/llama.h; then
    # Add LLAMA_SPLIT_MODE_BALANCED after LLAMA_SPLIT_MODE_TENSOR (handle variable whitespace)
    # Use sed with regex that matches any amount of whitespace between TENSOR and =
    sed -i '/LLAMA_SPLIT_MODE_TENSOR[[:space:]]*=[[:space:]]*3,/{
        s/LLAMA_SPLIT_MODE_TENSOR[[:space:]]*=[[:space:]]*3,/LLAMA_SPLIT_MODE_TENSOR = 3,\n        LLAMA_SPLIT_MODE_BALANCED = 4, \/\/ split layers across GPUs, balanced by estimated layer size/
    }' include/llama.h
    info "  Added LLAMA_SPLIT_MODE_BALANCED = 4 enum"
else
    error "Could not find LLAMA_SPLIT_MODE_TENSOR in llama.h"
    exit 1
fi

# ============================================================
# Patch 2: common/arg.cpp - Add "balanced" CLI option
# ============================================================
info "Patching common/arg.cpp..."

# Find the tensor help text line and add balanced option after it
if grep -q '"- tensor: split weights and KV across GPUs' common/arg.cpp; then
    # Fix: the tensor line may or may not have a trailing \n, and we need to add \n to tensor line
    # then add balanced line without trailing comma (it becomes the last help line)
    # Use Python for reliable multi-line replacement
    python3 << 'PYEOF'
import re
with open("common/arg.cpp", "r") as f:
    content = f.read()
# Find the tensor help line and fix it to have \n then add balanced line
old = '"- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)",'
new = '"- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)\\n"\n        "- balanced: split layers across GPUs, balanced by estimated layer size (MoE-aware)",'
if old in content and '"- balanced:' not in content:
    content = content.replace(old, new)
    with open("common/arg.cpp", "w") as f:
        f.write(content)
    print("  [OK] Added balanced help text")
else:
    print("  [SKIP] balanced help text already present or pattern not found")
PYEOF
    info "  Added balanced help text"
fi

# Add the balanced value handler after the tensor handler
if grep -q 'params.split_mode = LLAMA_SPLIT_MODE_TENSOR;' common/arg.cpp; then
    sed -i '/params.split_mode = LLAMA_SPLIT_MODE_TENSOR;/{n;s/}/} else if (value == "balanced") {\n                params.split_mode = LLAMA_SPLIT_MODE_BALANCED;\n            }/}' common/arg.cpp
    info "  Added balanced value handler"
else
    error "Could not find LLAMA_SPLIT_MODE_TENSOR handler in arg.cpp"
    exit 1
fi

# ============================================================
# Patch 3: common/fit.cpp - Skip fit check for balanced mode
# ============================================================
info "Patching common/fit.cpp..."

# Find the SPLIT_MODE_TENSOR check and add balanced mode check after it
if grep -q 'SPLIT_MODE_TENSOR.*abort' common/fit.cpp; then
    # Add balanced mode exception after tensor mode exception
    sed -i '/SPLIT_MODE_TENSOR.*abort/a\    }\n    if (mparams->split_mode == LLAMA_SPLIT_MODE_BALANCED) {\n        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_BALANCED, skipping fit check");' common/fit.cpp
    info "  Added balanced mode fit check exception"
else
    error "Could not find SPLIT_MODE_TENSOR check in fit.cpp"
    exit 1
fi

# ============================================================
# Patch 4: src/llama-model.cpp - Balanced layer assignment
# ============================================================
info "Patching src/llama-model.cpp..."

# Use Python for reliable multi-line code insertion (sed/awk are fragile with complex C++ code)
python3 << 'PYEOF'
import re
import sys

with open("src/llama-model.cpp", "r") as f:
    content = f.read()

# Check for required patterns
if "const int act_gpu_layers = devices.empty()" not in content:
    print("ERROR: Could not find act_gpu_layers line in llama-model.cpp", file=sys.stderr)
    sys.exit(1)

if "std::upper_bound(splits.begin()" not in content:
    print("ERROR: Could not find upper_bound in llama-model.cpp", file=sys.stderr)
    sys.exit(1)

# Patch 1: Insert balanced assignment code after act_gpu_layers line
balanced_code = '''
    // balanced assignment: estimate per-layer size and assign greedily to balance memory
    // this is useful for MoE models where expert layers dominate and sequential assignment is lopsided
    std::vector<int> layer_to_gpu(n_layer_all + 1, -1); // +1 for output layer
    if (split_mode == LLAMA_SPLIT_MODE_BALANCED && n_devices() > 1) {
        // estimate relative layer size by summing tensor sizes from GGUF metadata
        // this accounts for per-layer quantization depth (MXFP4 vs IQ2_XXS, etc.)
        std::vector<double> layer_weight(n_layer_all + 1, 1.0);
        for (int il = 0; il < n_layer_all; ++il) {
            double layer_size = 0.0;
            // iterate over all tensors in the model loader and sum sizes for this layer
            std::string layer_prefix = "blk." + std::to_string(il) + ".";
            for (auto & kv : ml.weights_map) {
                // check if tensor name starts with the layer prefix
                if (kv.first.compare(0, layer_prefix.size(), layer_prefix) == 0) {
                    // use tensor's nbytes for size estimation
                    layer_size += (double)ggml_nbytes(kv.second.tensor);
                }
            }
            layer_weight[il] = (layer_size > 0) ? layer_size : 1.0;
        }
        // output layer (embedding + lm head): moderate size
        layer_weight[n_layer_all] = 1.0;

        // greedy balanced assignment: assign each layer to the GPU with least total weight
        std::vector<double> gpu_weight(n_devices(), 0.0);
        for (int il = 0; il <= n_layer_all; ++il) {
            if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
                layer_to_gpu[il] = -1; // CPU
                continue;
            }
            // find GPU with minimum accumulated weight
            int min_gpu = 0;
            for (size_t g = 1; g < n_devices(); ++g) {
                if (gpu_weight[g] < gpu_weight[min_gpu]) {
                    min_gpu = (int)g;
                }
            }
            layer_to_gpu[il] = min_gpu;
            gpu_weight[min_gpu] += layer_weight[il];
        }

        LLAMA_LOG_INFO("%s: balanced layer assignment (MoE-aware, quantization-aware)\\n", __func__);
        for (size_t g = 0; g < n_devices(); ++g) {
            LLAMA_LOG_INFO("%s:   device %zu: estimated size %.2f MiB\\n",
                __func__, g, gpu_weight[g] / (1024.0 * 1024.0));
        }
    }
'''

# Find and insert after act_gpu_layers line
act_gpu_pattern = r'(    const int act_gpu_layers = devices\.empty\(\) \? 0 : std::min\(n_gpu_layers, n_layer_all \+ 1\);)'
match = re.search(act_gpu_pattern, content)
if match:
    insert_pos = match.end()
    content = content[:insert_pos] + balanced_code + content[insert_pos:]
    print("  [OK] Inserted balanced assignment code after act_gpu_layers")
else:
    print("ERROR: Could not match act_gpu_layers pattern", file=sys.stderr)
    sys.exit(1)

# Patch 2: Replace sequential layer_gpu assignment with balanced-aware version
old_assignment = r'const int layer_gpu = std::upper_bound\(splits\.begin\(\), splits\.begin\(\) \+ n_devices\(\), float\(il - i_gpu_start\)/act_gpu_layers\) - splits\.begin\(\);'
new_assignment = '''int layer_gpu;
        if (split_mode == LLAMA_SPLIT_MODE_BALANCED && layer_to_gpu[il] >= 0) {
            layer_gpu = layer_to_gpu[il];
        } else {
            layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        }'''

match = re.search(old_assignment, content)
if match:
    content = content[:match.start()] + new_assignment + content[match.end():]
    print("  [OK] Replaced sequential layer_gpu assignment with balanced-aware version")
else:
    print("ERROR: Could not match layer_gpu assignment pattern", file=sys.stderr)
    sys.exit(1)

with open("src/llama-model.cpp", "w") as f:
    f.write(content)

print("  [OK] llama-model.cpp patched successfully")
PYEOF

if [ $? -ne 0 ]; then
    error "Python patch for llama-model.cpp failed"
    exit 1
fi

# ============================================================
# Verify patches
# ============================================================
info "Verifying patches..."

ERRORS=0

if grep -q "LLAMA_SPLIT_MODE_BALANCED = 4" include/llama.h; then
    info "  ✓ llama.h: LLAMA_SPLIT_MODE_BALANCED enum added"
else
    error "  ✗ llama.h: LLAMA_SPLIT_MODE_BALANCED enum NOT found"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'LLAMA_SPLIT_MODE_BALANCED' common/arg.cpp; then
    info "  ✓ arg.cpp: balanced CLI handler added"
else
    error "  ✗ arg.cpp: balanced CLI handler NOT found"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'SPLIT_MODE_BALANCED.*skipping fit check' common/fit.cpp; then
    info "  ✓ fit.cpp: balanced mode fit check exception added"
else
    error "  ✗ fit.cpp: balanced mode fit check exception NOT found"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'balanced layer assignment.*quantization-aware' src/llama-model.cpp; then
    info "  ✓ llama-model.cpp: balanced assignment code added"
else
    error "  ✗ llama-model.cpp: balanced assignment code NOT found"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'layer_to_gpu\[il\]' src/llama-model.cpp; then
    info "  ✓ llama-model.cpp: layer_to_gpu lookup in get_layer_buft_list"
else
    error "  ✗ llama-model.cpp: layer_to_gpu lookup NOT found"
    ERRORS=$((ERRORS + 1))
fi

if [[ $ERRORS -gt 0 ]]; then
    error "$ERRORS patch(es) failed to apply correctly"
    error "Backups are available with suffix: ${BACKUP_SUFFIX}"
    error "Restore with: for f in include/llama.h common/arg.cpp common/fit.cpp src/llama-model.cpp; do cp \"\$f${BACKUP_SUFFIX}\" \"\$f\"; done"
    exit 1
fi

info ""
info "============================================"
info "All patches applied successfully!"
info "============================================"
info ""
info "Next steps:"
info "  cd build"
info "  cmake --build . --target llama-server -j \$(nproc)"
info ""
info "Usage:"
info "  ./bin/llama-server -m model.gguf --split-mode balanced --verbose"
info ""
info "Backups saved with suffix: ${BACKUP_SUFFIX}"
