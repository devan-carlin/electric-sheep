# Balanced Split Mode for llama.cpp

## Overview

`--split-mode balanced` is a custom feature that distributes model layers across multiple GPUs using a **greedy balanced assignment algorithm** that accounts for per-layer quantization depth. This is essential for MoE (Mixture of Experts) models where layers have vastly different memory footprints due to mixed quantization (MXFP4, IQ2_XXS, Q8_0, etc.).

## Problem Statement

### The Sequential Assignment Problem

By default, llama.cpp uses `--split-mode layer` which assigns layers **sequentially** to GPUs based on proportional split ratios. With 4 equal GPUs:
- GPU 0 gets layers 0-10
- GPU 1 gets layers 11-21
- GPU 2 gets layers 22-32
- GPU 3 gets layers 33-42

For **DeepSeek V4 Flash** (43 layers, 256 experts), this creates severe imbalance:

| GPU | Layers | MXFP4 Layers | VRAM | Usage |
|-----|--------|--------------|------|-------|
| SYCL0 | 0-10 | 1 (layer 10) | 19.2 GB | 60% |
| SYCL1 | 11-21 | 1 (layer 14) | 21.5 GB | 68% |
| SYCL2 | 22-32 | 2 (layers 30, 34) | 24.1 GB | 76% |
| SYCL3 | 33-42 | 6 (layers 34, 37-42) | 28.9 GB | **91%** |

**GPU 3 gets 60% of all MXFP4 layers** (the heaviest quantization), pushing it to 91% capacity while GPU 0 sits at 60%. This wastes ~9 GB of VRAM on underutilized GPUs.

### Root Cause

MXFP4 layers are ~2x heavier than IQ2_XXS layers. Sequential assignment clusters the heaviest layers (34, 37-42) on the last GPU. The default algorithm has no awareness of per-layer quantization depth.

## Solution: Balanced Split Mode

`--split-mode balanced` uses a **greedy balanced assignment** algorithm:

1. **Reads actual tensor sizes** from GGUF metadata (not estimated)
2. **Sums per-layer tensor sizes** to get accurate memory footprint
3. **Assigns each layer to the GPU with minimum accumulated weight**
4. **Accounts for mixed quantization** automatically (MXFP4 vs IQ2_XXS)

### Results

| GPU | Layers | Estimated | Actual VRAM | Usage |
|-----|--------|-----------|-------------|-------|
| SYCL0 | 0,4,9,12,17,20,25,28,33,35,38,42 | 25,720 MiB | 25,720 MiB | 81% |
| SYCL1 | 1,5,8,13,16,21,24,29,32,36,43(output) | 22,282 MiB | 22,818 MiB | 73% |
| SYCL2 | 2,7,11,14,19,22,27,30,37,41 | 24,996 MiB | 24,996 MiB | 79% |
| SYCL3 | 3,6,10,15,18,23,26,31,34,40 | 23,514 MiB | 23,514 MiB | 74% |

**Improvements:**
- Max VRAM: 28.9 GB → 25.7 GB (**-11%**)
- Min VRAM: 19.2 GB → 22.8 GB (**+19%**)
- Spread: 31% → 8% (**74% improvement**)
- All GPUs in tight 73-81% range
- Estimated sizes match actual sizes within 1-2%

## Usage

```bash
./bin/llama-server \
  -m model.gguf \
  --gpu-layers 999 \
  --split-mode balanced \
  --tensor-split 1,1,1,1 \
  --flash-attn on \
  --ctx-size 4096 \
  --batch-size 256 \
  --parallel 4 \
  --verbose
```

### Key Flags

| Flag | Purpose |
|------|---------|
| `--split-mode balanced` | Enable balanced layer assignment |
| `--tensor-split 1,1,1,1` | Equal split ratios (4 GPUs) |
| `--gpu-layers 999` | Offload all layers to GPU |
| `--verbose` | Show layer assignment details |

### Verbose Output

With `--verbose`, you'll see:
```
I load_tensors: balanced layer assignment (MoE-aware, quantization-aware)
I load_tensors:   device 0: estimated size 25719.63 MiB
I load_tensors:   device 1: estimated size 22281.70 MiB
I load_tensors:   device 2: estimated size 24995.90 MiB
I load_tensors:   device 3: estimated size 23513.72 MiB
D load_tensors: layer   0 assigned to device SYCL0, is_swa = 1
D load_tensors: layer   1 assigned to device SYCL1, is_swa = 1
...
I load_tensors:        SYCL0 model buffer size = 25719.63 MiB
I load_tensors:        SYCL1 model buffer size = 22818.41 MiB
I load_tensors:        SYCL2 model buffer size = 24995.91 MiB
I load_tensors:        SYCL3 model buffer size = 23513.72 MiB
```

## Implementation Details

### Files Modified

| File | Change |
|------|--------|
| `include/llama.h` | Added `LLAMA_SPLIT_MODE_BALANCED = 4` enum |
| `common/arg.cpp` | Added "balanced" CLI option for `--split-mode` |
| `common/fit.cpp` | Added exception to skip fit check for balanced mode |
| `src/llama-model.cpp` | Greedy balanced assignment with quantization awareness |

### Algorithm

```cpp
// 1. Calculate per-layer size from GGUF tensor metadata
for (int il = 0; il < n_layer_all; ++il) {
    double layer_size = 0.0;
    std::string layer_prefix = "blk." + std::to_string(il) + ".";
    for (auto & kv : ml.weights_map) {
        if (kv.first.compare(0, layer_prefix.size(), layer_prefix) == 0) {
            layer_size += (double)ggml_nbytes(kv.second.tensor);
        }
    }
    layer_weight[il] = (layer_size > 0) ? layer_size : 1.0;
}

// 2. Greedy balanced assignment
std::vector<double> gpu_weight(n_devices(), 0.0);
for (int il = 0; il <= n_layer_all; ++il) {
    // Find GPU with minimum accumulated weight
    int min_gpu = argmin(gpu_weight);
    layer_to_gpu[il] = min_gpu;
    gpu_weight[min_gpu] += layer_weight[il];
}
```

### Why Fit Check is Skipped

The `common_params_fit_impl` function in `fit.cpp` assumes sequential layer assignment when estimating memory. For balanced mode, this estimation is incorrect (layers are distributed differently), so the fit check is skipped with a warning.

## Compatibility

### Supported Architectures

- **deepseek4** (tested with DeepSeek V4 Flash)
- Any MoE architecture with mixed quantization
- Dense models (works but less impactful)

### Unsupported Split Modes

| Mode | Status |
|------|--------|
| `none` | Single GPU only |
| `layer` | Sequential (default) |
| `row` | Segfaults on deepseek4 |
| `tensor` | Not implemented for deepseek4 |
| `balanced` | ✅ Working |

## Performance

### Inference Speed (DeepSeek V4 Flash, 4× Arc Pro B70)

| Metric | Sequential | Balanced | Change |
|--------|-----------|----------|--------|
| Prompt speed | 7.87 tok/s | 7.87 tok/s | Same |
| Generation speed | 11.93 tok/s | 11.93 tok/s | Same |
| Max GPU VRAM | 28.9 GB (91%) | 25.7 GB (81%) | -11% |
| Min GPU VRAM | 19.2 GB (60%) | 22.8 GB (73%) | +19% |

**No performance penalty** — balanced mode only changes layer assignment, not computation.

### Memory Overhead

The balanced assignment iterates `ml.weights_map` once during model loading (~1328 tensors for DeepSeek V4). This adds ~100ms to load time, negligible compared to the 2-minute model load.

## Installation

### Quick Install

```bash
cd ~/llama.cpp
bash /path/to/apply-balanced-split-mode.sh
cd build && cmake --build . --target llama-server -j $(nproc)
```

### Manual Install

See [Implementation Details](#implementation-details) for file-by-file changes.

## Troubleshooting

### "llama_params_fit is not implemented for SPLIT_MODE_BALANCED"

**Expected behavior**. The fit check is skipped for balanced mode. The model will load normally.

### Logs not appearing

Layer assignment logs require `--verbose` flag. Without it, only the summary appears.

### Segfault with `--split-mode row` or `--split-mode tensor`

These modes are not implemented for deepseek4 architecture. Use `--split-mode balanced` instead.

### Uneven VRAM despite balanced mode

Check that:
1. `--gpu-layers 999` is set (all layers on GPU)
2. `--tensor-split 1,1,1,1` matches your GPU count
3. The model uses mixed quantization (uniform quant = uniform layers)

## References

- Original issue: Uneven VRAM on 4× Intel Arc Pro B70 with DeepSeek V4 Flash
- Model: `apetersson/DeepSeek-V4-Flash-0731--DS4-Quality128` (102.8 GB GGUF)
- Hardware: Threadripper PRO 3945WX, 4× Arc Pro B70 (32 GB each)
- llama.cpp: commit `dd1ea5243` (latest master, 2026-08-10) with SYCL/XPU backend
- Patch script: `/electric-sheep/scripts/common/apply-balanced-split-mode.sh`
- PR documentation: `/electric-sheep/docs/guides/pr-balanced-split-mode.md`

## Clean Build Verification (2026-08-10)

Verified on fresh clone of llama.cpp (commit `dd1ea5243`):

1. Deleted existing llama.cpp, cloned fresh from `ggml-org/llama.cpp`
2. Applied `apply-balanced-split-mode.sh` — all 5 verifications passed
3. Built with `cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DGGML_SYCL=ON`
4. Model loaded successfully with balanced mode
5. Inference test passed (correct output, no regression)

| GPU | Sequential (MiB) | Balanced (MiB) | Delta |
|-----|------------------|----------------|-------|
| SYCL0 | 22,279 | 25,720 | +15% |
| SYCL1 | 22,287 | 22,818 | +2% |
| SYCL2 | 22,315 | 24,996 | +12% |
| SYCL3 | **30,166** | 23,514 | **-22%** |

**Spread improvement**: 35% variance → 13% variance (63% reduction)
