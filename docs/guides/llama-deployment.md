# llama.cpp SYCL Deployment Guide — 4× Intel Arc Pro B70

End-to-end guide for building and running llama.cpp with SYCL backend on Ubuntu 26.04 with 4× Intel Arc Pro B70 GPUs.

**Project Root:** `~/electric-sheep/llama/`  
**Shared Models:** `~/electric-sheep/models/`

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Project Structure](#2-project-structure)
3. [Build llama.cpp](#3-build-llama-cpp)
4. [Download Models](#4-download-models)
5. [Run Inference](#5-run-inference)
6. [Multi-GPU Strategies](#6-multi-gpu-strategies)
7. [Performance Tuning](#7-performance-tuning)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Quick Start

```bash
# Setup (clone source, create structure)
bash ~/electric-sheep/build/ubuntu/04-setup-llama.sh

# Build (SYCL + FP16)
bash ~/electric-sheep/build/ubuntu/05-build-llama-cpp.sh

# Download a GGUF model
hf download <repo> --local-dir ~/electric-sheep/models/

# Run inference
source ~/electric-sheep/llama/set-env.sh
~/electric-sheep/llama/llama.cpp/build/bin/llama-cli \
    -m ~/electric-sheep/models/your-model.gguf \
    -ngl 99 -n 256 -sm layer
```

---

## 2. Project Structure

```
~/electric-sheep/llama/
├── llama.cpp/              # Cloned source (ggml-org/llama.cpp)
│   └── build/              # Build output (binaries)
├── set-env.sh              # Environment config (oneAPI + SYCL)
├── deepseek/               # DeepSeek-specific scripts + docs
│   ├── start-deepseek-v4-flash.sh
│   └── run-stats.md
└── 03-llama-cpp-deployment-guide.md

~/electric-sheep/models/    # Shared model repository (vLLM + llama.cpp)
├── unsloth-DeepSeek-V4-Flash-0731-GGUF/
└── ...
```

### Environment Config

`set-env.sh` loads oneAPI and configures SYCL with performance tuning:

```bash
# All 4 GPUs (default)
source ~/electric-sheep/llama/set-env.sh

# Specific GPUs
source ~/electric-sheep/llama/set-env.sh 0,1    # GPUs 0 and 1
source ~/electric-sheep/llama/set-env.sh 0      # GPU 0 only
```

---

## 3. Build llama.cpp

### Quick Build (One Command)

```bash
bash ~/electric-sheep/build/ubuntu/05-build-llama-cpp.sh
```

This script performs:
- Pre-flight checks (OS, GPU, oneAPI, build tools)
- Auto-installs missing dependencies (git, cmake, ninja-build)
- CMake configuration (SYCL + FP16 + Battlemage arch)
- Parallel build using all CPU cores
- Rebuild skip if source unchanged (commit hash comparison)

### Manual Build Steps

```bash
# 1. Load oneAPI environment
source ~/electric-sheep/llama/set-env.sh

# 2. Configure with CMake (SYCL + FP16)
cd ~/electric-sheep/llama/llama.cpp
cmake -B build \
    -G Ninja \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_F16=ON \
    -DGGML_SYCL_DEVICE_ARCH=bmg \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release

# 3. Build (uses all 24 cores)
cmake --build build -j $(nproc)
```

### Build Configuration Reference

| CMake Flag | Value | Purpose |
|------------|-------|---------|
| `GGML_SYCL` | `ON` | **Required** — enables SYCL backend |
| `GGML_SYCL_F16` | `ON` | **Recommended** — FP16 math for better performance |
| `GGML_SYCL_DEVICE_ARCH` | `bmg` | Battlemage architecture optimization |
| `GGML_SYCL_DNN` | `ON` (default) | Use oneDNN for GEMM operations |
| `CMAKE_C_COMPILER` | `icx` | Intel oneAPI C compiler |
| `CMAKE_CXX_COMPILER` | `icpx` | Intel oneAPI C++/SYCL compiler |

---

## 4. Download Models

llama.cpp uses **GGUF format** (not HuggingFace safetensors). Models are stored in `~/electric-sheep/models/` (shared with vLLM).

### Download Pre-Quantized GGUF Models

```bash
# DeepSeek V4-Flash (UD-IQ3_XXS — ~98GB, 4 shards)
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
    --include 'UD-IQ3_XXS/*' \
    --local-dir ~/electric-sheep/models/

# Qwen 3.6 27B (Q4_K_M — ~16GB, good balance)
hf download Qwen/Qwen3.6-27B-GGUF \
    --include '*q4_k_m.gguf' \
    --local-dir ~/electric-sheep/models/

# Gemma 4 27B (Q4_K_M — ~16GB)
hf download google/gemma-3-27b-it-GGUF \
    --include '*q4_k_m.gguf' \
    --local-dir ~/electric-sheep/models/
```

### Quantization Format Guide

| Format | Size (27B model) | Quality | VRAM | Best For |
|--------|-----------------|---------|------|----------|
| `Q8_0` | ~27 GB | Excellent | 27+ GB | Quality-critical |
| `Q6_K` | ~20 GB | Very Good | 20+ GB | Balanced |
| **`Q4_K_M`** | **~16 GB** | **Good** | **16+ GB** | **Recommended default** |
| `Q4_0` | ~14 GB | Acceptable | 14+ GB | VRAM-constrained |
| `Q3_K_M` | ~12 GB | Usable | 12+ GB | Maximum size fit |

---

## 5. Run Inference

### Single GPU (GPU 0 only)

```bash
source ~/electric-sheep/llama/set-env.sh 0

~/electric-sheep/llama/llama.cpp/build/bin/llama-cli \
    -m ~/electric-sheep/models/your-model.gguf \
    -ngl 99 \
    -n 512 \
    -c 8192 \
    --load-mode mmap \
    -p "Explain quantum computing in simple terms:"
```

### All 4 GPUs (Layer Split)

```bash
source ~/electric-sheep/llama/set-env.sh

~/electric-sheep/llama/llama.cpp/build/bin/llama-cli \
    -m ~/electric-sheep/models/your-model.gguf \
    -ngl 99 \
    -n 512 \
    -c 8192 \
    -sm layer \
    --load-mode mmap \
    -p "Explain quantum computing in simple terms:"
```

### llama-server (OpenAI-Compatible API)

```bash
source ~/electric-sheep/llama/set-env.sh

~/electric-sheep/llama/llama.cpp/build/bin/llama-server \
    -m ~/electric-sheep/models/your-model.gguf \
    -ngl 99 \
    -c 8192 \
    -sm layer \
    --port 8080 \
    --host 0.0.0.0
```

Then query with curl:
```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }'
```

### Command-Line Flags Reference

| Flag | Description | Recommended Value |
|------|-------------|-------------------|
| `-m` | Model file path | Path to `.gguf` file |
| `-ngl` | Layers to offload to GPU | `99` (all layers) |
| `-n` | Max tokens to generate | `256`–`2048` |
| `-c` | Context size | `4096`–`98304` |
| `-t` | CPU threads | `12` (half of 24 cores) |
| `-sm` | Split mode | `none`, `layer`, `tensor` |
| `--load-mode` | Loading mode | `mmap` (recommended) |
| `--flash-attn` | Flash attention | `auto` |
| `--batch-size` | Batch size | `4096` (for prompt processing) |

---

## 6. Multi-GPU Strategies

### Layer Split (Default, Most Stable)

Distributes model layers across GPUs sequentially. Each layer runs on one GPU.

```bash
source ~/electric-sheep/llama/set-env.sh
~/electric-sheep/llama/llama.cpp/build/bin/llama-cli -m model.gguf -ngl 99 -sm layer
```

**Pros:** Stable, works with all models, no communication overhead within layers.  
**Cons:** Load may be uneven if layers have different sizes.

### Tensor Parallelism (Experimental)

Shards each layer across all GPUs. Requires flash attention (auto-enabled).

```bash
source ~/electric-sheep/llama/set-env.sh 0,1
~/electric-sheep/llama/llama.cpp/build/bin/llama-cli -m model.gguf -ngl 99 -sm tensor
```

**Pros:** Better load balancing, optimized for 2 GPUs.  
**Cons:** Currently optimized for 2 GPUs; 4 GPUs fall back to generic all-reduce.

### GPU Pair Deployment (Two Models Simultaneously)

Run two independent models, each on a pair of GPUs:

```bash
# Terminal 1: Model A on GPUs 0,1
source ~/electric-sheep/llama/set-env.sh 0,1
~/electric-sheep/llama/llama.cpp/build/bin/llama-server -m model-a.gguf -ngl 99 -sm layer --port 8080

# Terminal 2: Model B on GPUs 2,3
source ~/electric-sheep/llama/set-env.sh 2,3
~/electric-sheep/llama/llama.cpp/build/bin/llama-server -m model-b.gguf -ngl 99 -sm layer --port 8081
```

### VRAM Budget per Strategy

| Strategy | GPUs Used | Max Model Size (per GPU: 31.89 GB) | Practical Limit |
|----------|-----------|-----------------------------------|-----------------|
| Single GPU | 1 | ~28 GB | Q4_K_M 27B model |
| Layer split (4 GPUs) | 4 | ~112 GB total | Q4_K_M 70B+ model |
| Tensor (2 GPUs) | 2 | ~56 GB total | Q4_K_M 40B model |
| Dual model (2+2) | 4 | ~28 GB each | Two Q4_K_M 27B models |

---

## 7. Performance Tuning

### Environment Variables

| Variable | Value | Effect |
|----------|-------|--------|
| `ZES_ENABLE_SYSMAN` | `1` | **Required** for multi-GPU memory reporting |
| `GGML_SYCL_ENABLE_FLASH_ATTN` | `1` | Flash attention — reduces memory usage |
| `GGML_SYCL_ENABLE_OPT` | `1` | Intel GPU optimizations |
| `GGML_SYCL_ENABLE_DNN` | `1` | Use oneDNN for GEMM |
| `GGML_SYCL_ENABLE_MKL_FA` | `1` | oneMKL flash attention for prompt processing |
| `GGML_SYCL_FA_ONEDNN_MAX_KV` | `24576` | Cap KV length (prevents watchdog resets) |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | `1` | Allow allocations > 4 GB |

### Known SYCL Limitations

- **q8_0 KV cache unsupported** — SYCL backend lacks `concat` kernel for q8_0. Use f16 (default) instead.
- **Flash attention** — works with `--flash-attn auto` but MKL FA path may not activate on all models.

---

## 8. Troubleshooting

### "No SYCL GPU devices detected"

```bash
# Check if oneAPI is loaded
source /opt/intel/oneapi/setvars.sh
sycl-ls | grep level_zero

# If empty, check user groups
groups $USER | grep -E "render|video"

# Add to groups if missing
sudo usermod -aG render,video $USER
# Then logout and login
```

### "can't allocate X Bytes of memory on device"

The model is too large for available VRAM. Options:

1. **Use more GPUs:** `source set-env.sh` (all 4)
2. **Use smaller quantization:** `Q4_K_M` → `Q3_K_M`
3. **Reduce context size:** `-c 4096` instead of `-c 32768`
4. **Enable host fallback:** `export GGML_SYCL_HOST_MEM_FALLBACK=1`

### "libsycl.so: cannot open shared object file"

```bash
source ~/electric-sheep/llama/set-env.sh
# Or add to ~/.bashrc for persistence:
echo 'source /opt/intel/oneapi/setvars.sh' >> ~/.bashrc
```

### Garbled output or crash with multiple GPUs

Try the host-forward memory copy method:
```bash
export GGML_SYCL_DEV2DEV_MEMCPY=2
```

### DEVICE_LOST during long-context inference

The GPU driver watchdog is resetting. Reduce oneDNN FA KV cap:
```bash
export GGML_SYCL_FA_ONEDNN_MAX_KV=24576
```

### Slow startup (JIT compilation)

First run is slow due to SYCL JIT compilation. Subsequent runs are faster. **Do not** use `SYCL_CACHE_PERSISTENT=1` — it causes crashes when code changes.

---

## Appendix: Comparison with vLLM

| Feature | llama.cpp | vLLM |
|---------|-----------|------|
| **Language** | C/C++ | Python |
| **Format** | GGUF only | HuggingFace (safetensors) |
| **Multi-GPU** | Layer split, tensor (2 GPU) | Tensor parallel (4 GPU) |
| **Throughput** | Lower (single-batch) | Higher (continuous batching) |
| **Latency** | Good for single request | Better for concurrent requests |
| **API** | llama-server (OpenAI-compatible) | OpenAI-compatible HTTP API |
| **Use Case** | Local inference, edge, low overhead | Production serving, high throughput |
| **Memory Efficiency** | Excellent (GGUF quantization) | Good (kv cache management) |

**Recommendation:** Use vLLM for production serving with concurrent requests. Use llama.cpp for local inference, testing, or when you need GGUF format compatibility (Ollama, LM Studio, etc.).
