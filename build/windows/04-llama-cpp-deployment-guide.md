# llama.cpp CUDA Deployment Guide — RTX 5090

End-to-end guide for building and running llama.cpp (upstream) and beellama.cpp (fork) with CUDA backend on Windows with NVIDIA RTX 5090.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Build llama.cpp](#2-build-llama-cpp)
3. [Download Models](#3-download-models)
4. [Run Inference](#4-run-inference)
5. [beellama.cpp Features](#5-beellama-cpp-features)
6. [Performance Tuning](#6-performance-tuning)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

### Required Software

| Component | Version | Install Source |
|-----------|---------|----------------|
| NVIDIA Driver | Latest Game Ready / Studio | [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx) |
| CUDA Toolkit | 12.6+ (13.1 tested) | [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) |
| CMake | 3.24+ (4.3.3 tested) | [cmake.org/download](https://cmake.org/download/) |
| Visual Studio 2022 | 17.10+ with C++ workload | [visualstudio.microsoft.com](https://visualstudio.microsoft.com/) |
| Git | Latest | [git-scm.com](https://git-scm.com/download/win) |
| Python | 3.10+ (optional) | [python.org](https://www.python.org/downloads/) |

### Verify Installation

```powershell
# Run the prerequisites check
.\01-install-prerequisites.ps1
```

This validates:
- NVIDIA GPU detection (nvidia-smi)
- CUDA toolkit (nvcc version)
- CMake version (≥ 3.24)
- MSVC build tools (Visual Studio or Build Tools)
- Git availability
- Disk space (≥ 20GB free)

### MSVC Build Tools Setup

If you don't have Visual Studio 2022:

1. Download [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Select **"Desktop development with C++"** workload
3. Ensure these components are checked:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools
   - Windows 10/11 SDK
   - CMake tools for Windows

### CUDA Architecture Reference

| GPU | Architecture | CMake Flag |
|-----|-------------|------------|
| RTX 5090 | Blackwell sm_100 | `-DCMAKE_CUDA_ARCHITECTURES=100` |
| RTX 4090 | Ada Lovelace sm_89 | `-DCMAKE_CUDA_ARCHITECTURES=89` |
| RTX 3090 | Ampere sm_86 | `-DCMAKE_CUDA_ARCHITECTURES=86` |

---

## 2. Build llama.cpp

### Quick Build (One Command)

```powershell
cd ~/electric-sheep/build/windows
.\02-build-llama-cpp.ps1
```

The script will:
1. Detect your GPU and set the correct CUDA architecture
2. Ask which builds to create (both, llama.cpp only, or beellama.cpp only)
3. Clone repos, configure CMake, and build with FlashAttention
4. Verify binaries are produced

### Manual Build Steps

#### llama.cpp (upstream)

```powershell
# Clone
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Configure (CUDA + FlashAttention + RTX 5090)
cmake -B build `
    -DGGML_CUDA=ON `
    -DGGML_NATIVE=ON `
    -DGGML_CUDA_FA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=100 `
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release --parallel
```

#### beellama.cpp (fork with KVarN, precision tail)

```powershell
# Clone
git clone https://github.com/Anbeeld/beellama.cpp.git
cd beellama.cpp

# Configure (same flags — beellama.cpp is a drop-in fork)
cmake -B build `
    -DGGML_CUDA=ON `
    -DGGML_NATIVE=ON `
    -DGGML_CUDA_FA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=100 `
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release --parallel
```

### Build Configuration Reference

| CMake Flag | Value | Purpose |
|------------|-------|---------|
| `GGML_CUDA` | `ON` | **Required** — enables CUDA backend |
| `GGML_NATIVE` | `ON` | **Recommended** — tune for host CPU (AVX2/FMA) |
| `GGML_CUDA_FA` | `ON` | **Recommended** — CUDA FlashAttention kernels |
| `GGML_CUDA_FA_ALL_QUANTS` | `ON` | Optional — build all FA quant pairs (larger build) |
| `CMAKE_CUDA_ARCHITECTURES` | `100` | Target GPU architecture (sm_100 for RTX 5090) |
| `CMAKE_BUILD_TYPE` | `Release` | Optimized build |

### Build Output

Both builds produce binaries in `build\bin\Release\`:

| Binary | Purpose |
|--------|---------|
| `llama-server.exe` | OpenAI-compatible inference server |
| `llama-cli.exe` | Interactive CLI inference |
| `llama-bench.exe` | Benchmarking tool |
| `llama-quantize.exe` | GGUF quantization utility |
| `llama-export.exe` | Model export/conversion |
| `llama-convert-hf-to-gguf.py` | HuggingFace → GGUF converter |

---

## 3. Download Models

llama.cpp uses **GGUF format**. Models can be downloaded from HuggingFace or converted from safetensors.

### Download Pre-Quantized GGUF Models

```powershell
# Create models directory
New-Item -ItemType Directory -Path "E:\llama.cpp\models" -Force

# Using huggingface-cli (requires Python + pip install huggingface_hub)
huggingface-cli download Qwen/Qwen3.6-27B-GGUF `
    --include '*q4_k_m.gguf' `
    --local-dir E:\llama.cpp\models

# Or using hf CLI tool
hf download Qwen/Qwen3.6-27B-GGUF `
    --include '*q5_k_m.gguf' `
    --local-dir E:\llama.cpp\models
```

### Quantization Format Guide

| Format | Size (27B model) | Quality | VRAM | Best For |
|--------|-----------------|---------|------|----------|
| `Q8_0` | ~27 GB | Excellent | 27+ GB | Quality-critical |
| `Q6_K` | ~20 GB | Very Good | 20+ GB | Balanced |
| **`Q5_K_M`** | **~18 GB** | **Good** | **18+ GB** | **Recommended default** |
| `Q4_K_M` | ~16 GB | Good | 16+ GB | VRAM-constrained |
| `Q4_0` | ~14 GB | Acceptable | 14+ GB | Maximum size fit |

### RTX 5090 VRAM Budget (32 GB)

| Model | Quant | Model Size | KV Cache (128K) | Total | Fits? |
|-------|-------|-----------|-----------------|-------|-------|
| Qwen 3.6 27B | Q5_K_M | ~18 GB | ~4 GB | ~22 GB | ✅ Yes |
| Qwen 3.6 27B | Q4_K_M | ~16 GB | ~3 GB | ~19 GB | ✅ Yes |
| Gemma 4 31B | Q5_K_M | ~20 GB | ~4 GB | ~24 GB | ✅ Yes |
| Gemma 4 31B | Q4_K_M | ~17 GB | ~3 GB | ~20 GB | ✅ Yes |
| Qwen 3.6 35B-A3B | Q5_K_M | ~22 GB | ~4 GB | ~26 GB | ✅ Tight |
| Qwen 3.6 35B-A3B | Q4_K_M | ~19 GB | ~3 GB | ~22 GB | ✅ Yes |

---

## 4. Run Inference

### llama-server (OpenAI-Compatible API)

```powershell
.\build\bin\Release\llama-server.exe `
    -m E:\llama.cpp\models\qwen3.6-27b-q5_k_m.gguf `
    -ngl 99 `
    -c 32768 `
    -fa on `
    --no-mmap `
    --jinja `
    -b 16384 `
    -t 16 `
    --port 8080
```

Then query with curl or any OpenAI-compatible client:
```powershell
curl http://localhost:8080/v1/chat/completions `
    -H "Content-Type: application/json" `
    -d '{
        "model": "llama",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }'
```

### llama-cli (Interactive)

```powershell
.\build\bin\Release\llama-cli.exe `
    -m E:\llama.cpp\models\qwen3.6-27b-q5_k_m.gguf `
    -ngl 99 `
    -c 32768 `
    -fa on `
    -n 512 `
    -p "Explain quantum computing in simple terms:"
```

### Command-Line Flags Reference

| Flag | Description | Recommended Value |
|------|-------------|-------------------|
| `-m` | Model file path | Path to `.gguf` file |
| `-ngl` | Layers to offload to GPU | `99` (all layers) |
| `-n` | Max tokens to generate | `256`–`8192` |
| `-c` | Context size | `32768`–`262144` |
| `-t` | CPU threads | `16` (half of 32 cores) |
| `-b` | Batch size | `16384` (prompt processing) |
| `-fa` | Flash attention | `on` |
| `--no-mmap` | Disable memory mapping | Use for large models |
| `--jinja` | Use Jinja chat template | Recommended |
| `--cache-type-k` | KV cache key dtype | `q4_0`, `kvarn4`, `f16` |
| `--cache-type-v` | KV cache value dtype | `q4_0`, `kvarn4`, `f16` |
| `--cache-ram` | KV cache size limit | `16384` (16 GiB) |
| `--kv-tail-tokens` | KVarN precision tail | `1024` (beellama only) |

---

## 5. beellama.cpp Features

beellama.cpp is a fork of llama.cpp with enhanced KV cache features:

### KVarN Quantization

Variable-precision KV cache — uses lower precision for older tokens, higher precision for recent tokens:

```powershell
.\build\bin\Release\llama-server.exe `
    -m model.gguf `
    --cache-type-k kvarn5 `
    --cache-type-v kvarn5 `
    --kv-tail-tokens 1024
```

| KVarN Level | Bytes/Element | VRAM Savings vs f16 |
|-------------|---------------|-------------------|
| `kvarn8` | 1.0 | 50% |
| `kvarn6` | 0.75 | 62.5% |
| `kvarn5` | 0.625 | 68.75% |
| `kvarn4` | 0.5 | 75% |
| `kvarn3` | 0.375 | 81.25% |

### Precision Tail (`--kv-tail-tokens`)

Keeps the last N tokens at full precision regardless of KV cache quantization:

```powershell
--kv-tail-tokens 1024  # Last 1024 tokens at full precision
```

### Independent K/V Cache Types

Set different quantization for K and V caches:

```powershell
--cache-type-k kvarn4 --cache-type-v kvarn6
```

### Comparison: llama.cpp vs beellama.cpp

| Feature | llama.cpp (upstream) | beellama.cpp (fork) |
|---------|---------------------|-------------------|
| KV-cache quantization | `q4`/`q5`/`q8` | `q2`–`q8` + `kvarn2`–`kvarn8` |
| KV-cache precision tail | ❌ | ✅ (`--kv-tail-tokens`) |
| Adaptive draft-max (DFlash) | ❌ | ✅ |
| Reasoning-loop protection | ❌ | ✅ |
| Independent K/V cache types | ❌ | ✅ |
| ggml version | 0.19.0 | 0.17.0 |
| Best for | Stable baseline, broad compatibility | Squeezing more speed/context from limited VRAM |

---

## 6. Performance Tuning

### Recommended Defaults for RTX 5090

```powershell
# Environment variables (set in PowerShell session or profile)
$env:TURBO_AUTO_ASYMMETRIC = "0"

# Server launch with optimal flags
.\build\bin\Release\llama-server.exe `
    -m model.gguf `
    -ngl 99 `
    -c 32768 `
    -fa on `
    --no-mmap `
    -b 16384 `
    -t 16 `
    --cache-type-k kvarn4 `
    --cache-type-v kvarn4 `
    --kv-tail-tokens 1024 `
    --cache-ram 16384
```

### Context Size vs VRAM Usage

| Context | KV Cache (q4_0, 27B) | KV Cache (kvarn4, 27B) | KV Cache (f16, 27B) |
|---------|---------------------|----------------------|-------------------|
| 32K | ~1.0 GiB | ~1.0 GiB | ~4.0 GiB |
| 64K | ~2.0 GiB | ~2.0 GiB | ~8.0 GiB |
| 128K | ~4.0 GiB | ~4.0 GiB | ~16.0 GiB |
| 256K | ~8.0 GiB | ~8.0 GiB | ~32.0 GiB |

### FlashAttention

Always enable for RTX 5090:
```powershell
-fa on
```

FlashAttention reduces memory usage during attention computation and improves throughput. Built with `-DGGML_CUDA_FA=ON`.

### Batch Size

Larger batch size = faster prompt processing:
```powershell
-b 16384  # Good default for 32GB VRAM
```

### Thread Count

Set to half your CPU cores for optimal CPU offload:
```powershell
-t 16  # For 32-thread CPU
```

---

## 7. Troubleshooting

### "CUDA error: out of memory"

The model + KV cache exceeds 32 GB VRAM. Options:

1. **Use smaller quantization:** `Q5_K_M` → `Q4_K_M`
2. **Use KVarN cache:** `--cache-type-k kvarn4 --cache-type-v kvarn4`
3. **Reduce context size:** `-c 16384` instead of `-c 131072`
4. **Enable precision tail:** `--kv-tail-tokens 1024` (beellama only)

### "prompt state size exceeds cache size limit"

Increase the KV cache RAM limit:
```powershell
--cache-ram 16384  # 16 GiB
# or
--cache-ram 32768  # 32 GiB
```

### "nvcc not found" during CMake configure

CUDA Toolkit not on PATH. Fix:
```powershell
# Add to current session
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"

# Or add to profile permanently
notepad $PROFILE
# Add: $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
```

### "MSB8020: The build tools for v143 are missing"

Visual Studio 2022 C++ build tools not installed. Install [Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++" workload.

### Slow first inference (JIT compilation)

CUDA kernels are compiled on first use. Subsequent inferences are fast. This is normal.

### "Failed to load model"

Check:
1. File path is correct (use full path)
2. File is a valid GGUF (not safetensors)
3. File is not corrupted (re-download if needed)

### Verify CUDA Device Selection

```powershell
.\build\bin\Release\llama-cli.exe --list-devices
```

Expected output:
```
Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32768 MiB, 32768 MiB free)
  CPU : CPU (65536 MiB, 65536 MiB free)
```

---

## Appendix: Launcher Script

The `Start-Llama-Server.ps1` launcher provides an interactive menu for:
- Selecting server build (llama.cpp vs beellama.cpp)
- Selecting model from `E:\llama.cpp\models`
- Choosing context size (32K–512K)
- Choosing KV cache type (kvarn3–kvarn8, q4_0–f16)
- Setting KV cache size limit
- Enabling vision (mmproj) support
- Configuring MTP/speculative decoding
- Gemma thinking mode toggle

```powershell
# Run the launcher
powershell -ExecutionPolicy Bypass -File "C:\path\to\scripts\Start-Llama-Server.ps1"
```
