# vLLM Deployment on Intel Arc Pro B70

## Overview

This repository contains the deployment scripts and configuration for running vLLM on **4× Intel Arc Pro B70 (Battlemage)** GPUs with **Intel quantized INT4 models**.

## Hardware

| Component | Specification |
|-----------|---------------|
| **GPU** | 4× Intel Arc Pro B70 (31.89 GiB VRAM each) |
| **CPU** | AMD Threadripper PRO 3945WX (12C/24T) |
| **RAM** | 247 GiB |
| **Storage** | Samsung 990 PRO 2TB NVMe |
| **OS** | Ubuntu 26.04 LTS (Kernel 7.0.0-29) |
| **oneAPI** | 2026.1.0 |
| **Python** | 3.12 (deadsnakes PPA) |

## Available Models

| Model | Parameters | Quantization | VRAM (TP=4) | VRAM (TP=2) |
|-------|-----------|--------------|-------------|-------------|
| **Qwen 3.6 27B** | 27B | INT4 AutoRound | ~3.5 GiB/GPU | ~7 GiB/GPU |
| **Qwen 3.6 35B-A3B** | 35B (MoE) | INT4 Mixed AutoRound | ~4.5 GiB/GPU | ~9 GiB/GPU |
| **Gemma 4 31B** | 31B | INT4 AutoRound V2 | ~4 GiB/GPU | ~8 GiB/GPU |
| **Gemma 4 26B-A4B** | 26B (MoE) | INT4 AutoRound | ~3.2 GiB/GPU | ~6.5 GiB/GPU |

## Quick Start

### 1. Install Prerequisites

```bash
sudo bash install-prerequisites.sh
```

Installs: system packages, Python 3.12, huggingface-cli, verifies GPUs and swap space.

### 2. Setup Project Directory

```bash
bash setup-project-directory.sh
```

Creates: virtual environment, environment configs, model startup scripts, downloads models.

### 3. Build vLLM from Source

```bash
bash build-vllm-xpu.sh
```

Compiles: PyTorch XPU, vllm-xpu-kernels, vLLM engine, enforces triton-xpu override.

### 4. Apply MoE Patch (if using 35B-A3B)

```bash
bash patch-vllm-moe-qzeros.sh
```

Fixes: `RuntimeError` on symmetric MoE expert layers with empty `qzeros`.

### 5. Launch a Model

```bash
# Baseline (no MTP)
bash ~/electric-sheep/vllm/start-qwen3.6-27b.sh

# MTP enabled (speculative decoding)
bash ~/electric-sheep/vllm/start-qwen3.6-27b-mtp.sh
```

## Project Structure

```
~/electric-sheep/vllm/
├── .venv/              # Python virtual environment
├── set-env-0123-gpu.sh # All 4 GPUs (TP=4, full context)
├── set-env-01-gpu.sh   # GPUs 0,1 (TP=2, reduced context)
├── set-env-23-gpu.sh   # GPUs 2,3 (TP=2, reduced context)
├── start-qwen3.6-27b.sh      # Qwen 3.6 27B INT4 (baseline)
├── start-qwen3.6-27b-mtp.sh  # Qwen 3.6 27B INT4 (MTP3 speculative)
├── start-qwen3.6-35b-a3b.sh  # Qwen 3.6 35B-A3B INT4 (MoE)
├── start-gemma4-31b.sh        # Gemma 4 31B INT4
├── start-gemma4-26b-a4b.sh   # Gemma 4 26B-A4B INT4 (MoE)
└── models/
    ├── Intel-Qwen3.6-27B-int4-AutoRound/
    ├── Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound/
    ├── Intel-gemma-4-31B-it-int4-AutoRound-V2/
    └── Intel-gemma-4-26B-A4B-it-int4-AutoRound/
```

## Deployment Modes

### Single Model (Full Context, 4 GPUs)

- **Tensor Parallelism:** 4
- **Context Window:** 232,144 tokens
- **Best for:** Maximum context, single-model throughput

### Dual Model (Reduced Context, 2 GPUs each)

- **Tensor Parallelism:** 2 per model
- **Context Window:** 32,768 tokens
- **Ports:** 8030 (GPUs 0,1) + 8031 (GPUs 2,3)
- **Best for:** Running two models simultaneously

## Performance Benchmarks

| Configuration | GPUs | Speed | Notes |
|--------------|------|-------|-------|
| Baseline (TP4) | 4 | ~50-55 tok/s | Standard vLLM serve |
| MTP3 (TP4) | 4 | ~65-75 tok/s | Speculative decoding enabled |
| TP2 Record (Steve Seguin) | 2 | 93 tok/s | PIECEWISE graph + INT8 LM head |

## MTP (Multi-Token Prediction) Configuration

The `start-qwen3.6-27b-mtp.sh` script enables speculative decoding optimizations:

```bash
export QWEN36_27B_ENABLE_MTP=1
export NUM_SPECULATIVE_TOKENS=3
export VLLM_XPU_GDN_PROMOTE_ACCEPTED_SPEC_STATE=1
export VLLM_XPU_GDN_NONSPEC_POSTPROCESS_ACCEPTED_STATE=0
export VLLM_XPU_LM_HEAD_INT8=1
export VLLM_XPU_LM_HEAD_INT8_SCALE_DTYPE=bf16
export COMPILATION_CONFIG='{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":8}'
```

## Known Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `No module named 'vllm_xpu_kernels._C'` | `.so` binaries not in `site-packages` | Manual `cp` fallback in build script |
| `ImportError: cannot import name 'intel' from 'triton._C'` | Standard `triton` overwrote `triton-xpu` | `pip install triton-xpu --force-reinstall --no-deps` |
| `RuntimeError: copy_() shape mismatch` on 35B-A3B | Empty `qzeros` on symmetric MoE layers | Run `patch-vllm-moe-qzeros.sh` |
| Pre-flight VRAM check fails at `0.90+` | Level-Zero driver reserves ~1.5 GiB/GPU | Use `--gpu-memory-utilization 0.80` |
| `resource_tracker: leaked shared_memory` | Python IPC cleanup after process exit | Safe to ignore |

## Acknowledgments

**Special thanks to [Steve Seguin](https://github.com/steveseguin) for his incredible work on the [b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab).** His research and benchmarking on Intel Arc B70 GPUs has been invaluable in understanding how to optimize vLLM for Battlemage hardware. The MTP speculative decoding configuration in this repo is directly inspired by his findings.

## License

This project is for internal deployment use. Model weights are subject to their respective licenses.
