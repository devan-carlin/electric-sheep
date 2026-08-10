# DeepSeek V4-Flash  — INT4 AutoRound Conversion Guide

**Source:** `apetersson/DeepSeek-V4-Flash-0731--FP8`  
**Output:** `~/electric-sheep/models/DeepSeek-V4-Flash-0731--FP8-int4-AutoRound/`  
**Backend:** vLLM XPU (4× Intel Arc Pro B70)  
**Date:** 2026-08-09

---

## Status

| Metric | Value |
|--------|-------|
| **State** | ❌ FAILED |
| **Elapsed** | ~13h 49m |
| **Cause** | Hard system crash (AI server) |
| **Restartable** | No (no checkpoint support) |

**Notes:**
- System crashed during quantization phase (after ~13h 49m elapsed)
- AutoRound does not support checkpoint/resume; must restart from scratch
- See system health check logs for crash root cause

---

## Overview

This document covers the process of converting the DeepSeek V4-Flash  FP8 model to INT4 AutoRound format for vLLM deployment on Intel XPU hardware.

### Why Convert?

| Format | Size | Use Case |
|--------|------|----------|
| **FP8 (source)** | ~98 GB | llama.cpp GGUF, high quality |
| **INT4 AutoRound (output)** | ~25 GB | vLLM XPU, 4× smaller, fast inference |

The INT4 AutoRound format enables:
- **4× smaller** model footprint (~25 GB vs ~98 GB)
- **vLLM compatibility** with `--tensor-parallel-size 4`
- **FP8 KV cache** support for extended context windows
- **High accuracy** via AutoRound's sign-gradient descent optimization

---

## Prerequisites

### Hardware

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU** | 4× Intel Arc Pro B70 (31.89 GiB each) | 128 GiB total VRAM |
| **RAM** | 247 GiB | Model loading + quantization overhead |
| **Disk** | ~200 GB free | Source (98 GB) + output (25 GB) + temp |

### Software

| Component | Version | Install |
|-----------|---------|---------|
| **Python** | 3.12 | `sudo apt install python3.12` |
| **vLLM venv** | `~/electric-sheep/vllm/.venv/` | Built via `03-build-vllm-xpu.sh` |
| **PyTorch XPU** | 2.x | Installed in vLLM venv |
| **auto-round** | Latest | Auto-installed by script |
| **optimum** | Latest | Auto-installed by script |
| **datasets** | Latest | Auto-installed by script |
| **hf CLI** | Latest | `pipx install huggingface_hub` |

---

## Conversion Process

### Quick Start

```bash
# Run the conversion script (max quality settings)
cd ~/electric-sheep/scripts/common
./04-convert-int4-autoround.sh apetersson/DeepSeek-V4-Flash-0731--FP8
```

### Script Pre-flight Checks

The script validates:
1. ✅ vLLM virtual environment exists
2. ✅ Python version (3.11 or 3.12 recommended)
3. ✅ PyTorch with XPU support
4. ✅ GPU count and total VRAM
5. ✅ Available RAM (>64 GB recommended)
6. ✅ hf CLI available
7. ✅ HF_TOKEN set (for gated models)
8. ✅ Disk space (>100 GB recommended)
9. ✅ `datasets` library installed
10. ✅ `auto-round` library installed
11. ✅ `optimum` library installed

### Quantization Settings

| Parameter | Value | Impact |
|-----------|-------|--------|
| **bits** | 4 | INT4 weight quantization |
| **group_size** | 128 | Standard group size for INT4 |
| **symmetric** | false | Asymmetric (better accuracy) |
| **iters** | 1000 | Maximum accuracy (4-5× slower) |
| **recipe** | auto-round-best | Highest accuracy recipe |
| **low_gpu_mem** | true | Saves ~20 GB VRAM, no accuracy loss |
| **torch.compile** | true | Accelerates quantization |
| **calibration samples** | 128 | From `timdettmers/openassistant-guanaco` |
| **sequence length** | 2048 | Calibration context window |

### Quality vs. Speed Trade-offs

| Setting | iters | Recipe | Time | Quality |
|---------|-------|--------|------|---------|
| **Maximum** | 1000 | auto-round-best | 8-12 hours | Best possible |
| **Standard** | 200 | auto-round | 2-4 hours | High quality |
| **Fast** | 50 | auto-round-light | 1-2 hours | Good |
| **Baseline** | 0 | auto-round-rtn | <1 hour | RTN (round-to-nearest) |

### Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| **Pre-flight checks** | ~30 seconds | Validates environment |
| **Model download** | 10-30 minutes | ~98 GB FP8 source |
| **Model loading** | 2-5 minutes | Deserializes weights to GPU/RAM |
| **Quantization** | 8-12 hours | Layer-by-layer with iters=1000 |
| **Saving output** | 5-10 minutes | Writes INT4 safetensors |
| **Total** | ~9-13 hours | Overnight run recommended |

### Monitoring Progress

```bash
# Check if process is alive
ps aux | grep python3 | grep -v grep

# Check CPU usage (should fluctuate 50-200%)
top -p $(pgrep -f "04-convert-int4-autoround")

# Check GPU utilization (one GPU at 100% with low_gpu_mem=true)
gputop

# Check output directory (files appear after quantization completes)
ls -lh ~/electric-sheep/models/DeepSeek-V4-Flash-0731--FP8-int4-AutoRound/
```

**Note:** With `low_gpu_mem_usage=True`, only one GPU will show 100% utilization at a time. The other GPUs hold model weights passively. Output files only appear after all layers are quantized.

---

## Output

### Location

```
~/electric-sheep/models/DeepSeek-V4-Flash-0731--FP8-int4-AutoRound/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model-00001-of-00004.safetensors  (~6 GB each)
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
└── model.safetensors.index.json
```

### Expected Size

| Component | Size |
|-----------|------|
| **Total output** | ~25 GB |
| **Per shard** | ~6 GB |
| **Compression ratio** | ~4× (98 GB → 25 GB) |

---

## Deployment with vLLM

### Start Server

```bash
# Activate vLLM environment
source ~/electric-sheep/vllm/.venv/bin/activate

# Start vLLM server with INT4 model
vllm serve ~/electric-sheep/models/DeepSeek-V4-Flash-0731--FP8-int4-AutoRound \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching
```

### Test Inference

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4-Flash-0731--FP8-int4-AutoRound",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 100
  }'
```

### Expected Performance

| Metric | Estimate | Notes |
|--------|----------|-------|
| **VRAM usage** | ~10-12 GB/GPU | INT4 model + FP8 KV cache |
| **Context window** | 32K-64K tokens | Depends on KV cache budget |
| **Throughput** | TBD | Benchmark after deployment |

---

## Troubleshooting

### Out of Memory (OOM)

If quantization fails with OOM:
```bash
# Force CPU-only loading (slower but safer)
# Edit the script, change device_map="auto" to device_map="cpu"
```

### Stuck on Loading

If the model loading phase takes >10 minutes:
```bash
# Check if process is still active
top -p $(pgrep -f "04-convert-int4-autoround")

# If CPU is 0% for >5 minutes, it may be stuck
kill $(pgrep -f "04-convert-int4-autoround")
```

### Slow Quantization

If quantization is too slow:
```bash
# Retry with fewer iterations (still high quality)
./04-convert-int4-autoround.sh apetersson/DeepSeek-V4-Flash-0731--FP8 --iters 200
```

### HF Token Required

For gated models (Llama, etc.):
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
./04-convert-int4-autoround.sh <model-repo>
```

---

## Comparison: INT4 AutoRound vs. GGUF

| Feature | INT4 AutoRound (vLLM) | UD-IQ3_XXS (llama.cpp) |
|---------|----------------------|------------------------|
| **Format** | safetensors | GGUF |
| **Size** | ~25 GB | ~98 GB |
| **Backend** | vLLM XPU | llama.cpp SYCL |
| **Tensor parallelism** | Native (TP=4) | Layer split (-sm layer) |
| **KV cache** | FP8 (efficient) | FP16 (default) |
| **Context** | 32K-64K | 64K-96K |
| **Speed** | TBD | ~13 t/s (no DSpark), ~16 t/s (DSpark) |
| **Quality** | INT4 AutoRound (high) | UD-IQ3_XXS (~3-bit, aggressive) |

---

## References

- [AutoRound GitHub](https://github.com/intel/auto-round)
- [AutoRound User Guide](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)
- [Source Model](https://huggingface.co/apetersson/DeepSeek-V4-Flash-0731--FP8)
- [Conversion Script](./04-convert-int4-autoround.sh)
