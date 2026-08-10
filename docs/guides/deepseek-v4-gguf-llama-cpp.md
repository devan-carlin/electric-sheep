# DeepSeek V4 GGUF Testing — llama.cpp on 4× Intel Arc Pro B70

**Date:** 2026-08-10  
**Model:** `apetersson/DeepSeek-V4-Flash-0731--DS4-Quality128` (GGUF)  
**Hardware:** Threadripper PRO 3945WX, 247 GiB RAM, 4× Intel Arc Pro B70 (31.89 GiB each)  
**llama.cpp:** b10331 (SYCL/XPU backend)  
**Goal:** Serve a 304B MoE model (/) on Intel GPUs via llama.cpp

---

## Summary

| Aspect | Result |
|--------|--------|
| Model loads on llama.cpp SYCL | ✅ Yes |
| Server starts and serves requests | ✅ Yes (~12.5 tok/s) |
| Memory balanced across 4 GPUs | ❌ No — uneven layer assignment |
| Tensor split mode (`--split-mode tensor`) | ❌ Not implemented for deepseek4 |
| Row split mode (`--split-mode row`) | ❌ Segfaults |
| Layer split mode (default) | ✅ Works, but lopsided |
| High context lengths | ⚠️ Limited by GPU OOM on heaviest GPU |

---

## Model Details

### GGUF File

| File | Size | SHA256 |
|------|------|--------|
| `DeepSeek-V4-Flash-0731--DS4-Quality128.gguf` | 102.8 GB | `2cfc36b7...` |
| `DeepSeek-V4-Flash-0731--DS4-Quality128-llamacpp-DSpark-support.gguf` | 7.3 GB | `0582de4d...` |

### Quantization Scheme (Mixed Precision)

- **MXFP4** on 10 critical layers (10, 14, 30, 34, 37–42)
- **Q8_0** on attention/shared-expert/output paths
- **IQ2_XXS / Q2_K** on remaining routed experts
- **Reported:** IQ2_XXS — 2.0625 bpw (284B params)

### Why This Model

- / variant of DeepSeek-V4-Flash
- 304B parameter MoE architecture
- Mixed precision GGUF designed to fit in 128 GiB total VRAM (~18 GiB overhead)
- DSpark companion file for speculative decoding (requires special llama.cpp build `fffbcbdb`)

---

## What Worked

### 1. Basic Serving (Default Layer Split)

```bash
source ~/electric-sheep/llama/set-env.sh
cd ~/llama.cpp/build
./bin/llama-server \
  -m ~/electric-sheep/models/DeepSeek-V4-Flash-0731--DS4-Quality128/DeepSeek-V4-Flash-0731--DS4-Quality128.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-layers 999 \
  --flash-attn on \
  --ctx-size 4096 \
  --batch-size 256 \
  --ubatch-size 256 \
  --parallel 4
```

**Result:** Model loaded in ~2 minutes, server started, generated text at ~12.5 tok/s.

**Problem:** One GPU consumed most VRAM (MoE expert layers assigned unevenly), leaving no headroom for large context lengths.

### 2. Quantized KV Cache + Unified Buffer

```bash
source ~/electric-sheep/llama/set-env.sh
cd ~/llama.cpp/build
./bin/llama-server \
  -m ~/electric-sheep/models/DeepSeek-V4-Flash-0731--DS4-Quality128/DeepSeek-V4-Flash-0731--DS4-Quality128.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-layers 999 \
  --flash-attn on \
  --ctx-size 8192 \
  --batch-size 256 \
  --ubatch-size 256 \
  --parallel 2 \
  --kv-unified \
  --cache-type-k q4_0 \
  --cache-type-v q4_0
```

**Result:** Model loaded successfully, 8192 context length, q4_0 quantized KV cache (~50% less memory for context), unified KV buffer shared across slots.

**Benefit:** More headroom for context by reducing KV cache memory pressure.

---

## What Didn't Work

### 1. Tensor Split Mode

```bash
--split-mode tensor --tensor-split 1,1,1,1
```

**Error:** `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'deepseek4'`

**Root cause:** llama.cpp's tensor parallelism code path has no implementation for the DeepSeek V4 (deepseek4) architecture. This is the ideal mode for MoE models — it splits both weights AND KV across all GPUs in parallel.

### 2. Row Split Mode

```bash
--split-mode row --tensor-split 1,1,1,1
```

**Error:** Segmentation fault (core dumped) during model loading.

**Root cause:** Row-based weight splitting (splitting weight matrices by rows across GPUs) crashes on deepseek4 architecture. Likely an incomplete implementation or unhandled tensor shape in the MoE expert layers.

### 3. vLLM with GGUF Plugin

```bash
vllm serve <model.gguf> --device xpu
```

**Error:** `RuntimeError: Unknown gguf model_type: deepseek_v4` in `vllm_gguf_plugin/weights_adapter/default.py:150`

**Root cause:** The `vllm-gguf-plugin` (v0.0.5) out-of-tree GGUF loader doesn't recognize the DeepSeek V4 architecture. Its `build_name_map()` function can't map DeepSeekV4 GGUF tensor names to vLLM's internal format.

**Note:** vLLM *does* have native DeepSeekV4 XPU support (`vllm/models/deepseek_v4/xpu/`), but only for non-GGUF model formats (safetensors, etc.).

### 4. AutoRound Quantization on Intel XPU

**Error:** AutoRound 0.14.2 is CUDA-only. Falls back to CPU-only execution.

**Root cause:** AutoRound's optimization loop uses CUDA kernels. No SYCL/XPU backend. Runs on 2 CPU cores despite `OMP_NUM_THREADS=24` being set (sequential Python loop is the bottleneck, not BLAS).

---

## Key Findings

### MoE Memory Distribution Problem

The default `--split-mode layer` (pipelined) assigns whole layers to GPUs in round-robin fashion. For MoE models:

- Expert FFN layers are disproportionately large (each expert is a full feed-forward network)
- Round-robin assignment doesn't account for layer size variance
- One GPU ends up with the biggest experts and runs near OOM
- No flag exists to target KV cache to specific GPUs in layer mode

### Available Mitigations

| Option | Effect | Tradeoff |
|--------|--------|----------|
| `--kv-unified` | Single shared KV buffer across slots | Less duplication, but still on GPUs |
| `--cache-type-k q4_0 --cache-type-v q4_0` | Quantized KV cache (~50% less memory) | Small quality impact on long contexts |
| `--no-kv-offload` | KV cache in system RAM instead of GPU | Slower KV access, but frees GPU VRAM |
| `--parallel 2` (fewer slots) | Less total KV cache needed | Fewer concurrent requests |
| `--ctx-size N` (smaller) | Less KV cache per slot | Shorter context window |

### Recommended Configuration for High Context

```bash
source ~/electric-sheep/llama/set-env.sh
cd ~/llama.cpp/build
./bin/llama-server \
  -m ~/electric-sheep/models/DeepSeek-V4-Flash-0731--DS4-Quality128/DeepSeek-V4-Flash-0731--DS4-Quality128.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-layers 999 \
  --flash-attn on \
  --ctx-size 32768 \
  --batch-size 256 \
  --ubatch-size 256 \
  --parallel 2 \
  --kv-unified \
  --no-kv-offload \
  --cache-type-k q4_0 \
  --cache-type-v q4_0
```

This keeps weights on GPU (still uneven) but moves KV cache to the 247 GiB system RAM, where there's plenty of headroom for large contexts.

---

## Environment Setup

### set-env.sh (Key Settings)

```bash
# oneAPI device selector — all 4 GPUs
export ONEAPI_DEVICE_SELECTOR="level_zero:0,1,2,3"

# SYCL optimizations
export GGML_SYCL_ENABLE_FLASH_ATTN=1
export GGML_SYCL_ENABLE_OPT=1
export GGML_SYCL_ENABLE_DNN=1

# Relaxed VRAM allocation limits (critical for large models)
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
```

### llama.cpp Build

- Source: `~/llama.cpp/` (ggml-org/llama.cpp, b10331)
- Build: `~/llama.cpp/build/`
- Binary: `~/llama.cpp/build/bin/llama-server`
- Backend: SYCL/XPU with Level Zero

---

## Performance Numbers

| Metric | Value |
|--------|-------|
| Model load time | ~2 minutes (102.8 GB file) |
| Token generation speed | ~12.5 tok/s (single request) |
| Peak RAM during load | ~101 GB (39% of 247 GB) |
| VRAM per GPU (loaded) | Uneven — one GPU near OOM |

---

## Open Issues / Future Work

1. **Tensor split mode for deepseek4** — needs upstream llama.cpp implementation. Track: `LLAMA_SPLIT_MODE_TENSOR` support for MoE architectures.
2. **Row split mode segfault** — likely a bug in the deepseek4 row-split code path. Worth filing an issue with llama.cpp.
3. **vllm-gguf-plugin DeepSeekV4 support** — needs `build_name_map()` update for deepseek_v4 tensor naming.
4. **Speculative decoding with DSpark** — requires llama.cpp build at commit `fffbcbdb` with DSpark support enabled.
5. **Better GPU memory monitoring** — `intel-smi` not installed on this system. Consider installing `intel-gpu-tools` for per-GPU VRAM visibility.

---

## Related Docs

- [llama.cpp Deployment Guide](./llama-deployment.md) — Build and general setup
- [vLLM Deployment Guide](./vllm-deployment.md) — vLLM on Intel Arc (non-GGUF models)
- [DeepSeek INT4 Conversion](./deepseek-int4-conversion.md) — AutoRound quantization (CUDA-only)
- [Architecture Overview](../architecture.md) — Hardware and project structure
