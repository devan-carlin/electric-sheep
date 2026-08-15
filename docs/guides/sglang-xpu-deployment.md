# SGLang XPU Deployment Guide

## Status: Experimental / Future Reference

**Created**: 2026-08-10  
**Last Reviewed**: 2026-08-10  
**Hardware**: 4× Intel Arc Pro B70 (32 GB each, Level Zero)  
**SGLang XPU Docs**: https://docs.sglang.io/docs/hardware-platforms/xpu

---

## Overview

SGLang is a high-performance LLM serving framework with XPU (Intel GPU) support. It offers features like RadixAttention (tree-based KV cache reuse), structured generation, and agentic workflow composition.

**Current assessment**: Not production-ready for our workload. Worth revisiting when quantized model support and speculative decoding land.

---

## Installation

### Prerequisites

- Intel oneAPI 2026.1.1+ (with `icpx` compiler)
- Conda environment
- PyTorch XPU build

### Install from Source

```bash
# Create and activate conda environment
conda create -n sgl-xpu python=3.12 -y
conda activate sgl-xpu

# Install PyTorch XPU (not CUDA)
pip3 install torch==2.12.0+xpu torchao==0.17.0+xpu torchvision==0.27.0+xpu torchaudio==2.11.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu

# Install dependencies (avoid CUDA-enabled triton)
pip3 install xgrammar --no-deps
pip3 install apache-tvm-ffi

# Clone and build SGLang
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# Use XPU-specific config
cd python
cp pyproject_xpu.toml pyproject.toml
pip install --upgrade pip setuptools
pip install -v . --extra-index-url https://download.pytorch.org/whl/xpu
```

### Docker Alternative

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build image
docker build -t sglang-xpu:latest -f xpu.Dockerfile .

# Run container
docker run \
    -it \
    --privileged \
    --ipc=host \
    --network=host \
    --user root \
    --group-add $(getent group video | cut -d: -f3) \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v /dev/shm:/dev/shm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    -e "HF_TOKEN=<your-token>" \
    sglang-xpu:latest /bin/bash
```

---

## Launching the Server

### Basic Launch

```bash
sglang serve \
    --model-path <MODEL_ID_OR_PATH> \
    --trust-remote-code \
    --disable-overlap-schedule \
    --device xpu \
    --host 0.0.0.0 \
    --tp 4 \
    --attention-backend intel_xpu \
    --page-size 64
```

### Multi-GPU (4× Arc B70)

```bash
# All GPUs
sglang serve \
    --model-path <MODEL> \
    --trust-remote-code \
    --device xpu \
    --tp 4 \
    --attention-backend intel_xpu \
    --host 0.0.0.0 \
    --port 30000
```

### With XPU Graph Capture (Experimental)

```bash
# Decode graph only
sglang launch_server \
    --model-path <MODEL> \
    --device xpu \
    --cuda-graph-backend-decode full

# Decode + Prefill (tc_piecewise)
sglang launch_server \
    --model-path <MODEL> \
    --device xpu \
    --cuda-graph-config '{"decode":{"backend":"full"},"prefill":{"backend":"tc_piecewise","tc_compiler":"eager"}}'

# With torch.compile for decode
sglang launch_server \
    --model-path <MODEL> \
    --device xpu \
    --enable-torch-compile
```

### Prefill-Decode Disaggregation (Experimental)

```bash
# Prefill server (GPU 0)
ZE_AFFINITY_MASK=0 UCX_POSIX_USE_PROC_LINK=n python -m sglang.launch_server \
    --model-path <MODEL> --trust-remote-code --device xpu \
    --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port 12335 --host 0.0.0.0 --port 30000

# Decode server (GPU 1-3)
ZE_AFFINITY_MASK=1,2,3 UCX_POSIX_USE_PROC_LINK=n python -m sglang.launch_server \
    --model-path <MODEL> --trust-remote-code --device xpu \
    --disaggregation-mode decode --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port 12335 --host 0.0.0.0 --port 30001

# Router
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode  http://127.0.0.1:30001 \
    --host 0.0.0.0 --port 8000
```

---

## Benchmarking

```bash
python -m sglang.bench_serving \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --num-prompts 1 \
    --request-rate inf \
    --random-range-ratio 1.0
```

---

## Current Limitations (2026-08-10)

| Limitation | Impact |
|-----------|--------|
| **No GGUF/quantized model support** | Cannot run quantized models. BF16 DeepSeek V4-Flash would need ~568GB VRAM (we have 128GB). |
| **No speculative decoding** | DSpark not implemented. Would lose 20-30% generation speed improvement. |
| **Only small models tested** | Verified on Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-1.5B. No MoE models tested. |
| **Source install only** | No pip wheel. Requires conda + custom build. |
| **XPU graph capture experimental** | Multiple backends (full, tc_piecewise, breakable) with opt-in configuration. |
| **No memory saver** | `--enable-memory-saver` not supported on XPU. |
| **No two-batch overlap** | `--enable-two-batch-overlap` not supported on XPU. |

---

## Comparison: SGLang vs llama.cpp (Current Setup)

| Feature | llama.cpp (current) | SGLang XPU |
|---------|-------------------|------------|
| Quantized models (GGUF) | ✅ Full support | ❌ Not supported |
| Speculative decoding | ✅ DSpark n_max=2 | ❌ Not implemented |
| MoE models | ✅ DeepSeek V4-Flash | ❌ Not tested |
| Multi-GPU split | ✅ Balanced mode (custom) | ✅ Tensor parallel |
| KV cache reuse | ❌ Page-based only | ✅ RadixAttention (tree-based) |
| Structured generation | ❌ External tools | ✅ Built-in (JSON, regex, grammar) |
| Agentic workflows | ❌ Manual | ✅ SGLang DSL |
| Generation speed | 13-14 tok/s (with DSpark) | Unknown (no speculation) |
| Model size limit | 102.8GB (quantized) | ~30GB (BF16, single GPU) |
| Maturity | Production-ready | Experimental on XPU |

---

## When to Revisit

Re-evaluate SGLang when:

1. **GGUF/quantized model support lands** — Critical for running large models within our 128GB VRAM budget
2. **Speculative decoding implemented** — DSpark provides 20-30% speedup; losing it is a significant regression
3. **MoE models tested on XPU** — DeepSeek V4-Flash is a 284B MoE model; SGLang needs to handle expert routing efficiently
4. **Memory saver support added** — Would enable larger batch sizes and context windows
5. **Stable release with XPU parity** — When XPU is no longer "experimental"

---

## Potential Value Proposition

If/when the limitations are addressed, SGLang could offer:

- **RadixAttention** — Better KV cache reuse for multi-turn conversations and shared prefixes
- **Structured generation** — Native JSON/regex/grammar constraints without external libraries
- **SGLang DSL** — Composable workflows for tool use, multi-step reasoning, and agentic patterns
- **Prefill-decode disaggregation** — Separate prefill and decode workloads across GPUs for better throughput
- **Better Python ecosystem** — Easier integration with Hugging Face, PEFT, and research tooling

---

## Notes

- SGLang XPU support is optimized for Intel Arc Pro B-Series and Arc B-Series
- Intel provides an optimized attention backend (`--attention-backend intel_xpu`)
- Page sizes supported: 32, 64, 128
- Tested models are all BF16, small dense models (3B-8B range)
- NIXL-based KV transfer for P/D disaggregation is experimental
- `UCX_POSIX_USE_PROC_LINK=n` required to avoid UCX shared-memory issues on XPU

---

## References

- SGLang XPU Docs: https://docs.sglang.io/docs/hardware-platforms/xpu
- SGLang GitHub: https://github.com/sgl-project/sglang
- PyTorch XPU Guide: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
- NIXL (KV transfer): https://github.com/ai-dynamo/nixl
