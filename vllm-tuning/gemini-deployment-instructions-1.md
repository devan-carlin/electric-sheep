







# Multi-GPU Intel Arc (XPU) vLLM Deployment Guide

This document provides a comprehensive, battle-tested deployment guide for running **vLLM** across **multi-GPU Intel Arc (Battlemage / Xe2)** setups using **Tensor Parallelism (`TP=4`)**.

It accounts for all compilation pitfalls, binary placement fixes, PyTorch dependency overrides, and VRAM memory boundary tuning discovered during configuration.

---

## Technical Overview & Stack Specs

* **Hardware**: 4x Intel Arc GPUs (e.g., Arc Pro B70 / Xe2)
* **OS / Environment**: Ubuntu Linux / Linux Mint (`ai-server`)
* **Python Runtime**: Python 3.12 Virtual Environment (`.venv-b70-minimax`)
* **Core Drivers & Runtime**: Intel oneAPI / Level-Zero (`xccl`, `ofi` fabric transport)
* **Serving Stack**: `vLLM` (XPU target), `vllm-xpu-kernels`, `triton-xpu` (v3.7.2)

---

## Step 1: Virtual Environment & Shell Initialization

Always ensure you are operating inside the dedicated virtual environment and sourcing your Intel oneAPI runtime variables before executing any build or python steps.

```bash
# 1. Navigate to the working vLLM directory
cd /mnt/fast-ai/vllm

# 2. Activate the target virtual environment
source ~/.venv-b70-minimax/bin/activate

# 3. Export global XPU target variables and source system environment profile
export VLLM_TARGET_DEVICE="xpu"
source /mnt/fast-ai/promoted-env.sh

```

---

## Step 2: C++/SYCL Kernel Binding Fix (`vllm-xpu-kernels`)

### Problem Encountered

When launching vLLM, you may see an error stating:

> `XPU platform is not available because: No module named 'vllm_xpu_kernels._C'`

### Root Cause

Building `vllm-xpu-kernels` compiles the Battlemage (`xe_2`) C++/SYCL shared objects into `build/lib.linux-x86_64-cpython-312/vllm_xpu_kernels/`, but standard `pip install` may fail to move these `.so` binaries into `site-packages`.

### Solution

Copy the compiled shared libraries directly into your active virtual environment's site-packages:

```bash
# 1. Compile the extensions inside the repository
cd /mnt/fast-ai/vllm-xpu-kernels
python3 setup.py build_ext --inplace

# 2. Copy compiled .so binaries directly to site-packages
cp /mnt/fast-ai/vllm-xpu-kernels/build/lib.linux-x86_64-cpython-312/vllm_xpu_kernels/*.so \
   /home/dc/.venv-b70-minimax/lib/python3.12/site-packages/vllm_xpu_kernels/

# 3. Verify the C++ extension loads cleanly
python3 -c "import vllm_xpu_kernels._C; print('Successfully loaded vllm_xpu_kernels._C!')"

```

---

## Step 3: Align `triton-xpu` Compiler Dependencies

### Problem Encountered

Importing `triton` fails with:

> `ImportError: cannot import name 'intel' from 'triton._C.libtriton'`

### Root Cause

Standard PyPI `pip install` commands can overwrite Intel's custom `triton-xpu` package with standard OpenAI CUDA/ROCm `triton`, stripping away Intel C++ bindings (`intel`).

### Solution

Force-reinstall the official `triton-xpu` wheel directly from the PyTorch XPU index:

```bash
# Force reinstall triton-xpu without modifying other dependencies
pip install triton-xpu --index-url https://download.pytorch.org/whl/xpu --force-reinstall --no-deps

# Verify Triton loads Intel C++ bindings correctly
python3 -c "import triton; print('Triton successfully imported from:', triton.__file__)"

```

---

## Step 4: VRAM Allocation & Context Tuning

### Memory Boundary Mechanics

* **Startup Driver Reserved VRAM**: ~1.5 GiB per GPU reserved by Level-Zero at initialization.
* **Effective Max Utilization Target**: Setting `--gpu-memory-utilization` above `0.95` can fail the pre-flight check because the driver exposes ~`28.79 GiB / 30.3 GiB` as available startup space.
* **KV Cache Calculation**:
* If utilization is set too low (e.g., `0.75`), vLLM bounds total allocation strictly, leaving negative KV cache headroom after model weights are loaded.
* For **27B INT4 models** (~17.7 GiB total across 4 GPUs, taking ~4.5 GiB/GPU), **`0.85`** provides the optimal balance, leaving **~17.8 GiB of KV cache per GPU**.



---

## Step 5: Verified Multi-GPU Launch Command

Execute the working production launch script to start the OpenAI-compatible REST server across all 4 GPUs:

```bash
# 1. Environment exports
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"

# 2. Launch 4-GPU Tensor Parallel Server
python3 -m vllm.entrypoints.openai.api_server \
    --model /home/dc/intel-vllm-01/models/intel-qwen3.6-27b-autoround-int4 \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 4 \
    --max-model-len 232144 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85

```

### Expected Startup Milestone

When startup is successful, the log will output:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     API server: HTTP server started on http://0.0.0.0:8030

```

---

## Step 6: Web UI Frontend Integration (Open-WebUI)

Deploy Open-WebUI via Docker, connecting directly to host port `8030`:

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8030/v1" \
  -e OPENAI_API_KEY="sk-vllm" \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

```

* Access the interface at `http://localhost:3000`.

To update Open-WebUI to a newer release without losing chat history:

```bash
docker rm -f open-webui
docker pull ghcr.io/open-webui/open-webui:main
# Re-run the docker run command above

```

---

## Operational Diagnostic Matrix (Known Logs & Warnings)

| Log Warning / Output | Cause | Operational Action |
| --- | --- | --- |
| `resource_tracker: leaked shared_memory objects` | Python IPC cleanup following a process termination. | **Safe to ignore.** Memory is freed automatically by OS. |
| `torch.distributed.all_gather_into_tensor is deprecated` | Upstream PyTorch function rename notice. | **Safe to ignore.** No functional or performance impact. |
| `xpu kernel topk_topp_sampler fallback` | Per-request seeds fallback to PyTorch native sampling. | **Safe to ignore.** Sampling takes <1% of execution time. |
| `XPU Graph support: single-GPU execution only` | XPU graph capture restrictions on multi-GPU arrays. | **Safe to ignore.** `FLASH_ATTN` handles multi-GPU execution. |
| `Triton kernel JIT compilation: batch_memcpy_kernel` | Dynamic memory layout compiled on first encounter. | **Safe to ignore.** Occurs once and caches permanently. |

---

## Performance Benchmarks (4x Intel Arc Pro B70 Array)

* **Decode Speed (Single Request)**: `50.5 - 51.9 tokens/sec`
* **Decode Speed (2 Parallel Streams)**: `40.8 - 41.0 tokens/sec`
* **Cached Prompt Throughput**: `~618 tokens/sec` (via Prefix Caching)
* **Total Cluster KV Cache**: `1,137,180 tokens` (~4.9x concurrency at max length)