# Server Configuration Baseline — `ai-server`

**Captured:** 2026-08-06  
**Hostname:** `ai-server`  
**User:** `user`

---

## 1. Operating System & Kernel

| Property | Value |
|---|---|
| Distribution | Ubuntu 26.04 LTS (Resolute) |
| Kernel | `7.0.0-29-generic` (PREEMPT_DYNAMIC) |
| Architecture | `x86_64` |

---

## 2. CPU

| Property | Value |
|---|---|
| Model | AMD Ryzen Threadripper PRO 3945WX 12-Cores |
| Cores / Threads | 12 cores / 24 threads (2 threads per core) |
| Sockets | 1 |
| Max MHz | 4427.77 |
| L1d / L1i Cache | 384 KiB (12 instances each) |
| L2 Cache | 6 MiB (12 instances) |
| L3 Cache | 64 MiB (4 instances) |
| NUMA Nodes | 1 |

**Instruction Set Flags (relevant):** `sse`, `sse2`, `ssse3`, `sse4_1`, `sse4_2`, `sse4a`, `avx`, `avx2`, `f16c`, `fma`

---

## 3. GPU Array

| Property | Value |
|---|---|
| Model | Intel Arc Pro B70 (Battlemage G31 — `8086:e223`) |
| Subsystem | ASRock Incorporation Device 6025 |
| Count | **4** |
| PCIe Slots | `43:00.0`, `47:00.0`, `63:00.0`, `67:00.0` |
| Kernel Driver | `xe` (Xe driver — NOT legacy `i915`) |
| VRAM Per GPU | **31.89 GiB** |
| Total VRAM | **~127.56 GiB** |

### Level Zero Runtime

| Property | Value |
|---|---|
| Runtime | Intel oneAPI Unified Runtime over Level-Zero V2 |
| Driver Version | `20.2.0 [1.15.38646+7]` |
| Devices Detected | `level_zero:0` through `level_zero:3` (all 4 GPUs) |

### OpenCL Runtime

| Property | Value |
|---|---|
| Platform | Intel(R) OpenCL Graphics |
| Driver Version | `26.22.38646.7` |
| Devices | 4 GPU devices + 1 CPU device |

### Installed GPU Driver Packages

| Package | Version |
|---|---|
| `intel-opencl-icd` | `26.22.38646.7-1~26.04~ppa1` |
| `libze-intel-gpu1` | `26.22.38646.7-1~26.04~ppa1` |
| `libze1` | `1.28.6-1~26.04~ppa1` |
| `libze-dev` | `1.28.6-1~26.04~ppa1` |
| `libze-intel-gpu-raytracing` | `1.2.4-1~26.04~ppa3` |

---

## 4. oneAPI Toolkit

| Property | Value |
|---|---|
| Toolkit Version | **2026.1.0** |
| DPC++/C++ Compiler | `2026.1.1-325` |
| Fortran Compiler | `2026.1.1-325` |
| MKL | `2026.1.0-236` (classic + SYCL) |
| oneCCL | `2022.0.0` |
| DNNL | `2026.0.2-40` |
| OpenMP | `2026.1.1-325` |
| TBB | `2023.1.0-151` |
| MPI | `2021.18.1-9` |
| VTune | `2026.3.0-5` |
| Dev Utilities | `2026.0.1-16` |

---

## 5. Memory & Swap

| Property | Value |
|---|---|
| Total RAM | **247 GiB** |
| Used RAM | ~132 GiB |
| Available RAM | ~115 GiB |
| Swap | **8 GiB** (`/swap.img`) |

---

## 6. Storage

| Property | Value |
|---|---|
| NVMe Model | Samsung SSD 990 PRO 2TB |
| Total Capacity | 1.8 TB |
| Root (`/`) | 1.8 TB (LVM: `ubuntu-vg/ubuntu-lv`) — 694 GB available (61% used) |
| Boot | 2 GB partition |
| EFI | 1 GB partition |

### Model Storage Path

| Path | Purpose |
|---|---|
| `~/electric-sheep/vllm/models/` | Model storage location |

---

## 7. Python Environment

| Property | Value |
|---|---|
| System Python | 3.14.4 |
| Venv Python | **3.12.13** (via deadsnakes PPA) |
| Venv Path | `~/electric-sheep/vllm/.venv` |
| pip | 26.2.1 |

---

## 8. Core AI Stack Versions

| Package | Version | Notes |
|---|---|---|
| **torch** | `2.13.0+xpu` | Intel XPU build |
| **torchvision** | `0.28.0+xpu` | |
| **torchaudio** | `2.11.0+xpu` | |
| **vllm** | `0.26.1rc1.dev380+gc2d800904.xpu` | Compiled from source (`--no-build-isolation`) |
| **vllm-xpu-kernels** | `0.1.13.dev5+g5085fdd` | Compiled from source; `__version__` attribute not exposed on dev builds |
| **triton** | `3.7.1` | |
| **triton-xpu** | `3.7.2` | Intel XPU backend override |
| **transformers** | `5.14.1` | |
| **numpy** | `2.3.5` | |
| **safetensors** | `0.8.0` | |
| **tokenizers** | `0.22.2` | |
| **compressed-tensors** | `0.17.0` | |
| **auto-round-lib** | `0.14.2` | |
| **xgrammar** | `0.2.3` | |
| **lm-format-enforcer** | `0.11.3` | |
| **oneccl** | `2022.0.0` | Intel collective communications |
| **onemkl-sycl-\*** | `2026.0.0` | BLAS, DFT, LAPACK, RNG, Sparse |

---

## 9. PyTorch XPU Verification

```
XPU Available: True
XPU Device Count: 4
  GPU 0: Intel(R) Arc(TM) Pro B70 Graphics  —  31.89 GiB
  GPU 1: Intel(R) Arc(TM) Pro B70 Graphics  —  31.89 GiB
  GPU 2: Intel(R) Arc(TM) Pro B70 Graphics  —  31.89 GiB
  GPU 3: Intel(R) Arc(TM) Pro B70 Graphics  —  31.89 GiB
```

---

## 10. Runtime Services

| Service | Status | Port | Details |
|---|---|---|---|
| **vLLM API Server** | Running | `0.0.0.0:8030` | PID 30533 |
| **Open-WebUI** | Running (healthy, 4 hrs) | `0.0.0.0:3000 → 8080` | Docker container |
| **Docker** | v29.7.1 | — | |

---

## 11. Network

| Interface | IP | State |
|---|---|---|
| `enp1s0` | `192.168.x.x/22` | UP |
| `docker0` | `172.17.x.x/16` | UP |
| `lo` | `127.0.0.1/8` | — |

---

## 12. Environment Variables (from `~/electric-sheep/vllm/set-env.sh`)

```bash
# Device Bindings
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3
UR_L0_SYNC_MODE=BLOCKING
TORCH_LLM_ALLREDUCE=1
CCL_ZE_IPC_EXCHANGE=pidfd
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

# vLLM Execution
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_ENGINE_ITERATION_TIMEOUT_S=300
TRITON_CACHE_DIR=/tmp/triton_cache
UVICORN_KEEP_ALIVE_TIMEOUT=300
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_TARGET_DEVICE=xpu

# Model Path
MODEL_PATH=~/electric-sheep/vllm/models/Intel-Qwen3.6-27B-int4-AutoRound
```

---

## 13. Verified Production Launch Command

The most stable start command discovered through iterative tuning:

```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/set-env.sh
export VLLM_TARGET_DEVICE="xpu"

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/vllm/models/Intel-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 4 \
    --max-model-len 232144 \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

### Parameter Rationale

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `4` | Matches 4× B70 GPU array |
| `--max-model-len` | `232144` | Qwen 3.x native context window |
| `--max-num-batched-tokens` | `65536` | High batch window for throughput; fits within fp8 KV cache budget |
| `--max-num-seqs` | `16` | Concurrency ceiling that fits within KV cache budget |
| `--kv-cache-dtype` | `fp8` | Reduces KV cache memory by ~50% vs fp16, critical for 27B+ models |
| `--enable-prefix-caching` | *(flag)* | Reuses KV cache for repeated prompt prefixes; ~618 tok/s on cached prompts |
| `--gpu-memory-utilization` | `0.80` | Conservative ceiling; leaves headroom for prefix caching + large batch tokens |
| `--generation-config` | `vllm` | Lets vLLM manage generation defaults (temperature, top_p, etc.) internally |
| `--reasoning-parser` | `qwen3` | Enables structured reasoning output parsing |
| `--tool-call-parser` | `qwen3_coder` | Enables tool-calling function parsing |

---

## 14. Known Issues & Observations

| Issue | Status | Notes |
|---|---|---|
| `vllm_xpu_kernels.__version__` AttributeError | **Resolved (false alarm)** | Dev builds don't expose `__version__`. The module and `_C` sub-module import cleanly. `.so` binaries are correctly placed in `site-packages`. |
| `triton` vs `triton-xpu` conflict | **Resolved** | Force-reinstall `triton-xpu` from PyTorch XPU index with `--force-reinstall --no-deps` |
| vLLM version check partial failure | **Observed** | `import vllm_xpu_kernels` fails in standalone check — likely path/state issue when vLLM server isn't running |
| `--gpu-memory-utilization` ceiling | **Documented** | `0.85` is the reliable max; `0.90+` causes pre-flight VRAM check failures due to Level-Zero driver reservation (~1.5 GiB/GPU) |
| Swap space | **8 GiB only** | May be insufficient for large model loading; consider expanding to 64+ GiB |
| 35B-A3B MoE `qzeros` crash | **Patched** | `inc_wna16_linear.py` assumes all layers have matching `qzeros` shapes. Symmetric expert layers have empty `qzeros`, causing `torch.copy_()` RuntimeError. Fixed via `patch-vllm-moe-qzeros.sh`. |

---

## 14. Summary Spec Sheet

```
┌─────────────────────────────────────────────────────────────┐
│  ai-server — Intel Arc B70 × 4 vLLM Deployment             │
├─────────────────────────────────────────────────────────────┤
│  OS:          Ubuntu 26.04 LTS (Kernel 7.0.0-29)           │
│  CPU:         AMD Threadripper PRO 3945WX (12C/24T)        │
│  RAM:         247 GiB (8 GiB swap)                         │
│  GPU:         4× Intel Arc Pro B70 (31.89 GiB each)        │
│  Storage:     Samsung 990 PRO 2TB NVMe                     │
│  Python:      3.12.13 (venv)                               │
│  PyTorch:     2.13.0+xpu                                   │
│  vLLM:        0.26.1rc1.dev380 (compiled from source)      │
│  Kernels:     vllm-xpu-kernels 0.1.13.dev5                 │
│  Triton:      3.7.2 (triton-xpu)                           │
│  oneAPI:      2026.1.0                                     │
│  L0 Driver:   26.22.38646.7                                │
│  Driver:      xe (kernel module)                           │
│  Network:     192.168.x.x/22                               │
│  Services:    vLLM :8030  │  Open-WebUI :3000              │
└─────────────────────────────────────────────────────────────┘
```
