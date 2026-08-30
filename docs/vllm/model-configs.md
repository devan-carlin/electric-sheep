# Model Configuration Reference

**Project:** `~/electric-sheep/vllm/` (vLLM) + `~/electric-sheep/llama/` (llama.cpp)  
**Shared Models:** `~/electric-sheep/models/`  
**Hardware:** 4× Intel Arc Pro B70 (31.89 GiB each, `xe` driver)  
**VRAM Budget:** ~25.5 GiB/GPU at `0.80` utilization

---

## Available Models

| Model | Parameters | Quantization | Est. Size | Est. VRAM (TP=4) | Est. VRAM (TP=2) |
|---|---|---|---|---|---|
| **Qwen 3.6 27B** | 27B | INT4 AutoRound | ~14 GiB | ~3.5 GiB/GPU | ~7 GiB/GPU |
| **Qwen 3.6 35B-A3B** | 35B (MoE) | INT4 Mixed AutoRound | ~18 GiB | ~4.5 GiB/GPU | ~9 GiB/GPU |
| **Gemma 4 31B** | 31B | INT4 AutoRound V2 | ~16 GiB | ~4 GiB/GPU | ~8 GiB/GPU |
| **Gemma 4 26B-A4B** | 26B (MoE) | INT4 AutoRound | ~13 GiB | ~3.2 GiB/GPU | ~6.5 GiB/GPU |
| **DeepSeek V4-Flash** | ~38B (MoE) | UD-IQ3_XXS GGUF | ~98 GiB (4 shards) | ~8 GiB/GPU | N/A (TP=4 only) |

---

## Deployment Configurations

### 1. Full 4-GPU Deployment (Single Model, Full Context)

**Best for:** Maximum context window, single-model throughput

**GPU Allocation:** 0, 1, 2, 3  
**Tensor Parallelism:** 4  
**Context Window:** 232,144 tokens

#### Qwen 3.6 27B (Recommended Primary)

```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-0123-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound \
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

#### Qwen 3.6 35B-A3B (MoE — requires patch)

> **Prerequisite:** Run `bash ~/electric-sheep/build/ubuntu/03.1-patch-vllm-moe-qzeros.sh` before first launch.

```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-0123-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound \
    --served-model-name qwen3.6-35b-a3b \
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

#### Gemma 4 31B

```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-0123-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-gemma-4-31B-it-int4-AutoRound-V2 \
    --served-model-name gemma-4-31b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 4 \
    --max-model-len 232144 \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

#### Gemma 4 26B-A4B

```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-0123-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-gemma-4-26B-A4B-it-int4-AutoRound \
    --served-model-name gemma-4-26b-a4b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 4 \
    --max-model-len 232144 \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

#### DeepSeek V4-Flash (GGUF — 4 shards, UD-IQ3_XXS)

> **Prerequisites:**
> 1. Install GGUF plugin: `pip install vllm-gguf-plugin`
> 2. Pass `--tokenizer` with the base model repo (GGUF tokenizer is slow/buggy)
> 3. Use `--hf-config-path` if the model is not supported by HuggingFace
> 4. Point `--model` to the **first shard file** (not the directory)

```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-0123-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf \
    --tokenizer deepseek-ai/DeepSeek-V4-Flash-0731 \
    --hf-config-path deepseek-ai/DeepSeek-V4-Flash-0731 \
    --served-model-name deepseek-v4-flash \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.95
```

> **Notes:**
> - GGUF support in vLLM is **experimental and under-optimized**
> - Uses `0.95` GPU memory utilization (larger model needs headroom)
> - 64K context window (UD-IQ3_XXS is aggressive ~3-bit quantization)
> - TP=4 only — too large for TP=2 on B70s

---

### 2. Dual-Model Deployment (Two Models, 2 GPUs each)

**Best for:** Running two models simultaneously across the GPU array

**GPU Allocation:** 0,1 (Model A) + 2,3 (Model B)  
**Tensor Parallelism:** 2 per model  

#### Option A: Reduced Context (Higher Concurrency)
*Context: 32,768 tokens | Concurrency: `--max-num-seqs 16`*

**Terminal 1 — GPUs 0,1 (Qwen 27B):**
```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-01-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
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

**Terminal 2 — GPUs 2,3 (Qwen 35B-A3B or Gemma):**
*(See previous section for model-specific commands on port 8031)*

---

#### Option B: Full Context (Single Concurrent Request)
*Context: 128,000 tokens | Concurrency: `--max-num-seqs 1`*

> **VRAM Reality Check:** Full 232k context on TP=2 requires ~36.5 GB/GPU (KV cache + weights), which exceeds the B70's 31.89 GB physical limit. Capping at 128k keeps you within the `0.80` utilization budget (~23.1 GB/GPU). If you absolutely need 232k, you must bump to `--gpu-memory-utilization 0.95` and accept pre-flight OOM risk.

**Terminal 1 — GPUs 0,1 (Qwen 27B, 128k Context):**
```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-01-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 2 \
    --max-model-len 128000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

**Terminal 2 — GPUs 2,3 (Qwen 35B-A3B, 128k Context):**
```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-23-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound \
    --served-model-name qwen3.6-35b-a3b \
    --host 0.0.0.0 \
    --port 8031 \
    --tensor-parallel-size 2 \
    --max-model-len 128000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

---

## Parameter Reference

| Flag | 4-GPU Value | 2-GPU Value | Rationale |
|---|---|---|---|
| `--tensor-parallel-size` | `4` | `2` | Matches GPU count |
| `--max-model-len` | `232144` | `32768` | Full context vs. reduced for dual-model |
| `--max-num-batched-tokens` | `65536` | `16384` | Scaled to KV cache budget |
| `--max-num-seqs` | `16` | `16` | Concurrency ceiling |
| `--kv-cache-dtype` | `fp8` | `fp8` | ~50% memory reduction vs fp16 |
| `--gpu-memory-utilization` | `0.80` | `0.80` | Conservative ceiling for stability |
| `--enable-prefix-caching` | *(flag)* | *(flag)* | Prompt prefix reuse |
| `--generation-config` | `vllm` | `vllm` | Internal generation defaults |

---

## Notes

- **35B-A3B MoE:** Requires `patch-vllm-moe-qzeros.sh` before first launch. The patch guards against empty `qzeros` tensors on symmetric expert layers.
- **Gemma 4 31B:** No special patching required. Standard INT4 AutoRound V2 quantization.
- **DeepSeek V4-Flash (GGUF):** Requires `vllm-gguf-plugin` (`pip install vllm-gguf-plugin`). Always pass `--tokenizer` with the base model repo. Point `--model` to the first shard `.gguf` file (not the directory). GGUF support is experimental/under-optimized in vLLM.
- **Port conflicts:** Dual-model deployments use `8030` (GPUs 0,1) and `8031` (GPUs 2,3).
- **Open-WebUI:** Point to the desired port (`8030` or `8031`) via `OPENAI_API_BASE_URL`.
