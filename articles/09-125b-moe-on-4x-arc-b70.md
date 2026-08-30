# 125B MoE at 53 tok/s on 4x Intel Arc Pro B70 (Non-NVIDIA)

Running a frontier model at high speeds on consumer-pro Intel GPUs. Implementation details follow.

---

Qwen3.8-Flash-Next is a 125B-parameter Mixture-of-Experts (MoE) model with 6B active parameters. This model typically requires datacenter-grade GPU hardware.

This implementation uses **4x Intel Arc Pro B70** GPUs, providing 128GB of total VRAM. It supports **256K context** with tool calling and reasoning, achieving a decoding speed of **53.4 tokens/second**.

This is a functional server, not a benchmark curiosity. Achieving this required significant Intel XPU development.

## The result

Test parameters: Single stream, 512 tokens, greedy decoding, median of 3 runs:

| Metric | Value |
|--------|-------|
| Decode throughput | **53.4 tok/s** |
| Time to first token | ~0.11s |
| Context | 256K |
| KV cache | fp8 |
| Baseline (llama.cpp, same box) | ~29 tok/s |

vLLM provides ~1.8x the throughput of the llama.cpp baseline on identical hardware.

## The hardware

- 4x Intel Arc Pro B70 (Xe3), 32GB each = 128GB VRAM
- 247GB host RAM
- oneAPI Level-Zero 20.2.0

Host RAM capacity is critical. See the memory distribution below.

## The stack

- vLLM `0.26.1rc1.dev500+gc39076fef` (patched)
- torch `2.13.0+xpu`
- vllm-xpu-kernels `0.1.12` (pinned — version 0.1.13.2 is broken)
- Python 3.12

This implementation uses a **patched** version of vLLM.

## Implementation details

### 1. Model Porting

The `qwen4_exp` architecture was not natively supported in vLLM. The following components were ported:

- 48 layers: 36 Gated DeltaNet (linear attention) + 12 full-attention (QSA)
- 512-expert MoE, top-10 + 1 shared
- A novel **hyper-connection** residual mechanism (4 parallel streams, low-rank mixer, deviation-form RMSNorm)

### 2. XPU Kernels

- W4A16 dense GEMM (oneDNN `int4_gemm_w4a16`)
- W4A16 MoE grouped GEMM (CUTLASS-SYCL)
- GDN linear-attention XPU forward path

A broken kernel wheel (0.1.13.2) prevented implementation. Pinning to version 0.1.12 resolved this issue.

### 3. W4A16 Quantization

W4A16 (int4 group-128) was derived from the official BF16 source. No FP8 double-quant was used. The `quantization_config` ignore list maintains sensitive components (embeddings, MTP, PLE, linear attention, shared expert, hyper-connections) in BF16.

### 4. Memory Management: 96GB PLE Table

The model includes an N-gram Embedding (PLE) layer backed by a **96GB hash table**. This table resides in host RAM rather than VRAM.

| Pool | Sized for | Holds |
|------|-----------|-------|
| VRAM (128GB) | weights + KV cache | 77GB W4A16 weights (sharded 4 ways) + fp8 KV + activations |
| Host RAM (247GB) | the PLE table | 96GB, memory-mapped, shared across all 4 ranks |

The table is memory-mapped (mmap) from host RAM. Each decoding step gathers only 16 rows per token (a few KB) to transfer to the GPU. This requires both high VRAM and high host RAM capacity. A system with 128GB VRAM but only 64GB RAM would accommodate the weights but not the table.

### 5. XPU Graph Mode

Enabling `VLLM_XPU_ENABLE_XPU_GRAPH=1` increases throughput from 7 tok/s (eager) to 43+ tok/s. Host-side PLE synchronization only breaks the graph at the PLE layer; the remaining 47 layers remain captured. This setting is essential for high throughput.

## The MTP caveat

The model includes a multi-token-prediction (MTP) head for speculative decoding. While the port is correct and lossless (34% draft acceptance), it is unreliable on XPU. The GDN `causal_conv1d` kernel cannot mix speculative and non-speculative tokens in a single call, causing mid-request crashes.

Consequently, MTP-enabled mode is slower than the baseline due to eager-drafter overhead. MTP is disabled by default. Resolving this kernel limitation is a high-priority improvement for XPU.

## Validation

The model was validated by having it generate a one-shot Flappy Bird clone successfully, demonstrating practical usability.

## Summary

- A frontier MoE model is practical on 4x B70 hardware.
- The remaining gaps for first-class XPU support are specific: the GDN spec-decode kernel, the drafter compile guard, and kernel version management.
- These port and kernel improvements are being submitted to the Intel B70 inference team.

## Links

- Model + recipe (W4A16, XPU): [HuggingFace repo](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16)
- Patched vLLM (installable fork branch): [`devan-carlin/vllm` @ `xpu-qwen4exp`](https://github.com/devan-carlin/vllm/tree/xpu-qwen4exp)
- XPU gaps + upstreamable items (public issue): [intel/llm-scaler#649](https://github.com/intel/llm-scaler/issues/649)
- Operations guide (clean rebuild, bug log, launch): [qwen4exp-vllm-operations.md](https://github.com/devan-carlin/electric-sheep/blob/main/docs/qwen4exp-vllm-operations.md)

*The vision tower is now in: the serving recipe is multimodal (Qwen3-VL ViT reused verbatim). Images are served alongside text.*