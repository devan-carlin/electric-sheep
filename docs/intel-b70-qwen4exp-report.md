# Intel Arc Pro B70 Inference — Qwen3.8-Flash-Next on vLLM/XPU

Report for the Intel B70 inference team.

## Summary

A frontier 125B MoE model (Qwen3.8-Flash-Next, 6B active) achieves **53.4 tok/s decode** on **4x Intel Arc Pro B70 (128GB total VRAM)** using a patched vLLM. This performance matches or exceeds single high-end NVIDIA GPU performance for this model class and is ~2x the llama.cpp baseline (~27 tok/s) on the same hardware.

This implementation required significant XPU-side development. This report details the implementation requirements and the remaining XPU gaps for upstreaming.

## Result

- Model: Qwen3.8-Flash-Next, 125B MoE / 6B active, W4A16 (int4 group-128)
- Stack: vLLM built from upstream main @ c39076fef, torch 2.13.0+xpu, vllm-xpu-kernels 0.1.12, oneAPI Level-Zero 20.2.0
- Config: TP4 + expert parallel, fp8 KV, 256K context, XPU graph mode
- **53.4 tok/s decode** (single stream, 512 tokens, greedy, median of 3)
- TTFT ~0.11s
- 96GB PLE n-gram table mmap'd from host RAM (247GiB machine)

## What it took

1. **Model port** — Ported the `qwen4_exp` architecture to vLLM. This includes 48 layers (36 GDN linear-attention + 12 full-attention/QSA), 512-expert MoE (top-10 + 1 shared), and the hyper-connection residual mechanism (4 parallel streams, low-rank mixer, deviation-form RMSNorm).
2. **XPU kernels** — Verified W4A16 dense GEMM (oneDNN `int4_gemm_w4a16`) and W4A16 MoE grouped GEMM (CUTLASS-SYCL) on B70. Implemented the GDN linear-attention XPU forward path. Resolved "missing ops" by pinning `vllm-xpu-kernels` to version 0.1.12 (0.1.13.2 is broken).
3. **Quantization** — Performed single-quant int4 from the official BF16 source (no FP8 double-quant). Used a `quantization_config` ignore list to keep mtp, ple, visual, linear_attn, shared_expert, gate, hyper_connection, and indexer in BF16.
4. **PLE offload** — Mmap'd the 96GB n-gram table from host RAM, sharing it across all 4 TP ranks via the page cache. Per-step gather is minimal (16 rows/token).
5. **Graph mode** — Enabled `VLLM_XPU_ENABLE_XPU_GRAPH=1`, which is 6x faster than eager mode (43 vs 7 tok/s). PLE host-side syncs only break the graph at the PLE layer; the remaining 47 layers remain captured.

## XPU gaps (upstreamable items)

### 1. GDN `causal_conv1d` spec-decode limitation (highest value)

`RuntimeError: causal_conv1d does not support spec-decode and non-spec (prefill + decode) tokens in the same invocation; the spec path and the non-spec path are mutually exclusive`

- Location: `vllm/_xpu_ops.py:212 _gdn_attention_core_xpu_impl` (`torch.ops._xpu_C.gdn_attention`)
- Impact: MTP speculative decoding is correct and lossless (34% draft acceptance) but crashes when a step mixes spec and non-spec tokens in a single GDN call. This scheduling-dependent crash makes MTP unusable by default on XPU.
- This is a compiled-kernel limitation in the XPU GDN op, not a Python bug. A kernel fix would unblock speculative decoding for all GDN models on XPU.

### 2. Drafter dynamic-shape compile guard

A compiled drafter forward pass using two tensors with independent dynamic dim-0 symbols triggers a `ConstraintViolationError` during element-wise combination (the fc add). 
- Workaround: Run the drafter in eager mode (negligible cost; 1 of 49 layers). 
- Fix: A cleaner fix in the XPU compile path would allow compiled drafters.

### 3. qwen4_exp model port

The complete `qwen4_exp` model definition (attention, GDN wiring, MoE block, hyper-connection, PLE offload, MTP drafter) is a self-contained, upstreamable contribution to vLLM. It tests the W4A16 and GDN XPU paths end-to-end.

### 4. vllm-xpu-kernels version pinning

Version 0.1.13.2 is broken due to missing op registrations; use 0.1.12. A supported-version matrix is recommended to prevent "missing ops" errors for downstream users.

## Why this matters for the B70 line

- Demonstrates that frontier MoE models are practical on 4x B70, supporting 256K context with tool calling and reasoning.
- The identified gaps represent the work required to move from workarounds to first-class support. Item 1 (GDN spec-decode) is the highest-priority kernel fix.

## Reproduction

Refer to `qwen4exp-vllm-operations.md` for rebuild instructions, launch commands, and full bug logs. 
Benchmark script: `/tmp/bench_decode.py` (streaming, TTFT-separated, cache-busting nonce, greedy).