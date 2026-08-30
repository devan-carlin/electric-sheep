# Draft comment for intel/llm-scaler#649 — REVIEW BEFORE POSTING

Status update: the `qwen4_exp` port is complete and running on 4x Arc Pro B70.

## What's done

- Full `qwen4_exp` model implementation for vLLM: 48 layers (36 GDN linear-attention + 12 QSA full-attention), 512-expert MoE (top-10 + 1 shared), hyper-connection (gated-residual) branches, PLE n-gram table offload to host RAM, MTP drafter.
- W4A16 (int4 group-128, compressed-tensors) serving path on XPU, including the W4A16 dense + MoE grouped GEMM kernels and the GDN XPU forward.
- Expert parallelism (whole-expert distribution, since moe_intermediate 640 is not divisible by the 128-wide quant group under TP).
- PLE table (96GB) memory-mapped from host RAM, shared page cache across all TP ranks.
- MTP speculative decoding: correct and lossless, 34% draft acceptance.

## Measured (4x Arc Pro B70, 128GB VRAM, 247GB RAM)

- 53.4 tok/s decode (single stream, 512 tokens, greedy, median of 3)
- ~0.11s TTFT, 256K context, fp8 KV cache
- ~1.8x the llama.cpp baseline on the same box (~29 tok/s)
- Stack: vLLM 0.26.1rc1 (base c39076fef), torch 2.13.0+xpu, vllm-xpu-kernels 0.1.12, oneAPI Level-Zero 20.2.0

## Where the code is

- Upstream-based port (vllm-project/vllm @ c39076fef + a 16-file patch): https://github.com/devan-carlin/vllm/tree/xpu-qwen4exp (branch `xpu-qwen4exp`, one commit, 16 files; verified byte-identical to a working build and reproduced in a fresh venv)
- Model + recipe (W4A16 weights, PLE table, model card with hardware expectations): https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16
- Operations guide (clean rebuild, bug log, launch config): https://github.com/devan-carlin/electric-sheep/blob/main/docs/qwen4exp-vllm-operations.md
- One-shot installer: https://github.com/devan-carlin/electric-sheep/blob/main/vllm/setup-vllm-xpu.sh

The port targets upstream vLLM, where XPU is now a first-class target. Happy to open an upstream PR and/or rebase onto the llm-scaler-vllm tree for the next release.

## Two specific XPU kernel gaps found (blocking first-class MTP)

1. GDN `causal_conv1d` spec-decode limitation — `torch.ops._xpu_C.gdn_attention` rejects a call that mixes spec-decode and non-spec (prefill+decode) tokens: "the spec path and the non-spec path are mutually exclusive". MTP is correct/lossless but crashes whenever a step mixes the two (scheduling-dependent), so it's unusable as a default on XPU. A kernel fix would unblock speculative decoding for all GDN models on XPU.
2. Drafter dynamic-shape compile guard — a compiled drafter forward combining two tensors with independent dynamic dim-0 symbols (element-wise fc add) triggers a torch ConstraintViolationError; we run the drafter eager as a workaround. A cleaner fix in the XPU compile path would let drafters stay compiled.

## Reproduction

4x Arc Pro B70 + >=128GB host RAM. Launch: TP4 + expert parallel, fp8 KV, 256K ctx, XPU graph mode. Full flag set in the operations doc. PLE table is required for correct output (disabling it runs but produces wrong results).