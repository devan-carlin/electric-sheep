# Feature request draft: intel/llm-scaler

Title: [Feature] Qwen3.8-Flash-Next (qwen4_exp) support on XPU / Arc Pro B70

---

## Request

Add support for Qwen3.8-Flash-Next (`model_type: qwen4_exp`) to `llm-scaler-vllm` on Arc Pro B70.

## Model

- 125B MoE, ~6B active, 48 layers, 512 experts/layer, `moe_intermediate_size: 640`
- Hybrid attention: 36 GDN linear-attention layers + 12 QSA full-attention layers (KV cache only on the 12)
- PLE n-gram tables: 51B params (~102 GB BF16), random-access lookup, offloaded to host RAM at inference
- MTP speculative head (1 layer, BF16)
- Native context: 262144
- Checkpoints:
  - `Qwen/Qwen3.8-Flash-Next` (official, FP8)
  - `VnimanieAI/Qwen3.8-Flash-Next-W4A16` (INT4 group-128, compressed-tensors; targets consumer GPUs without FP8)

## Current state (verified 2026-08-27)

- Not in upstream vLLM main (checked `vllm-src` @ `c39076feff`): no `qwen4_exp` in the model registry. The architecture only exists in a vendor private dev build (`vllm/vllm-openai:qwen38-flash-next`, CUDA-only, no XPU variant on Docker Hub).
- Not in `llm-scaler-vllm` (checked repo code, branches, and issues).
- Runs on XPU today only via llama.cpp (PR #27742, `qwen4exp` arch): 4x Arc Pro B70, 262K context, q8_0 KV, ~29 t/s decode, PLE tables in host RAM.

## What would be needed (suggested priority order)

1. `qwen4_exp` model implementation in the vLLM fork: hybrid GDN + QSA layers, PLE n-gram tables, MTP head, hyper-connection (gated-residual) branches
2. PLE CPU offload (equivalent of `VLLM_PLE_CPU_OFFLOAD`): 102 GB of tables in host RAM, ~110 GB free RAM required
3. Expert parallelism: mandatory for this checkpoint family — `moe_intermediate_size` 640 shards to 160 (TP=4) / 80 (TP=8), neither divisible by the 128-wide quantization group; experts must be distributed whole
4. FP8 checkpoint path: B70 supports FP8, and the official FP8 checkpoint avoids the CUDA-only Marlin INT4 dependency
5. W4A16 INT4 group-128 (compressed-tensors) kernels for XPU: lower priority, Marlin has no XPU equivalent
6. Decode graph capture without inductor: the upstream recipe is `mode: 0` + decode-only graphs; inductor compilation of this architecture hangs (data-dependent control flow in QSA index selection)

## Reference implementations

- llama.cpp PR #27742 (`qwen4exp`, SYCL) — production-proven on 4x B70, includes the PLE-in-RAM layout and q8_0 KV
- VnimanieAI model card deployment notes (CUDA side: EP, PLE offload, decode-only graphs, MTP depth 4)

## Hardware

- 4x Intel Arc Pro B70 (32 GB each)
- 247 GB host RAM
- Ubuntu 24.04, oneAPI 2026.1

## Why it matters

- 6B-active MoE: 125B-class quality at a fraction of the decode cost
- On XPU the only working path today is llama.cpp; vLLM would add prefix caching, MTP speculative decoding, multi-user concurrency, and structured output (xgrammar)
- The model is new (released 2026-08-27) and high-profile; early XPU support would keep the B70 stack current