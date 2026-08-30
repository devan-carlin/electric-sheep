# More MoE Experts ≠ Better Answers. I Benchmarked 4 Strategies to Prove It.

**The common assumption: activating more experts per token gives richer, more specialized knowledge. The data says otherwise.**

---

## The Setup

- **Model**: Intel Qwen3.6-35B-A3B (256 experts, 40 layers, INT4 quantized)
- **Hardware**: 4× Intel Arc Pro B70 GPUs (128 GB VRAM total)
- **Framework**: vLLM 0.26.1, FlashAttention v2, TP=4
- **Test**: 8 prompts × 4 TopK variants (8, 16, 32, 64 experts per token)

## The Results at a Glance

| Variant | Throughput | Slowdown vs. TopK=8 | Verdict |
|---------|-----------|-------------------|----------|
| **TopK=8** | **81.4 tok/s** | — | ✅ Best overall |
| TopK=16 | 77.2 tok/s | 5.2% | ✅ Near-identical quality |
| TopK=32 | 71.1 tok/s | 12.7% | ✅ Near-identical quality |
| TopK=64 | 60.4 tok/s | 25.8% | ⚠️ Slower, no quality gain |

## The Performance Cliff

Each doubling of active experts costs more than the last:

- **8 → 16 experts**: 5.2% slowdown
- **16 → 32 experts**: 7.9% slowdown
- **32 → 64 experts**: 15.0% slowdown

TopK=64 is **25.8% slower** than TopK=8 — and the degradation accelerates with each doubling.

## The Quality Paradox

Here's the surprising part: **all four variants produce nearly identical quality.**

- **All 32 prompts passed** (8 prompts × 4 TopK variants)
- **No arithmetic errors** in any variant
- **No measurable quality degradation** at TopK=64
- **TopK=8, 16, 32, 64** all produce correct, detailed outputs

This contradicts the assumption that more experts = better answers. The data suggests the gating network's top-8 selection already captures the most relevant experts for these tasks.

## Why This Happens

The non-linear slowdown suggests:

1. **Diminishing returns on expert relevance** — The top-8 experts capture most of the signal. Experts 9–64 contribute marginal value but full compute cost.
2. **Compute-bound routing** — Each additional expert adds MoE layer overhead (routing, aggregation, communication across TP ranks).
3. **Accelerating degradation** — The 15% drop at 32→64 (vs 5–8% at lower ranges) suggests the tail experts are mostly noise.

## The Takeaway

**`num_experts_per_tok` is not a "set it to the max" parameter.**

- **TopK=8** is the sweet spot for most workloads (fastest, no quality loss)
- **TopK=16** costs ~5% for marginal gains in complex reasoning tasks
- **TopK=32** costs ~13% with no measurable quality improvement
- **TopK=64** costs ~26% with no quality gain — avoid unless you have specific edge cases

The data suggests the gating network's top-8 selection is already highly effective. More experts = more compute, not more intelligence.

## The Patches Required

This benchmark required three upstream patches:

1. **qzeros guard** — Intel INT4 MoE models crash vLLM on empty zero-point tensors
2. **TopK kernel extensions** — vllm-xpu-kernels only supported TopK ≤ 10 (Intel added TopK=16 in August 2026; we added TopK=32 and 64)
3. **Config symlinks** — variant-specific configs for each TopK value

All patches + full benchmark suite: [github.com/devan-carlin/electric-sheep](https://github.com/devan-carlin/electric-sheep)

Full technical analysis with per-prompt data, kernel diffs, and quality scoring methodology: [benchmark-analysis.md](https://github.com/devan-carlin/electric-sheep/blob/main/bench/results/SUMMARY.md)

---

*Devan Carlin — independent researcher, 4× Intel Arc Pro B70 GPUs. [github.com/devan-carlin/electric-sheep](https://github.com/devan-carlin/electric-sheep)*
