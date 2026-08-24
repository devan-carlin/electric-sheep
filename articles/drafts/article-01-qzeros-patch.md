# vLLM Crashes on Intel INT4 MoE Models. Here's the 6-Line Fix.

**Running Intel's quantized MoE models through vLLM? You've probably hit this error — and the fix is simpler than you'd expect.**

---

## The Error

```
RuntimeError: copy_() shape mismatch, 
copying tensor with shape [1, 1, 0, 1] to tensor with shape [1, 1, 4096, 1]
```

Cryptic. No stack trace. No indication of which layer failed. Just a crash.

## What's Actually Happening

Intel's AutoRound quantization produces **mixed-precision** models — some layers are INT4, others stay FP16. The INT4 layers use two types of quantization:

- **Asymmetric** — needs a zero-point offset (`qzeros` tensor)
- **Symmetric** — no zero-point needed (`qzeros` is empty)

vLLM's Intel quantization backend blindly copies `qzeros` without checking if it's empty. Empty tensor → shape mismatch → crash.

## The Fix

**File:** `vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py`

Replace the unconditional `qzeros.copy_()` with a guarded version:

```python
if (
    hasattr(layer, "qzeros") and layer.qzeros is not None
    and layer.qzeros.numel() > 0
    and hasattr(ark_linear, "qzeros") and ark_linear.qzeros is not None
    and ark_linear.qzeros.numel() > 0
    and layer.qzeros.shape == ark_linear.qzeros.shape
):
    ark_linear.qzeros.copy_(layer.qzeros.detach())
else:
    ark_linear.qzeros = None  # Symmetric layer, no zero-point needed
```

**That's it.** Seven conditions, one copy. If the tensor is empty or the shapes don't match, set `qzeros` to `None` instead of crashing.

## Why This Affects You

- Mixed-precision quantization is the standard for running large MoE models on consumer hardware
- Intel's AutoRound produces these models **by design**
- Without this patch, **any** Intel INT4 MoE model with symmetric layers crashes in vLLM

## Quick Check: Do You Have the Patch?

```python
from vllm.model_executor.layers.quantization.inc.schemes.inc_wna16_linear import INCWNA16Linear
import inspect

source = inspect.getsource(INCWNA16Linear.apply_weights)
print("✓ Patched" if "layer.qzeros.numel() > 0" in source else "✗ Not patched")
```

## The Bigger Story

This was one of three patches needed to run MoE benchmarks on Intel Arc hardware. The other two:

1. **TopK kernel patches** — vllm-xpu-kernels only supported TopK ≤ 10. Added 16, 32, 64.
2. **Config symlinks** — variant-specific configs for each TopK value.

Full technical deep-dive and benchmark results: [github.com/devan-carlin/electric-sheep](https://github.com/devan-carlin/electric-sheep)

---

*Devan Carlin — independent researcher, 4× Intel Arc Pro B70 GPUs. [github.com/devan-carlin/electric-sheep](https://github.com/devan-carlin/electric-sheep)*
