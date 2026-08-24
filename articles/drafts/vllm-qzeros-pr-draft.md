# vLLM PR / Issue Draft — qzeros crash on symmetric (empty-qzeros) INT4 layers

> **How to use this file**
> - **Option A (recommended): open a PR.** Fork `vllm-project/vllm`, apply the diff in
>   §3 to `vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py`,
>   commit, push, and open a PR. Paste §1 as the PR description.
> - **Option B: open an issue first** (if you want to gauge interest / can't test on
>   upstream CI). Paste §1 as the issue body and attach the diff from §3.
> - The diff below is against **upstream `main`** (verified: the original block matches
>   upstream exactly). Your local build is `0.26.1rc1.dev500+gc39076fef`.
> - Replace the `@devan-carlin` handle and the repro model name with whatever you want
>   public before posting.

---

## §1 — PR / Issue description (paste this)

**Title:**
`[XPU][INC] Fix RuntimeError when loading INT4 checkpoints with symmetric (empty-qzeros) layers`

**Body:**

### Summary

Loading an Intel AutoRound INT4 checkpoint that contains **symmetrically-quantized layers**
crashes in `INCARKLinearMethod.process_weights_after_loading` with:

```
RuntimeError: copy_() shape mismatch,
copying tensor with shape [1, 1, 0, 1] to tensor with shape [1, 1, 4096, 1]
```

The crash happens because the code copies `layer.qzeros` into `ark_linear.qzeros` whenever
`layer.qzeros is not None` — but for **symmetric** layers the `qzeros` tensor *exists* and is
simply **empty** (`numel() == 0`). The `is not None` guard passes, the `copy_()` runs, and the
shape mismatch (0 vs. the real feature dim) raises.

This affects any mixed-precision INT4 MoE checkpoint where some layers are asymmetric
(non-empty `qzeros`) and others are symmetric (empty `qzeros`) — which is exactly what Intel's
AutoRound produces by design (e.g. attention layers asymmetric, expert layers symmetric).

### Root cause

`vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py`,
`INCARKLinearMethod.process_weights_after_loading`:

```python
ark_linear.qweight.copy_(qweight_src)
if hasattr(layer, "qzeros") and layer.qzeros is not None:
    ark_linear.qzeros.copy_(layer.qzeros.detach())
else:
    ark_linear.qzeros = None
ark_linear.scales.copy_(layer.scales.detach())
```

The guard checks `is not None` but not whether the tensor is **non-empty** or **shape-compatible**.
A symmetric layer's `qzeros` is a registered, non-`None`, zero-element tensor, so it falls into
the `copy_()` branch and fails.

### Fix

Only copy when both tensors are present, non-empty, and shape-compatible; otherwise set
`ark_linear.qzeros = None` (the symmetric path, which is already handled downstream):

```python
ark_linear.qweight.copy_(qweight_src)

# Safely handle qzeros to prevent empty/mismatched tensor copy crashes on symmetric layers
if (
    hasattr(layer, "qzeros")
    and layer.qzeros is not None
    and layer.qzeros.numel() > 0
    and hasattr(ark_linear, "qzeros")
    and ark_linear.qzeros is not None
    and ark_linear.qzeros.numel() > 0
    and layer.qzeros.shape == ark_linear.qzeros.shape
):
    ark_linear.qzeros.copy_(layer.qzeros.detach())
else:
    ark_linear.qzeros = None

ark_linear.scales.copy_(layer.scales.detach())
```

This is a strict superset of the existing guard: asymmetric layers (non-empty, matching
`qzeros`) behave exactly as before; symmetric layers now fall through to `qzeros = None`
instead of crashing.

### Reproduction

- **Hardware:** 4× Intel Arc Pro B70 (Battlemage, XPU), vLLM `0.26.1rc1.dev500+gc39076fef`
- **Model:** an Intel AutoRound INT4 MoE checkpoint with mixed symmetric/asymmetric layers
  (e.g. `Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound`, `--quantization auto-round`)
- **Command:**
  ```bash
  vllm serve <model> --quantization auto-round --tensor-parallel-size 4
  ```
- **Result:** crashes during weight loading with the `copy_() shape mismatch` error above,
  before the server starts.

### Verification

After the patch, the same model loads and serves on XPU. (I can add a unit test under
`tests/quantization/` that constructs a layer with an empty `qzeros` and asserts
`process_weights_after_loading` does not raise — happy to add it if maintainers want it.)

### Notes

- The sibling XPU path `INCXPULinearMethod` is unaffected: it hard-codes a symmetric
  `qzeros = tensor([8])` and never copies from `layer.qzeros`.
- No behavior change for fully-asymmetric or fully-symmetric checkpoints.

---

## §2 — Why this is a real, upstreamable bug (context for you, don't paste)

- The original guard is genuinely insufficient — it's not a local-env quirk. Anyone loading a
  mixed-symmetry AutoRound INT4 checkpoint on the ARK/XPU path hits it.
- The fix is minimal, defensive, and backward-compatible (strict superset of the old guard).
- It's the kind of small, well-scoped fix maintainers like to merge.

## §3 — Exact diff (apply to upstream `main`)

File: `vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py`
Class: `INCARKLinearMethod` → method `process_weights_after_loading`

```diff
             ark_linear.qweight.copy_(qweight_src)
-            if hasattr(layer, "qzeros") and layer.qzeros is not None:
-                ark_linear.qzeros.copy_(layer.qzeros.detach())
-            else:
-                ark_linear.qzeros = None
+
+            # Safely handle qzeros to prevent empty/mismatched tensor copy crashes on symmetric layers
+            if (
+                hasattr(layer, "qzeros")
+                and layer.qzeros is not None
+                and layer.qzeros.numel() > 0
+                and hasattr(ark_linear, "qzeros")
+                and ark_linear.qzeros is not None
+                and ark_linear.qzeros.numel() > 0
+                and layer.qzeros.shape == ark_linear.qzeros.shape
+            ):
+                ark_linear.qzeros.copy_(layer.qzeros.detach())
+            else:
+                ark_linear.qzeros = None
+
             ark_linear.scales.copy_(layer.scales.detach())
```

## §4 — Suggested commit message

```
[XPU][INC] Guard qzeros copy against empty/symmetric layers

INCARKLinearMethod.process_weights_after_loading copied layer.qzeros
whenever it was not None. Symmetric INT4 layers register a non-None but
empty (numel()==0) qzeros tensor, so the copy_() raised a shape-mismatch
RuntimeError and model loading failed.

Only copy when both qzeros tensors are present, non-empty, and
shape-compatible; otherwise set ark_linear.qzeros = None (the existing
symmetric path). No behavior change for asymmetric checkpoints.
```

## §5 — Before you post (checklist)

- [ ] **Decide PR vs. issue.** A PR is stronger (it's a real fix). If you open a PR, you'll
      want to run it against upstream CI — the XPU-specific path may not be exercised by
      default CI, so be ready to note "verified locally on Arc Pro B70."
- [ ] **Make the repro model public or anonymize.** If `Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound`
      isn't a public HF repo, either link it or describe it generically ("an Intel AutoRound
      INT4 MoE checkpoint with mixed symmetric/asymmetric layers").
- [ ] **Confirm the GitHub repo in your article is public** before the article links it.
- [ ] **Fix the article's "Quick Check" snippet** — it imports `INCWNA16Linear`, which does not
      exist. The real class is `INCARKLinearMethod`. Corrected check:
      ```python
      from vllm.model_executor.layers.quantization.inc.schemes.inc_wna16_linear import INCARKLinearMethod
      import inspect
      src = inspect.getsource(INCARKLinearMethod.process_weights_after_loading)
      print("✓ Patched" if "layer.qzeros.numel() > 0" in src else "✗ Not patched")
      ```
- [ ] **Line-count consistency** (article): title says "6-Line Fix," body says "Seven
      conditions." The added guard is 7 conditions / ~10 lines. Pick one framing.
