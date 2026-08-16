# Patching vLLM's MoE Kernel to Support TopK=16, 32, and 64

*The XPU kernel only accepted a handful of expert-routing widths. We extended
it — and the reason is a compile-time template parameter you can't just
"bump up."*

> **Status: outline.** Real and verified (all TopK values validated); needs a
> final pass before posting.

---

## The hook

We wanted to benchmark a 35B MoE model at different `num_experts_per_tok`
settings — top-8, top-16, top-32, top-64. top-8 worked. Everything higher
crashed:

```
RuntimeError: error: not support TOPK=16
RuntimeError: Unsupported TopK value
```

The Intel XPU kernel library (`vllm-xpu-kernels`) only supported TopK values of
**1, 2, 4, 6, 7, 8, 10**. Anything else was rejected by a hardcoded dispatch
table.

## Why it's hardcoded (and why that's not a bug)

The dispatch is an `if/else if` chain in the SYCL kernel source:

```cpp
#define DISPATCH_TOPK_LAUNCH(TA, TS, TopK)              \
  if (TopK == 1)  { LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 1);  } \
  else if (TopK == 2) { LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 2); } \
  else if (TopK == 4) { LAUNCH_REMAP_HIDDEN_STATES(TA, TS, 4); } \
  /* ... 6, 7, 8, 10 ... */
```

`TopK` is a **compile-time template parameter**, not a runtime value. That's
not an accident or laziness — it's required. The kernels use it for:

- `#pragma unroll` loops (the compiler needs a constant trip count)
- stack array allocation: `int moe_ids[TOPK]`, `float scores[TOPK]`

You can't allocate a stack array with a runtime size in a SYCL kernel. So the
set of supported TopK values is *baked into the compiled binary*. To support a
new value, you add a branch **and recompile**.

## The fix

Two kernel files carry the dispatch tables:

- `csrc/moe/remap_hidden_states.cpp` (line ~518) — remaps hidden states during
  expert routing
- the grouped-GEMM / topk-selection kernel

For each, we added the new branches (`16`, `32`, `64`) to the dispatch macro and
rebuilt the kernel from source with the Intel oneAPI compiler (`icx`/`icpx`).
The new kernel version: `0.1.13.dev8+gd0dc965`.

After the rebuild, all four variants ran:

| Variant | Experts per token | Before | After |
|---------|-------------------|--------|-------|
| top-8   | 8  | ✅ | ✅ |
| top-16  | 16 | ❌ | ✅ |
| top-32  | 32 | ❌ | ✅ |
| top-64  | 64 | ❌ | ✅ |

## The general lesson

When a GPU kernel rejects a value you want, the first question is **is that
value a compile-time constant or a runtime one?** If it's compile-time (and in
SYCL/CUDA it often is, for unrolling and stack sizing), then "just pass a
bigger number" won't work — you have to extend the dispatch and rebuild.

This is a recurring theme in the XPU ecosystem: the kernels are more
conservative than the CUDA equivalents, and extending them means working in
C++/SYCL, not Python. It's a different skill, and it's worth knowing where the
boundary is.

## Takeaway

- **A kernel rejecting a value is often a compile-time constraint, not a
  runtime check.** Unrolling and stack arrays need constants.
- **Extending a GPU kernel = edit the dispatch + recompile.** Budget for the
  C++/SYCL toolchain, not just the Python side.
- **Know which values your backend supports before you design a benchmark
  around them.** We found out at runtime; a quick look at the dispatch table
  would have told us up front.
