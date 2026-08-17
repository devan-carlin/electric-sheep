# INT8 W8A16 Quantization for Qwen3.5-family Models on Intel Arc B70 XPU

**Date:** 2026-08-17
**Model:** `Qwen3.8-27B-heretic-ara` (qwen3_5 arch, 27B, 52 GB bf16)
**Hardware:** 4× Intel Arc Pro B70 (32 GB each), Threadripper PRO 3945WX, 247 GB RAM
**Status:** WORKING — model quantized, loads, and runs at 47.8 tok/s (TP=4, cudagraph), matching the INT4 baseline (48.0 tok/s)

---

## Summary

This document covers **producing** an INT8 W8A16 checkpoint that the patched vLLM XPU
path can load and run. It is the companion to
[`xpu-int8-w8a16-kernel-gap.md`](./xpu-int8-w8a16-kernel-gap.md), which covers the
**kernel** (the C++/Python changes that make `int8_gemm_w8a16` exist and run).

- **kernel-gap doc** = how to *run* an INT8 W8A16 model (the kernel + the byte-order/fpmath bugs).
- **this doc** = how to *make* the checkpoint (the RTN quantizer + the three config gotchas
  that make a qwen3_5 multimodal checkpoint actually load).

The quantizer is a small, self-contained RTN (round-to-nearest) script — no AutoRound, no
llm-compressor, no calibration data. It reads a bf16 safetensors model and writes a
compressed-tensors `pack-quantized` checkpoint that the patched vLLM loads directly.

## Why RTN and not AutoRound / llm-compressor

| Option | Why not (for this target) |
|--------|---------------------------|
| **AutoRound INT8** | Its INT8 export writes scales as **E8M0-encoded uint8** (`(scale + E8M0_EXPONENT_BIAS).clamp(...).to(torch.uint8)`), not plain bf16. The XPU kernel expects bf16 `weight_scale`. Would need a decode fix on top. |
| **llm-compressor W8A16** | Heavier dependency + calibration pass. Produces the same `pack-quantized` layout, but for a one-off local model the RTN script is simpler and has no extra install. |
| **RTN (this script)** | Writes bf16 scales directly → format-compatible out of the box. No calibration. ~3 min for a 27B model. Round-trip dequant rel err 0.004 (= 0.5/127, the theoretical RTN floor). |

RTN is the right tool here: the goal is a working, fast checkpoint, not a quality-maximized
one. (If you later want better quality, AutoRound is the upgrade path — but fix the E8M0
scale encoding first.)

## The checkpoint format (what the kernel consumes)

For each quantized 2D linear `[N, K]` (bf16), the checkpoint stores two tensors:

| Tensor | dtype | shape | meaning |
|--------|-------|-------|---------|
| `weight_packed` | **int32** | `[N, K/4]` | 4 int8 values packed per int32, little-endian. Each byte = `int8_value + 128` (the `uint8b128` offset convention, **not** two's complement). |
| `weight_scale` | **bf16** | `[N, K/128]` | Per-group (group_size=128) `absmax / 127`. |

8-bit symmetric → compressed-tensors scalar type `uint8b128`. No zero points.

> The `+128` offset is the single most important byte-level fact. The C++ kernel treats the
> unpacked values as **signed** int8, so the Python `process_weights_after_loading` must do
> `(packed.view(uint8).to(int16) - 128).to(int8)` — a plain `.view(int8)` reinterpret is wrong
> for any byte ≥ 128. (That bug is documented in the kernel-gap doc.)

## Patch / change inventory

| # | Target | What it does |
|---|--------|--------------|
| 1 | `quantize-w8a16.py` (new script) | RTN W8A16 group-128 quantizer. bf16 → `weight_packed` + `weight_scale`. |
| 2 | `config.json` → `quantization_config.quant_method` | **Must** be `compressed-tensors`, or every layer builds as plain bf16 and load crashes. |
| 3 | `config.json` → `quantization_config.ignore` | Two regexes: keep only language-model linears quantized; keep the sub-32 `in_proj_ba` bf16. |
| 4 | `quantize-w8a16.py` → `LINEAR_SUFFIXES` | Excludes `in_proj_a`/`in_proj_b` so their source weights stay bf16 (couples with #3). |
| 5 | Runtime env (`set-env-0123-gpu.sh`) | `VLLM_XPU_ENABLE_XPU_GRAPH=1` + comm-capture vars — required for cudagraph (2.4× throughput). |

Items 1–4 are the "patch" (checkpoint + config). Item 5 is a runtime requirement, not a code
change, but it is the difference between 19.6 and 47.8 tok/s, so it is documented here.

---

## Detailed descriptions

### 1. The RTN quantizer (`quantize-w8a16.py`)

**Location:** `vllm/vllm-src-int8-test/quantize-w8a16.py`
**Usage:** `python quantize-w8a16.py <src_dir> <dst_dir> [--group-size 128]`

Per-group quantization core (group_size=128, symmetric):

```python
def quantize_group128(w, group_size):
    """w: [N, K] bf16 -> (weight_packed int32 [N, K/4], weight_scale bf16 [N, K/G])."""
    N, K = w.shape
    wf = w.to(torch.float32)
    wg = wf.view(N, K // group_size, group_size)                # [N, G, gs]
    amax = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)  # [N, G, 1]
    scale_f = amax / 127.0                                       # [N, G, 1] f32
    q = torch.round(wg / scale_f).clamp(-127, 127).to(torch.int8)  # [N, G, gs]
    q = q.view(N, K)                                            # [N, K]
    u8 = (q.to(torch.int16) + 128).to(torch.uint8)              # [N, K]  (uint8b128 offset)
    packed = u8.view(torch.int32).contiguous()                  # [N, K/4]
    scale = scale_f.to(w.dtype).view(N, K // group_size).contiguous()  # [N, K/G]
    return packed, scale
```

Three bugs were found and fixed while bringing this up (all in this function):

1. **Broadcasting** — dividing the full `[N, K]` weight by `amax` `[N, G, 1]` fails. Divide the
   grouped view `wg` instead.
2. **Packing shape** — `q.view(N, K//4, 4).view(torch.int32)` collapses only the last dim,
   giving `[N, K/4, 1]`. Keep `q` as `[N, K]` and `.view(torch.int32)` directly → `[N, K/4]`.
3. **Dividing by the wrong value** — dividing by `amax` (not `scale = amax/127`) puts `q` in
   `[-1, 1]` instead of `[-127, 127]` (rel err 0.99). Divide by `scale_f`.

The script also:
- Copies all non-safetensors files (config, tokenizer, etc.).
- Processes each safetensors shard, replacing each target `.weight` with `.weight_packed` +
  `.weight_scale` and keeping everything else.
- Rebuilds `model.safetensors.index.json` (metadata values **stringified** — safetensors
  requires str→str metadata).
- Writes the `quantization_config` into `config.json` (see #2 and #3).

**Result for heretic-ara:** 400 tensors packed, 31.62 GB (from 52 GB bf16).

### 2. `quant_method: compressed-tensors` is mandatory

**Symptom when missing:**
```
AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'
```
during weight loading (both TP workers).

**Cause:** vLLM uses `quantization_config.quant_method` to select the quant config class.
Without it, `get_quant_method` returns `UnquantizedLinearMethod` for every layer, so each
linear is built with a plain `weight` parameter (no `weight_packed`). When the loader then
tries to load the checkpoint's `weight_packed` tensor, `load_weights` does
`param = getattr(self, name, self)` — the layer has no `weight_packed` attribute, so `param`
falls back to `self` (the layer object), and `param.data` raises the AttributeError.

**Fix:** add to `quantization_config`:
```json
"quant_method": "compressed-tensors",
"version": "0.1.0",
"quantization_status": "compressed"
```

### 3. The ignore list matches the *construction* prefix, not the weight name

This is the subtle one. The `ignore` list is evaluated by `should_ignore_layer` **at layer
build time**, against the layer's **construction prefix** — not the final vLLM weight name.

For the qwen3_5 **multimodal** wrapper (`Qwen3_5ForConditionalGeneration`), the language model
is nested:

```
Qwen3_5ForConditionalGeneration   (prefix "")
  └─ self.language_model          (prefix "language_model")
       └─ Qwen3_5ForCausalLM
            └─ self.model         (prefix "language_model.model")
                 └─ self.layers   (prefix "language_model.model.layers")
```

So the linear layers' construction prefix is `language_model.model.layers.N.<proj>`, **not**
`model.layers.N.<proj>`. (The `hf_to_vllm_mapper` strips `model.language_model.` → `model.`
only for *weight loading*; the ignore check happens earlier, at construction.)

A naive `re:^(?!model\.layers\.)` therefore matches (ignores) **every** language layer, because
none of them start with `model.layers.` — the whole model builds as bf16 and load crashes.

**Fix:** anchor on the real construction prefix. The `.*` makes it robust to any leading prefix:
```json
"re:^(?!.*language_model\\.model\\.layers\\.)"
```
This quantizes everything under `language_model.model.layers.*` and keeps visual / MTP /
embed / norm / lm_head as bf16.

> **How to find your model's construction prefix:** trace the `__init__` chain from the
> registered arch class, following `maybe_prefix(prefix, name)` calls. The top-level model is
> built with `prefix=""` (see `model_loader/utils.py::initialize_model`). For a *non*-multimodal
> model the prefix is usually just `model.layers.N.*`.

### 4. Fused-layer ignore checks the *unfused* shard names

The GDN `in_proj_ba` layer is a `MergedColumnParallelLinear` merging `in_proj_a` + `in_proj_b`
(N=48 each). N=48 is **never a multiple of 32** at any TP (48 → 24 → 12), so the
`int8_gemm_w8a16` kernel's `in/out % 32 == 0` constraint rejects it:
```
XPUw8a16IntLinearKernel cannot implement due to: in/out sizes (5120, 48) must be multiples of 32
```

Two coupled fixes are needed:

**(a) Ignore the layer** — but for a *fused* layer, `should_ignore_layer` does **not** check the
fused name. It expands to the unfused shard names (`in_proj_b`, `in_proj_a`) and checks *those*
against the ignore list. So `re:.*in_proj_ba` does **not** work (the shard names don't contain
`in_proj_ba`). The rule must match the shard names:
```json
"re:.*in_proj_[ab]$"
```

**(b) Keep the source weights bf16** — since the layer is built as bf16, the checkpoint must
contain plain `in_proj_a.weight` / `in_proj_b.weight` (bf16), **not** packed tensors. So
`in_proj_a` / `in_proj_b` are removed from `LINEAR_SUFFIXES` in the quantizer. If you forget
this, the checkpoint has packed tensors for a layer that expects bf16 → load mismatch.

**Final ignore list:**
```json
"ignore": [
  "re:^(?!.*language_model\\.model\\.layers\\.)",
  "re:.*in_proj_[ab]$"
]
```

### 5. Cudagraph requires the XPU graph env vars

`enforce_eager=False` alone is **not** enough on XPU. Without the env vars, vLLM logs
`Skipping CUDA graph capture ... cudagraph_mode was not manually set to NONE` and runs
effectively eager.

**Fix:** source `vllm/env/set-env-0123-gpu.sh`, which sets:
```bash
export VLLM_XPU_ENABLE_XPU_GRAPH=1        # the cudagraph toggle
export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1   # capture comm ops in the graph
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3
# + UR_L0_SYNC_MODE, TORCH_LLM_ALLREDUCE, CCL_*, ZE_AFFINITY_MASK, etc.
```

The `FakeTensor` impl for `_xpu_C::int8_gemm_w8a16` (added to the main venv's `_xpu_ops.py`,
documented in the kernel-gap doc) is what lets the torch.compile memory-profiling pass succeed
during capture. Both are required.

---

## Results

| Config | tok/s | Notes |
|--------|-------|-------|
| INT8 W8A16 heretic-ara, TP=4, **cudagraph ON** (main venv) | **47.80** | 512 tokens, 10.71 s. The target. |
| INT8 W8A16 heretic-ara, TP=4, cudagraph OFF (main venv) | 19.58 | Same run without the env vars — shows the 2.4× gap. |
| INT4 AutoRound Qwen3.8-27B, TP=4 (baseline) | 48.0 | Served server, stress-test suite. |
| INT8 W8A16 lued, TP=2, eager (test venv) | 10.30 | Earlier bring-up, different TP/venv — not comparable to the TP=4 numbers. |
| INT4 AutoRound, TP=2, eager (test venv) | 9.23 | Same-TP control for the lued run. |

**Bottom line:** INT8 W8A16 == INT4 at TP=4 with cudagraph. The quantization is not the
bottleneck; the environment (cudagraph, TP) is.

## Validation

- **Correctness:** `diag-int8-model.py <model> <tp>` → prompt "What is 17 * 23?" → output
  `'\n\n391'` (correct) in both the test venv (TP=2) and main venv (TP=4).
- **Quantizer self-test:** round-trip dequant rel err 0.004 (= 0.5/127 RTN floor); byte
  convention matches the known-good lued checkpoint (byte = int8+128, 4 per int32 LE);
  scale == per-group absmax/127.

## Reproducing

```bash
# 1. Quantize (bf16 -> INT8 W8A16)
source /home/dc/electric-sheep/vllm/.venv-int8-test/bin/activate
cd /home/dc/electric-sheep/vllm/vllm-src-int8-test
python quantize-w8a16.py \
  /mnt/data/models/Qwen3.8-27B-heretic-ara \
  /mnt/data/models/Qwen3.8-27B-heretic-ara-int8-w8a16

# 2. Sanity check (test venv, TP=2, eager)
source /opt/intel/oneapi/setvars.sh
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 CUDA_VISIBLE_DEVICES="" \
  python diag-int8-model.py /mnt/data/models/Qwen3.8-27B-heretic-ara-int8-w8a16 2

# 3. Benchmark (main venv, TP=4, cudagraph)
source /home/dc/electric-sheep/vllm/.venv/bin/activate
source /opt/intel/oneapi/setvars.sh
source /home/dc/electric-sheep/vllm/env/set-env-0123-gpu.sh
CUDA_VISIBLE_DEVICES="" \
  python bench-decode-cg.py /mnt/data/models/Qwen3.8-27B-heretic-ara-int8-w8a16 4 cg
```

## Files

| File | Purpose |
|------|---------|
| `vllm/vllm-src-int8-test/quantize-w8a16.py` | The RTN quantizer. |
| `vllm/vllm-src-int8-test/diag-int8-model.py` | Correctness check (raw token IDs + decode). |
| `vllm/vllm-src-int8-test/bench-decode-cg.py` | Throughput bench with `eager`/`cg` mode arg. |
| `vllm/env/set-env-0123-gpu.sh` | XPU graph + device env vars (cudagraph enabler). |
| `/mnt/data/models/Qwen3.8-27B-heretic-ara-int8-w8a16/` | The quantized checkpoint (31.62 GB). |

## Related

- [`xpu-int8-w8a16-kernel-gap.md`](./xpu-int8-w8a16-kernel-gap.md) — the kernel (C++/Python)
  that runs this checkpoint, including the fpmath and byte-order bugs.
- [`INT4-QUANTIZATION-PATCHES.md`](./INT4-QUANTIZATION-PATCHES.md) — the INT4 AutoRound patch
  set (the alternative quantization path).
