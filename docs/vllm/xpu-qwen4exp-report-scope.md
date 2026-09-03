# XPU qwen4exp Re-Port Scope (vLLM upstream main)

**Date:** 2026-09-01
**Trigger:** Backlog item "Full vLLM rebuild from upstream main."
**Finding:** Upstream main (`7c5dc57`) has merged qwen4_exp support, but XPU is
not supported. Our 17-file port patch is now obsolete. A re-port is required.

---

## TL;DR

Upstream merged qwen4_exp as a first-class model package
(`vllm/models/qwen4_exp/`) with `nvidia/` and `amd/` variants. The dispatcher
raises `NotImplementedError` on XPU:

> "Qwen4Exp currently supports CUDA and ROCm only"

There is **no `xpu/` variant**. Our XPU port (17 files, ~15K lines) needs to be
re-ported into upstream's new structure. The hard parts:

1. **PLE table host-RAM offload** (95GB table; upstream is GPU-resident FP8 only).
2. **W4A16 quantization** (upstream qwen4_exp is FP8 only; we serve W4A16).
3. **GDN linear attention XPU path** (our `qwen_gdn_linear_attn.py`, 1806 lines).
4. **MTP spec-decode wiring** (our `qwen4_exp_mtp.py`, 333 lines).

**Effort:** multi-hour focused work (5-10h), not an unattended night build.
The current venv (`c39076fef`, ~53 tok/s) still works, so there is no urgency.

---

## What upstream merged

`vllm/models/qwen4_exp/` package:

- `common/hyperconnection.py` — `HyperConnectionBase`, `GatedResidual`,
  `GroupedGemmaRMSNorm`, `HyperConnectionConfig`.
- `common/ple.py` — `PLEVocabParallelEmbedding`, shard-overlap helpers.
- `common/qsa_cache.py` — QSA cache.
- `nvidia/model.py` (1065 lines) — `Qwen4ExpForCausalLM`,
  `Qwen4ExpForConditionalGeneration`, `Qwen4ExpModel`, `Qwen4ExpDecoderLayer`,
  `Qwen4ExpMixtureOfExperts`, `Qwen4ExpProcessingInfo`.
- `nvidia/ple_layer.py` — `Qwen4ExpPLELayer` (GPU-resident FP8 PLE embedding,
  short-conv attention backend).
- `nvidia/mtp.py` — `Qwen4ExpMTP`.
- `amd/` — ROCm variant.
- `config.py` — `Qwen4ExpConfig`, `Qwen4ExpTextConfig`, `Qwen4ExpVisionConfig`.

Registry entries for `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration`,
`Qwen4ExpMTP` (pointing at `vllm.models.qwen4_exp`).

XPU infrastructure (already upstream, no re-port needed):
`xpu_worker.py`, `xpu_model_runner.py`, `platforms/xpu.py`, `xpu_moe.py`,
`linear/scaled_mm/xpu.py`, `linear/mixed_precision/xpu.py`,
`xpu_communicator.py`, `xpumem.py`, `_xpu_ops.py`, `flash_attn.py` XPU backend.

---

## What's already upstream (free wins from the rebuild)

- **Mamba XPU ptr fix** (`_reinterpret_u64_as_i64` in `mamba_utils.py`,
  line 186, both call sites). Our `mamba-xpu-ptr.patch` is superseded.
- **K3 mamba metadata prep** (#52388, 6.6-7.6x kernel speedup).
- **State-copy race fix** (#50729).
- **3D-grid tiling** (#49436).
- **XPU infrastructure** (worker, model runner, platform, MoE, linear kernels,
  communicator, memory allocator).

The rebuild picks these up for free. The re-port is only about the qwen4_exp
model package XPU variant + our XPU-specific fixes.

---

## The re-port surface

Our XPU-specific files (from the current venv `vllm-fresh-venv`) and how each
maps to the re-port:

| File (venv path) | Lines | Disposition |
|---|---|---|
| `model_executor/models/qwen4_exp.py` | 1440 | **Re-port** to `models/qwen4_exp/xpu/model.py` |
| `model_executor/models/qwen4_exp_mtp.py` | 333 | **Re-port** to `models/qwen4_exp/xpu/mtp.py` |
| `models/qwen4_exp/__init__.py` (upstream) | ~50 | **Edit** — add XPU branch to dispatcher |
| `model_executor/kernels/linear/mixed_precision/xpu.py` | 310 | Keep (already XPU, upstream has it) |
| `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | 1806 | Keep (our GDN XPU path) |
| `model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py` | 475 | Keep (our W4A16 scheme) |
| `_xpu_ops.py` | 1342 | Keep (upstream has it) |
| `v1/attention/backends/flash_attn.py` | 1784 | Keep (upstream has XPU backend) |
| `v1/spec_decode/llm_base_proposer.py` | 1894 | **Re-apply** our `image_token_index` arch fix |
| `v1/worker/mamba_utils.py` | 1308 | Use upstream (has ptr fix) |
| `config/speculative.py` | 1515 | **Re-apply** our MTP wiring |
| `config/model.py` | 2479 | **Re-apply** our qwen4_exp config bits |
| `model_executor/models/qwen3_next.py` | 963 | **Re-apply** our GDN base changes |
| `model_executor/models/registry.py` | 1523 | Use upstream (has qwen4_exp entries) |
| `transformers_utils/config.py` | 1249 | **Re-apply** our qwen4_exp config |
| `transformers_utils/configs/qwen4_exp.py` | 322 | Use upstream (has it) |
| `model_executor/kernels/linear/__init__.py` | 1207 | **Re-apply** our W4A16 registration |

The bulk is the new `xpu/` model variant: our 1440-line `qwen4_exp.py` +
333-line `qwen4_exp_mtp.py`, adapted to upstream's class hierarchy
(`GatedResidual` instead of our `HyperConnection`, upstream `Qwen4ExpConfig`,
etc.).

---

## The key deltas (the hard parts)

### 1. PLE table host-RAM offload (biggest delta)

Our `Qwen4ExpPLE` memory-maps a **95GB** table from host RAM
(`PLE_TABLE_PATH=/mnt/data/ple_cache/ple_table_qwen4exp.pt`). Upstream's
`Qwen4ExpPLELayer` is **GPU-resident FP8** (no host-RAM offload; the table must
fit in GPU memory).

- **Option A (recommended):** port our `Qwen4ExpPLE` offload into upstream's
  `Qwen4ExpDecoderLayer` interface. Preserves the proven 95GB-table approach.
- **Option B:** use upstream's GPU-resident FP8 PLE. The 95GB table would need
  to fit in 4x32GB = 128GB GPU memory (tight), and it changes the quant path.
  Bigger change, unproven.

### 2. W4A16 quantization

We serve W4A16 (int4 weights, bf16 activations). Upstream qwen4_exp is FP8 only.
Wire our W4A16 linear kernel (`mixed_precision/xpu.py` + `inc_wna16_linear.py`)
into upstream's model `load_weights` + linear layers. Moderate change.

### 3. GDN linear attention XPU path

Our `qwen_gdn_linear_attn.py` (1806 lines) implements the GDN linear-attention
XPU path. Upstream uses the standard attention backend. Wire our GDN path into
upstream's attention layers. Moderate change.

### 4. MTP spec-decode wiring

Our `qwen4_exp_mtp.py` (333 lines) is the MTP draft model. Upstream has
`nvidia/mtp.py` + `amd/mtp.py` but no `xpu/mtp.py`. Create the XPU MTP variant
and wire it into the spec-decode proposer. Moderate change. (Recall: MTP is a
net ~15% slowdown on this model, so it stays disabled by default.)

---

## Concrete plan

1. **Create the `xpu/` model variant.**
   - Copy our `qwen4_exp.py` to `vllm/models/qwen4_exp/xpu/model.py`.
   - Adapt to upstream's structure: use `GatedResidual` (from
     `common/hyperconnection.py`) in place of our `HyperConnection`; use
     upstream `Qwen4ExpConfig` (from `config.py`).
   - Keep our PLE offload (`Qwen4ExpPLE`), W4A16 linear kernel, GDN path.
   - Copy our `qwen4_exp_mtp.py` to `vllm/models/qwen4_exp/xpu/mtp.py`.
   - Add `vllm/models/qwen4_exp/xpu/__init__.py`.
2. **Update the dispatcher.** Edit `vllm/models/qwen4_exp/__init__.py` to add an
   XPU branch importing from `xpu/model.py` + `xpu/mtp.py` (replace the
   `NotImplementedError` for `is_xpu()`).
3. **Re-apply our XPU-specific fixes** (the "Re-apply" rows in the table above):
   `llm_base_proposer.py` arch fix, `speculative.py` MTP wiring, `model.py`
   config, `qwen3_next.py` GDN base, `linear/__init__.py` W4A16 registration,
   `transformers_utils/config.py`.
4. **Build + test.**
   - Build vllm from main + re-port patch (arc-b70 `build-vllm-xpu.sh`).
   - Keep vllm-xpu-kernels pinned at 0.1.12 (do NOT rebuild from main).
   - Fresh venv, point at the PLE table.
   - Smoke test (text + image), benchmark vs the 53.4 tok/s baseline.
5. **Swap production venv** if green; keep the old one a day as fallback.

---

## Effort estimate

- Re-port the model variant (`xpu/model.py` + `xpu/mtp.py`): **2-4h** (the bulk).
- Update dispatcher + re-apply XPU fixes: **1-2h**.
- Build + test + debug XPU-specific issues: **2-4h**.

**Total: 5-10h of focused work.** Not unattended-friendly.

---

## Risks

- **Structure drift:** upstream's class hierarchy differs from our old port;
  interface mismatches may surface during the re-port.
- **PLE offload integration:** our custom `Qwen4ExpPLE` must fit upstream's
  `Qwen4ExpDecoderLayer` forward path (PLE runs before the attention mix).
- **W4A16 + upstream model:** upstream model is FP8; our W4A16 `load_weights`
  + linear kernel must be wired in without breaking the FP8 path.
- **GDN path integration:** our custom GDN attention must coexist with
  upstream's attention backend selection.
- **Kernels pin:** the arc-b70 build script rebuilds kernels from main; we must
  pin to 0.1.12 to match the current venv (the build script needs a tweak or a
  post-build `pip install` of the 0.1.12 wheel).

---

## Recommendation

The re-port is worth doing to pick up the upstream improvements (K3 mamba
metadata 6.6-7.6x, state-copy race fix, 3D-grid tiling, XPU infra updates). But
it is a 5-10h focused effort, and the current venv is stable at ~53 tok/s.

**Defer the rebuild until the re-port is scheduled as its own effort.** The
current venv (`c39076fef`) is the production path; nothing is broken. When we
do proceed, follow the plan order above and pin kernels to 0.1.12.