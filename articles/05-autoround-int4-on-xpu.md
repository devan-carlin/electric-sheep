# Quantizing a 27B Hybrid Model to 4-Bit on Intel Arc

*AutoRound W4A16 on a Gated-DeltaNet hybrid — and the specific layers you must
leave at 16-bit or the model falls apart.*

> **Status: outline.** The technical core is real and verified; this needs a
> final pass (numbers, before/after quality, a stronger hook) before posting.

---

## The hook

A 27B model is 52 GB in BF16. In INT4 it's ~18 GB — small enough to fit on two
32 GB GPUs with room for a 256K context. The catch: it's a *hybrid*
architecture (3:1 linear-attention / full-attention), and not every layer
survives 4-bit the same way. Quantize the wrong ones and the model degrades in
ways that are hard to see until you benchmark it.

## The recipe

- **Method:** AutoRound, W4A16, `group_size=128`, symmetric,
  `format=auto_round:auto_gptq` (vLLM-loadable).
- **Hardware:** 4× Intel Arc Pro B70. The 52 GB BF16 model is distributed
  across all four via `--device_map auto` during calibration.
- **Runtime:** 1–3 hours (calibration + per-layer GPTQ tuning).

## The non-obvious part: which layers stay 16-bit

The Gated-DeltaNet (linear attention) layers have small, numerically sensitive
gate projections. Quantizing them to 4-bit measurably hurts quality. The fix is
to **keep them at 16-bit**:

- `model.language_model.layers.N.linear_attn.in_proj_a` (48 layers)
- `model.language_model.layers.N.linear_attn.in_proj_b` (48 layers)
- `mtp.fc` (the multi-token-prediction projection)
- `lm_head` (left unquantized, matching the reference Intel model)

These are tiny tensors — a few MB each — so keeping them at 16-bit costs almost
nothing in memory but preserves the quality that 4-bit would destroy.

**The general lesson:** in a hybrid or unusual architecture, the *small*
projections are often the *sensitive* ones. A blanket "quantize everything to
4-bit" is the wrong default. Identify the numerically fragile layers and
exempt them.

## The XPU-specific patches

Running AutoRound on the XPU backend required a set of patches to bridge
AutoRound, transformers, and the Intel hardware. (Full inventory in
`../docs/vllm/INT4-QUANTIZATION-PATCHES.md`.) The recurring themes:

- **INT4 tensors can't be `nn.Parameter`** — the fused-MoE replacement path
  has to be no-op'd.
- **Multi-device tensor mismatches** — scale tensors and module moves assume a
  single device; XPU's multi-GPU layout breaks that assumption.
- **FP8 kernel dispatch** — some deepgemm FP8 paths don't autograd on XPU and
  must be disabled.

Each patch is small, but there are ~10 of them, and they're all in
`site-packages` — meaning they're **ephemeral** and lost on a venv rebuild.
(See the companion note on why `site-packages` patches are a trap.)

## Takeaway

- **4-bit is a 3× memory win, but only if you exempt the fragile layers.** In
  hybrid architectures, the small gate/projection tensors are the ones to keep
  at 16-bit.
- **Quantization on a non-CUDA backend is a patching exercise.** Budget time
  for the framework/hardware impedance mismatches, not just the quantization
  itself.
- **Catalog your `site-packages` patches.** They don't survive a rebuild. If
  you need them again, you want them written down.
