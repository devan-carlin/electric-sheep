# Removing a Model's Refusals Without Fine-Tuning — on Consumer Intel GPUs

*Abliteration is a surgical strike on a model's weights. We ran it on a 27B
hybrid model across four Intel Arc GPUs, and de-risked it with a 4B probe first.*

---

Every capable LLM has a refusal behavior: a region of its weight space that,
when a prompt trips it, produces "I can't help with that." **Abliteration**
removes that behavior by editing the weights directly — no fine-tuning, no
training data, no gradient descent. The result is a model that answers
questions the original would have refused.

The tool we used is **Heretic** (`p-e-w/heretic`). It's a *directional*
ablation, which makes it far less destructive than the older manual approach.
But it was built around PyTorch on CUDA. Our GPUs are Intel Arc Pro B70 (XPU).
And our model is a hybrid architecture that Heretic only *partially* claims to
support.

So before we burned 6–10 hours on the 27B, we built a probe.

## What Heretic actually does

It does not fine-tune. It finds a *direction* in the model's activation space
that corresponds to refusal, and removes it.

1. **Residual directions.** For each transformer layer, compute the
   difference-of-means between first-token hidden states on *refusal-triggering*
   prompts vs. *harmless* prompts. That difference is a vector pointing "toward
   refusal."
2. **Search.** An Optuna TPE loop (default 200 trials) searches over which
   direction index to ablate and the shape/position of the ablation weight
   kernel, **co-minimizing** two objectives: refusal rate *and* KL divergence
   from the original model. The KL term is what keeps it from lobotomizing the
   model.
3. **Orthogonalize.** The chosen direction is removed by orthogonalizing the
   attention out-projection and the MLP down-projection against it —
   implemented as a LoRA that is then **merged** into the weights.
4. **Export.** The merged model (or the LoRA adapter) is written out.

The headline number from Heretic's own evaluation: **KL divergence 0.16** vs.
**0.45–1.04** for manual abliteration, at the same level of refusal
suppression. Same censorship removed, a third of the intelligence damage.

## The two unknowns — and why a probe

Running this on our box had two things we had not verified:

1. **XPU compatibility.** Heretic is built around CUDA. PyTorch's XPU backend
   is mature — we'd already run AutoRound quantization on it — but Heretic
   specifically was untested.
2. **The `qwen3_5` hybrid architecture.** Our target, Qwen 3.8 27B, is a 3:1
   Gated-DeltaNet (linear attention) / full-attention hybrid with an MTP layer.
   Heretic's README says it supports "some hybrid models like Qwen3.5" — not
   all research architectures.

Both unknowns are cheap to resolve with a **small model of the same
architecture**: **Qwen 3.5 4B**. Same `qwen3_5` hybrid, same MTP, same thinking
model — but 4B parameters (~8 GB in BF16), so it fits on a single 32 GB GPU and
runs in 15–30 minutes instead of 6–10 hours.

The probe is *representative*, not a proxy. If the architecture is the problem,
the 4B hits it. If the XPU backend is the problem, the 4B hits it. Either way,
we find out in half an hour instead of half a day.

## The pipeline

| Phase | What | Model | GPU | Time |
|-------|------|-------|-----|------|
| 0 | Isolated venv (XPU torch + heretic-llm) | — | none | 5–10 min |
| 1 | **Probe** — validate XPU + `qwen3_5` | Qwen 3.5 4B | 1 | 15–30 min |
| 2 | **Main** — abliterate the target | Qwen 3.8 27B | 4 | 6–10 h |
| 3 | Verify + re-quantize | Qwen 3.8 27B-heretic | 1–4 | 10 min + 5.5 h |

**Gate:** Phase 2 only starts if Phase 1 produces a working decensored model.
No probe, no 27B run.

## Why the gate matters

The 27B run is the expensive one. It loads a 52 GB BF16 model across four GPUs
and runs the Optuna search for hours. If it's going to fail, you want it to
fail on the 4B, where "failure" costs you a coffee, not a workday.

This is a general pattern worth stealing: **when you have two unverified
unknowns and an expensive operation, build the smallest thing that exercises
both unknowns, and gate the expensive thing on it.** The probe is not a
shortcut around the real work. It's the real work, made cheap.

## Takeaway

- **Abliteration is weight surgery, not training.** It removes a behavior by
  editing directions in activation space, and the KL-divergence term is what
  keeps the model coherent.
- **De-risk with a same-architecture small model.** A 4B probe of the same
  hybrid architecture resolves both the backend and the architecture unknowns
  in 30 minutes.
- **Gate expensive runs on cheap probes.** The gate is the whole point. If the
  probe fails, you saved a day. If it passes, you're confident enough to spend
  it.
