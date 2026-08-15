# Heretic Abliteration Process — Qwen3.8-27B on 4× Arc Pro B70 (XPU)

A comprehensive, reproducible process to remove the refusal behavior ("abliteration")
from `Qwen3.8-27B` using the **Heretic** tool (`p-e-w/heretic`, `pip install heretic-llm`),
with a **small-model probe first** to de-risk the two unknowns before committing to the
long 27B run.

---

## 1. TL;DR

| Phase | What | Model | GPU need | Est. time |
|---|---|---|---|---|
| 0 | Isolated venv (XPU torch + heretic-llm) | — | none | ~5–10 min |
| 1 | **Probe** — validate XPU + `qwen3_5` arch | Qwen3.5-4B | 1 GPU | ~15–30 min |
| 2 | **Main** — abliterate the target | Qwen3.8-27B | 4 GPUs | ~6–10 h |
| 3 | Verify + (optional) re-quantize | Qwen3.8-27B-heretic | 1–4 GPU | ~10 min + 5.5 h |

**Gate:** Phase 2 only starts if Phase 1 succeeds (model loads on XPU, abliteration loop
runs, a decensored model is produced).

---

## 2. What Heretic does (recap)

Heretic is a **fully automatic directional-ablation** tool. It does *no* fine-tuning:

1. **Residual directions** — for each transformer layer, compute the difference-of-means
   between first-token hidden states on "harmful" (refusal-triggering) vs "harmless" prompts.
2. **Optuna TPE search** — over `n_trials` (default 200), try combinations of
   (direction index, ablation weight-kernel shape/position) and **co-minimize**
   refusal rate + KL-divergence from the original model.
3. **Orthogonalize** the attention out-projection and MLP down-projection matrices against
   the chosen direction(s) — implemented as a LoRA that is then merged into the weights.
4. **Export** the merged model (or LoRA adapter).

Result: refusals removed, with far less intelligence damage than manual abliteration
(Heretic's own numbers: KL 0.16 vs 0.45–1.04 for manual, same refusal suppression).

---

## 3. The two risks — and why a probe

Running Heretic on this box has two unverified unknowns:

1. **XPU compatibility.** Heretic is built around PyTorch on CUDA. Our GPUs are Intel Arc
   Pro B70 (XPU). PyTorch's XPU backend is mature (AutoRound just proved layer-wise work
   runs on it), but Heretic specifically is unverified.
2. **`qwen3_5` hybrid architecture.** Qwen3.8-27B is a 3:1 Gated-DeltaNet (linear attn) /
   full-attention hybrid. Heretic's README says it supports "some hybrid models like
   Qwen3.5" but not all research architectures.

A **probe on a small model of the same architecture** resolves both cheaply before we burn
6–10 hours on the 27B.

### Probe model: `Qwen/Qwen3.5-4B`

- **Same `qwen3_5` architecture** as the 27B (Gated DeltaNet + Gated Attention hybrid,
  MTP, thinking model) → the test is *representative*, not a proxy.
- **4B params (~8 GB BF16)** → fits on a single 34 GB GPU, runs fast.
- **Thinking model** → exercises Heretic's `chain_of_thought_skips` (`think` block handling).
- The Qwen3.5 family (0.8B/2B/4B/9B) is the smallest `qwen3_5` line; 4B is the sweet spot
  (big enough to be a real test, small enough to be quick).

> Note: the 4B probe uses **one** GPU, so it does *not* exercise multi-GPU `device_map`.
> That is a separate, smaller risk for the 27B (see §7 Fallbacks) — AutoRound already proved
> Accelerate's `device_map="auto"` works on this XPU stack.

---

## 4. Phase 0 — Isolated venv (XPU torch)

We do **not** install Heretic into the working vLLM venv (`/home/dc/electric-sheep/vllm/.venv`)
— Heretic pulls `transformers`, `accelerate`, `peft`, `bitsandbytes`, `optuna`, etc., and an
upgrade could destabilize the benchmarking environment. Instead: a throwaway venv.

**Why a separate venv + XPU torch:**
- The vLLM venv has `torch 2.13.0+xpu` (Intel build). A plain `pip install heretic-llm`
  would pull **CUDA** torch. We install the **XPU** torch first from Intel's index
  (`https://download.pytorch.org/whl/xpu`) so the venv is XPU-native.
- `heretic-llm` requires `transformers~=5.6` (we have 5.14.1 — compatible), `accelerate~=1.13`,
  `optuna~=4.7`, `peft~=0.19`, `bitsandbytes~=0.49`, `numpy~=2.2`.

**Script:** `heretic-setup-venv.sh` → creates `/home/dc/electric-sheep/vllm/heretic-venv`,
installs XPU torch then `heretic-llm`, and verifies `torch.xpu.is_available()` + `import heretic`.

This phase needs **no GPU** (pip + network only), so it can run while the INT4 quantization
finishes.

---

## 5. Phase 1 — Probe (Qwen3.5-4B)

**Script:** `heretic-probe-qwen3.5-4b.sh`

Steps:
1. Download `Qwen/Qwen3.5-4B` to `/home/dc/electric-sheep/models/Qwen3.5-4B` (if absent).
2. Run `heretic` with **headless CLI flags** (kebab-case). Every interactive prompt has a
   config equivalent; setting them all makes the run fully unattended (CLI has the highest
   config priority, so no `config.toml` is needed):
   ```bash
   heretic \
     --model /home/dc/electric-sheep/models/Qwen3.5-4B \
     --device-map auto \
     --quantization none \
     --n-trials 20 --n-startup-trials 10 \
     --seed 42 \
     --checkpoint-action restart \
     --trial-index 0 \
     --model-action save \
     --save-directory /home/dc/electric-sheep/models/Qwen3.5-4B-heretic-probe \
     --export-strategy merge
   ```
   - `--n-trials 20` (reduced — we only need a signal, not the optimum)
   - `--checkpoint-action restart` → skip the "continue / restart" prompt
   - `--trial-index 0` → pick the best (Pareto) trial, skip the "which trial" prompt
   - `--model-action save` + `--save-directory` + `--export-strategy merge` → save the
     merged model and exit, skipping the action + export prompts
3. Run under the XPU env (1 GPU).
4. Log to `/tmp/heretic-probe.log`.

**Pass criteria (all must hold):**
- [ ] Model loads on XPU without error (no CUDA-only op, no OOM on 1 GPU).
- [ ] Residual directions computed for all layers.
- [ ] Optuna study runs ≥ 1 trial to completion (abliterate + evaluate).
- [ ] A merged model is written to the save directory (safetensors + config.json).
- [ ] No crash in the `qwen3_5` GDN layer handling.

**Fail →** stop, diagnose (see §8), do **not** start the 27B.

---

## 6. Phase 2 — Main run (Qwen3.8-27B)

**Script:** `heretic-qwen3.8-27b.sh` (only run after Phase 1 passes)

Steps:
1. Confirm all 4 GPUs are free (no vLLM server, quant finished).
2. Run `heretic` with headless CLI flags (same pattern as the probe, full quality):
   ```bash
   heretic \
     --model /home/dc/electric-sheep/models/Qwen3.8-27B \
     --device-map auto \
     --max-memory 0=30GB 1=30GB 2=30GB 3=30GB \
     --quantization none \
     --n-trials 200 --n-startup-trials 60 \
     --seed 42 \
     --checkpoint-action restart \
     --trial-index 0 \
     --model-action save \
     --save-directory /home/dc/electric-sheep/models/Qwen3.8-27B-heretic \
     --export-strategy merge
   ```
   - `--max-memory 0=30GB 1=30GB 2=30GB 3=30GB` → spread the 52 GB model across the 4
     GPUs (Accelerate `device_map="auto"`), leaving headroom for activations.
     (If the CLI can't parse the dict form, fall back to a `config.toml` with
     `max_memory = { "0" = "30GB", "1" = "30GB", "2" = "30GB", "3" = "30GB" }`.)
   - `--n-trials 200` (full quality; override with `N_TRIALS` env in the script)
   - `offload_outputs_to_cpu` is `true` by default (cuts peak VRAM during residual analysis)
3. Run under the XPU env (all 4 GPUs visible).
4. Log to `/tmp/heretic-27b.log`.

**What to watch in the log:**
- `Loading good/bad prompts` — HF datasets `mlabonne/harmless_alpaca` / `mlabonne/harmful_behaviors`.
- `Determining optimal batch size` — auto-tunes throughput.
- `Calculating per-layer residual directions` — one pass over 400+400 prompts.
- `Running trial N of 200` — each prints `KeywordRate` + `KLDivergence`.
- `Optimization finished!` → best trial selected → `Model saved to ...`.

**Time estimate:** residual pass (~15 min) + 200 trials. Each trial generates ~200 short
responses (batched). At 27B batched throughput, expect **~6–10 h total**. It's resumable
(checkpointed Optuna study) — if interrupted, re-run with `CHECKPOINT_ACTION=continue`.

---

## 7. Phase 3 — Verify (+ optional re-quantize)

1. **Sanity-load** the abliterated model in the vLLM venv (same flags as the other 27B
   models, `--served-model-name qwen-256k`).
2. **A/B behavior:** a few prompts that the censored model refuses should now be answered;
   a few normal prompts (math/code) should be unchanged in quality.
3. **Optional — INT4:** run the existing `quantize-qwen3.8-27b-int4.sh` against the
   abliterated model (change `MODEL=` to the `-heretic` dir) to get an uncensored INT4.
   Same ~5.5 h, same GDN `in_proj_a/b` + `mtp.fc` exclusions.

---

## 8. Fallbacks / troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `pip install heretic-llm` replaces XPU torch with CUDA | dep resolution | Re-run `pip install torch==2.13.0+xpu torchvision==0.28.0+xpu --index-url https://download.pytorch.org/whl/xpu --force-reinstall` |
| `torch.xpu.is_available()` False in heretic venv | XPU torch not installed / env not sourced | Verify install; source `set-env-0123-gpu.sh` (sets `ZE_AFFINITY_MASK`, `ONEAPI_DEVICE_SELECTOR`) |
| Probe OOM on 1 GPU | vision encoder + activations | Set `quantization = "bnb_4bit"` in probe config (if bnb works on XPU) or reduce `max_batch_size` |
| `qwen3_5` / GDN layer error in probe | arch not supported | Check Heretic issues; try a text-only `qwen3_5` variant; or fall back to a dense model probe to at least validate XPU, then assess 27B risk separately |
| 27B `device_map="auto"` doesn't spread across 4 GPUs | Accelerate XPU device naming | Set explicit `max_memory` with correct XPU device keys; or load with `quantization="bnb_4bit"` on 1 GPU; or CPU-offload tail layers |
| 27B run too slow | `n_trials` too high | Re-run with `N_TRIALS=100` (halves time, minor quality cost) |
| Run interrupted | OOM / preemption | Re-run with `CHECKPOINT_ACTION=continue` (resumes the Optuna study) |

---

## 9. File manifest

| File | Purpose |
|---|---|
| `HERETIC-PROCESS.md` | This document |
| `heretic-setup-venv.sh` | Phase 0 — isolated XPU venv + heretic-llm |
| `heretic-probe-qwen3.5-4b.sh` | Phase 1 — small-model probe (gate) |
| `heretic-qwen3.8-27b.sh` | Phase 2 — main 27B abliteration |
| `heretic-venv/` | (created) isolated venv |
| `models/Qwen3.5-4B-heretic-probe/` | (created) probe output |
| `models/Qwen3.8-27B-heretic/` | (created) final abliterated 27B |
| `/tmp/heretic-probe.log`, `/tmp/heretic-27b.log` | (created) run logs |

> Heretic writes its Optuna study checkpoint to `./checkpoints/` relative to the cwd
> (the `vllm/` dir when the scripts run). That's what makes `CHECKPOINT_ACTION=continue`
> resumption work — don't delete it between runs of the same model.

---

## 10. Next steps (execution order)

1. **Now (no GPU):** run `heretic-setup-venv.sh` + download `Qwen3.5-4B` — in parallel with
   the finishing INT4 quantization.
2. **When quant finishes (~04:20):** verify the INT4 model, free the GPUs.
3. **Probe:** run `heretic-probe-qwen3.5-4b.sh`. Review against the pass criteria (§5).
4. **If pass:** run `heretic-qwen3.8-27b.sh` (overnight).
5. **When done:** verify (§7), optionally re-quantize to INT4.
