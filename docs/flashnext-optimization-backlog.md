# Qwen3.8-Flash-Next (qwen4exp) — Optimization Project Backlog

Status: **active** (2026-08-26). Living doc — update the findings log and
backlog as work lands. Everything below is verified against the local tree
unless marked "estimate".

**How to pick this back up:**
1. Read §1 (current state) and §4 (findings log).
2. `bash ~/electric-sheep/serve/start-flashnext-llama.sh status` — confirm the
   server is up (:8090, alias `flash-next`).
3. Run the baseline bench (§7) to re-anchor the numbers.
4. Work the backlog top-down (§5). Tier 1 needs no code changes.

---

## 1. Current state (2026-08-26)

### Hardware
- 4× Intel Arc Pro B70 (Battlemage, PCI 8086:E223), 31.89 GiB VRAM each
  (127.56 GiB total). `xe` driver, renderD128-131 = SYCL0-3.
- 247 GiB RAM, Threadripper PRO 3945WX, Ubuntu 26.04 (kernel 6.8+ →
  device-accessible host memory, zero-copy PLE reads).
- **No GPU↔GPU P2P** (see F1). All cross-device traffic stages through RAM.

### Build
- Tree: `~/electric-sheep/llama/llama.cpp`, branch **pr-27742**
  (head `035e22731`, unmerged PR "model: add Qwen3.8-Flash-Next (qwen4exp)").
- SYCL build: IntelLLVM 2026.1.1, Release, `GGML_SYCL=ON`,
  DNN/F16/GRAPH/HOST_MEM_FALLBACK/LEVEL_ZERO_API=ON, TARGET=INTEL.
- Binary: `build/bin/llama-server`.
- **Never `git pull`/checkout master without re-checking out pr-27742 and
  rebuilding** (`cmake --build build -j$(nproc)`), or qwen4exp support is lost.
- Local b10355 patches (balanced split + DSV4 conversion) are stashed:
  `llama.cpp-local-patches-balanced-split-dsv4.patch`.

### Model
- `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL, 4 shards ~104 GB in
  `~/electric-sheep/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/`.
- 125B total / 6B active MoE + 51B PLE n-gram table + 4B MTP head (unwired).
- Key hparams: 48 layers, 262144 ctx, n_embd 2560, 24 Q heads / 2 KV heads,
  head dim 256, full attention every 4th layer (12 QSA layers, 36 Gated
  DeltaNet layers), 512 experts / 10 active, expert FFN 640.
- PLE table: 320,001,446 rows × 160 dim ≈ **28.7 GB** at Q4_K_XL, single
  indivisible tensor, used at layer 0 only.

### Running config (the default, ~27.3 t/s)
```
llama-server -m <UD-Q4_K_XL shard 1> \
  -ngl 99 -fa on -c 262144 -np 1 -b 2048 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  --jinja --reasoning on --alias flash-next \
  --host 0.0.0.0 --port 8090
```
- Env (from `~/electric-sheep/llama/set-env.sh` + launch script):
  `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`, `ZES_ENABLE_SYSMAN=1`,
  `GGML_SYCL_ENABLE_FLASH_ATTN/OPT/DNN/MKL_FA=1`,
  `GGML_SYCL_FA_ONEDNN_MAX_KV=65536`,
  `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`,
  `LLAMA_ATTN_ROT_DISABLE=1` (required for q8_0 KV, see F5).
- Layout: layer mode, default free-memory split across all 4 GPUs,
  PLE table in host RAM (zero-copy device reads).
- Launch: `bash ~/electric-sheep/serve/start-flashnext-llama.sh {start|stop|status}`.
  Log: `~/electric-sheep/serve/logs/llama_8090.log`.
  Toggles: `FLASHNEXT_PLE_DEV` (CPU|SYCL0..3), `FLASHNEXT_TSPLIT`,
  `FLASHNEXT_CTX`, `FLASHNEXT_PORT`, `FLASHNEXT_GPUS`.

### Measured performance
| Config | Decode | Notes |
|---|---|---|
| f16 KV, PLE in RAM | 28.95 t/s | prior session |
| q8_0 KV, PLE in RAM | **27.28 t/s** | current default (300 tok / 11.00 s) |
| q8_0 KV, PLE on GPU3 | 24.74 t/s | all-VRAM; slower, more CPU (F6) |
| prompt eval | ~50 t/s | |

- Bandwidth ceiling estimate: ~150-170 t/s (active weights ~3.4 GB/token
  spread over 4 GPUs). Current efficiency ~15-18%.
- CPU cost: ~36 CPU-seconds per 3 s of wall time during generation
  (12×, 40 threads) — kernel submission is a first-class cost.

---

## 2. Architecture primer (why this model is unusual)

qwen4exp breaks several assumptions the llama.cpp multi-GPU code makes:

- **Hyper-connections (HC)** replace every layer norm. The residual is `hc`
  parallel streams `[n_embd, hc, T]`. Each block: grouped RMSNorm → low-rank
  down/silu/up gate → per-stream inject. `build_hc_mix` in
  `src/models/qwen4exp.cpp` is ~10 small ops per call, called **twice per
  layer** (before attention, before MoE) → ~960 tiny ops/token just for HC.
- **PLE (per-layer token embeddings)**: n-gram hash table (28.7 GB) gathered
  at layer 0 via `get_rows`, then a small attention/conv block. The table is
  one tensor that cannot be split; its placement (RAM vs GPU) is the main
  layout decision (F6).
- **QSA (query-sparse attention)** on the 12 full-attention layers: an
  indexer (4 heads × 128 dim, top_k 2048) scores blocks of `compress_ratio`
  tokens; attention runs over the top blocks + tail. QKV weight has a
  **doubled Q dim** (`n_embd_head_k * n_head * 2` = 12288, q|gate
  interleaved per head) — this is what breaks tensor-split (F3).
- **Gated DeltaNet** on the other 36 layers: fixed-size recurrent state
  (not per-token), so KV-cache math only applies to the 12 QSA layers.
- **MoE**: 512 experts, 10 active, expert FFN 640. `mul_mat_id` path.
- **MTP head** (4B) exists in the model but the PR sets `no_mtp=True` —
  not wired into the graph.
- KV cache @262K: 12 QSA layers × (K+V 2048 B/token + indexer 1024 B/token)
  ≈ 9.7 GB f16 / ~4.9 GB q8_0. DeltaNet state is constant-size.

---

## 3. Performance model (where the time goes)

Per decode token, in order:
1. PLE gather (layer 0): `get_rows` into the 28.7 GB table. RAM-resident
   table = zero-copy device read (fast). GPU-resident table = host-staged
   cross-device copy (slow, F6).
2. 48 layers, each: HC mix → attention (QSA or DeltaNet) → HC mix → MoE.
   In layer mode the layers are distributed ~12 per GPU; each GPU handoff is
   a synchronous host-staged copy (GPU→RAM→GPU, ~50-100 µs + stall).
3. ~2000 small kernel submissions per token, all submitted from the CPU.
   The CPU is the pacing bottleneck (12× utilization).

The bottleneck is **latency, not bandwidth**: kernel launch overhead +
per-GPU pipeline syncs (GPU0→1→2→3 strictly sequential, no overlap) +
host-staged cross-device copies. This is why bandwidth-side levers (heavier
quant, KV quant) barely move decode, and why latency-side levers (draft
model, P2P, overlap, fusion) are the real targets.

---

## 4. Findings log (verified facts)

- **F1 — P2P is disabled in the SYCL backend.**
  `syclDeviceEnablePeerAccess` is commented out
  (`ggml/src/ggml-sycl/ggml-sycl.cpp:3042-3048`). Cross-device copies fall to
  the host-staged path (`ggml-sycl.cpp:700-730`): malloc → GPU→RAM → RAM→GPU,
  each with a synchronous `.wait()`. Whether the Arc Pro B70 + xe driver can
  actually do P2P is **untested** — see the probe in §7.3.
- **F2 — SYCL comm allreduce is N=2 only.**
  `ggml_backend_sycl_comm_allreduce_tensor`
  (`ggml-sycl.cpp:6470-6740`) handles exactly 2 backends, F32/F16 contiguous
  only. For N=4, `comm_init` returns null and the meta backend uses its
  generic butterfly fallback (`ggml/src/ggml-backend-meta.cpp`,
  `allreduce_fallback`): `ceil(log2(4)) = 2` steps × 4 cross-device copies
  per allreduce, each host-staged.
- **F3 — tensor mode crashes on a shape assert, root cause known.**
  `llama_meta_device_get_split_state` → `get_split_segments`
  (`src/llama-model.cpp:~655`) asserts
  `ne[axis] == n_embd + 2*n_embd_gqa` (= 2560 + 2×512 = 3584) for
  `attn_qkv.weight`. qwen4exp's QSA QKV has doubled Q heads
  (`src/models/qwen4exp.cpp:135`: `n_embd_head_k * n_head * 2` = 12288), so
  `ne[1] = 13312` and the assert fires. Fix is a small arch special-case
  (segments `{12288,1},{512,2}`, granularity 512/256) — but see §6: even
  fixed, tensor mode is ~5-10× slower than layer mode on this hardware.
- **F4 — row mode SIGSEGVs during load.** Undiagnosed SYCL backend bug in
  the split-buffer path (`ggml-sycl.cpp:1038-1300`,
  `ggml_backend_sycl_split_buffer_init_tensor` / `get_row_split`). Needs a
  core dump + debugger session. Low priority given F3/F2.
- **F5 — q8_0 KV crashes the QSA graph; workaround in place.**
  Quantized KV enables `attn_rot_k` (Hadamard K-rotation input,
  `src/llama-kv-cache.cpp:313-336`), which builds `self_k_rot`; the QSA path
  RoPEs K before caching and asserts `self_k_rot == nullptr`
  (`src/models/qwen4exp.cpp:544`). Fixed by `LLAMA_ATTN_ROT_DISABLE=1` in the
  launch script (scoped to flash-next, not shared `set-env.sh`, so other
  models' q4_0 KV keeps rotation). f16 KV never triggers it.
- **F6 — PLE placement: RAM beats GPU3.**
  PLE on GPU3 + layers on GPUs 0-2 (`--tensor-split 1,1,1,0` +
  `-ot per_layer_token_embd=SYCL3`): the PLE gather (layer 0 on GPU 0) copies
  GPU3→RAM→GPU0 per token (F1) → 24.74 t/s + more CPU. PLE in RAM:
  zero-copy device-accessible reads → 27.28 t/s. **RAM is the default.**
  This flips if P2P lands (F1).
- **F7 — CPU submission is a first-class cost.** ~12× CPU utilization during
  decode. Governor/pinning (T1.2) and op fusion (T2.3) both attack this.
- **F8 — CLI quirk:** this build registers `--device` (long form only) for
  the device list; `-dev`/`--dev` fail with "invalid argument".
- **F9 — oneDNN FA KV cap.** `GGML_SYCL_FA_ONEDNN_MAX_KV` (read at
  `ggml-sycl.cpp:313`) was 24576; raised to 65536 in `set-env.sh`. Above the
  cap, attention falls back to a slower path. Set to 0 (unlimited) if
  >64K-context workloads become common.
- **F10 — sampling (Unsloth recs):** thinking temp 1.0 / top_p 0.95 /
  top_k 20; instruct temp 0.7 / top_p 0.80 / presence_penalty 1.5.
  Set per-request. `reasoning_effort` via `--chat-template-kwargs`.

---

## 5. Optimization backlog (prioritized)

### Tier 1 — config only, no code changes

- **T1.1 — Draft-model speculative decoding. BIGGEST AVAILABLE LEVER.**
  `--draft-model <small gguf>` (e.g. a 4B Qwen GGUF). The big model verifies
  2-4 draft tokens per step in one pass; decode is latency-bound so this
  compounds. Realistic 1.5-2.5× if acceptance is good.
  - Steps: find/quantize a compatible draft (same tokenizer family), add
    `--draft-model` + `--draft-max-n` (try 3) to the launch script behind an
    env toggle, A/B with the §7 bench.
  - Watch: draft weights add VRAM; hybrid-attention (DeltaNet) draft
    support in this build is untested — verify the draft's recurrent state
    is handled (check `llama-speculative` / graph builder for qwen4exp
    target + draft arch pairing). If the PR's draft path doesn't cover
    qwen4exp as the *target*, this may need a small patch — log it in §4.
  - Effort: half a day. Impact: potentially 1.5-2.5×.
- **T1.2 — CPU governor + core pinning.**
  `performance` governor; pin the 4 SYCL submission threads to dedicated
  cores (isolcpus or taskset on the server pid). Attacks F7.
  - Effort: 30 min. Impact: 2-5% (estimate).
- **T1.3 — `GGML_SYCL_FA_ONEDNN_MAX_KV=0`** (unlimited oneDNN FA fast path).
  Only matters for >64K-context workloads. One-line change in `set-env.sh`.
- **T1.4 — q4_0 KV** (`-ctk q4_0 -ctv q4_0`). Halves KV reads again.
  Only matters at long context (at 262K filled, KV reads ≈ 4.8 GB/token ≈
  10 ms/token → would cap ~100 t/s). No-op at short context. Quality
  tradeoff — A/B output quality before adopting.

### Tier 2 — SYCL backend patches (local llama.cpp tree, rebuild required)

- **T2.1 — P2P probe + enable. STRUCTURAL LEVER; unblocks T2.2, T2.3, §6,
  and flips F6 (PLE back on GPU3).**
  - Step 1 (10-line test program): two SYCL contexts (SYCL0, SYCL1), call
    `ext_oneapi_can_access_peer` / attempt a peer USM memcpy. If the xe
    driver says no, this lever is dead and so is tensor mode.
  - Step 2: if yes, uncomment/fix `syclDeviceEnablePeerAccess`
    (`ggml-sycl.cpp:3042-3048`), rebuild, re-run the PLE-on-GPU3 A/B
    (expect 24.7 → ~29+ t/s) and the tensor-mode A/B (§6).
  - Effort: probe 1 h; enable + verify half a day. Impact: unknown until
    probed; potentially the largest structural win.
- **T2.2 — Pipeline overlap (double-buffered GPU handoffs).**
  Layer mode runs GPU0→1→2→3 strictly sequential; each token pays 3
  synchronous host-staged handoffs. Async copy + events: start GPU1's first
  layer while GPU0 finishes its last. Hides ~2 of 3 stalls.
  - Location: `ggml-sycl.cpp` copy path + the scheduler's per-device
    submission. Moderate work.
  - Effort: 1-2 days. Impact: 10-20% (estimate).
- **T2.3 — Op reduction / kernel fusion.**
  The HC mixer alone is ~10 small ops × 2 × 48 layers; the PLE path adds
  gather + conv + n-gram attention. Each op = a full SYCL submission from
  the CPU (F7). Fusing `build_hc_mix` (norm → lora → silu → sigmoid → mul →
  stream collapse) into 1-2 kernels cuts launch latency and CPU cost.
  - Location: `src/models/qwen4exp.cpp` graph builder + new SYCL kernels.
    Deepest backend work of the three.
  - Effort: days. Impact: 10-30% (estimate).
- **T2.4 — N=4 comm allreduce.**
  Extend `ggml_backend_sycl_comm_allreduce_tensor` beyond N=2 (ring or
  butterfly over the existing dev2dev path). Only worth it if tensor mode
  is revived after T2.1. Parked until then.

### Tier 3 — PR / upstream dependent (nothing to do locally yet)

- **T3.1 — MTP head wiring. UNBLOCKED 2026-08-28.** PR #27742 merged
  (qwen4exp MTP wiring); local tree is on `pr-27742` with a build. The 4B
  MTP head is in the model. Self-speculative decoding (no separate draft
  model) is now the cleanest version of T1.1 — promote to Tier 1 and run.
- **T3.2 — mmproj / vision.** Vision tower (Qwen3-VL ViT) is in the PR; no
  projector file published yet. Feature, not speed.
- **T3.3 — Upstream qwen4exp fixes.** Any graph-builder simplification or
  SYCL-backend improvement merged into the PR (or into master once it lands)
  is free speed. Rebase + rebuild + re-bench after each PR update.

### Dead ends (documented — do not retry without new info)

| Item | Verdict | Evidence |
|---|---|---|
| Tensor parallelism | ~5-10× slower than layer mode | §6 full analysis |
| Row split | SIGSEGV during load, undiagnosed | F4 |
| PLE on GPU3 (no P2P) | 24.7 vs 27.3 t/s | F6, A/B measured |
| Heavier weight quant (Q3/IQ3) | decode is latency-bound, no gain | §3 |
| Bigger `-ub` | only affects prompt eval | measured |
| f16 vs q8_0 KV (short ctx) | 28.95 vs 27.28 t/s — q8_0 keeps the RAM headroom for PLE; f16 only wins at short ctx where KV is irrelevant | A/B measured |

---

## 6. Tensor parallelism deep-dive (the honest answer)

**Can it be fixed? Yes. Should it be? No — not on this hardware.**

1. **The crash is a shape mismatch, not a deep bug.** `get_split_segments`
   (`src/llama-model.cpp:~655`) asserts `ne[axis] == n_embd + 2*n_embd_gqa`
   (= 3584) for `attn_qkv.weight`; qwen4exp's doubled Q heads make
   `ne[1] = 13312` (F3). A ~15-line arch special-case in
   `get_split_segments` + `get_split_granularity` (Q segment 12288 @
   granularity 512, K/V 512 @ 256) fixes the assert.
2. **Only 2 KV heads.** K/V can't split across 4 GPUs (512 / 4 = 128, not a
   multiple of the 256 head dim) → K/V mirrored, only Q actually splits.
3. **Allreduce cost dominates.** N=4 → generic butterfly fallback (F2):
   8 host-staged cross-device copies per allreduce. Every split matmul
   (QKV, attn out, indexer projs, HC loras ×2, MoE gate/up/down) needs one:
   ~8-15 per layer × 48 layers = several hundred per token → ~3000-4000
   host-staged copies/token → ~150-400 ms of pure communication per token.
   Ceiling: **< 5 t/s** vs 27.3 t/s for layer mode.
4. **Verdict:** layer mode (current default) is the optimal layout for this
   hardware. Tensor mode becomes viable only if T2.1 (P2P) lands AND T2.4
   (N=4 allreduce) is written.

---

## 7. Verification recipes

### 7.1 Baseline bench (decode t/s)
```bash
# server must be running: bash ~/electric-sheep/serve/start-flashnext-llama.sh status
bash ~/electric-sheep/serve/bench-flashnext.sh layer
# or manually: 300-token generation, time it
curl -s http://127.0.0.1:8090/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"Write a short story about a lighthouse."}],"max_tokens":300,"temperature":1.0,"top_p":0.95,"top_k":20}'
```
Reference: 300 tok / 11.00 s = 27.28 t/s (q8_0 KV, PLE in RAM).

### 7.2 A/B methodology
- One variable at a time; restart the server between configs.
- Warm up (20-token generation) before timing.
- 300-token runs; report t/s from the API response timing or the log.
- Check output quality on any KV/quant change (F5/T1.4).
- Record results in §1's table and in `serve/bench_results/`.

### 7.3 P2P probe (T2.1 step 1)
Minimal SYCL program: create contexts for SYCL0 and SYCL1, query
`ext_oneapi_can_access_peer` (or attempt a small peer USM memcpy +
`wait()`). Run with `source ~/electric-sheep/llama/set-env.sh 0,1`.
Result decides whether T2.1/T2.4/§6 are alive or dead.

### 7.4 CPU cost check
```bash
# during a 300-token generation:
pid=$(pgrep -f "llama-server.*--port 8090" | head -1)
top -H -p $pid -bn1 | head -50   # per-thread CPU
```
Reference: ~36 CPU-s per 3 s wall (12×).

---

## 8. Watchlist (things that may improve naturally)

Give it a few days and re-check:
- **PR #27742 activity** — MTP wiring (T3.1), qwen4exp graph fixes,
  reviewer-driven simplifications. Rebase + rebuild + re-bench on updates.
- **llama.cpp master SYCL backend** — N>2 comm allreduce, pipeline overlap,
  P2P enablement upstream. Once the PR merges, master changes apply.
- **xe driver updates** — P2P support on Battlemage would flip F1/F6/T2.1.
- **oneDNN updates** — flash-attention kernel improvements on Arc.
- **New GGUF quants** — unsloth may publish better Q4_K_XL variants or
  PLE-specific quantization (the 28.7 GB table is the single biggest
  memory consumer; a PLE-only lower-quant build would free RAM/VRAM).
- **Speculative decoding for hybrid models** — upstream work on draft
  support for DeltaNet/hybrid targets would de-risk T1.1.

---

## 9. File map

| Path | What |
|---|---|
| `~/electric-sheep/serve/start-flashnext-llama.sh` | launch script (toggles in header) |
| `~/electric-sheep/serve/bench-flashnext.sh` | split-mode bench harness |
| `~/electric-sheep/serve/logs/llama_8090.log` | server log |
| `~/electric-sheep/llama/set-env.sh` | SYCL env (MAX_KV=65536 etc.) |
| `~/electric-sheep/llama/llama.cpp` | pr-27742 tree (do not checkout master) |
| `~/electric-sheep/llama/llama.cpp-local-patches-balanced-split-dsv4.patch` | stashed local patches |
| `~/electric-sheep/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/` | model shards |
| `~/electric-sheep/serve/SERVICES.md` | service inventory (canonical; `~/docs/server-services.md` deleted 2026-08-30) |
| `~/electric-sheep/docs/guides/llama-deployment.md` | general llama.cpp SYCL guide |
| `~/electric-sheep/BACKLOG.md` | personal cross-repo backlog (pointer added) |

### Key source locations (llama.cpp tree)
| Location | What |
|---|---|
| `ggml/src/ggml-sycl/ggml-sycl.cpp:3042-3048` | P2P enable (commented out) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:700-730` | dev2dev memcpy (host-staged fallback) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:1038-1300` | split buffer (row mode, F4 suspect) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:6470-6740` | comm allreduce (N=2 only) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:313` | `GGML_SYCL_FA_ONEDNN_MAX_KV` read |
| `ggml/src/ggml-backend-meta.cpp` (~2300) | butterfly allreduce fallback |
| `src/llama-model.cpp:367-830` | `llama_meta_device_get_split_state` (F3) |
| `src/models/qwen4exp.cpp:135` | QKV shape (doubled Q dim) |
| `src/models/qwen4exp.cpp:~170-230` | `build_hc_mix` (fusion target, T2.3) |
| `src/models/qwen4exp.cpp:544` | QSA `self_k_rot` assert (F5) |
| `src/llama-kv-cache.cpp:313-336` | `attn_rot_k` logic + `LLAMA_ATTN_ROT_DISABLE` |