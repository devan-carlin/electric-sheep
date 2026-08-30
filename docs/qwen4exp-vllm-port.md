# Project: qwen4_exp on vLLM XPU (Qwen3.8-Flash-Next)

Goal: serve `VnimanieAI/Qwen3.8-Flash-Next-W4A16` (125B MoE, W4A16) via vLLM
on 4x Arc Pro B70 as `qwen-256k` (port 8000, 256K ctx, fp8 KV).
Baseline to beat: llama.cpp port 8090 (~27 t/s, single stream).
vLLM value prop: prefix caching, concurrency, xgrammar — not raw speed.

Status: COMPLETE (serving on 4x B70; see qwen4exp-vllm-operations.md).
Companion issue: intel/llm-scaler#649.

## Why vLLM (and why it was blocked)

- `qwen4_exp` arch absent from vLLM (installed + upstream main @ c39076feff).
- Phase 0 (done) proved the kernel layer is NOT the blocker:
  - W4A16 dense GEMM (oneDNN `int4_gemm_w4a16`) — verified on B70.
  - W4A16 MoE grouped GEMM (CUTLASS-SYCL `cutlass_grouped_gemm_interface`) — verified on B70.
  - GDN linear attention has an XPU forward path (`forward_xpu`).
  - Root cause of "missing ops" was a broken custom vllm-xpu-kernels 0.1.13.2
    build; the pinned 0.1.12 wheel registers everything.
- Remaining work is Python: model definition + PLE offload + registration.

## Architecture facts (from config.json + weight index + llama.cpp PR #27742)

Text model (`Qwen4ExpForConditionalGeneration`, text_config model_type `qwen4_exp_text`):
- 48 layers: 36 `linear_attention` (GDN) + 12 `full_attention` (QSA),
  `full_attention_interval: 4` (layer_types list in config).
- hidden 2560, vocab 248320, max_pos 262144, bf16.
- Full attention: q_proj [12288, 2560] = 48 heads x 256 (q+gate interleaved per
  head, like Qwen3Next), k/v [512, 2560] = 2 heads x 256, o_proj [2560, 6144]
  = 24 heads x 256. q_norm/k_norm RMSNorm over head_dim 256.
  RoPE: partial_rotary_factor 0.25 (first 64 of 256 dims), mrope_interleaved,
  mrope_section [11, 11, 10], theta 1e7.
  Output gate: sigmoid (config `output_gate_type: sigmoid`).
  QSA indexer (12 layers): index_qk_proj [640, 2560] (4 q heads x 128 + 1 k
  head x 128), q/k_layernorm [128]; sparse block selection
  (indexer_budget 2048, compress_ratio 4). v1: use DENSE attention instead.
- GDN linear attention (36 layers): in_proj_qkv [10240, 2560]
  (q 2048 + k 2048 + v 6144: 16 k-heads x 128, 48 v-heads x 128),
  in_proj_z [6144, 2560], in_proj_a [48, 2560], in_proj_b [48, 2560],
  conv1d [10240, 1, 4], A_log [48], dt_bias [48], norm [128] (RMSNormGated,
  sigmoid gate), out_proj [2560, 6144]. Matches Qwen3.5 GDN layout
  (separate in_proj_qkv/z/b/a, gqa_interleaved_layout=False).
- MoE (all 48 layers): 512 experts, top-10, moe_intermediate 640,
  shared_expert_intermediate 640 + shared_expert_gate [1, 2560] (sigmoid).
  Expert weights W4A16 int4 group-128 (weight_packed int32 + weight_scale bf16
  + weight_shape int32). Shared expert / router / GDN / attention stay BF16
  (quant ignore list).
- Hyper-connections (NEW, no vLLM equivalent): state is `hc_count=4` parallel
  residual streams [T, 2560, 4] (hc_dim 10240). Per block:
  grouped RMSNorm over each stream (gamma [10240], folded 1+w) ->
  low-rank down [320, 10240] -> silu(x/hc) -> up [10240, 320] -> sigmoid gate
  -> x*gate -> mean-collapse streams -> mixed [T, 2560] feeds the block;
  block output written back via inject [4, 10240] weights:
  w = 2*sigmoid(inject/hc); new_state = state + block_out * w.
  Two HC modules per layer (attn_hyper_connection, mlp_hyper_connection) +
  top-level hyper_connection_mixer (acts as the final output norm; there is
  NO separate final RMSNorm).
- PLE (layer index 1 only, ple_layer_ids=[2] is 1-based): n-gram hash table.
  128 shards x ~2.5M rows x 160 cols bf16 = ~51.2B params (~102GB).
  16 heads (ngram_size 3 -> (3-1)*8 heads), per-head offsets/vocab_sizes (i64),
  layer_multipliers (i64, ~2^45). Row index per token:
  mixed = t0*m0 ^ t1*m1 ^ t2*m2 (uint64, EOS resets window);
  row = mixed % vocab[h] + offset[h]. Gather 16 rows -> emb [T, 16*160] ->
  key_proj [10240, 2560] (on hc_dim!) / value_proj [2560, 2560] ->
  grouped norms -> per-stream dot gate (signed sqrt + sigmoid) ->
  dilated depthwise conv1d (kernel 4, dilation 3) -> add to hidden.
  MUST live in host RAM (102GB > 128GB GPU pool once weights are in).
  Per-step gather is tiny (16 rows/token) — compute indices host-side,
  gather on CPU, ship result to GPU.
- MTP head (1 layer, BF16, fused gate_up_proj [512, 1280, 2560]):
  speculative-decoding only. SKIP in v1 (llama.cpp also runs no_mtp).
- Vision tower (27 blocks): added after v1 — the Qwen3-VL ViT is reused verbatim; the serving recipe is now multimodal.

## v1 scope decisions

- Text-only (no vision), no MTP, dense attention on the 12 QSA layers
  (indexer = approximation; true sparse indexer is a later phase).
- PLE table in host RAM, per-step CPU gather.
- TP=4 + expert parallelism (640/TP breaks group-128; EP gives 128
  experts/GPU). KV cache fp8 (XPU equivalent of llama.cpp q8_0).
- Not bit-exact vs llama.cpp (dense vs sparse indexer) — acceptable.

## Task breakdown (todo list)

Phase A — skeleton (forward pass runs, PLE stubbed to zero):
1. `Qwen4ExpConfig` (+ text/vision split) in `transformers_utils/configs/`,
   register in configs `__init__.py`.
2. `HyperConnectionMixer` module (mix + combine + head norm).
3. `Qwen4ExpAttention` (full attn: fused q+gate, GQA 24:2, q/k-norm,
   partial MRoPE, sigmoid gate, dense attention).
4. GDN layer: reuse `QwenGatedDeltaNetAttention` (gqa_interleaved_layout=False,
   output_gate_type=sigmoid) + weight remap in_proj_qkv/z/b/a.
5. MoE block: reuse `Qwen3NextSparseMoeBlock` (W4A16 via compressed-tensors).
6. `Qwen4ExpDecoderLayer` (HC mix -> attn -> HC combine -> HC mix -> MoE ->
   HC combine) + `Qwen4ExpModel` (embed -> 48 layers -> head HC mix -> lm_head).
7. `Qwen4ExpForCausalLM` + registry entry + `packed_modules_mapping`
   (q/k/v_proj separate in ckpt; in_proj remap; gate/up not stacked in ckpt).
8. Load checkpoint, iterate on weight-loading errors.
9. Smoke test: short generation, sanity-check coherence.

Phase B — PLE host-RAM offload:
10. PLE table loader: keep shards in pinned host RAM (skip GPU load via
    skip_prefixes / custom loader).
11. Host-side n-gram row-index computation (per request, incremental across
    chunks; EOS window reset; predecessor history like llama.cpp).
12. CPU gather of 16 rows/token -> to GPU -> key/value proj + gate + conv.
13. Verify layer-1 output matches a CPU reference on a fixed prompt.

Phase C — serve + tune:
14. Launcher script (XPU env pattern from start-qwen.sh), served name
    `qwen-256k`, port 8000, TP=4 + EP, fp8 KV, 256K ctx.
15. Throughput/quality comparison vs llama.cpp baseline (port 8090).
16. Update issue #649 + feature-request doc with final status.

Later (out of v1 scope): true QSA sparse indexer backend, MTP speculative
decoding, vision tower, W8A16 variant (int8_gemm_w8a16 not registered in
0.1.12).

## Key files

Edit location (running, patched copy — edits take effect on next server start):
- Model: `~/vllm-fresh-venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen4_exp.py`
- Config: `~/vllm-fresh-venv/lib/python3.12/site-packages/vllm/transformers_utils/configs/qwen4_exp.py`
- Registry: `~/vllm-fresh-venv/lib/python3.12/site-packages/vllm/model_executor/models/registry.py`
- Configs `__init__`: `~/vllm-fresh-venv/lib/python3.12/site-packages/vllm/transformers_utils/configs/__init__.py`

Reference / version control:
- The venv is a clean build from upstream `c39076fef` + the 16-file
  `vllm/patches/qwen4exp-xpu-port.patch`. The patch is the source of truth;
  re-apply it (or rebuild via `vllm/setup-vllm-xpu.sh`) rather than editing
  site-packages by hand. `vllm/vllm-src/` (clean checkout) was deleted
  2026-08-30 — re-clone if a clean tree is needed for diffing.
- Base refs: `qwen3_next.py` (MoE block, attention, model shell),
  `qwen3_5.py` (GDN wiring + multimodal wrapper pattern),
  `qwen_gdn_linear_attn.py` (GDN layer, XPU forward)
- Op map ground truth: `llama/llama.cpp/src/models/qwen4exp.cpp`
- Checkpoint: `models/VnimanieAI-Qwen3.8-Flash-Next-W4A16/`
- Phase 0 tests: `vllm/tests/phase0_w4a16_test.py`, `vllm/tests/phase0_w4a16_moe_test2.py`

## Environment notes

- venv: `~/vllm-fresh-venv/` (vLLM `0.1.dev1+gc39076fef`, torch 2.13.0+xpu,
  vllm-xpu-kernels 0.1.12 installed). The original `vllm/.venv` (0.26.1rc1)
  was deleted 2026-08-30.
- Install: clean build from upstream `c39076fef` + `qwen4exp-xpu-port.patch`
  (see `vllm/setup-vllm-xpu.sh`). Python edits in site-packages take effect on
  the next server start (no reinstall).
- XPU env: ZE_AFFINITY_MASK + ONEAPI_DEVICE_SELECTOR (filtered space),
  UR_L0_SYNC_MODE=BLOCKING, VLLM_WORKER_MULTIPROC_METHOD=spawn,
  VLLM_XPU_ENABLE_XPU_GRAPH=1 (see serve/fallback/start-qwen.sh).
- Watch items: 0.1.12 reports B70 (Xe3) as is_xe2_arch=True (kernels work
  regardless); int8_gemm_w8a16 has 0 schemas in 0.1.12.

## Log

- 2026-08-27: Phase 0 complete (W4A16 dense + MoE grouped verified on B70;
  root cause = broken 0.1.13.2 wheel, fixed by pinned 0.1.12). Issue #649
  updated. Port started: config + model skeleton (Phase A).
- 2026-08-30: Port complete and serving on 4x B70 (53.4 tok/s). Vision tower
  added (Qwen3-VL ViT reused verbatim) — recipe is now multimodal. MTP drafter
  wired (correct/lossless, 34% acceptance; kept off by default on XPU). Shipped:
  16-file patch, fork branch `devan-carlin/vllm @ xpu-qwen4exp`, HF model card,
  status posted to intel/llm-scaler#649. -2 () variant
  quantized + served + verified (refusal 0/66).