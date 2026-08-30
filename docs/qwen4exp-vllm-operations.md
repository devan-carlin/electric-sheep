# Qwen3.8-Flash-Next on vLLM / Intel Arc Pro B70 — Operations Guide

Companion to `qwen4exp-vllm-port.md` (design + architecture). This document provides instructions for clean rebuilds, bug logging, and serving.

## What this is

Serves `Qwen3.8-Flash-Next` (125B MoE / 6B active, W4A16) on 4x Intel Arc Pro B70 using a patched vLLM.
- **Endpoint**: `qwen-256k` on port 8000.
- **Configuration**: 256K context, fp8 KV, TP4 + expert parallel.

**Performance (Measured 2026-08-29, single stream, 512 tokens, greedy):**

- **53.4 tok/s decode** (median of 3 runs; TTFT ~0.11s).
- **llama.cpp baseline**: ~29 tok/s (~1.8x slower).
- **MTP speculative decoding**: Correct and lossless with 34% draft acceptance, but **unreliable on XPU** (see bug log). Do not use as the default.

## Hardware

- 4x Intel Arc Pro B70 (Xe3), 32GB each (**128GB total VRAM**).
- 247GiB host RAM (contains the 96GB PLE table).
- oneAPI Level-Zero 20.2.0.

## The stack (exact versions)

- vLLM `0.26.1rc1.dev500+gc39076fef` (venv `electric-sheep/vllm/.venv`).
- torch `2.13.0+xpu`.
- vllm-xpu-kernels `0.1.12` (pinned; `0.1.13.2` is broken).
- Python 3.12.

The venv is a **non-editable patched copy** of a clean upstream checkout (`vllm/vllm-src`, commit `c39076feff`). Local XPU patches are located in `site-packages/vllm`. Do **NOT** re-sync from `vllm-src` or you will overwrite the patches. Python edits in `site-packages` take effect upon the next server restart.

## Clean rebuild (from a fresh machine)

1. **oneAPI + venv**: Install oneAPI base + runtime. Create the venv, install torch `2.13.0+xpu`, vLLM, and **pin vllm-xpu-kernels to 0.1.12**.
2. **Apply local XPU patches**: Apply the `vllm/patches/` diffs to `site-packages/vllm`. These add W4A16 linear/MoE kernels and the GDN XPU forward path.
3. **Add the qwen4_exp port** (new files in `site-packages`):
   - `model_executor/models/qwen4_exp.py` (model + MTP drafter).
   - `transformers_utils/configs/qwen4_exp.py` (config).
   - Registry entries: `Qwen4ExpForCausalLM` / `Qwen4ExpMTP`.
   - `config/speculative.py` (architecture rewrite + `hc_count`/`hc_lowrank` promotion).
4. **Quantize from BF16 source**: Use `serve/bf16_to_w4a16.py` to create `devancarlin-Qwen3.8-Flash-Next-W4A16-BF16src` (77GB, 17 shards). The `quantization_config` ignore list keeps `mtp`, `ple`, `visual`, `linear_attn`, `shared_expert`, `gate`, `hyper_connection`, and `indexer` in BF16.
5. **Build the PLE table**: Run `phase_b_ple_table_prep.py` to generate `/mnt/data/ple_cache/ple_table_qwen4exp.pt` (96GB, mmap'd from host RAM, shared across all 4 ranks). Note: `QWEN4EXP_DISABLE_PLE=1` produces **incorrect** output.
6. **Launch** (see below).

## Launch

```
cd ~/electric-sheep/serve
setsid bash ./start-qwen256k-vllm.sh start > /tmp/vllm_baseline.log 2>&1 < /dev/null & disown
```

**Key flags (in `start-qwen256k-vllm.sh`):**
`--tensor-parallel-size 4 --enable-expert-parallel --dtype bfloat16 --max-model-len 262144 --max-num-seqs 4 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --generation-config vllm`.

**Environment variables:**
- `VLLM_XPU_ENABLE_XPU_GRAPH=1` (Graph mode is 6x faster than eager: 43 vs 7 tok/s).
- `UR_L0_SYNC_MODE=BLOCKING`.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

**Note on detachment**: Use `setsid ... < /dev/null & disown`. Using `nohup` in a shared persistent terminal can lead to process-group cleanup, which kills the server and breaks the web UI with "model not found".

**MTP (optional, not recommended on XPU):**
`QWEN256K_SPEC_FLAGS='--speculative-config {"method":"mtp","num_speculative_tokens":1}'`.

## Serving notes

- **Multimodal**: `Qwen4ExpForConditionalGeneration` is a real wrapper around the Qwen3-VL vision tower (`Qwen3_VisionTransformer`) + the `Qwen4ExpForCausalLM` text model. All 333 `visual.*` tensors load; images are served via the Qwen3-VL processor (`image_url` content parts). Text-only prompts work unchanged.
- **Degenerate tail fix**: At greedy (temperature 0), the model may emit a foreign-language degenerate tail. Set **temperature 0.7 + reasoning "high"** in the web UI to resolve this.
- **Open WebUI**: Use container `open-webui` (image `ghcr.io/open-webui/open-webui:main`), mapping port 3000->8080. Use a named volume `open-webui` to preserve data. Use `--add-host host.docker.internal:host-gateway` and `OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1`.
- **VRAM Management**: After any restart, verify that no orphan `EngineCore`/`Worker` processes are holding VRAM before relaunching.

## Bug log (issues hit and fixed)

1. **Broken vllm-xpu-kernels 0.1.13.2**: Caused "missing ops" errors. Fixed by pinning to 0.1.12 to ensure W4A16 dense/MoE kernels and GDN XPU forward are registered.
2. **OOM from missing `quantization_config`**: vLLM attempted to load int4 weights as bf16. Fixed by including the config in the checkpoint.
3. **MTP weight-load crash** (`no module named 'language_model'`): The drafter was fed the full checkpoint stream. Fixed by filtering the stream (`mtp.*` -> `model.*`) and applying the main model's stacked (qkv/gate_up) mapper.
4. **MTP compile `ConstraintViolationError`**: The compiled drafter forward combined two tensors with independent dynamic dim-0 symbols. Fixed by running the drafter in **eager** mode (removed `@support_torch_compile`). This has negligible cost (1 of 49 layers) while the target remains compiled and XPU-graphed.
5. **MTP 0% draft acceptance**: Pre-fc norms used raw gamma, but the model stores RMSNorm gammas as **deviations** (true gamma = `1 + w`). Fixed both pre-fc norms to `1 + w`, resulting in 34% acceptance.
6. **`hc_count` buffer bug**: The drafter's hidden buffer was sized using a missing top-level `hc_count`. Fixed by promoting `hc_count`/`hc_lowrank` from `text_config` to the top level of the draft config (buffer 2560 -> 10240).
7. **MTP unreliable on XPU (Accepted limitation)**: `RuntimeError: causal_conv1d does not support spec-decode and non-spec (prefill + decode) tokens in the same invocation` in `vllm/_xpu_ops.py:212 _gdn_attention_core_xpu_impl`. This is a pre-existing XPU GDN kernel limitation. MTP works until a step mixes spec/non-spec tokens in one GDN call. Fixing this requires a compiled-kernel change in the venv.

## MTP throughput verdict

MTP-on (eager drafter) achieves 44.2 tok/s, compared to the baseline of ~53 tok/s. While correct and lossless, the eager-drafter overhead offsets the 34% acceptance gain, making it net slower. **Keep MTP off by default on XPU.**

## Backlog

- **Multimodality**: Port the vision tower (Qwen3.5-vision-lineage encoder + image processor + MRoPE wiring).
- **QSA sparse indexer**: Implement a true backend (current version uses dense attention on the 12 QSA layers).
- **W8A16 variant**: `int8_gemm_w8a16` is not yet registered in 0.1.12.
- **Upstreaming**: Port `qwen4_exp`, XPU kernel work, and the GDN spec-decode fix.