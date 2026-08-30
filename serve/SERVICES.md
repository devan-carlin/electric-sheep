# AI Server Service Map

4x Intel Arc Pro B70 (32 GB, XPU/Level Zero). One service per GPU.
Last updated: 2026-08-28.

## Active services

| Port | GPU | Service | Model | Context | Use for |
|------|-----|---------|-------|---------|---------|
| 8188 | 0 | ComfyUI A | (image models) | - | Image generation, workflow A |
| 8189 | 1 | ComfyUI B | (image models) | - | Image generation, workflow B |
| 8088 | 2 | llama.cpp | Qwen3.6-35B-A3B Aggressive ( Q4_K_P, MoE) | 256K | Book writing, long-context prose |
| 8089 | 3 | llama.cpp | Gemma4 26B-A4B ( Balanced Q4_K_P) | 256K | VN writing, daily chat |

Open WebUI (Docker) points at `host.docker.internal:8089` and `:8088`.
Model IDs in WebUI: `gemma`, `qwen` (version-agnostic on purpose - swap the
model in the launcher, never the name).

## Why these choices

### 8088 - Qwen (llama.cpp, was vLLM)

- Served name `qwen` (version-agnostic). Current model: **Qwen3.6-35B-A3B
  Aggressive** ( Q4_K_P, MoE 35B total / 3B active) - swapped in from
  Qwen3.8-27B dense on 2026-08-25 for MoE decode throughput.
- Switched from vLLM -ara (2026-08-25). vLLM capped at 128K on one
  card; llama.cpp runs the full 256K on-GPU (q4_0 KV ~4.8 GiB, ~5 GB spare).
- Config: q4_0 KV, 256K ctx, np=1, thinking ON (`--reasoning on
  --reasoning-format deepseek --reasoning-budget 2048`). Thoughts ->
  `reasoning_content`, answer -> `content`.
- Trade-off vs vLLM: no prefix caching, no tool-call parsers, no concurrency.
  vLLM fallback: `fallback/start-qwen.sh` (-ara int4, 128K, fp8 KV).

### 8089 - Gemma (llama.cpp)

- Served name `gemma` (version-agnostic). Current model: **Gemma4 26B-A4B**
  ( Balanced Q4_K_P).
- Won a blind prose A/B over the 31B QAT MTP (4/5 prompts) and decodes
  ~2.5-3x faster. Locked in 2026-08-25.
- Config: q4_0 KV, 256K ctx, np=1. KV ~16.9 GiB - top of the on-GPU budget,
  small CPU spill. Thinking ON, same flags as the qwen slot (uniform
  `reasoning_content`/`content` split on both endpoints).
- vLLM 31B (llmfan46-gemma-4-31b-qat-int4) remains a manual fallback for
  >256K context or MTP speculative decoding.

### 8188/8189 - ComfyUI

- Two independent instances for parallel image work.
- Managed by `start-all.sh` (kills old instances on start).

### Qwen3.8-Flash-Next (4x GPU, on demand)

- 125B MoE / 6B active + PLE n-gram table. Served two ways, both 4x GPU:
  - **vLLM** `:8000` alias `qwen-256k` (W4A16). The engine with the HC + PLE
    `1 + w` gamma fixes (2026-08-28). Runs in XPU graph mode
    (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, no `--enforce-eager`): PLE does host-side
    n-gram hashing with GPU->CPU syncs, but vLLM's piecewise capture
    (FULL_AND_PIECEWISE) breaks the graph only at PLE and captures the other
    47 layers. Measured 2026-08-28: 7.0 tok/s eager vs 53.2 tok/s graph in
    production (7.6x). Cost: ~2-4 min graph compile at startup. vLLM's
    "XPU Graph ... single-GPU only" warning is a red herring; works at TP4.
  - **llama.cpp** `:8090` alias `flash-next` (Q4_K_XL). PLE table in host RAM.
- One front door: `start-qwen-256k.sh` (`start|stop|restart|status|smoke|logs`).
  It delegates to the two launchers above; env overrides pass through
  (`QWEN256K_*` / `FLASHNEXT_*`).
- Not in the always-on service map: it needs all 4 GPUs, so it runs on demand
  and is stopped before the per-GPU slots (8088/8089) are used.

## Launch scripts

| Script | What it does |
|--------|--------------|
| `start-all.sh` | Start/stop/status all 4 services. Subcommands: `start`, `stop`, `status`, `restart-gemma`, `restart-qwen` |
| `start-qwen-256k.sh` | **Front door for Qwen3.8-Flash-Next.** Delegates to the vLLM + llama launchers; adds unified `status`/`smoke`/`logs`/`restart`. `start [vllm\|llama]`, `stop [vllm\|llama\|all]`, `status`, `smoke`, `logs` |
| `start-qwen-256k-vllm.sh` | Flash-Next via vLLM, :8000, alias `qwen-256k` (W4A16, 4x GPU, 256K, fp8 KV, XPU graph mode). The engine with the HC + PLE `1 + w` gamma fixes |
| `start-qwen-256k--vllm.sh` | Flash-Next **-2** () via vLLM, :8000, alias `qwen-256k` (W4A16 from BF16, 4x GPU, 256K, fp8 KV). Same flags as the base launcher; needs all 4 GPUs, so stop `qwen-256k` first |
| `start-flashnext-llama.sh` | Flash-Next via llama.cpp, :8090, alias `flash-next` (Q4_K_XL, 4x GPU, q8_0 KV, PLE table in RAM) |
| `start-gemma-llama.sh` | Gemma slot on GPU 3 :8089, alias `gemma` (start/stop/status) |
| `start-qwen-llama.sh` | Qwen slot on GPU 2 :8088, alias `qwen` (start/stop/status). Model pinned in the script; swap via QWEN_LLAMA_MODEL_DIR/MODEL_FILE |
| `fallback/start-qwen.sh` | vLLM Qwen -ara fallback (128K, fp8 KV) |
| `bench-throughput.sh` | Throughput benchmarks (GEMMA_TOK path may be stale) |

Env overrides: `QWEN_LLAMA_GPU/PORT/CTX/ALIAS/MODEL_DIR/MODEL_FILE/MMPROJ/REASONING/REASONING_FORMAT/REASONING_BUDGET`,
`GEMMA_LLAMA_GPU/PORT/CTX/ALIAS/REASONING/REASONING_FORMAT/REASONING_BUDGET`, `QWEN_PORT/GPU`, `GEMMA_PORT/GPU`,
`COMFY_PORT_A/B`, `COMFY_GPU_A/B`.

## Logs

- `logs/llama_8088.log` - Qwen3.6 llama.cpp
- `logs/llama_8089.log` - Gemma4 llama.cpp
- `logs/comfyui_8188.log`, `logs/comfyui_8189.log` - ComfyUI
- `logs/vllm_8088.log` - vLLM Qwen (when using the fallback)
- `logs/vllm_8000.log` - Qwen3.8-Flash-Next vLLM (qwen-256k)
- `logs/llama_8090.log` - Qwen3.8-Flash-Next llama.cpp (flash-next)

## Known gotchas

- `GGML_SYCL_DISABLE_OPT` was a dead env var (build only reads
  `GGML_SYCL_ENABLE_OPT`). Removed from `.bashrc` 2026-08-25.
- KV cache types on this SYCL build: q4_0 and q5_0 work; q8_0 and iq4_nl
  segfault in sustained decode. f16 works but costs 2x memory.
- Per-GPU VRAM probe: `torch.xpu.mem_get_info` with
  `env -u ONEAPI_DEVICE_SELECTOR` (the pinned selector shifts device indices).
- The Qwen3.8 GGUF has an embedded chat template. Do not pass an external
  `--chat-template` - it overrides the embedded one and the model echoes the
  template source. (Root cause of a 2026-08-25 incident: `strings | grep -q`
  under `pipefail` dies with SIGPIPE and inverts the detection test.)
- `llama-server` defaults to 4 slots; always pass `-np 1` for single-user to
  keep KV on-GPU.