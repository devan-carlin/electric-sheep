# AI Server Handoff

Read this first. It tells you what is running, where, how to call it, and how to
manage it. Written for an operator (human or LLM) with no prior context.

## What this is

A 4x Intel Arc Pro B70 (32 GB each, XPU / Level Zero) AI server. One service per
GPU. Two of the four are LLM chat endpoints that speak the OpenAI API; two are
image-generation (ComfyUI) instances.

- Host OS: Ubuntu, Linux.
- The LLM endpoints are `llama.cpp` `llama-server` processes. They expose a
  standard OpenAI-compatible API. No API key is required (open on the host).
- A Docker `open-webui` container is the human-facing chat UI. It talks to the
  two LLM endpoints over `host.docker.internal`. You do not need it to use the
  models directly - hit the endpoints yourself.

## The two LLM endpoints (what you will actually use)

| Port | GPU | Model ID | Model (weights) | Context | Use for |
|------|-----|----------|-----------------|---------|---------|
| 8088 | 2 | `qwen`  | Qwen3.6-35B-A3B Aggressive (, Q4_K_P, MoE 35B/3B-active) | 256K | Book writing, long-context prose, fast bulk generation |
| 8089 | 3 | `gemma` | Gemma4 26B-A4B ( Balanced, Q4_K_P) | 256K | VN writing, daily chat, general prose |

Both are OpenAI-compatible. Both are multimodal (accept images via `mmproj`).
Both run with thinking/reasoning ON (`--reasoning on --reasoning-format
deepseek --reasoning-budget 2048`), so `reasoning_content` holds the model's
thoughts and `content` holds the final answer. The 2048 budget caps thinking
length; thinking tokens count against `max_tokens`.

### Model IDs are version-agnostic on purpose

`qwen` and `gemma` are stable names. The actual weights behind them can be
swapped (e.g. `qwen` is currently Qwen3.6-35B-A3B but could become a future
Qwen) without changing the name. Always send `model: "qwen"` or `model: "gemma"`
- never a versioned name. If you need to know which weights are live, query
`/v1/models` or read the launch script (below).

## How to call them

Base URL: `http://<host>:<port>/v1`. From the host itself use `127.0.0.1`.
No auth header needed.

### List models
```
curl -s http://127.0.0.1:8088/v1/models
```

### Chat completion (the main call)
```
curl -s http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      {"role": "user", "content": "Your prompt here"}
    ],
    "max_tokens": 2048,
    "temperature": 0.7
  }'
```
Response is standard OpenAI shape: `choices[0].message.content` is the final
answer. `choices[0].message.reasoning_content` holds the model's reasoning
(thinking is on, capped at 2048 tokens). `choices[0].finish_reason` is `stop`
(model finished) or `length` (hit `max_tokens`).

Note: thinking tokens count against `max_tokens`. For reasoning-heavy prompts
send a larger `max_tokens` (e.g. 4096+), or the answer can be cut off with
`finish_reason: "length"`.

### Recommended sampling
- Prose / creative: `temperature` 0.7, `top_p` 0.9.
- Factual / deterministic: `temperature` 0.0.
- Long outputs: raise `max_tokens`. Context is 256K, so you can send very long
  prompts, but each extra prompt token costs prefill time.

### Throughput (so you can set expectations)
- `qwen` (MoE): ~80 tokens/s decode. Fast. Good for bulk / long generation.
- `gemma` (dense): ~54 tokens/s decode.
- These are single-stream (one request at a time per endpoint, `np=1`). Do not
  expect concurrent-request speedups; queue requests if you need many.

## The two image endpoints (ComfyUI)

| Port | GPU | Use for |
|------|-----|---------|
| 8188 | 0 | ComfyUI instance A - image generation, workflow A |
| 8189 | 1 | ComfyUI instance B - image generation, workflow B |

These are full ComfyUI web apps, not simple APIs. They have their own UI at
`http://<host>:8188` and `:8189`. Use them for image generation workflows. They
are independent of the LLM endpoints.

## Managing the services

All launch scripts live in `/home/dc/electric-sheep/serve/`.

| Script | What it does |
|--------|--------------|
| `start-all.sh` | Manage all 4 services. `start` / `stop` / `status` / `restart-gemma` / `restart-qwen` |
| `start-qwen-llama.sh` | The `qwen` endpoint (GPU 2 :8088). `start` / `stop` / `status`. Model pinned in-script. |
| `start-gemma-llama.sh` | The `gemma` endpoint (GPU 3 :8089). `start` / `stop` / `status`. |
| `fallback/start-qwen.sh` | vLLM fallback for the qwen slot (128K, fp8 KV). Only if you need prefix caching / tool parsers / concurrency. |

Common operations:
```
bash /home/dc/electric-sheep/serve/start-all.sh status        # are they up?
bash /home/dc/electric-sheep/serve/start-all.sh restart-qwen  # restart qwen only
bash /home/dc/electric-sheep/serve/start-all.sh restart-gemma # restart gemma only
bash /home/dc/electric-sheep/serve/start-all.sh stop          # stop everything
bash /home/dc/electric-sheep/serve/start-all.sh start         # start all 4
```

### Swapping the model behind `qwen` or `gemma`
The weights are pinned inside the launcher. To point `qwen` at a different GGUF:
```
QWEN_LLAMA_MODEL_DIR=/path/to/model-dir \
QWEN_LLAMA_MODEL_FILE=/path/to/model.gguf \
QWEN_LLAMA_MMPROJ=/path/to/mmproj.gguf \
bash /home/dc/electric-sheep/serve/start-qwen-llama.sh start
```
The name stays `qwen`. Same pattern with `GEMMA_LLAMA_*` for the gemma slot.
Other overrides: `QWEN_LLAMA_GPU`, `QWEN_LLAMA_PORT`, `QWEN_LLAMA_CTX`,
`QWEN_LLAMA_REASONING` (`on`/`off`/`auto`, default `on`),
`QWEN_LLAMA_REASONING_FORMAT` (`none`/`deepseek`/`deepseek-legacy`, default
`deepseek`), `QWEN_LLAMA_REASONING_BUDGET` (thinking token cap, default 2048;
`-1` = unlimited, `0` = no thinking). Same trio with the `GEMMA_LLAMA_` prefix
for the gemma slot. `start-all.sh` forwards all of these.

## Logs

- `/home/dc/electric-sheep/serve/logs/llama_8088.log` - qwen
- `/home/dc/electric-sheep/serve/logs/llama_8089.log` - gemma
- `/home/dc/electric-sheep/serve/logs/comfyui_8188.log`, `comfyui_8189.log`

Decode speed and draft stats appear in the llama logs as `print_timing` lines
(`... tokens per second`).

## Models on disk

Root: `/home/dc/electric-sheep/models/` (a symlink to `/mnt/data/models`).
- `-Qwen3.6-35B-A3B---Aggressive/` - the live `qwen` weights (Q4_K_P + f16 mmproj)
- `-Gemma4-26B-A4B---Balanced/` - the live `gemma` weights (Q4_K_P + f16 mmproj)
- Other dirs are fallbacks / experiments (vLLM safetensors, other quants).

## Gotchas (read before debugging)

- **Per-GPU VRAM probe**: use
  `env -u ONEAPI_DEVICE_SELECTOR /home/dc/electric-sheep/vllm/.venv/bin/python -c "import torch; f,t=torch.xpu.mem_get_info(torch.device('xpu:2')); print((t-f)/1024**3)"`
  You MUST `env -u ONEAPI_DEVICE_SELECTOR` or the device indices shift and you
  read the wrong card. System Eye / `zes` reports a unified pool - useless per-GPU.
- **KV cache types** on this SYCL build: `q4_0` and `q5_0` work. `q8_0` and
  `iq4_nl` segfault in sustained decode. Do not switch KV type without re-testing.
- **Chat template**: the GGUFs carry an embedded template. Never pass an external
  `--chat-template` - it overrides the embedded one and the model echoes the
  template source instead of answering.
- **`-np 1`** is always set (single user). If you see KV spilling to CPU or slow
  decode, that is expected at 256K for the dense `gemma` model.
- **Reasoning is ON by design** on both endpoints (deepseek format, 2048
  token budget). Thoughts arrive in `reasoning_content`, the answer in
  `content`. If `content` comes back empty, the tokens went to
  `reasoning_content` - raise `max_tokens` or lower the budget via
  `*_LLAMA_REASONING_BUDGET` (or set `*_LLAMA_REASONING=off` and restart).

## Quick health check (run this first)
```
for p in 8088 8089 8188 8189; do
  echo -n "port $p: "
  curl -s -o /dev/null --max-time 2 -w "%{http_code}" http://127.0.0.1:$p/ && echo " up" || echo " DOWN"
done
```
All four should report `up`. If an LLM port is down, `bash start-all.sh
restart-qwen` (or `restart-gemma`) and re-check.