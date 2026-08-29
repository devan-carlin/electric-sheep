---
library_name: vllm
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - intel-arc
  - xpu
  - w4a16
  - moe
  - long-context
  - 256k
inference_provider: community
---

# Qwen3.8-Flash-Next — W4A16 (Intel Arc Pro B70 / XPU)

W4A16 (int4 group-128, compressed-tensors) quantization of **Qwen3.8-Flash-Next** (125B MoE / 6B active). Derived from official BF16 weights. Single-quant int4; no FP8 double-quant.

This repository provides a **community quantization and XPU serving recipe**. The base model, architecture, and license are provided by Qwen (Qwen Community License 1.0). Refer to the base model card for the full architecture description.

## Changes from base model

- Expert weights quantized to int4 group-128 (W4A16) via `compressed-tensors`.
- `quantization_config` ignore list maintains these in BF16: `lm_head`, `embed_tokens`, `mtp.*`, `ple.*`, `visual.*`, `*.gate`, `hyper_connection*`, `indexer*`, `linear_attn.*`, `shared_expert*`.
- 17 safetensors shards, ~77GB total.

## Hardware requirements (confirmed)

Configuration used for benchmarking:

- **4x Intel Arc Pro B70** (Xe3), 32GB each = **128GB total VRAM**
- **>=128GB host RAM** (247GiB on reference system) — required for the PLE n-gram table
- oneAPI Level-Zero 20.2.0

Performance measured on this hardware (single stream, 512 tokens, greedy):

- **53.4 tok/s decode**, TTFT ~0.11s
- 256K context, fp8 KV cache

### Memory distribution

Memory is split across two pools for different purposes:

| Pool | Sized for | Contents |
|---|---|---|
| **VRAM** (4x32GB = 128GB) | weights + KV cache | 77GB W4A16 weights (sharded 4 ways via TP+EP, ~19GB/GPU), fp8 KV cache, activations, and XPU graph workspace |
| **Host RAM** (>=128GB) | PLE table | 96GB n-gram table (memory-mapped, shared page cache across all 4 ranks) |

- **The PLE table resides in host RAM.** It is mmap'd from host RAM. Each decode step transfers only 16 rows/token (~few KB) to the GPU.
- A system with 128GB VRAM but only 64GB RAM will hold the weights but fail to map the table. **Both ~128GB VRAM and >=128GB RAM are required.**
- `--gpu-memory-utilization 0.85` provides VRAM headroom for the 256K context KV cache.

## Dependency: patched vLLM

Stock vLLM is not compatible. This model requires a patched vLLM with the `qwen4_exp` model port and XPU W4A16 / GDN kernels:

- vLLM `0.26.1rc1.dev500+gc39076fef`
- torch `2.13.0+xpu`
- vllm-xpu-kernels `0.1.12` (pinned; `0.1.13.2` is broken)

Port and kernel patches are provided as diffs (see companion operations doc / PR).

## PLE n-gram table (required, included in repo)

The model uses an N-gram Embedding (PLE) layer backed by a ~96GB hash table. **The table is required for correct output.** Disabling it (`QWEN4EXP_DISABLE_PLE=1`) will result in incorrect output.

This repository **includes the table** (`ple_table_qwen4exp.pt`) for out-of-the-box use. It is memory-mapped from host RAM and shared across all tensor-parallel ranks via the page cache; it is never loaded onto the GPU.

- Set `PLE_TABLE_PATH` to the `ple_table_qwen4exp.pt` file in this repo.
- To rebuild the table (e.g., after a base-model update), run the `phase_b_ple_table_prep.py` script on a machine with sufficient RAM, then point `PLE_TABLE_PATH` to the result.

## Launch

```
python -m vllm.entrypoints.openai.api_server \
  --model <this-repo> \
  --served-model-name qwen-256k \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --generation-config vllm
```

Environment: `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `UR_L0_SYNC_MODE=BLOCKING`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `PLE_TABLE_PATH=/path/to/ple_table.pt`.

## Notes

- **Text-only.** This recipe does not include the vision tower; use the base model for multimodal tasks.
- **Serving tip:** Greedy decoding (temp 0) may produce a degenerate foreign-language tail. Use temperature 0.7 and high reasoning effort to resolve this.
- **MTP speculative decoding** is supported by the port but is unreliable on XPU due to GDN `causal_conv1d` kernel limitations. Keep it disabled.

## License

Qwen Community License 1.0 (see `LICENSE`). Derivative distribution is permitted. Display the model name if MAU >100M or monthly revenue >$20M. MaaS / AI-assistant commercial use requires a separate license from Qwen.