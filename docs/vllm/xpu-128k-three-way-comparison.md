# XPU 128k-Context Benchmark: vLLM vs SGLang vs OpenVINO

Date: 2026-08-17
Hardware: 4x Intel Arc Pro B70 (Battlemage Xe2, 32 GB, PCIe), Threadripper PRO 3945WX, 247 GB RAM, Ubuntu 26.04, oneAPI 2026.1.0
Model: Qwen3.8-27B (hybrid GDN: mamba linear-attention + full attention, MoE)

## Test setup

| | vLLM | SGLang | OpenVINO |
|---|---|---|---|
| Version | 0.26.1rc1.dev500 | 0.5.16 (editable, fdebc93) | nightly dev20260814 |
| Torch | 2.13.0+xpu | 2.13.0+xpu | n/a |
| Model file | -ara **int4** (auto-round w4g128) | -ara **bf16** | base Qwen3.8-27B **int4** / **int8** |
| Context window | 131072 | 131072 | default (short prompt) |
| Prompt / gen | 4096 / 128 | 4096 / 128 | 141 / 128 |
| KV cache | fp8, prefix caching on | bf16, radix cache off | n/a |
| Runs | 3, avg of runs 1-2 (run 0 = warmup) | 3, avg of runs 1-2 | 3, avg of runs 1-2 |

## Results

### vLLM (int4 auto-round, 128k window, 4096-prompt / 128-gen)

| TP | load s | ttft s | prefill tok/s | decode tok/s |
|----|--------|--------|---------------|--------------|
| 1  | 55.1   | 2.94   | 1391          | 22.1         |
| 2  | 60.1   | 1.86   | 2196          | 36.4         |
| 4  | 60.1   | 1.31   | 3121          | 53.7         |

Scaling TP=1 -> TP=4: prefill 2.2x, decode 2.4x.

### SGLang (bf16, 128k window, 4096-prompt / 128-gen, --disable-radix-cache, eager defaults)

| TP | load s | ttft s | prefill tok/s | decode tok/s |
|----|--------|--------|---------------|--------------|
| 1  | n/a    | n/a    | n/a           | n/a          |
| 2  | 46.1   | 3.64   | 1126          | 14.4         |
| 4  | 142.2  | 2.13   | 1923          | 12.4         |

TP=1 impossible: bf16 weights are 52 GB, one card holds ~30.3 GB.
TP=2 -> TP=4: prefill 1.7x, decode flat (14.4 -> 12.4).
NOTE: this run used eager defaults (triton attention, no graph capture).

### SGLang long-context sweep (bf16, TP=4, OPTIMIZED XPU config)

Config: `--attention-backend intel_xpu --page-size 64 --cuda-graph-backend-decode
full`, bf16 KV, --disable-radix-cache, 128k window, gen=128. Decode = server-log
steady-state gen throughput (all batches under the captured XPU graph).

| prompt | ttft s | prefill tok/s | decode tok/s |
|--------|--------|---------------|--------------|
| 4k     | 1.11   | 3674          | 34.9         |
| 16k    | 5.57   | 2942          | 34.3         |
| 32k    | 17.5   | 1873          | 33.8         |
| 64k    | 39.0   | 1680          | 32.7         |

- Optimized config lifts decode ~2.8x vs eager defaults (12.4 -> 34.9 at 4k).
- Decode is FLAT across context: 34.9 -> 32.7 (4k -> 64k, -6%).
- Prefill degrades with context: 3674 -> 1680 (2.2x).
- At 4k, SGLang prefill (3674) beats vLLM int4 (3121); vLLM decode (53.7)
  beats SGLang (34.9) by ~1.5x.
- Data: /tmp/sglang-64k-matrix.json, runner /home/dc/sglang/bench-128k/run_sweep.py.

### OpenVINO (base model, 141-prompt / 128-gen)

| Config | device | load s | decode tok/s | total tok/s |
|--------|--------|--------|--------------|-------------|
| int4 1gpu | GPU.0 | 16.1 | 28.4 | 59.6 |
| int4 2gpu | HETERO:GPU.0,1 | 27.9 | 28.3 | 59.5 |
| int4 4gpu | HETERO:GPU.0-3 | 28.1 | 28.2 | 59.4 |
| int8 1gpu | GPU.0 | 20.2 | 17.7 | 37.1 |
| int8 2gpu | HETERO:GPU.0,1 | 31.2 | 17.6 | 37.0 |
| int8 4gpu | HETERO:GPU.0-3 | 31.4 | 17.6 | 37.0 |

HETERO multi-GPU gives zero gain over 1 GPU for both int4 and int8.

## Caveats (read before comparing across rows)

1. **Different quantizations.** vLLM ran int4 (18 GB), SGLang ran bf16 (52 GB), OpenVINO ran int4/int8 of the base model. int4 auto-round does not work on SGLang XPU: the XPU sgl_kernel has no GPTQ W4A16 gemm (no gptq_gemm/gptq_shuffle exports) and Marlin is CUDA-only. The vLLM-vs-SGLang decode gap (53.7 vs 12.4 at TP=4) is therefore mostly quantization + engine, not a clean apples-to-apples number.
2. **Different prompt lengths.** vLLM/SGLang used 4096-token prompts at a 128k window; OpenVINO used 141-token prompts. OpenVINO prefill is not comparable, and its decode was measured with a tiny KV footprint.
3. **OpenVINO HETERO is not tensor parallelism.** Layers are split across devices, not weights; the flat 1/2/4-GPU numbers confirm no real scaling.
4. **OpenVINO is a nightly build** (dev20260814), experimental on XPU.
5. SGLang CUDA graphs are disabled in this config (see server_args in /tmp/sglang-128k-tp4.log), which likely suppresses decode further.

## Findings

- **vLLM int4 is the clear winner for this workload on XPU.** 53.7 decode / 3121 prefill tok/s at TP=4, with near-linear scaling from TP=1.
- **SGLang decode is weak on this model on XPU** (12-14 tok/s bf16). The hybrid GDN (mamba) architecture plus triton attention backend and disabled CUDA graphs all hurt. Prefill scales better (1.7x TP=2->4) than decode (flat).
- **OpenVINO int4 is competitive on short-prompt decode** (28.4 tok/s, 1 GPU) but does not scale across GPUs and was not tested at long context.
- For interactive use at 128k context on this box: vLLM TP=4 int4. For short-prompt, single-GPU, low-latency: OpenVINO int4 is a reasonable alternative.

## Artifacts

- vLLM matrix: /tmp/vllm-128k-matrix.json (runner: /home/dc/electric-sheep/vllm/bench-128k/run_matrix.py)
- SGLang matrix: /tmp/sglang-128k-matrix.json (runner: /home/dc/sglang/bench-128k/run_matrix.py)
- OpenVINO matrix: /tmp/ov-bench-matrix.json
- SGLang server logs: /tmp/sglang-128k-tp{2,4}.log

## SGLang XPU install notes (for reproducibility)

- torch 2.13.0+xpu required (bundles intel_sycl_rt 2026.0.0, ABI-compatible with system libsycl 2026.1). torch 2.12.0+xpu + kernel built vs system libsycl 2026.1 -> UR_RESULT_ERROR_UNINITIALIZED.
- sgl_kernel 0.11.0 rebuilt from /tmp/sgl-kernel-xpu against torch 2.13.0.
- xgrammar 0.2.1 hard import; its CUDA triton dep clobbers triton-xpu -> force-reinstall triton-xpu 3.7.2 with --no-deps.
- Local patch in python/sglang/srt/layers/quantization/marlin_utils.py: get_device_capability returns (None, None) on XPU, crashed `major * 10 + minor` in two places; added None-guard so Marlin is cleanly rejected and GPTQ fallback path is taken. Needs re-applying on a fresh checkout; candidate for upstream PR.
- Hybrid GDN on XPU: default 'extra_buffer' mamba strategy asserts (CUDA/MUSA/NPU/ROCm only); 'no_buffer' forces disable_overlap_schedule (~5x slower decode). Best config: --disable-radix-cache (mamba resolution returns early, overlap + CUDA graphs stay on).
