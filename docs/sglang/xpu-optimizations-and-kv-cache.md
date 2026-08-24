# SGLang on Intel XPU — Optimizations & KV-Cache Options

Date: 2026-08-17
Hardware: 4x Intel Arc Pro B70 (Battlemage Xe2, 32 GB, PCIe)
SGLang: 0.5.16 (editable, commit fdebc93), torch 2.13.0+xpu, sgl_kernel 0.11.0 (rebuilt vs 2.13.0)
Sources: docs.sglang.io — `hardware-platforms/xpu`, `advanced_features/quantized_kv_cache`, `advanced_features/attention_backend`, `advanced_features/server_arguments`

## 1. XPU-specific optimizations we were NOT using

Our 128k benchmark ran SGLang with **defaults**, which on XPU means the generic
`triton` attention backend and **no graph capture**. The official XPU page
documents opt-in optimizations that were left off. These are the likely cause
of the weak decode (12-14 tok/s).

### 1a. Attention backend: `intel_xpu` (recommended) vs `triton` (default)
- The XPU launch example uses `--attention-backend intel_xpu` with
  `--page-size` in `[32, 64, 128]`.
- Our server_args showed `attention_backend='triton'` (generic fallback).
- **Trade-off (see section 2):** `intel_xpu` does NOT support quantized KV
  cache; `triton` does. So this is a speed-vs-KV-memory choice.

### 1b. XPU Graph capture — OFF by default on XPU (opt-in)
Both phases are disabled by default, which is exactly what our run had
(`cuda_graph_config backend='disabled'`).

| Phase   | Backend        | Flag |
|---------|----------------|------|
| Decode  | `full`         | `--cuda-graph-backend-decode full` |
| Prefill | `tc_piecewise` | `--cuda-graph-backend-prefill tc_piecewise` |
| Prefill | `breakable`    | `--cuda-graph-backend-prefill breakable` |

- Decode `full` = one `torch.xpu.XPUGraph` per batch size, captured at startup.
- Prefill `tc_piecewise` = `torch.compile` + XPU graph, one segment per
  token-length bucket. `--cuda-graph-tc-compiler inductor` for higher-quality
  code at longer startup (default `eager`).
- Prefill `breakable` = segmented `XPUGraph` capture, eager break points at
  attention/MoE boundaries, no `torch.compile`.
- One-shot JSON: `--cuda-graph-config '{"decode":{"backend":"full"},"prefill":{"backend":"tc_piecewise","tc_compiler":"eager"}}'`
- `--enable-torch-compile` adds a compile pass on top of the decode graph
  (mutually exclusive with prefill `tc_piecewise`).
- Capture buckets: `--cuda-graph-bs-decode 1 2 4 8`,
  `--cuda-graph-bs-prefill 64 128 256 512` (prefill defaults derive from
  `--chunked-prefill-size`).

### 1c. Recommended XPU launch (per docs)
```bash
sglang serve \
    --model-path <MODEL> \
    --trust-remote-code \
    --disable-overlap-schedule \
    --device xpu \
    --host 0.0.0.0 \
    --tp 2 \
    --attention-backend intel_xpu \
    --page-size 64
```
Note: the docs example includes `--disable-overlap-schedule`. For our hybrid-GDN
(mamba) model we instead used `--disable-radix-cache` (overlap stays on, mamba
pool is a physical per-request store). Re-test whether overlap can stay enabled
with the intel_xpu backend.

### 1d. Install note
Docs say `pip install xgrammar --no-deps` — the clean way to avoid the
CUDA-triton clobber we hit (had to force-reinstall triton-xpu 3.7.2).

### 1e. Optimized model list (XPU)
Only Llama-3.2-3B / Llama-3.1-8B / Qwen2.5-1.5B (BF16, verified on B580).
**Qwen3.8-27B is NOT on the tuned list** — we're past the optimized set, which
also contributes to the weak numbers.

## 2. Quantized KV cache — options & XPU reality

### 2a. Formats SGLang supports (general)
| Flag | Format | Notes |
|------|--------|-------|
| `--kv-cache-dtype fp8_e5m2` | FP8 E5M2 | larger range (±57344), lower precision |
| `--kv-cache-dtype fp8_e4m3` | FP8 E4M3 | higher precision (±240), **recommended** |
| `--kv-cache-dtype nvfp4` | FP4 E2M1 | experimental, max savings |
| `--kv-cache-dtype fp4_mx_block16` | FP4 E2M1 block-16 | experimental, auto scaling |

- Memory: BF16→FP4 ≈ 3.56x more tokens; FP4 ≈ 1.78x more than FP8.
- FP8 needs scaling factors (per-tensor only): from checkpoint (`k_scale`/
  `v_scale`) or `--quantization-param-path` JSON. **Defaults to 1.0 if absent
  → accuracy risk.** FP4 computes block scaling on-the-fly (no external file).
- Accuracy (docs, large models): FP8 E4M3 ≈ BF16 on gsm8k/gpqa; FP4 degrades
  on hard reasoning (aime25: 0.77→0.60 Qwen3-235B). Smaller models degrade more.
  Long-context accumulates error.

### 2b. CRITICAL: backend support matrix (FP8/FP4 KV)
From the attention-backend page. Only some backends support quantized KV, and
**dequant must be fused with the attention kernel or it's extremely slow.**

| Backend | Page>1 | FP8 KV | FP4 KV | Runs on XPU? |
|---------|--------|--------|--------|--------------|
| FlashInfer | ✅ | ✅ | ❌ | no (CUDA) |
| FA3 / FA4 | ✅ | ✅/❌ | ❌/✅ | no (CUDA) |
| **Triton** | ❌ | **✅** | **✅** | **yes (our default)** |
| Torch Native (SDPA) | ❌ | ✅ | ✅ | yes |
| FlexAttention | ❌ | ❌ | ✅ | yes |
| TRTLLM MHA | ✅ | ✅ | ✅ | no (CUDA) |
| AITER / Wave | ✅ | ✅/❌ | ❌ | no (ROCm) |
| Ascend | ✅ | ❌ | ❌ | no (NPU) |
| **Intel XPU** | ✅ | **❌** | **❌** | **yes (docs-recommended)** |
| Intel AMX (CPU) | ❌ | ❌ | ❌ | CPU |

**The catch:** the docs-recommended `intel_xpu` backend does **NOT** support
quantized KV cache. The backends that DO support FP8/FP4 KV (Triton, SDPA,
FlexAttention) are the generic ones. So on XPU:
- Want **fast attention** → `intel_xpu` backend, but KV must be **BF16**.
- Want **quantized KV** (more context/concurrency) → `triton`/`sdpa` backend,
  accept slower attention.

### 2c. What this means for our setup
- Our 128k run used `triton` + **BF16 KV** (no `--kv-cache-dtype` set). We could
  have used `--kv-cache-dtype fp8_e4m3` with triton to halve KV memory — but at
  128k window with max-seqs 8 we weren't KV-memory-bound, so it wouldn't have
  helped throughput (and may have slowed attention via dequant).
- vLLM, by contrast, ran **fp8 KV** (its XPU path supports it) — one reason its
  long-context numbers are stronger.
- If we later push SGLang to higher concurrency or longer context where KV
  memory binds, the XPU options are: (a) triton + fp8_e4m3 KV, or (b) intel_xpu
  + BF16 KV and rely on TP to shard KV.

## 3. Multi-GPU on XPU
- `--tp N` is the mechanism (docs example: `--tp 2  # using multi GPUs`).
- Device selection via `ZE_AFFINITY_MASK` (shown in the P/D-disaggregation
  examples: `=0` prefill GPU, `=1` decode GPU). Same var we use.
- Generic caveat: "peer access not supported between these two devices" → add
  `--enable-p2p-check`.
- P/D disaggregation (NIXL) is experimental, tested only on 0.6B/7B models.

## 4. Action items for the 64k re-run
1. Re-run SGLang with `--attention-backend intel_xpu --page-size 64`
   `--cuda-graph-backend-decode full` to measure how much decode recovers vs the
   12-14 tok/s eager baseline. (KV stays BF16 — intel_xpu can't do quantized KV.)
2. Optionally a second SGLang config: `triton` + `--kv-cache-dtype fp8_e4m3` to
   compare the quantized-KV path.
3. Keep `--disable-radix-cache` for the hybrid-GDN model; re-test overlap.

## Artifacts
- XPU docs: /tmp/sgl-xpu-docs.md
- KV cache docs: /tmp/sgl-kv-cache-docs.md
- Attention backend docs: /tmp/sgl-attn-docs.md
- Server args docs: /tmp/sgl-server-args.md
- SGLang runner: /home/dc/sglang/bench-128k/run_matrix.py
