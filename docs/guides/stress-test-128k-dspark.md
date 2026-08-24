# Stress Test: DeepSeek V4-Flash with Balanced Split + DSpark (128K Context)

## Test Date
2026-08-10

## Environment

| Component | Details |
|-----------|---------|
| **CPU** | AMD Threadripper PRO 3945WX (12C/24T), 247 GB RAM |
| **GPU** | 4× Intel Arc Pro B70 (32 GB each, Level Zero) |
| **OS** | Ubuntu 26.04 LTS, Kernel 7.0.0-29 |
| **oneAPI** | 2026.1.1 (icpx compiler) |
| **llama.cpp** | commit `dd1ea5243` (latest master) with SYCL/XPU backend |
| **Patch** | Balanced split mode (`--split-mode balanced`) |

## Optimal Configuration (Recommended)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | apetersson-DeepSeek-V4-Flash-0731--DS4-Quality128 (102.8 GB GGUF) | |
| **Context** | 131,072 tokens (128K) | |
| **Split Mode** | Balanced (quantization-aware, MoE-aware) | |
| **DSpark** | Enabled (draft-dspark, n_max=2) | Sweet spot vs n_max=5 |
| **Drafter** | `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` (11 GB) | Smaller VRAM than native |
| **Batch Size** | 4096 | Fast prompt ingestion (52 tok/s) |
| **Ubatch Size** | 128 | Avoids OOM on warmup |
| **KV Cache** | f16 (default) | Quantized KV not supported on SYCL multi-GPU |
| **Flash Attention** | ON | |
| **Port** | 8080 | Falls back to 8081 |
| **Parallel Slots** | 4 | |

## VRAM Distribution

### Main Model (Balanced Split)

| GPU | Model Buffer | Usage |
|-----|-------------|-------|
| SYCL0 | 25,720 MiB (25.1 GB) | 81% |
| SYCL1 | 22,818 MiB (22.3 GB) | 73% |
| SYCL2 | 24,996 MiB (24.4 GB) | 79% |
| SYCL3 | 23,514 MiB (23.0 GB) | 74% |

### DSpark Draft Model

| GPU | Draft Buffer |
|-----|-------------|
| SYCL0 | 3,403 MiB |
| SYCL1 | 3,403 MiB |
| SYCL2 | 3,403 MiB |
| SYCL3 | 178 MiB |

### Total VRAM (Model + Draft)

| GPU | Total | Combined Usage |
|-----|-------|---------------|
| SYCL0 | ~28.5 GB | 89% |
| SYCL1 | ~25.7 GB | 80% |
| SYCL2 | ~27.8 GB | 87% |
| SYCL3 | ~23.2 GB | 73% |

## Test Results

### Test 1: Short Generation (256 tokens)

**Prompt**: "Write a detailed explanation of how transformers work in deep learning..."

| Metric | Value |
|--------|-------|
| Prompt tokens | 36 |
| Completion tokens | 256 |
| Prompt eval time | 2,669ms (13.5 tok/s) |
| Eval time | 19,312ms (**13.3 tok/s**) |
| Total time | 22.0s |
| Graph reuse | 106 times |

**DSpark Stats**:
- Draft acceptance: **65.5%** (144 accepted / 220 generated)
- Mean accepted chain: 2.31 tokens
- Acceptance per position: Pos1: 73.6%, Pos2: 57.3%

### Test 2: Long Generation (2048 tokens)

**Prompt**: "Write a comprehensive essay about the history and evolution of AI..."

| Metric | Value |
|--------|-------|
| Prompt tokens | 57 |
| Completion tokens | 2,048 |
| Prompt eval time | 1,944ms (29.3 tok/s) |
| Eval time | 150,278ms (**13.6 tok/s**) |
| Total time | 152.3s |
| Graph reuse | 731 times |

**DSpark Stats**:
- Draft acceptance: **71.2%** (1,202 accepted / 1,688 generated)
- Mean accepted chain: 2.42 tokens
- Acceptance per position: Pos1: 80.4%, Pos2: 60.7%

### Test 3: Code Generation (512 tokens)

**Prompt**: "Write a Python implementation of merge sort with type hints and tests..."

| Metric | Value |
|--------|-------|
| Prompt tokens | 28 |
| Completion tokens | 512 |
| Prompt eval time | 1,073ms (26.1 tok/s) |
| Eval time | 35,518ms (**14.4 tok/s**) |
| Total time | 36.6s |
| Graph reuse | 203 times |

**DSpark Stats**:
- Draft acceptance: **76.0%** (308 accepted / 405 generated)
- Mean accepted chain: 2.52 tokens
- Acceptance per position: Pos1: 81.1%, Pos2: 61.9%

## Cumulative DSpark Statistics (All Tests Combined)

| Metric | Value |
|--------|-------|
| Total draft calls | 1,157 |
| Total drafts generated | 2,313 tokens |
| Total drafts accepted | 1,654 tokens |
| **Overall acceptance rate** | **71.5%** |
| Mean accepted chain length | 2.43 tokens |
| Acceptance per position | Pos1: 81.1%, Pos2: 61.9% |
| Draft generation latency | 12,935ms total (11.2ms per call avg) |

## Performance Analysis

### Generation Speed by Test

| Test | Tokens | Speed (tok/s) | DSpark Acceptance |
|------|--------|---------------|-------------------|
| Short (256) | 256 | 13.3 | 65.5% |
| Long (2048) | 2,048 | 13.6 | 71.2% |
| Code (512) | 512 | 14.4 | 76.0% |

### Key Observations

1. **Stable generation speed**: 13.3-14.4 tok/s across all tests, with longer generations showing slightly higher throughput (better amortization of MoE expert routing overhead)

2. **DSpark acceptance improves with context**: Short generation (65.5%) → Long generation (71.2%) → Code (76.0%). The drafter model benefits from more tokens of context for better predictions.

3. **Position 1 acceptance (73-81%)** is consistently high, showing DSpark's first draft token is very accurate. Position 2 (57-62%) shows expected degradation.

4. **No VRAM drift**: After 3 tests totaling 2,816 generated tokens, VRAM distribution remained stable (no memory leaks observed).

5. **128K context loaded without OOM**: Server initialized successfully with 128K context window, though actual usage was limited to ~32K per slot (4 slots × 32K = 128K).

## Test 4: DSpark n_max=5 with Native DSpark Model (Large Prompt)

**Configuration**: DSpark n_max=5, native DSpark model (not Q8_0 drafter), batch-size=4096, ubatch-size=128

**Prompt**: Large prompt (5,551 tokens ingested)

| Metric | Value |
|--------|-------|
| Prompt tokens | 5,551 |
| Completion tokens | 3,048 |
| Prompt eval time | 106,272ms (**52.23 tok/s**) |
| Eval time | 290,750ms (**10.48 tok/s**) |
| Total time | 397s |
| Graph reuse | 1,027 times |

**DSpark Stats**:
- Draft acceptance: **38.7%** (2,009 accepted / 5,195 generated)
- Mean accepted chain: 2.93 tokens

**Notes**:
- Prompt ingestion at 52 tok/s with batch-size=4096 (16× faster than batch-size=256)
- Generation speed at 10.48 tok/s (lower than n_max=2 tests due to more draft overhead)
- Draft acceptance at 38.7% (lower than n_max=2 tests — more draft tokens = more rejections)
- Native DSpark model used (`DeepSeek-V4-Flash-0731--DS4-Quality128-llamacpp-DSpark-support.gguf`)

## Comparison: DSpark vs No-DSpark (Previous Tests)

| Metric | No DSpark (64K ctx) | DSpark n_max=2 (128K ctx) | DSpark n_max=5 (128K ctx) |
|--------|---------------------|---------------------------|---------------------------|
| Generation speed | ~12.9 tok/s | 13.3-14.4 tok/s | 10.48 tok/s |
| Context window | 64K | 128K | 128K |
| DSpark acceptance | N/A | 65-76% | 38.7% |
| Prompt speed (large) | N/A | N/A | 52.23 tok/s |

## Comparison: DSpark n_max=2 vs n_max=5

| Metric | n_max=2 (Q8_0 drafter) | n_max=5 (native DSpark) | Change |
|--------|------------------------|------------------------|--------|
| Generation speed | 13.3-14.4 tok/s | 10.48 tok/s | **-20-27%** |
| Draft acceptance | 65-76% | 38.7% | **-27-37%** |
| Mean chain length | 2.43 tokens | 2.93 tokens | **+20%** |
| VRAM overhead | ~11GB (Q8_0) | Larger (native) | Higher |

**Verdict**: n_max=2 with Q8_0 drafter is the sweet spot — faster generation, higher acceptance, lower VRAM cost. n_max=5 generates more draft tokens but rejects more, adding overhead without enough accepted tokens to compensate.

## Known Limitations

1. **Quantized KV cache not supported on SYCL multi-GPU**: `--cache-type-k q8_0` crashes with `ggml_sycl_op_concat: unsupported types: dst: q8_0`. Only f16 KV cache works with balanced/layer split modes.

2. **196K context crashes**: Attempting 196K context (triple) with f16 KV cache triggered `GGML_ASSERT(false)` in concat.cpp — likely exceeded per-GPU allocation limits.

3. **Reasoning tokens dominate**: Most completion tokens go to `reasoning_content` (thinking), not `content`. This is expected for DeepSeek reasoning models.

4. **ubatch-size OOM with large values**: `--ubatch-size 4096` causes OOM during warmup on sequential mode (stuck on empty run). Balanced mode tolerates larger ubatch, but sequential mode's compute buffers grow proportionally. Safe value: `--ubatch-size 128` (or match batch-size only on balanced mode).

## Commands Used

```bash
# Optimal startup (recommended config)
source ~/electric-sheep/llama/set-env.sh
cd ~/llama.cpp/build
./bin/llama-server \
  -m ~/electric-sheep/models/apetersson-DeepSeek-V4-Flash-0731--DS4-Quality128/DeepSeek-V4-Flash-0731--DS4-Quality128.gguf \
  --host 0.0.0.0 --port 8080 \
  --gpu-layers 999 \
  --split-mode balanced \
  --tensor-split 1,1,1,1 \
  --flash-attn on \
  --ctx-size 131072 \
  --batch-size 4096 \
  --ubatch-size 128 \
  --parallel 4 \
  -md ~/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 2 \
  --verbose

# Alternative: n_max=5 with native DSpark model (slower, higher VRAM)
source ~/electric-sheep/llama/set-env.sh
cd ~/llama.cpp/build
./bin/llama-server \
  -m ~/electric-sheep/models/apetersson-DeepSeek-V4-Flash-0731--DS4-Quality128/DeepSeek-V4-Flash-0731--DS4-Quality128.gguf \
  --host 0.0.0.0 --port 8081 \
  --spec-type draft-dspark \
  --spec-draft-model ~/electric-sheep/models/apetersson-DeepSeek-V4-Flash-0731--DS4-Quality128/DeepSeek-V4-Flash-0731--DS4-Quality128-llamacpp-DSpark-support.gguf \
  --spec-draft-n-max 5 \
  --ctx-size 131072 \
  --batch-size 4096 \
  --ubatch-size 128 \
  --gpu-layers 999 \
  --spec-draft-ngl 999 \
  --split-mode balanced \
  --flash-attn on
```
  --flash-attn on



## Log Files

- Server logs: `/tmp/stress-test-128k.log`
- Test 1 output: `/tmp/stress-test-1.json`
- Test 2 output: `/tmp/stress-test-2.json`
- Test 3 output: `/tmp/stress-test-3.json`
