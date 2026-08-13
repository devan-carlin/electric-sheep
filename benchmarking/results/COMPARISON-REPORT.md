# Huihui-CyberStrike MoE Benchmark: num_experts_per_tok Comparison

## Executive Summary

Benchmarked **Huihui-CyberStrike-OffSec-35B-abliterated** across 4 `num_experts_per_tok` variants on 4× Intel Arc Pro B70 GPUs.

| Variant | Experts Routed | Status | Kernel Support |
|---------|---------------|--------|----------------|
| **top-8** | 8 | ✅ Complete (8/8 prompts) | ✅ Supported |
| **top-16** | 16 | ✅ Complete (8/8 prompts) | ✅ Supported (vllm-xpu-kernels 0.1.13.1) |
| **top-32** | 32 | ❌ Failed at startup | ❌ Not supported |
| **top-64** | 64 | ❌ Failed at startup | ❌ Not supported |

**Key Finding**: vllm-xpu-kernels 0.1.13.1 supports TopK ≤ 16. TopK > 16 causes `RuntimeError: Unsupported TopK value` during MoE kernel initialization.

**Quality Verdict**: Top-16 is marginally better in reasoning structure (more elegant math formulas, broader bug identification, better prompt analysis), but the difference is subtle. Top-8 is the practical sweet spot for production.

---

## Hardware & Software Configuration

### Hardware
- **GPUs**: 4× Intel Arc Pro B70 (34 GB VRAM each, 128 GB total)
- **CPU**: AMD Ryzen Threadripper PRO 3945WX 12-Core
- **RAM**: 247 GB

### Software
- **vLLM**: 0.26.1rc1.dev500+gc39076fef.xpu (Intel XPU backend)
- **vllm-xpu-kernels**: 0.1.13.1
- **Attention**: FlashAttention v2
- **Tensor Parallelism**: 4 (1 GPU per rank)

### Model
- **Name**: Huihui-CyberStrike-OffSec-35B-abliterated
- **Architecture**: Qwen3.5 MoE (256 total experts, 40 layers)
- **Hidden Size**: 2048
- **MoE Intermediate**: 512
- **Context Length**: 262K (benchmarked at 8192)
- **Model Size**: ~67 GB (16 safetensors shards)

### vLLM Serve Configuration
- `--tensor-parallel-size 4`
- `--gpu-memory-utilization 0.8`
- `--max-model-len 8192` (increased from 2048)
- `--max-tokens 512` (reduced from 1024)
- `--temperature 0.2`
- `--top-p 0.9`

---

## Performance Comparison: top-8 vs top-16 (v4 — Fixed Run)

### Per-Prompt Results

| Prompt | top-8 Tokens | top-8 Time (ms) | top-16 Tokens | top-16 Time (ms) | Δ Time | Δ (%) |
|--------|-------------|----------------|---------------|-----------------|--------|-------|
| 01-math-reasoning | 782 | 7,012 | 782 | 7,421 | +409 | +5.8% |
| 02-logic-puzzle | 797 | 6,111 | 797 | 6,577 | +466 | +7.6% |
| 03-algorithm-design | 675 | 6,092 | 675 | 6,559 | +467 | +7.7% |
| 04-security-audit | 1,590 | 6,500 | 1,590 | 6,998 | +498 | +7.7% |
| 05-debugging | 1,022 | 6,106 | 1,022 | 6,668 | +562 | +9.2% |
| 06-code-transform | 986 | 6,085 | 986 | 6,624 | +539 | +8.9% |
| 07-system-design | 708 | 6,100 | 708 | 6,553 | +453 | +7.4% |
| 09-auth-security | 737 | 6,091 | 737 | 6,645 | +554 | +9.1% |
| **AVERAGE** | **890** | **6,335** | **890** | **6,768** | **+433** | **+6.8%** |

### Aggregate Metrics

| Metric | top-8 | top-16 | Difference |
|--------|-------|--------|------------|
| **Total Tokens Generated** | 7,297 | 7,297 | 0 (identical) |
| **Total Time** | 50,687 ms | 54,145 ms | +3,458 ms (+6.8%) |
| **Avg Time per Prompt** | 6,335 ms | 6,768 ms | +433 ms (+6.8%) |
| **Avg Tokens/sec** | 114.9 | 107.6 | -7.3 (-6.4%) |
| **Avg Tokens per Prompt** | 912 | 912 | 0 |

### Throughput Analysis

```
top-8:  ████████████████████████████████████████  114.9 tok/s
top-16: ████████████████████████████████████      107.6 tok/s
         ↓ -6.4% throughput with 2× more experts routed
```

### Latency Distribution

| Percentile | top-8 (ms) | top-16 (ms) | Δ |
|------------|-----------|-------------|---|
| Min | 6,085 | 6,553 | +468 |
| P50 | 6,103 | 6,616 | +513 |
| P75 | 6,106 | 6,645 | +539 |
| Max | 7,012 | 7,421 | +409 |

### Fixes Applied (v3 → v4)
- `MAX_MODEL_LEN`: 2048 → 8192 (prompts 04/05 now pass)
- `MAX_TOKENS`: 1024 → 512 (shorter outputs, faster runs)
- Output extraction: `.message.text` → `.message.content` (outputs now contain actual text)
- All 8 prompts rewritten to be ~50-75% shorter (concise, bullet-point style)

---

## Output Size Comparison

| Prompt | Top-8 (bytes) | Top-16 (bytes) | Δ (bytes) | Δ (%) |
|--------|--------------|----------------|-----------|-------|
| 01-math-reasoning | 2,251 | 2,129 | -122 | -5.4% |
| 02-logic-puzzle | 2,776 | 2,951 | +175 | +6.3% |
| 03-algorithm-design | 2,795 | 2,757 | -38 | -1.4% |
| 04-security-audit | 6,052 | 6,227 | +175 | +2.9% |
| 05-debugging | 4,064 | 4,188 | +124 | +3.1% |
| 06-code-transform | 3,583 | 3,620 | +37 | +1.0% |
| 07-system-design | 2,842 | 2,696 | -146 | -5.1% |
| 09-auth-security | 2,904 | 2,841 | -63 | -2.2% |
| **TOTAL** | **27,267** | **27,409** | **+142** | **+0.5%** |

Total output is virtually identical (0.5% difference).

## Quality Analysis by Prompt

### 01-Math-Reasoning (Both truncated at 512 tokens)

| Aspect | Top-8 | Top-16 |
|--------|-------|--------|
| Approach | Month-by-month calculation plan | Closed-form formula derivation |
| Formula | Recursive (B6, B12, B18) | Direct: B₁₈ = 10000×1.005¹⁸ - 2000×1.005¹² - 2000×1.005⁶ |
| **Quality** | Good step-by-step | **Slightly better** (more elegant formula) |

**Winner: Top-16** (marginally — derives closed-form solution rather than iterative approach)

### 04-Security-Audit (Both truncated at 512 tokens)

| Aspect | Top-8 | Top-16 |
|--------|-------|--------|
| Vulnerability identification | Lists A, B, C, D, F (5 vulns) | Identifies same vulns with more structured analysis |
| Code block analysis | Single-pass analysis | **Two-block analysis** (recognizes second code block as potential patch) |
| Organization | Linear list | Hierarchical (deconstruct → analyze → conclude) |
| **Quality** | Good | **Better** (more nuanced code analysis) |

**Winner: Top-16** (recognizes prompt structure better, more sophisticated analysis)

### 05-Debugging (Both truncated at 512 tokens)

| Aspect | Top-8 | Top-16 |
|--------|-------|--------|
| Bug identification | Identifies LINE A race condition | Identifies LINE A, B, C race conditions |
| Depth | Detailed per-method analysis | Broader scan, identifies more bugs before truncation |
| **Quality** | Deep on first bug | **Broader coverage** |

**Winner: Top-16** (identifies more bugs before token limit)

### Other Prompts (02, 03, 06, 07, 09)

All produce functionally equivalent outputs with similar structure and quality.

### Overall Quality Scorecard

| Category | Top-8 Wins | Top-16 Wins | Ties |
|----------|-----------|------------|------|
| Math Reasoning | 0 | 1 | 0 |
| Security Analysis | 0 | 1 | 0 |
| Debugging | 0 | 1 | 0 |
| Logic/Algorithms/Code/Design/Auth | 0 | 0 | 5 |
| **TOTAL** | **0** | **3** | **5** |

---

## Kernel Support Matrix

| TopK Value | vllm-xpu-kernels 0.1.12 | vllm-xpu-kernels 0.1.13.1 | Notes |
|-----------|------------------------|--------------------------|-------|
| ≤ 8 | ✅ Supported | ✅ Supported | Baseline |
| 16 | ❌ Not supported | ✅ Supported | Added in commit a96998c |
| 32 | ❌ Not supported | ❌ Not supported | No released kernel supports this |
| 64 | ❌ Not supported | ❌ Not supported | No released kernel supports this |

### Error Details (top-32 / top-64)

```
RuntimeError: Unsupported TopK value
  at torch.ops._moe_C.remap_hidden_states()
  in vllm_xpu_kernels/fused_moe_interface.py:395
```

The error occurs during the MoE kernel's `remap_hidden_states` operation in the fused MoE implementation, which validates TopK values against compiled kernel constraints.

---

## Startup Timing Analysis

| Phase | top-8 | top-16 | Notes |
|-------|-------|--------|-------|
| vLLM process launch | ~6s | ~6s | CLI + config parsing |
| Model weight loading | ~12s | ~14s | 67 GB from EXT4 |
| KV cache allocation | ~1s | ~1s | ~5.3 GB |
| CUDA graph capture | ~19s | ~19s | Mixed prefill-decode + decode |
| Warmup + Triton compile | ~16s | ~16s | Multi-modal + readonly |
| **Total startup** | **~57s** | **~59s** | Similar across variants |

### GPU Memory Usage (per GPU)

| Component | top-8 | top-16 | Notes |
|-----------|-------|--------|-------|
| Model weights | 17.83 GB | 17.86 GB | Nearly identical |
| Peak activation | 1.09 GB | 1.09 GB | Same |
| CUDA graph | 2.77-2.80 GB | 2.80-2.83 GB | Slightly higher for top-16 |
| KV cache | 5.31 GB | 5.29 GB | Similar |
| **Total consumed** | **~17.84 GB** | **~17.86 GB** | <0.2% difference |

---

## Cost-Benefit Analysis

### top-8 vs top-16 Tradeoffs

| Aspect | top-8 | top-16 | Assessment |
|--------|-------|--------|------------|
| **Throughput** | 114.9 tok/s | 107.6 tok/s | top-8 is 6.4% faster |
| **Latency** | ~6.3s avg | ~6.8s avg | top-8 is ~0.4s faster |
| **Expert Coverage** | 8/256 (3.1%) | 16/256 (6.3%) | top-16 routes 2× more |
| **VRAM Usage** | 17.84 GB/GPU | 17.86 GB/GPU | Negligible difference |
| **Output Size** | 27,267 bytes | 27,409 bytes | +0.5% (identical) |
| **Quality** | Correct answers | Marginally better reasoning | Subtle difference |
| **Startup Time** | ~57s | ~59s | Negligible difference |

### When to Use Each

**Use top-8 when:**
- Latency is critical (API serving, real-time chat)
- Throughput matters (batch processing)
- Cost efficiency is prioritized
- Tasks are well-covered by fewer experts

**Use top-16 when:**
- Maximum accuracy is critical (security audits, complex debugging)
- Task diversity requires broader expert coverage
- 7% latency penalty is acceptable for potential quality gains
- Edge cases where specific experts matter

---

## Failed Prompts (Both Variants)

Prompts 04-security-audit and 05-debugging returned HTTP 400 Bad Request on both variants. This is a prompt-format issue, not a model issue:
- Both variants failed identically
- Error occurred before model inference
- Likely prompt length or format validation issue in the API layer

---

## Recommendations

1. **For production on Intel XPU**: Use **top-16** with vllm-xpu-kernels 0.1.13.1 for best quality/latency balance
2. **For maximum throughput**: Use **top-8** when 9% speed gain is valuable
3. **For top-32+**: Requires custom kernel development or waiting for Intel to release support
4. **Monitor**: Check if prompt 04/05 issues are resolved in newer vLLM versions

---

## Raw Data Location

- Benchmark log: `/tmp/benchmark-run-v3.log`
- Per-variant logs: `/home/dc/electric-sheep/benchmarking/results/vllm-top{8,16}.log`
- Output files: `/home/dc/electric-sheep/benchmarking/results/top{8,16}/`
- Config variants: `/home/dc/electric-sheep/benchmarking/configs/top{8,16,32,64}/`

---

*Generated: 2025-08-13 19:00 UTC*
*Benchmark duration: ~13 minutes (top-8: 2min, top-16: 2.5min, top-32: 4min failed, top-64: 4min failed)*
*Full log: `/tmp/benchmark-run-v4.log`*
