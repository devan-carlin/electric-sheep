# MoE Expert Routing Benchmark

Head-to-head comparison of `num_experts_per_tok` (TopK) values for Mixture-of-Experts models on Intel Arc Pro B70 GPUs.

## What This Is

A benchmarking suite that tests how many MoE experts should be activated per token. The results challenge the assumption that "more experts = better outputs."

## Hardware

- **GPUs**: 4× Intel Arc Pro B70 (34 GB VRAM each, 128 GB total)
- **CPU**: AMD Ryzen Threadripper PRO 3945WX 12-Core
- **RAM**: 247 GB
- **Framework**: vLLM 0.26.1rc1 (XPU backend, FlashAttention v2, TP=4)

## Model

**Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound** — Qwen3.5 MoE architecture with 256 total experts across 40 layers, INT4 quantized to ~21 GB.

## Quick Start

```bash
# 1. Source GPU environment
source ../vllm/set-env-0123-gpu.sh

# 2. Activate vLLM virtual environment
source ../vllm/.venv/bin/activate

# 3. Run all variants (top-8, 16, 32, 64)
./scripts/run-benchmark.sh

# 4. Run specific variants only
./scripts/run-benchmark.sh 8 16
```

## Results

See `results/SUMMARY.md` for the latest benchmark run summary.

### Key Finding: The MoE Routing Quality Paradox

More active experts did **not** improve output quality — they degraded it.

| Variant | Avg Throughput | Quality Score | vs TopK=8 |
|---------|---------------|---------------|-----------|
| TopK=8 | 83.0 tok/s | 4.0/5 | — |
| TopK=16 | 77.0 tok/s | 4.1/5 | -7% slower |
| TopK=32 | 66.0 tok/s | 4.0/5 | -20% slower |
| TopK=64 | 49.0 tok/s | 3.4/5 | -41% slower |

**Recommendation:** TopK=8 is the sweet spot for production. TopK=16 is reasonable for reasoning-heavy workloads. TopK=64 shows measurable quality degradation.

## Directory Structure

```
benchmarking/
├── configs/           # Variant-specific model configs
│   ├── top8/          # num_experts_per_tok=8
│   ├── top16/         # num_experts_per_tok=16
│   ├── top32/         # num_experts_per_tok=32
│   └── top64/         # num_experts_per_tok=64
├── prompts/           # Test prompts (8 categories)
├── results/           # Benchmark outputs
│   ├── outputs/       # Full model responses
│   ├── timings/       # JSON timing data
│   └── SUMMARY.md     # Aggregated results
└── scripts/
    └── run-benchmark.sh  # Main benchmark script
```

## Prerequisites

1. **vLLM with XPU support** installed in a virtual environment
2. **vllm-xpu-kernels** patched for TopK > 10 (see `../docs/guides/moe-topk-kernel-patch.md`)
3. **qzeros patch** applied for Intel INT4 MoE models (see `../docs/guides/vllm-deployment.md`)
4. **Model weights** downloaded and symlinked in each config directory

## Test Prompts

8 prompts spanning different capability categories:

1. **Math reasoning** — compound interest calculations
2. **Logic puzzle** — Zebra Puzzle (15-clue constraint satisfaction)
3. **Algorithm design** — data structure implementation
4. **Security audit** — vulnerability identification
5. **Debugging** — race condition detection
6. **Code transformation** — refactoring exercise
7. **System design** — architecture planning
8. **Auth security** — authentication patterns

## Patches Required

This benchmark requires three patches to upstream vLLM/vllm-xpu-kernels:

1. **qzeros guard** — `inc_wna16_linear.py`: Handle empty qzeros tensors in mixed-precision MoE layers
2. **TopK kernel extensions** — `remap_hidden_states.cpp` + `moe_gather.cpp`: Add TopK=16, 32, 64 dispatch branches
3. **Config symlinks** — Each TopK variant needs variant-specific `config.json` with correct `num_experts_per_tok`

Full patch guides: [moe-topk-kernel-patch.md](../docs/guides/moe-topk-kernel-patch.md)

## License

[TODO: Add license]
