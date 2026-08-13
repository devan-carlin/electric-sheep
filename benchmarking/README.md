# Huihui-CyberStrike MoE Expert Routing Benchmark

## Objective

Test how increasing `num_experts_per_tok` (active experts per token) affects output quality and performance on the Huihui-CyberStrike-OffSec-35B-abliterated model running on 4× Intel Arc Pro B70 GPUs (128 GB VRAM).

## Model

| Property | Value |
|---|---|
| **Name** | Huihui-CyberStrike-OffSec-35B-abliterated |
| **Architecture** | Qwen3.5 MoE |
| **Total params** | ~35B |
| **Experts** | 256 total |
| **Layers** | 40 (linear + full attention mix) |
| **Hidden size** | 2048 |
| **MoE intermediate** | 512 |
| **Size** | 67 GB (16 shards) |

## Test Matrix

| Variant | `num_experts_per_tok` | Est Active Params | Expected Impact |
|---|---|---|---|
| **top-8** | 8 | ~3B | Baseline (trained routing) |
| **top-16** | 16 | ~6 | 2× active params, double expert compute |
| **top-32** | 32 | ~12B | 4× active params, significant slowdown |
| **top-64** | 64 | ~24B | 8× active params, near-dense behavior |

## Test Prompts (9 total)

| # | Prompt | Category | Tests |
|---|---|---|---|
| 01 | Multi-step math & word problems | Reasoning | Math, logic, optimization |
| 02 | Five houses logic puzzle | Reasoning | Constraint satisfaction |
| 03 | Algorithm design & analysis | Reasoning | CS fundamentals, Big-O |
| 04 | Vulnerability audit | Security | Code security analysis |
| 05 | Race condition debugging | Debugging | Concurrency bugs |
| 06 | Python to Go translation | Code transform | Cross-language reasoning |
| 07 | URL shortener design | System design | Architecture thinking |
| 09 | Auth implementation review | Security | Auth patterns, OWASP |

## How to Run

```bash
# Run all variants (8, 16, 32, 64)
./scripts/run-benchmark.sh

# Run specific variants only
./scripts/run-benchmark.sh 8 32

# Run just the baseline
./scripts/run-benchmark.sh 8
```

## Output Structure

```
results/
├── SUMMARY.md              # Aggregated results table
├── outputs/
│   ├── top8_01-math-reasoning.md
│   ├── top16_01-math-reasoning.md
│   ├── top32_01-math-reasoning.md
│   ├── top64_01-math-reasoning.md
│   └── ... (all variant × prompt combos)
└── timings/
    ├── top8_01-math-reasoning.json
    ├── top16_01-math-reasoning.json
    └── ... (timing data for each run)
```

## Evaluation Criteria

1. **Correctness**: Are math answers right? Does the logic puzzle solve correctly?
2. **Completeness**: Does the model address all parts of multi-part prompts?
3. **Coherence**: Is the output well-structured and non-repetitive?
4. **Code quality**: Are code examples syntactically correct and idiomatic?
5. **Performance**: tok/s, latency, VRAM usage per variant

## Notes

- All variants share the same weight files (symlinked) — only `config.json` differs
- vLLM is restarted between variants to ensure clean GPU state
- Temperature=0.2 for deterministic-ish outputs across variants
- Max tokens=2048 per response
