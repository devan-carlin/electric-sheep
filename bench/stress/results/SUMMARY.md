# Stress Test Results Summary

Aggregated comparison of model outputs across all test categories.

## How to Use

1. Run tests: `./scripts/run-test.sh <test.md> <model_name>`
2. Score results: `./scripts/score-test.sh <output_file.md>`
3. Compare models: `./scripts/compare-models.sh <model_a> <model_b>`
4. Update this file with the comparison output.

## Comparison Table

| Model | Platform | Quantization | Completeness | Correctness | Runnable | Restraint | Average |
|-------|----------|--------------|--------------|-------------|----------|-----------|---------|
| Qwen 3.6 27B | vLLM XPU | FP16 (baseline) | — | — | — | — | — |
| Qwen 3.6 27B | vLLM XPU | INT4 AutoRound | — | — | — | — | — |
| Qwen 3.6 27B | llama.cpp SYCL | GGUF Q4_K_M | — | — | — | — | — |

## Notes

- Score each criterion on a 1–5 scale.
- Run each test 3 times per model to account for temperature variance.
- Record the median score, not the mean (outliers skew results).
