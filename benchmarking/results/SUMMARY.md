# Intel Qwen3.6-35B-A3B MoE Expert Routing Benchmark

## Model
- **Name**: Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound
- **Architecture**: Qwen3.5 MoE (256 experts, 40 layers)
- **Total params**: ~35B (INT4 quantized, ~21 GB)
- **Test date**: $(date)

## Test Matrix

| Variant | `num_experts_per_tok` | Est Active Params |
|---|---|---|
| top-8 | 8 | ~3B |
| top-16 | 16 | ~6B |
| top-32 | 32 | ~12B |
| top-64 | 64 | ~24B |

## Performance Summary

| Variant | Prompt | Tokens | Time (ms) | Gen tok/s |
|---|---|---|---|---|
| top-16 | 01-math-reasoning | 782 | 7305 | 70.0 |
| top-16 | 02-logic-puzzle | 797 | 6480 | 79.0 |
| top-16 | 03-algorithm-design | 675 | 6466 | 79.1 |
| top-16 | 04-security-audit | 1590 | 6839 | 74.8 |
| top-16 | 05-debugging | 1022 | 6507 | 78.6 |
| top-16 | 06-code-transform | 986 | 6540 | 78.2 |
| top-16 | 07-system-design | 708 | 6454 | 79.3 |
| top-16 | 09-auth-security | 737 | 6489 | 78.9 |
| top-32 | 01-math-reasoning | 782 | 7896 | 64.8 |
| top-32 | 02-logic-puzzle | 797 | 7056 | 72.5 |
| top-32 | 03-algorithm-design | 675 | 7004 | 73.1 |
| top-32 | 04-security-audit | 1590 | 7431 | 68.9 |
| top-32 | 05-debugging | 1022 | 7113 | 71.9 |
| top-32 | 06-code-transform | 986 | 7105 | 72.0 |
| top-32 | 07-system-design | 708 | 7028 | 72.8 |
| top-32 | 09-auth-security | 737 | 7020 | 72.9 |
| top-64 | 01-math-reasoning | 782 | 9102 | 56.2 |
| top-64 | 02-logic-puzzle | 797 | 8261 | 61.9 |
| top-64 | 03-algorithm-design | 675 | 8282 | 61.8 |
| top-64 | 04-security-audit | 1590 | 8781 | 58.3 |
| top-64 | 05-debugging | 1022 | 8407 | 60.9 |
| top-64 | 06-code-transform | 986 | 8403 | 60.9 |
| top-64 | 07-system-design | 708 | 8258 | 62.0 |
| top-64 | 09-auth-security | 737 | 8332 | 61.4 |
| top-8 | 01-math-reasoning | 782 | 6588 | 77.7 |
| top-8 | 02-logic-puzzle | 797 | 6187 | 82.7 |
| top-8 | 03-algorithm-design | 675 | 6164 | 83.0 |
| top-8 | 04-security-audit | 1590 | 6584 | 77.7 |
| top-8 | 05-debugging | 1022 | 6271 | 81.6 |
| top-8 | 06-code-transform | 986 | 6249 | 81.9 |
| top-8 | 07-system-design | 708 | 6114 | 83.7 |
| top-8 | 09-auth-security | 737 | 6149 | 83.2 |

## Output Comparison

See individual output files in `results/outputs/` for full responses.

### Files
- `top{8,16,32,64}_{prompt}.md` — full output for each variant × prompt combination

## Conclusions

(To be filled after manual review of outputs)
