# Huihui-CyberStrike MoE Expert Routing Benchmark

## Model
- **Name**: Huihui-CyberStrike-OffSec-35B-abliterated
- **Architecture**: Qwen3.5 MoE (256 experts, 40 layers)
- **Total params**: ~35B
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
| top-16 | 01-math-reasoning | 782 | 7421 | 68.9 |
| top-16 | 02-logic-puzzle | 797 | 6577 | 77.8 |
| top-16 | 03-algorithm-design | 675 | 6559 | 78.0 |
| top-16 | 04-security-audit | 1590 | 6998 | 73.1 |
| top-16 | 05-debugging | 1022 | 6668 | 76.7 |
| top-16 | 06-code-transform | 986 | 6624 | 77.2 |
| top-16 | 07-system-design | 708 | 6553 | 78.1 |
| top-16 | 09-auth-security | 737 | 6645 | 77.0 |
| top-8 | 01-math-reasoning | 782 | 7012 | 73.0 |
| top-8 | 02-logic-puzzle | 797 | 6111 | 83.7 |
| top-8 | 03-algorithm-design | 675 | 6092 | 84.0 |
| top-8 | 04-security-audit | 1590 | 6500 | 78.7 |
| top-8 | 05-debugging | 1022 | 6106 | 83.8 |
| top-8 | 06-code-transform | 986 | 6085 | 84.1 |
| top-8 | 07-system-design | 708 | 6100 | 83.9 |
| top-8 | 09-auth-security | 737 | 6091 | 84.0 |

## Output Comparison

See individual output files in `results/outputs/` for full responses.

### Files
- `top{8,16,32,64}_{prompt}.md` — full output for each variant × prompt combination

## Conclusions

(To be filled after manual review of outputs)
