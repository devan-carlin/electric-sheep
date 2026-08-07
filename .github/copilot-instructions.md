# Copilot Instructions — `electric-sheep` Workspace

## Privacy & Redaction

- **Never** include real usernames, internal IP addresses, or personally identifiable information in generated documents.
- Redact internal IPs as `192.168.x.x` or `172.17.x.x`.
- Replace real usernames with `user` or `dc` (generic placeholder).
- Hardware IDs, PCIe slots, and GPU device IDs are safe to include (local topology, not externally identifiable).

## Project Structure Conventions

- **Project root:** `~/electric-sheep/vllm/`
- **Virtual environment:** `~/electric-sheep/vllm/.venv/` (Python 3.12)
- **Environment config:** `~/electric-sheep/vllm/set-env.sh`
- **Model storage:** `~/electric-sheep/vllm/models/`
- **Do NOT** reference legacy paths (`~/.venv-b70-minimax`, `/mnt/fast-ai/`, `/home/dc/intel-vllm-01/`) in new documents.

## Hardware Context

- **OS:** Ubuntu 26.04 LTS (Kernel 7.0.0-29)
- **CPU:** AMD Threadripper PRO 3945WX (12C/24T)
- **RAM:** 247 GiB
- **GPU:** 4× Intel Arc Pro B70 (31.89 GiB each, `xe` driver)
- **Storage:** Samsung 990 PRO 2TB NVMe
- **oneAPI:** 2026.1.0, Level Zero driver 26.22.38646.7

## vLLM Deployment Rules

- **Target device:** `VLLM_TARGET_DEVICE=xpu`
- **Tensor parallelism:** `--tensor-parallel-size 4` (or `2` for dual-model)
- **GPU memory utilization:** `0.80` (conservative ceiling; `0.90+` fails pre-flight)
- **KV cache dtype:** `fp8` (reduces memory by ~50% vs fp16)
- **Prefix caching:** `--enable-prefix-caching`
- **Python version:** 3.12 (via deadsnakes PPA on Ubuntu 26.04)

## Available Models

1. **Qwen 3.6 27B INT4** (`Intel/Qwen3.6-27B-int4-AutoRound`) — Primary model, full context
2. **Qwen 3.6 35B-A3B INT4** (`Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound`) — MoE, requires `patch-vllm-moe-qzeros.sh`
3. **Gemma 4 31B INT4** (`Intel/gemma-4-31B-it-int4-AutoRound-V2`) — Standard INT4, no patching needed
4. **Gemma 4 26B-A4B INT4** (`Intel/gemma-4-26B-A4B-it-int4-AutoRound`) — MoE, standard INT4, no patching needed

See `model-configs.md` for launch commands, VRAM budgets, and dual-model deployment strategies.
See `vllm-deployment-guide.md` for the definitive end-to-end deployment instructions.

## Known Patches

- **35B-A3B MoE qzeros crash:** Intel's `35B-A3B-int4-mixed-AutoRound` uses asymmetric quantization on attention layers and symmetric (empty `qzeros`) on expert layers. vLLM's `inc_wna16_linear.py` crashes on `torch.copy_()` due to shape mismatch. Run `patch-vllm-moe-qzeros.sh` after compiling vLLM to apply the guarded copy fix.

## Document Generation Rules

- New documents should use the current project structure only — no legacy paths.
- Keep reference/historical documents until deployment is validated end-to-end.
- Use clear, actionable commands with explanations for each step.
- Include verification gates after critical steps (e.g., XPU device count check).
