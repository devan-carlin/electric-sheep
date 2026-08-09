# Copilot Instructions — `electric-sheep` Workspace

## Privacy & Redaction

- **Never** include real usernames, internal IP addresses, or personally identifiable information in generated documents.
- Redact internal IPs as `192.168.x.x` or `172.17.x.x`.
- Replace real usernames with `user` or `dc` (generic placeholder).
- Hardware IDs, PCIe slots, and GPU device IDs are safe to include (local topology, not externally identifiable).

## Project Structure Conventions

- **Setup scripts:** `~/ubuntu-b70/` (numbered: `01-`, `02-`, `03-`, ...)
- **vLLM project root:** `~/electric-sheep/vllm/`
- **vLLM virtual environment:** `~/electric-sheep/vllm/.venv/` (Python 3.12)
- **vLLM environment config:** `~/electric-sheep/vllm/set-env-*.sh`
- **llama.cpp project root:** `~/electric-sheep/llama/`
- **llama.cpp source:** `~/electric-sheep/llama/llama.cpp/` (cloned repo, build in `build/`)
- **llama.cpp environment config:** `~/electric-sheep/llama/set-env.sh`
- **llama.cpp DeepSeek scripts:** `~/electric-sheep/llama/deepseek/`
- **Model storage:** `~/electric-sheep/models/` (shared by vLLM and llama.cpp)
- **Do NOT** reference legacy paths (`~/.venv-b70-minimax`, `/mnt/fast-ai/`, `/home/dc/intel-vllm-01/`, `~/llama.cpp/`) in new documents.

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

### vLLM (safetensors)

1. **Qwen 3.6 27B INT4** (`Intel/Qwen3.6-27B-int4-AutoRound`) — Primary model, full context
2. **Qwen 3.6 35B-A3B INT4** (`Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound`) — MoE, requires `patch-vllm-moe-qzeros.sh`
3. **Gemma 4 31B INT4** (`Intel/gemma-4-31B-it-int4-AutoRound-V2`) — Standard INT4, no patching needed
4. **Gemma 4 26B-A4B INT4** (`Intel/gemma-4-26B-A4B-it-int4-AutoRound`) — MoE, standard INT4, no patching needed

### llama.cpp (GGUF)

5. **DeepSeek V4-Flash UD-IQ3_XXS** (`unsloth/DeepSeek-V4-Flash-0731-GGUF`) — ~98 GB, 4 shards, 96K context, ~13 t/s decode

See `ubuntu-b70/vllm/06-model-configs.md` for vLLM launch commands and VRAM budgets.
See `ubuntu-b70/llama/03-llama-cpp-deployment-guide.md` for llama.cpp deployment.
See `ubuntu-b70/llama/deepseek/run-stats.md` for DeepSeek performance benchmarks.

## CLI Tools

- **HuggingFace CLI:** `huggingface-cli` is deprecated. Use `hf` instead (e.g., `hf download <repo> --local-dir <path>`). The `hf` CLI is installed via pipx on Ubuntu 26.04.
- **pipx:** Required for system-wide Python CLI tools on Ubuntu 26.04 (PEP 668 blocks system pip).

## Known Patches

- **35B-A3B MoE qzeros crash:** Intel's `35B-A3B-int4-mixed-AutoRound` uses asymmetric quantization on attention layers and symmetric (empty `qzeros`) on expert layers. vLLM's `inc_wna16_linear.py` crashes on `torch.copy_()` due to shape mismatch. Run `patch-vllm-moe-qzeros.sh` after compiling vLLM to apply the guarded copy fix.
- **vllm-gguf-plugin:** Install with `pip install --no-deps vllm-gguf-plugin` to avoid pip resolving deps that overwrite locally-built vllm-xpu-kernels.

## llama.cpp SYCL Rules

- **Build path:** Source at `~/electric-sheep/llama/llama.cpp/`, build output in `build/`
- **Environment:** `source ~/electric-sheep/llama/set-env.sh` (loads oneAPI + SYCL tuning)
- **GPU selector argument:** `set-env.sh 0,1` for GPUs 0+1, `set-env.sh 0` for GPU 0 only
- **q8_0 KV cache unsupported:** SYCL backend lacks `concat` kernel for q8_0. Use f16 (default).
- **Flash attention:** Use `--flash-attn auto` (requires a value, not a flag).
- **Load mode:** Use `--load-mode mmap` (not deprecated `--mmap`).
- **Layer split mode:** `-sm layer` for multi-GPU (most stable on SYCL).
- **DeepSeek V4-Flash:** 96K context (`-c 98304`), batch size 4096, 12 CPU threads. GPU 2 is the VRAM bottleneck (88% at 96K).

## Performance Notes

- **DeepSeek V4-Flash (llama.cpp):** 58 t/s prompt processing, ~13 t/s token generation, ~336W total power draw
- **vLLM gguf-plugin:** Use `--no-deps` flag to avoid overwriting local kernels

## Document Generation Rules

- New documents should use the current project structure only — no legacy paths.
- Keep reference/historical documents until deployment is validated end-to-end.
- Use clear, actionable commands with explanations for each step.
- Include verification gates after critical steps (e.g., XPU device count check).
