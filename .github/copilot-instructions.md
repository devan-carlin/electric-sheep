# Copilot Instructions — `electric-sheep` Workspace

## Privacy & Redaction

- **Never** include real usernames, internal IP addresses, or personally identifiable information in generated documents.
- Redact internal IPs as `192.168.x.x` or `172.17.x.x`.
- Replace real usernames with `user` or `dc` (generic placeholder).
- Hardware IDs, PCIe slots, and GPU device IDs are safe to include (local topology, not externally identifiable).

## Project Structure Conventions

- **Setup scripts:** `~/electric-sheep/build/ubuntu/` (numbered: `01-`, `02-`, `03-`, ...)
- **Common utilities:** `~/electric-sheep/build/common/` (platform-agnostic)
- **Windows scripts:** `~/electric-sheep/build/windows/` (PowerShell)
- **Runtime config docs:** `~/electric-sheep/docs/vllm/` (model-configs.md, server-config-baseline.md, deepseek-run-stats.md)
- **Benchmarks (all evaluation):** `~/electric-sheep/bench/` (throughput suite at root, `stress/` 53-prompt quality suite, `ab/` blind A/B tests, `vllm-128k/` long-context matrix)
- **Documentation:** `~/electric-sheep/docs/` (architecture.md, guides/)
- **Server launchers (all 4 GPUs):** `~/electric-sheep/serve/` (`start-all.sh` + per-slot `start-*-llama.sh`; vLLM fallbacks in `serve/fallback/`)
- **vLLM project root:** `~/electric-sheep/vllm/` (venv, source, quant, env configs - no launchers)
- **vLLM virtual environment:** `~/electric-sheep/vllm/.venv/` (Python 3.12). The **verified production venv** is `~/vllm-fresh-venv/` (fresh gate build `0.1.dev1+gc39076fef.d20260829.xpu` with the qwen4exp patch applied) — use this one for serving Qwen3.8-Flash-Next.
- **vLLM quantization:** `~/electric-sheep/vllm/quantize/` (AutoRound INT4 scripts)
- **vLLM environment config:** `~/electric-sheep/vllm/env/set-env-*.sh`
- **vLLM experimental:** `~/electric-sheep/vllm/experimental/` (one-off reap/slice/test/fix scripts)
- **vLLM port tests:** `~/electric-sheep/vllm/tests/` (phase0..phase_d dev-test/diagnostic scripts for the qwen4exp port)
- **vLLM patch reference:** `~/electric-sheep/docs/vllm/` (INT4-QUANTIZATION-PATCHES.md, PATCHES-DIFF.md)
- **llama.cpp project root:** `~/electric-sheep/llama/`
- **llama.cpp source:** `~/electric-sheep/llama/llama.cpp/` (cloned repo, build in `build/`)
- **llama.cpp environment config:** `~/electric-sheep/llama/set-env.sh`
- **llama.cpp DeepSeek launcher:** `~/electric-sheep/serve/fallback/start-deepseek-v4-flash.sh`
- **Model storage:** `~/electric-sheep/models/` (shared by vLLM and llama.cpp). This is a **symlink to `/mnt/data/models`** (fast NVMe) — both paths are identical.
- **Do NOT** reference legacy paths (`~/.venv-b70-minimax`, `/mnt/fast-ai/`, `/home/dc/intel-vllm-01/`, `~/llama.cpp/`, `~/ubuntu-b70/`, `~/windows-5090/`, `~/code-8-7-26/`) in new documents.

## Model Directory Naming Standard

- **Format:** `<hf-username>-<hf-repo-name>` — the HuggingFace repo path with `/` replaced by `-`.
- **Drop redundant author prefix:** if the repo name already starts with the author's name, omit the author (e.g. `huihui-ai/Huihui--...` → `Huihui--...`).
- **GGUF repos:** drop the trailing `-GGUF` suffix from the directory name (keep it on the files).
- **Preserve case** exactly as the HF repo name (e.g. `Qwen3.8-27B`, `gemma-4-31b`).
- **Examples:**
  - `Intel/gemma-4-26B-A4B-it-int4-AutoRound` → `Intel-gemma-4-26B-A4B-it-int4-AutoRound`
  - `devan-carlin/Qwen3.8-27B--ara-int4-AutoRound` → `devan-carlin-Qwen3.8-27B--ara-int4-AutoRound`
  - `huihui-ai/Huihui-gemma-4-26B-A4B-it-qat-q4_0-unquantized--GGUF` → `Huihui-gemma-4-26B-A4B-it-qat-q4_0-unquantized-`
- **Quantized-from-source suffix:** when a checkpoint is quantized from a local source, append the source dtype: `-W4A16-BF16src` (quantized from BF16) or `-W4A16-FP8src` (quantized from FP8). E.g. `devan-carlin-Qwen3.8-Flash-Next-W4A16-BF16src`, `devan-carlin-Qwen3.8-Flash-Next--2-W4A16-BF16src`.

## Launcher Script & Alias Naming

- **Script format:** `start-<alias>[-<variant>]-<engine>.sh` in `serve/` (e.g. `start-qwen-256k-vllm.sh`, `start-qwen-256k--vllm.sh`). Front-door scripts that dispatch to multiple engines drop the engine suffix (`start-qwen-256k.sh`).
- **Common alias rule:** any 256k-context model served via vLLM on `:8000` is served under the alias **`qwen-256k`** — that is the name used in other apps (Open WebUI etc.). The variant (e.g. ``) goes in the script name, not the alias. Whichever 256k model is up on `:8000` is what other apps see as `qwen-256k`.
- **Logs:** launchers `exec` the server with output to `serve/logs/vllm_${PORT}.log` (e.g. `vllm_8000.log`). Do not leave a `LOG=` variable unused — the redirect must actually target `$LOG`.
- **Refusal benchmark results:** `results-<variant>.json` in `~/neon-demon/refusal-benchmark/` (e.g. `results--2.json`, `results-base.json`).

## Hardware Context

### AI Server (Ubuntu — Intel Arc B70)

- **OS:** Ubuntu 26.04 LTS (Kernel 7.0.0-29)
- **CPU:** AMD Threadripper PRO 3945WX (12C/24T)
- **RAM:** 247 GiB
- **GPU:** 4× Intel Arc Pro B70 (31.89 GiB each, `xe` driver)
- **Storage:** Samsung 990 PRO 2TB NVMe
- **oneAPI:** 2026.1.0, Level Zero driver 26.22.38646.7

### Workstation (Windows — RTX 5090)

- **GPU:** NVIDIA RTX 5090 (32 GB, Blackwell sm_100)
- **CUDA:** 13.1
- **Build:** MSVC 2022, CMake 4.x
- **Power tuning:** MSI Afterburner (70% power cap ≈ 315W, ~10% throughput reduction)

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

See `docs/vllm/model-configs.md` for vLLM launch commands and VRAM budgets.
See `docs/guides/llama-deployment.md` for llama.cpp deployment.
See `docs/vllm/deepseek-run-stats.md` for DeepSeek performance benchmarks.

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
