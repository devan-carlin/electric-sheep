# Electric Sheep

Local LLM inference across two GPU platforms: NVIDIA RTX 5090 (CUDA) and 4× Intel Arc Pro B70 (SYCL/XPU).

This project documents the build, deployment, and tuning processes for running large language models on self-hosted hardware. The goal is straightforward — maximize model capacity and throughput while maintaining full control over the stack.

---

## Hardware

### Workstation — RTX 5090 (Windows)

| Spec | Detail |
|------|--------|
| GPU | NVIDIA RTX 5090 |
| VRAM | 32 GB GDDR7 |
| Architecture | Blackwell (sm_100) |
| CUDA | 13.1 |
| Build | MSVC 2022, CMake 4.x |

CUDA remains the most mature platform for LLM inference. Single-GPU deployment is straightforward, driver support is solid, and the ecosystem (vLLM, llama.cpp, Ollama) is optimized first for NVIDIA. The RTX 5090 handles most 70B-class INT4 models comfortably within 32 GB.

Power-tuned via MSI Afterburner (70% cap ≈ 315W) for ~10% throughput reduction with significantly lower thermal output.

### AI Server — 4× Intel Arc Pro B70 (Ubuntu)

| Spec | Detail |
|------|--------|
| GPUs | 4× Intel Arc Pro B70 |
| VRAM | 128 GB total (32 GB each) |
| Architecture | Battlemage (Xe) |
| Driver | Level Zero / oneAPI 2026.1.0 |
| CPU | AMD Threadripper PRO 3945WX (12C/24T) |
| RAM | 247 GB |

Four Arc B70 GPUs provide 128 GB of aggregate VRAM at a cost comparable to a single RTX 5090. The Intel ecosystem is less mature — you'll encounter driver quirks and missing optimizations — but the trajectory is positive. vLLM XPU support is functional, llama.cpp's SYCL backend handles multi-GPU layer splitting, and oneAPI tooling improves with each release. This platform excels at running larger models via tensor parallelism and hosting multiple models simultaneously.

---

## Platform Notes

CUDA is the safe choice. If you need reliability and broad framework support, NVIDIA is the default. The Arc B70 platform trades some convenience for VRAM density — expect to spend more time on driver configuration and troubleshooting, but the capacity per dollar is significantly higher. Both platforms are production-capable; the choice depends on whether you prioritize ease of use or raw VRAM budget.

---

## Project Structure

```
electric-sheep/
├── build/
│   ├── ubuntu/          # AI Server setup (numbered, sequential)
│   ├── windows/         # Workstation setup (PowerShell)
│   └── common/          # Shared utilities (both platforms)
├── configs/
│   ├── vllm/            # vLLM runtime configs (model + server)
│   └── llama/           # llama.cpp runtime configs
├── docs/
│   ├── architecture.md  # Hardware specs + project layout
│   ├── guides/          # Deployment + technique guides
│   └── vllm/            # vLLM patch reference (INT4 quant, patch diffs)
├── vllm/                # vLLM runtime (created by setup scripts)
│   ├── launch/          # start-*.sh launchers (per model) + interactive launchers
│   ├── quantize/        # AutoRound INT4 quantization scripts
│   ├── env/             # set-env-*.sh XPU environment configs
│   └── experimental/    # one-off scripts (reap/slice, test, fix)
├── llama/               # llama.cpp runtime (created by setup scripts)
├── models/              # Shared model storage (symlink -> /mnt/data/models)
├── bench/             # all evaluation: throughput suite, stress tests, A/B, 128k matrix
├── models/              # Shared model storage (symlink -> /mnt/data/models)
└── articles/            # Draft articles (issues, optimizations, experiments)
```

---

## Getting Started

### On the AI Server (Ubuntu + Intel Arc B70)

**Full setup from scratch:**

```bash
# Option A: Interactive (asks what to build)
bash ~/electric-sheep/build/common/build-all.sh

# Option B: Build everything at once
bash ~/electric-sheep/build/common/build-all.sh --all
```

**Step-by-step:**

```bash
# 1. System packages, Python 3.12, oneAPI
bash ~/electric-sheep/build/ubuntu/01-install-prerequisites.sh

# 2. Create vLLM project directory + virtual environment
bash ~/electric-sheep/build/ubuntu/02-setup-project-directory.sh

# 3. Build vLLM with XPU support
bash ~/electric-sheep/build/ubuntu/03-build-vllm-xpu.sh

# 4. Patch MoE models (required for Intel 35B-A3B)
bash ~/electric-sheep/build/ubuntu/03.1-patch-vllm-moe-qzeros.sh

# 5. Set up llama.cpp
bash ~/electric-sheep/build/ubuntu/04-setup-llama.sh
bash ~/electric-sheep/build/ubuntu/05-build-llama-cpp.sh

# 6. Download models
bash ~/electric-sheep/build/ubuntu/06-download-models.sh
```

**Convert a model to INT4:**

```bash
# Max quality (slower, ~12-16 hours for 30B+ models)
bash ~/electric-sheep/build/common/convert-int4-autoround.sh <huggingface-repo>

# Parallel mode (faster, ~2-4 hours, uses all 4 GPUs)
bash ~/electric-sheep/build/common/convert-int4-autoround-parallel.sh <huggingface-repo> --parallel --batch-size 4
```

### On the Workstation (Windows + RTX 5090)

**Build llama.cpp with CUDA:**

```powershell
cd ~/electric-sheep/build/windows
.\01-install-prerequisites.ps1
.\02-build-llama-cpp.ps1
```

**Set up Ollama:**

```powershell
.\03-setup-ollama.ps1
```

See `04-llama-cpp-deployment-guide.md` and `05-ollama-deployment-guide.md` for runtime configuration.

### Common Utilities

| Script | Purpose |
|--------|---------|
| `build/common/build-all.sh` | Orchestrate full build pipeline |
| `build/common/convert-int4-autoround.sh` | Quantize models to INT4 (single GPU) |
| `build/common/convert-int4-autoround-parallel.sh` | Quantize models to INT4 (multi-GPU) |
| `build/common/setup-gpu-power-limits.sh` | GPU power tuning |
| `build/common/tailscale-setup.md` | Mesh network between machines |

---

## Available Models

### vLLM (safetensors)

| Model | Format | Notes |
|-------|--------|-------|
| Qwen 3.6 27B INT4 | AutoRound | Primary model, full context |
| Qwen 3.6 35B-A3B INT4 | AutoRound (MoE) | Requires MoE patch |
| Gemma 4 31B INT4 | AutoRound | No patching needed |
| Gemma 4 26B-A4B INT4 | AutoRound (MoE) | No patching needed |

### llama.cpp (GGUF)

| Model | Format | Notes |
|-------|--------|-------|
| DeepSeek V4-Flash | UD-IQ3_XXS | ~98 GB, 4 shards, 96K context |

See `configs/vllm/model-configs.md` for launch commands and VRAM budgets.

---

## Runtime Configs

Launchers live in `serve/` (live stack: `start-all.sh` + `start-*-llama.sh`; vLLM
fallbacks in `serve/fallback/`), XPU environment configs in `vllm/env/`, and
reference configs in `configs/`:

```bash
# Start the full 4-GPU stack (2x ComfyUI + 2x llama.cpp)
bash ~/electric-sheep/serve/start-all.sh

# Or run a vLLM fallback launcher (sources the matching env config)
bash ~/electric-sheep/serve/fallback/start-qwen3.8-27b-int4.sh

# Or source the env config manually and serve any model
source ~/electric-sheep/vllm/env/set-env-0123-gpu.sh
vllm serve <model-path> --tensor-parallel-size 4 --kv-cache-dtype fp8

# Start llama.cpp DeepSeek
bash ~/electric-sheep/configs/llama/deepseek/start-deepseek-v4-flash.sh
```

---

## Docs & Guides

| Document | What it covers |
|----------|----------------|
| `docs/architecture.md` | Hardware specs, project layout, runtime paths |
| `docs/guides/vllm-deployment.md` | vLLM XPU deployment on Intel Arc |
| `docs/guides/llama-deployment.md` | llama.cpp SYCL deployment on Intel Arc |
| `docs/guides/deepseek-int4-conversion.md` | Full INT4 quantization walkthrough |
| `docs/guides/arc-b70-power-tuning.md` | sysfs power caps + frequency limits for Intel Arc B70 |
| `docs/guides/rtx-5090-power-tuning.md` | MSI Afterburner power limits for RTX 5090 |
| `docs/guides/tailscale-setup.md` | Mesh networking between machines |

---

## Model Evaluation

The `bench/stress/` directory contains 53 prompts across 13 categories for evaluating model output quality:

- **Game prompts** (12) — single-file HTML apps (Tetris, Minesweeper, Snake, Sudoku, etc.)
- **Compliance** (4) — strict JSON output, agentic planning, technical writing, API docs
- **Multi-file projects** (4) — full-stack CRUD, CLI tools, library APIs, microservices
- **Code transformation** (3) — cross-language conversion, refactoring, type inference
- **Debugging** (5) — race conditions, memory leaks, SQL optimization, CI diagnosis
- **Long context** (2) — needle-in-haystack, cross-file reference tracking
- **Data & infrastructure** (4) — ETL pipelines, Docker Compose, CI/CD, SQL migrations
- **Security** (7) — vulnerability audits, crypto review, OAuth, supply chain, Terraform
- **Reasoning & math** (4) — multi-step math, logic puzzles, algorithm design, scheduling
- **System design** (3) — URL shortener, rate limiter, event processing pipeline

See `bench/stress/README.md` for the full test index and helper scripts.

Use these to compare output quality between models, quantization levels, or platforms.

---

## Notes

- **vLLM Python:** 3.12 via deadsnakes PPA on Ubuntu 26.04
- **HF CLI:** Use `hf download` (not deprecated `huggingface-cli`)
- **Tailscale:** Mesh network between AI Server and Workstation
- **Models directory:** Shared between vLLM and llama.cpp (`~/electric-sheep/models/`)

---

---

*Questions or issues? Open a PR or reach out directly.*