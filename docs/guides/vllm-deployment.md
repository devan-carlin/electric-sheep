# vLLM Deployment Guide — Intel Arc B70 × 4

**Target:** Ubuntu 26.04 LTS (Resolute)  
**Hardware:** 4× Intel Arc Pro B70 (31.89 GiB each, `xe` driver)  
**Python:** 3.12 (deadsnakes PPA)  
**Project Root:** `~/electric-sheep/vllm/`

---

## Phase 1: System Preparation

Ensure base drivers, oneAPI toolkit, and Python 3.12 are installed.

```bash
# 1. System packages & deadsnakes PPA
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake git git-lfs clinfo libdrm-dev software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update

# 2. Python 3.12 & Level Zero runtime
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev \
    intel-opencl-icd libze-intel-gpu1 libze1 libze-dev

# 3. oneAPI Toolkit (if not already installed)
sudo apt-get install -y intel-oneapi-toolkit
```

---

## Phase 2: Project Setup & Model Download

Run the setup script to create the project structure, virtual environment, environment configs, and download all 4 models.

```bash
bash ~/electric-sheep/scripts/ubuntu/02-setup-project-directory.sh
```

**What this creates:**
```
~/electric-sheep/vllm/
├── .venv/
├── set-env-0123-gpu.sh   # All 4 GPUs (TP=4)
├── set-env-01-gpu.sh     # GPUs 0,1 (TP=2)
├── set-env-23-gpu.sh     # GPUs 2,3 (TP=2)
└── models/
    ├── Intel-Qwen3.6-27B-int4-AutoRound/
    ├── Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound/
    ├── Intel-gemma-4-31B-it-int4-AutoRound-V2/
    └── Intel-gemma-4-26B-A4B-it-int4-AutoRound/
```

---

## Phase 3: Virtual Environment & Stack Compilation

Activate the venv and compile the XPU stack from source.

```bash
# 1. Activate environment
source ~/electric-sheep/vllm/.venv/bin/activate

# 2. Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch XPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 4. Clone source repositories
mkdir -p ~/electric-sheep/vllm/vllm-src
cd ~/electric-sheep/vllm/vllm-src
git clone https://github.com/vllm-project/vllm-xpu-kernels.git
git clone https://github.com/vllm-project/vllm.git

# 5. Compile vllm-xpu-kernels
cd ~/electric-sheep/vllm/vllm-src/vllm-xpu-kernels
pip install setuptools setuptools-scm cmake ninja packaging psutil
pip install . --no-build-isolation

# 6. Compile vLLM engine
cd ~/electric-sheep/vllm/vllm-src/vllm
pip install setuptools-rust
pip install -r requirements-build.txt 2>/dev/null || pip install -r requirements/build.txt 2>/dev/null || true
export VLLM_TARGET_DEVICE="xpu"
pip install . --no-build-isolation

# 7. Enforce triton-xpu override (critical for Intel backend)
pip install triton-xpu --index-url https://download.pytorch.org/whl/xpu --force-reinstall --no-deps
```

**Fallback:** If `import vllm_xpu_kernels._C` fails after compilation, manually sync the `.so` binaries:
```bash
cp ~/electric-sheep/vllm/vllm-src/vllm-xpu-kernels/build/lib.linux-x86_64-cpython-312/vllm_xpu_kernels/*.so \
   ~/electric-sheep/vllm/.venv/lib/python3.12/site-packages/vllm_xpu_kernels/
```

---

## Phase 4: Verification & Patching

### Hardware Verification Gate

```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-0123-gpu.sh

python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('XPU Available:', torch.xpu.is_available())
print('XPU Device Count:', torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    print(f'  GPU {i}: {torch.xpu.get_device_name(i)}')
"
```
**Gate Requirement:** Must output `XPU Device Count: 4` with all 4 B70 GPUs listed.

### Apply MoE Patch (Required for 35B-A3B)

If you plan to run the Qwen 35B-A3B MoE model, apply the guarded `qzeros` copy fix:

```bash
bash ~/electric-sheep/scripts/ubuntu/03.1-patch-vllm-moe-qzeros.sh
```

---

## Phase 5: Launch & Deployment

Choose your deployment mode. See `model-configs.md` for full parameter rationale.

### Option A: Single Model (Full Context, 4 GPUs)

```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-0123-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 4 \
    --max-model-len 232144 \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

### Option B: Dual Model (Reduced Context, 2 GPUs each)

**Best for:** Running two models simultaneously with higher concurrency (16 seqs) but reduced context (32k tokens).

**Terminal 1 — GPUs 0,1 (Qwen 27B):**
```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-01-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

**Terminal 2 — GPUs 2,3 (Qwen 35B-A3B):**
```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-23-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound \
    --served-model-name qwen3.6-35b-a3b \
    --host 0.0.0.0 \
    --port 8031 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

### Option C: Dual Model (Full Context, 2 GPUs each)

**Best for:** Extended context window on both models, limited to 1 concurrent request per model. 

> **VRAM Reality Check:** Full 232k context on TP=2 requires ~36.5 GB/GPU (KV cache + weights), which exceeds the B70's 31.89 GB physical limit. The commands below cap context at **128k** to stay within the `0.80` utilization budget (~23.1 GB/GPU). If you absolutely need 232k, change `--max-model-len 232144` and bump to `--gpu-memory-utilization 0.95` (accepts pre-flight OOM risk).

**Terminal 1 — GPUs 0,1 (Qwen 27B, 128k Context):**
```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-01-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 2 \
    --max-model-len 128000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

**Terminal 2 — GPUs 2,3 (Qwen 35B-A3B, 128k Context):**
```bash
source ~/electric-sheep/vllm/.venv/bin/activate
source ~/electric-sheep/vllm/env/set-env-23-gpu.sh

python3 -m vllm.entrypoints.openai.api_server \
    --model ~/electric-sheep/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound \
    --served-model-name qwen3.6-35b-a3b \
    --host 0.0.0.0 \
    --port 8031 \
    --tensor-parallel-size 2 \
    --max-model-len 128000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm
```

---

## Phase 6: Web UI & Operations

### Deploy Open-WebUI

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8030/v1" \
  -e OPENAI_API_KEY="sk-vllm" \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```
Access at `http://localhost:3000`.

### Canary Test

```bash
curl http://localhost:8030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Analyze the specs of 4x Intel Arc B70 GPUs working in tensor parallelism."}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

---

## Troubleshooting & Known Issues

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'vllm_xpu_kernels._C'` | `.so` binaries not copied to `site-packages` | Run manual `cp` fallback from Phase 3 |
| `ImportError: cannot import name 'intel' from 'triton._C'` | Standard `triton` overwrote `triton-xpu` | `pip install triton-xpu --index-url https://download.pytorch.org/whl/xpu --force-reinstall --no-deps` |
| `RuntimeError: copy_() shape mismatch` on 35B-A3B | Empty `qzeros` on symmetric MoE expert layers | Run `patch-vllm-moe-qzeros.sh` |
| Pre-flight VRAM check fails at `0.90+` | Level-Zero driver reserves ~1.5 GiB/GPU | Use `--gpu-memory-utilization 0.80` |
| `resource_tracker: leaked shared_memory` | Python IPC cleanup after process exit | Safe to ignore; OS reclaims memory |

---

## Reference Documents

- `06-server-config-baseline.md` — Hardware & software baseline snapshot
- `05-model-configs.md` — Full model matrix, VRAM budgets, and launch commands
- `02-setup-project-directory.sh` — Automated project & model provisioning
- `05-patch-vllm-moe-qzeros.sh` — MoE qzeros guarded copy fix
- `07-vllm-deployment-guide.md` — This guide
- `.github/copilot-instructions.md` — Workspace rules & conventions
