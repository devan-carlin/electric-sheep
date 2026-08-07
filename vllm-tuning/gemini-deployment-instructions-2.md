Here is the complete, consolidated, end-to-end execution guide for your **Ubuntu 26.04 (Resolute) system with 4x Intel Arc Pro B70 GPUs**.

This refactored blueprint incorporates every workaround we established—including adding the deadsnakes PPA for Python 3.12 compatibility, pre-injecting build tools like `psutil` and `setuptools-rust`, compiling without build isolation, and protecting your custom kernels from dependency overwrites.

---

## 📋 Comprehensive Execution Blueprint

### Phase 1: Driver Infrastructure & Package Provisioning

Register the necessary legacy runtime repository paths and deploy the Level-Zero interface layers natively targeted by the core build layers.

```bash
# 1. Update the base package cache and acquire system utilities
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake git git-lfs clinfo libdrm-dev software-properties-common

# 2. Register the deadsnakes PPA to bring Python 3.12 compatibility to Ubuntu 26.04
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update

# 3. Install Python 3.12 toolchains and correct Level-Zero runtime loaders
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev \
    intel-opencl-icd libze-intel-gpu1 libze1 libze-dev

```

---

### Phase 2: Workspace Setup & Model Acquisition

Instantiate the virtual environment using the correct Python binary, pull down the targeted model weights into your workspace, and draft the runtime engine flags.

#### Step 1: Environment Instantiation & Base Layer Mapping

```bash
# Spawn the environment explicitly targeting Python 3.12
rm -rf ~/.venv-b70-minimax
python3.12 -m venv ~/.venv-b70-minimax
source ~/.venv-b70-minimax/bin/activate

# Upgrade primary packaging components
pip install --upgrade pip setuptools wheel

# Install the upstream hardware-accelerated PyTorch XPU binaries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

```

#### Step 2: Weight Cloning

```bash
# Clone the INT4 AutoRound quantized MiniMax weights into the NVMe directory
mkdir -p /mnt/fast-ai/models
cd /mnt/fast-ai/models
git lfs install
git clone https://huggingface.co/Lasimeri/MiniMax-M2.7-int4-AutoRound

```

#### Step 3: Runtime Flag Export Script

Create the system execution harness script at `/mnt/fast-ai/promoted-env.sh`:

```bash
cat << 'EOF' > /mnt/fast-ai/promoted-env.sh
#!/usr/bin/env bash
# Device Allocation (4x B70 GPUs linked via Level-Zero fabric)
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3
export UR_L0_SYNC_MODE=BLOCKING
export TORCH_LLM_ALLREDUCE=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

# Engine Guardrails & Optimizations
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300
export TRITON_CACHE_DIR=/tmp/triton_cache
export UVICORN_KEEP_ALIVE_TIMEOUT=300
export VLLM_XPU_ENABLE_XPU_GRAPH=1

# Targeted Weights Location
export MODEL_PATH="/mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound"
EOF

chmod +x /mnt/fast-ai/promoted-env.sh

```

---

### Phase 3: Hardware Verification Gate

Ensure you have sourced your execution variables and execute a rapid sanity check to confirm that PyTorch maps all 4 Battlemage XPUs flawlessly:

```bash
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh

python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('XPU Available:', torch.xpu.is_available())
print('XPU Device Count:', torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    print(f'GPU {i}:', torch.xpu.get_device_name(i))
"

```

* **Gate Requirement:** The script must return `XPU Device Count: 4` alongside four lines identifying the `Intel(R) Arc(TM) Pro B70 Graphics` chips.

---

### Phase 4: Compiling the Stack Natively from Source

To prevent dependency resolution sandboxing, we pre-inject build utilities and execute using non-isolated commands directly inside our verified virtual environment.

#### Step 1: Clone Repositories

```bash
cd /mnt/fast-ai
git clone https://github.com/vllm-project/vllm-xpu-kernels.git
git clone https://github.com/vllm-project/vllm.git

```

#### Step 2: Build the Specialized C++/SYCL Kernels

```bash
cd /mnt/fast-ai/vllm-xpu-kernels

# Inject psutil so the build orchestration engine can map Threadripper core threads
pip install setuptools setuptools-scm cmake ninja packaging psutil

# Compile the native hardware wheel using existing environment dependencies
pip install . --no-build-isolation

```

#### Step 3: Compile the Core vLLM Engine

```bash
cd /mnt/fast-ai/vllm

# Inject high-performance tokenization dependencies (Rust bindings required by setup)
pip install setuptools-rust

# Install core runtime configuration dependencies
pip install -r requirements-build.txt 2>/dev/null || pip install -r requirements/build.txt 2>/dev/null || true

# Target the hardware environment and compile the final backend interface
export VLLM_TARGET_DEVICE="xpu"
pip install . --no-build-isolation

```

#### Step 4: Re-link Custom Kernels (Correction Workaround)

Because vLLM's dependency script attempts to pull down and overwrite your work with an out-of-date precompiled `0.1.12` release, enforce your custom build back over the top:

```bash
cd /mnt/fast-ai/vllm-xpu-kernels
pip install . --no-build-isolation --no-deps

```

---

### Phase 5: Verification, Execution, & Canary Run

#### Step 1: Structural Verification Gate

Confirm all compiled bindings resolve without symbol errors:

```bash
source /mnt/fast-ai/promoted-env.sh

python3 -c "
import torch, vllm, vllm_xpu_kernels
print('Core Engine Operational Status: Verified')
print('vLLM Version:', vllm.__version__)
print('XPU Kernels Version:', vllm_xpu_kernels.__version__)
"

```

#### Step 2: Fire Up the Server Engine

Launch the server across all 4 GPUs using 4-way tensor parallelism:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound \
    --served-model-name minimax-m2.7 \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85

```

#### Step 3: Send the Canary Prompt

Once the logs report that the server is listening, open a separate terminal tab and execute a test generation run:

```bash
curl http://localhost:8030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.7",
    "messages": [{"role": "user", "content": "Explain the advantages of tensor parallelism across multiple GPUs."}],
    "max_tokens": 128
  }'

```

*Reminder from the optimization lab parameters: **Discard the timing metrics on the very first prompt.** The initial response handles the Triton JIT compilation passes. Evaluate output generation speeds beginning on the second request!*