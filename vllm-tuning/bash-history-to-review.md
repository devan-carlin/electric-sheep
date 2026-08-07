sudo apt-get update && sudo apt-get install -y     build-essential cmake git git-lfs clinfo     intel-opencl-icd libze-intel-gpu1 libze1 libze-dev     python3-pip python3-venv libdrm-dev
# 1. Create and activate a dedicated virtual environment
python3 -m venv ~/.venv-b70-minimax
source ~/.venv-b70-minimax/bin/activate
pip install --upgrade pip setuptools wheel
# 2. Clone the quantized MiniMax weights into your /mnt/fast-ai workspace
mkdir -p /mnt/fast-ai/models
cd /mnt/fast-ai/models
git lfs install
git clone https://huggingface.co/Lasimeri/MiniMax-M2.7-int4-AutoRound
cat << 'EOF' > /mnt/fast-ai/promoted-env.sh
#!/usr/bin/env bash
# Device Bindings across your 4x B70 GPUs
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3
export UR_L0_SYNC_MODE=BLOCKING
export TORCH_LLM_ALLREDUCE=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

# vLLM Execution Parameters
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300
export TRITON_CACHE_DIR=/tmp/triton_cache
export UVICORN_KEEP_ALIVE_TIMEOUT=300
export VLLM_XPU_ENABLE_XPU_GRAPH=1

# Model Path Target
export MODEL_PATH="/mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound"
EOF

chmod +x /mnt/fast-ai/promoted-env.sh
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
python3 -c "
import torch
print('PyTorch XPU Device Count:', torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    print(f'GPU {i}:', torch.xpu.get_device_name(i))
"
# Ensure you are inside the environment
source ~/.venv-b70-minimax/bin/activate
# 1. Install the specialized Intel XPU PyTorch wheels
pip install torch==2.11.0+xpu torchvision==0.22.0+xpu torchaudio==2.7.0+xpu     --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# 2. Install Intel Extension for PyTorch (IPEX)
pip install intel-extension-for-pytorch==2.11.0+xpu
# 3. Install standard requirements for vLLM
pip install accelerator-toolkit transformers accelerate
source ~/.venv-b70-minimax/bin/activate
# Install PyTorch with native Intel XPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install torch torchvision torchaudio --index-url https://pytorch-extension.intel.com/release-whl/stable/bmg/us/
pip install transformers accelerate intel-extension-for-pytorch
source ~/.venv-b70-minimax/bin/activate
pip install transformers accelerate intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
source ~/.venv-b70-minimax/bin/activate
pip install transformers accelerate
source /mnt/fast-ai/promoted-env.sh
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('XPU Available:', torch.xpu.is_available())
print('XPU Device Count:', torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    print(f'GPU {i}:', torch.xpu.get_device_name(i))
"
cd /mnt/fast-ai
git clone https://github.com/vllm-project/vllm-xpu-kernels.git
git clone https://github.com/vllm-project/vllm.git
cd /mnt/fast-ai/vllm-xpu-kernels
pip install -r requirements-build.txt
# Compile the native C++/SYCL kernels
pip install .
cd /mnt/fast-ai/vllm-xpu-kernels
# 1. Pre-install required build layout tools locally
pip install setuptools setuptools-scm cmake ninja packaging
# 2. Compile and install using the existing environment (--no-build-isolation)
pip install . --no-build-isolation
# 1. Install Python 3.12 and dev tools
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
# 2. Deactivate and remove the 3.14 venv
deactivate 2>/dev/null || true
rm -rf ~/.venv-b70-minimax
# 3. Create fresh venv explicitly using Python 3.12
python3.12 -m venv ~/.venv-b70-minimax
source ~/.venv-b70-minimax/bin/activate
# 4. Verify version output says 3.12.x
python3 --version
# Install tool configuration utilities
sudo apt-get update && sudo apt-get install -y software-properties-common
# Add the deadsnakes PPA for older Python versions
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
# Install Python 3.12 and build tools
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
# 1. Clean up old paths
rm -rf ~/.venv-b70-minimax
# 2. Spawn environment explicitly targeting 3.12
python3.12 -m venv ~/.venv-b70-minimax
source ~/.venv-b70-minimax/bin/activate
# 3. Verify it says 3.12.x
python3 --version
# Install PyTorch XPU
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install setuptools setuptools-scm cmake ninja packaging
# Compile vllm-xpu-kernels
cd /mnt/fast-ai/vllm-xpu-kernels
pip install . --no-build-isolation
source ~/.venv-b70-minimax/bin/activate
cd /mnt/fast-ai/vllm-xpu-kernels
# 1. Install psutil so the setup script can count CPU threads
pip install psutil
# 2. Rerun the direct environment compilation
pip install . --no-build-isolation
cd /mnt/fast-ai/vllm
# 1. Pre-install the core engine runtime layout dependencies
pip install -r requirements-xpu.txt
# 2. Lock the hardware acceleration target and compile without build isolation
export VLLM_TARGET_DEVICE="xpu"
pip install . --no-build-isolation
cd /mnt/fast-ai/vllm
# 1. Install setuptools-rust and build dependencies
pip install setuptools-rust wheel
# 2. Install core build requirements if present
pip install -r requirements-build.txt 2>/dev/null || pip install -r requirements/build.txt 2>/dev/null || true
# 3. Compile vLLM for XPU
export VLLM_TARGET_DEVICE="xpu"
pip install . --no-build-isolation
cd /mnt/fast-ai/vllm-xpu-kernels
pip install . --no-build-isolation --no-deps
source /mnt/fast-ai/promoted-env.sh
python3 -c "
import torch, vllm, vllm_xpu_kernels
print('\n========================================')
print('Core Engine Operational Status: VERIFIED')
print('PyTorch XPU Count:', torch.xpu.device_count())
print('vLLM Version:', vllm.__version__)
print('XPU Kernels Version:', vllm_xpu_kernels.__version__)
print('========================================\n')
"
python3 -m vllm.entrypoints.openai.api_server     --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 2048     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85
# Sourced env flags to ensure Level Zero bindings are active
source /mnt/fast-ai/promoted-env.sh
# Start the server explicitly pointing to the XPU device target
VLLM_TARGET_DEVICE="xpu" python3 -m vllm.entrypoints.openai.api_server     --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 2048     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85
# 1. Export the variables globally to the current shell state
export VLLM_TARGET_DEVICE="xpu"
export VLLM_LOGGING_LEVEL="DEBUG"
# 2. Re-verify the env profile state
source /mnt/fast-ai/promoted-env.sh
# 3. Fire up a single-GPU test command first to verify hardware attachment
python3 -m vllm.entrypoints.openai.api_server     --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 1     --max-model-len 2048     --gpu-memory-utilization 0.85
python3 -c "import vllm_xpu_kernels; print(dir(vllm_xpu_kernels)); print(vllm_xpu_kernels.__file__)"
find /mnt/fast-ai/vllm-xpu-kernels/ -name "*.so"
cp /mnt/fast-ai/vllm-xpu-kernels/build/lib.linux-x86_64-cpython-312/vllm_xpu_kernels/*.so    /home/dc/.venv-b70-minimax/lib/python3.12/site-packages/vllm_xpu_kernels/
python3 -c "import vllm_xpu_kernels._C; print('Successfully loaded vllm_xpu_kernels._C'/home/dc/intel-vllm-01/start_env.sh')"
# 1. Run the test with clean syntax
python3 -c "import vllm_xpu_kernels._C; print('Successfully loaded vllm_xpu_kernels._C'/home/dc/intel-vllm-01/start_env.sh')"
# 2. Source your legacy environment setup file if needed
source /home/dc/intel-vllm-01/start_env.sh
curl http://localhost:8030/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "minimax-m2.7",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Analyze the specs of 4x Intel Arc B70 GPUs working in tensor parallelism."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
sudo poweroff
sudo nvtop
source ~/.venv-b70-minimax/bin/activate
python3 -c "import vllm_xpu_kernels._C; print('Successfully loaded vllm_xpu_kernels._C'/home/dc/intel-vllm-01/start_env.sh')"
# 1. Activate the proper venv
source ~/.venv-b70-minimax/bin/activate
# 2. Copy compiled .so binaries into site-packages (if not already copied)
cp /mnt/fast-ai/vllm-xpu-kernels/build/lib.linux-x86_64-cpython-312/vllm_xpu_kernels/*.so    /home/dc/.venv-b70-minimax/lib/python3.12/site-packages/vllm_xpu_kernels/
python3 -c "import vllm_xpu_kernels._C; print('Successfully loaded vllm_xpu_kernels._C'/home/dc/intel-vllm-01/start_env.sh')"
python3 -c "import vllm_xpu_kernels._C; print('Successfully loaded vllm_xpu_kernels._C'/home/dc/intel-vllm-01/start_env.sh')"
python3 -c "import vllm_xpu_kernels._C; print('Successfully loaded vllm_xpu_kernels._C'/home/dc/intel-vllm-01/start_env.sh')"
python3 -c "import triton; print(triton.__file__)"
docker run -d -p 3000:8080   --add-host=host.docker.internal:host-gateway   -e OPENAI_API_BASE_URL="http://host.docker.internal:8030/v1"   -e OPENAI_API_KEY="sk-vllm"   -v open-webui:/app/backend/data   --name open-webui   --restart always   ghcr.io/open-webui/open-webui:main
# Enter the running container as root
sudo docker exec -it lsv-vllm-primary bash
# Install py-spy inside the container
pip install py-spy
# Dump the live stack trace for the EngineCore process (PID 175 from your screenshot) and its workers
py-spy dump --pid 175 --subprocesses
btop
sudo btop
'/home/dc/intel-vllm-01/start_env.sh'
# Unset the proxy variables inside the container
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# Now install py-spy
pip install py-spy
# Step 1: Unset Intel corporate proxy and install py-spy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
pip install py-spy
# Step 2: Find the main EngineCore PID
ps aux | grep EngineCore
py-spy dump --pid 14074 --subprocesses
sudo py-spy dump --pid 14074 --subprocesses
sudo ./intel-vllm-01/.venv-intel-vllm/bin/py-spy dump --pid 14074 --subprocesses
sudo docker exec -it lsv-vllm-primary bash -c "unset http_proxy https_proxy; pip install py-spy &>/dev/null; py-spy dump --pid \$(pgrep -f EngineCore) --subprocesses"
# Check Ubuntu OS Release Version
lsb_release -a
# Check Kernel Version (Looking for 6.8+ or HWE kernel)
uname -r
# Verify Level Zero / OpenCL ICD Driver Installations
dpkg -l | grep -E "intel-level-zero|level-zero|intel-opencl"
# 1. Verify 4 GPUs on PCIe bus via lspci
lspci -nn | grep -iE "Display|VGA|3D"
# 2. Verify GPU Kernel Driver Binding (i915 or xe)
lspci -k | grep -A 3 -iE "Arc|B70|Display"
# 3. List SYCL Devices (Ensures Level Zero runtime detects all 4 GPUs)
sycl-ls
# 4. Check OpenCL / Level Zero Compute Devices
clinfo | grep -E "Platform Name|Device Name|Total Local Memory"
# Display CPU architecture, core/thread count, and flags
lscpu | grep -E "Model name|CPU\(s\):|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|Flags"
# Verify NUMA topology (important for multi-GPU memory locality)
numactl --hardware 2>/dev/null || lscpu | grep "NUMA"
# Check System RAM & Active Swap Space
free -h
# Inspect active swap partitions/files
swapon --show
# 1. Check disk capacity on mount point
df -h /mnt/fast-ai
# 2. Test NVMe Sequential Write Speed (10 GB test file)
dd if=/dev/zero of=/mnt/fast-ai/test_speed.img bs=1G count=10 conv=fdatasync status=progress
# 3. Test NVMe Sequential Read Speed
dd if=/mnt/fast-ai/test_speed.img of=/dev/null bs=1G status=progress
# Clean up test file
rm -f /mnt/fast-ai/test_speed.img
sudo fallocate -l 64G /swap_b70.img
sudo chmod 600 /swap_b70.img
sudo mkswap /swap_b70.img
sudo swapon /swap_b70.img
# Verify swap is now ~72 GB total
free -h
# Create directory and assign ownership to your user (dc)
sudo mkdir -p /mnt/fast-ai/models
sudo chown -R dc:dc /mnt/fast-ai
# Verify directory exists
df -h /mnt/fast-ai
sudo apt-get update && sudo apt-get install -y     build-essential cmake git git-lfs clinfo     intel-opencl-icd intel-level-zero-gpu level-zero     python3-pip python3-venv libdrm-dev
sudo apt-get update && sudo apt-get install -y     build-essential cmake git git-lfs clinfo     intel-opencl-icd libze-intel-gpu1 libze1 libze-dev     python3-pip python3-venv libdrm-dev
# 1. Jump back to your vLLM directory and activate your compilation environment
cd /mnt/fast-ai/vllm
source ~/.venv-b70-minimax/bin/activate
# 2. Inject your system environment flags for the 4x B70 GPUs
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
# 3. Fire up the server across all 4 cards
python3 -m vllm.entrypoints.openai.api_server     --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 2048     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85
source ~/.venv-b70-minimax/bin/activate
# Install the correct triton-xpu wheel from PyTorch's XPU index
pip install triton-xpu --index-url https://download.pytorch.org/whl/xpu --force-reinstall --no-deps
python3 -c "import triton; print('Triton successfully imported from:', triton.__file__)"
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
# 3. Fire up the server across all 4 cards
python3 -m vllm.entrypoints.openai.api_server     --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 2048     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85
# 1. Enforce V0 engine architecture and device target
export VLLM_USE_V1=0
export VLLM_TARGET_DEVICE="xpu"
# 2. Source your environment flags
source /mnt/fast-ai/promoted-env.sh
# 3. Launch the server with KV cache tuning
python3 -m vllm.entrypoints.openai.api_server     --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 2048     --max-num-batched-tokens 4096     --gpu-memory-utilization 0.75     --enforce-eager
# 1. Environment exports
export VLLM_USE_V1=0
export VLLM_TARGET_DEVICE="xpu"
source /mnt/fast-ai/promoted-env.sh
# 2. Launch with explicit V0 engine flag and tuned memory caps
python3 -m vllm.entrypoints.openai.api_server     --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 2048     --max-num-batched-tokens 4096     --gpu-memory-utilization 0.90     --swap-space 16     --enforce-eager     --v0
# 1. Clear out explicit V1 force variables
unset VLLM_USE_V1
export VLLM_TARGET_DEVICE="xpu"
source /mnt/fast-ai/promoted-env.sh
# 2. Fire up the engine targeting a lean memory window
python3 -m vllm.entrypoints.openai.api_server     --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 1024     --max-num-batched-tokens 2048     --gpu-memory-utilization 0.95     --enforce-eager
# 1. Ensure system target strings are set
export VLLM_TARGET_DEVICE="xpu"
source /mnt/fast-ai/promoted-env.sh
# 2. Open up the allocation pool and shorten the initial length constraint
python3 -m vllm.entrypoints.openai.api_server     --model /mnt/fast-ai/models/MiniMax-M2.7-int4-AutoRound     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 512     --max-num-batched-tokens 1024     --gpu-memory-utilization 0.98     --enforce-eager
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/intel-qwen3.6-27b-autoround-int4     --served-model-name minimax-m2.7     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 2048     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/intel-qwen3.6-27b-autoround-int4     --served-model-name qwen3.6-27b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-40b-heretic     --served-model-name qwen3.6-27b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-40b-heretic     --served-model-name qwen3.6-27b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.90
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-40b-heretic     --served-model-name qwen3.6-27b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-40b-heretic     --served-model-name qwen3.6-27b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --hf-overrides '{"fix_mistral_regex": true}'
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-40b-heretic     --served-model-name qwen3.6-27b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --hf-overrides '{"fix_mistral_regex": true}'
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-40b-heretic     --served-model-name qwen3.6-40b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-seqs 128     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --hf-overrides '{"fix_mistral_regex": true}'
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-40b-heretic     --served-model-name qwen3.6-40b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-seqs 128     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --hf-overrides '{"fix_mistral_regex": true}'
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-27b-711     --served-model-name qwen3.6-40b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-seqs 16     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --hf-overrides '{"fix_mistral_regex": true}'
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-27b-711     --served-model-name qwen3.6-40b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-seqs 16     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'     --hf-overrides '{"fix_mistral_regex": true}'
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-27b-711     --served-model-name qwen3.6-40b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-seqs 16     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --spec-method mtp     --spec-tokens 2     --hf-overrides '{"fix_mistral_regex": true}'
nano /mnt/fast-ai/vllm/vllm/v1/worker/mamba_utils.py
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-27b-711     --served-model-name qwen3.6-40b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-seqs 16     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --spec-method mtp     --spec-tokens 2     --hf-overrides '{"fix_mistral_regex": true}'
nano /mnt/fast-ai/vllm/vllm/v1/worker/mamba_utils.py
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-27b-711     --served-model-name qwen3.6-40b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-seqs 16     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --spec-method mtp     --spec-tokens 2     --hf-overrides '{"fix_mistral_regex": true}'
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/davidau-qwen3.6-27b-711     --served-model-name qwen3.6-40b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-seqs 16     --kv-cache-dtype fp8     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85     --hf-overrides '{"fix_mistral_regex": true}'
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/intel-qwen3.6-27b-autoround-int4     --served-model-name qwen3.6-27b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --max-num-batched-tokens 8192     --gpu-memory-utilization 0.85
sudo nvtop
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/intel-qwen3.6-27b-autoround-int4     --served-model-name qwen3.6-27b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-batched-tokens 65536     --max-num-seqs 16     --enable-prefix-caching     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --gpu-memory-utilization 0.85     --generation-config vllm
source ~/.venv-b70-minimax/bin/activate
source /mnt/fast-ai/promoted-env.sh
export VLLM_TARGET_DEVICE="xpu"
python3 -m vllm.entrypoints.openai.api_server     --model /home/dc/intel-vllm-01/models/intel-qwen3.6-27b-autoround-int4     --served-model-name qwen3.6-27b     --host 0.0.0.0     --port 8030     --tensor-parallel-size 4     --max-model-len 232144     --max-num-batched-tokens 65536     --max-num-seqs 16     --enable-prefix-caching     --reasoning-parser qwen3     --trust-remote-code     --enable-auto-tool-choice     --tool-call-parser qwen3_coder     --gpu-memory-utilization 0.85     --generation-config vllm
huggingface-cli download Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound     --local-dir /home/dc/intel-vllm-01/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound     --local-dir-use-symlinks False
hf download Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound     --local-dir /home/dc/intel-vllm-01/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound     --local-dir-use-symlinks False
hf download Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound     --local-dir /home/dc/intel-vllm-01/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound
hf download Intel/Qwen3.6-27B-int4-AutoRound     --local-dir /home/dc/intel-vllm-01/models/Intel-Qwen3.6-27B-int4-AutoRound
sudo poweroff
