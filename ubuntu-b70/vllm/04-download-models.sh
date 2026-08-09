#!/usr/bin/env bash
# ============================================
# Download Models for vLLM XPU
# ============================================
# Downloads pre-configured models with VRAM
# capacity checks to ensure they'll fit on
# the available GPU memory.
#
# Prerequisites: 01, 02, 03 must be run first
#
# Usage:
#   ./04-download-models.sh              # Create configs + download all models that fit
#   ./04-download-models.sh --list       # Show available models + fit status
#   ./04-download-models.sh --status     # Check already-downloaded models
#   ./04-download-models.sh --configs    # Create configs/startups only (no downloads)
#   ./04-download-models.sh qwen3.6-27b  # Download specific model
# ============================================

set -e

# Detect if run via sudo
if [ -n "$SUDO_USER" ]; then
    export HOME=$(eval echo ~$SUDO_USER)
    export PIP_CACHE_DIR="/tmp/pip-cache-sudo"
fi

VLLM_DIR="$HOME/electric-sheep/vllm"
VENV_DIR="$VLLM_DIR/.venv"
MODEL_DIR="$HOME/electric-sheep/models"
GPU_MEMORY_UTIL=0.80

# Helper function to write files reliably (each argument becomes one line)
write_file() {
    local filepath="$1"
    shift
    printf '%s\n' "$@" > "$filepath"
}

# Graceful error handler
fail() {
    echo ""
    echo "=========================================="
    echo "  ERROR: $1"
    echo "=========================================="
    echo "Press Enter to close..."
    read -r
    exit 1
}

# -------------------------------------------
# Pre-flight checks
# -------------------------------------------
echo "=========================================="
echo "  Model Downloader — Pre-flight"
echo "=========================================="

[ -d "$VENV_DIR" ] || fail "Virtual environment not found. Run 02-setup-project-directory.sh first."
echo "✓ Virtual environment found"

source "$VENV_DIR/bin/activate" 2>/dev/null || fail "Cannot activate venv"

# Check vLLM is installed
python3 -c "import vllm" 2>/dev/null || fail "vLLM not installed. Run 03-build-vllm-xpu.sh first."
echo "✓ vLLM installed"

# Check huggingface_hub (use python module, not CLI — more reliable inside venv)
python3 -c "from huggingface_hub import HfApi" 2>/dev/null || fail "huggingface_hub not found. Run: pip install huggingface_hub"
echo "✓ huggingface_hub available"

# Check GPU memory
gpu_count=$(python3 -c "import torch; print(torch.xpu.device_count())" 2>/dev/null || echo "0")
[ "$gpu_count" -eq 0 ] && fail "No XPU devices detected"
echo "✓ $gpu_count XPU GPU(s) available"

# Calculate available VRAM per GPU (in GB)
vram_per_gpu=$(python3 -c "
import torch
props = torch.xpu.get_device_properties(0)
total_gb = props.total_memory / 1e9
usable_gb = total_gb * $GPU_MEMORY_UTIL
print(f'{usable_gb:.1f}')
" 2>/dev/null || echo "0")

vram_summary=$(python3 -c "
import torch
gpu_count = torch.xpu.device_count()
props = torch.xpu.get_device_properties(0)
total_per_gpu = props.total_memory / 1e9
usable_per_gpu = total_per_gpu * $GPU_MEMORY_UTIL
total_all = total_per_gpu * gpu_count
usable_all = usable_per_gpu * gpu_count
print(f'{total_per_gpu:.1f}|{usable_per_gpu:.1f}|{total_all:.1f}|{usable_all:.1f}')
" 2>/dev/null || echo "0|0|0|0")

vram_per_gpu=$(echo "$vram_summary" | cut -d'|' -f2)
total_vram_per_gpu=$(echo "$vram_summary" | cut -d'|' -f1)
total_vram_all=$(echo "$vram_summary" | cut -d'|' -f3)
usable_vram_all=$(echo "$vram_summary" | cut -d'|' -f4)

echo "✓ VRAM: ${total_vram_per_gpu}GB/GPU (${vram_per_gpu}GB usable), ${total_vram_all}GB total (${usable_vram_all}GB usable)"
echo ""

# -------------------------------------------
# Model definitions
# -------------------------------------------
# Format: alias|repo_id|est_vram_tp4|est_vram_tp2|tp_size|notes
declare -a MODELS=(
    "qwen3.6-27b|Intel/Qwen3.6-27B-int4-AutoRound|3.5|7.0|4|Primary model, full context"
    "qwen3.6-35b-a3b|Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound|4.5|9.0|4|MoE, may need patch"
    "gemma-4-31b|Intel/gemma-4-31B-it-int4-AutoRound-V2|4.0|8.0|4|Standard INT4"
    "gemma-4-26b-a4b|Intel/gemma-4-26B-A4B-it-int4-AutoRound|3.2|6.5|4|MoE, standard INT4"
    "qwen3.6-35b-quark|nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8|5.0|10.0|4|Quark W8A8 INT8, experimental"
)

# -------------------------------------------
# Helper functions
# -------------------------------------------
print_header() {
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

# Check if a model will fit on available VRAM
# Args: est_vram_per_gpu
will_fit() {
    local est_vram="$1"
    python3 -c "
est = float('$est_vram')
avail = float('$vram_per_gpu')
exit(0 if est <= avail else 1)
" 2>/dev/null
}

# Download a single model
# Args: alias repo_id est_vram_tp4
download_model() {
    local alias="$1"
    local repo="$2"
    local est_vram="$3"
    local local_name="${repo//\//-}"  # "Intel/Qwen3.6-27B..." → "Intel-Qwen3.6-27B..."
    local local_path="$MODEL_DIR/$local_name"

    # Get total model size from HuggingFace (sum all files)
    local model_size
    model_size=$(python3 -c "
from huggingface_hub import HfApi
try:
    api = HfApi()
    info = api.model_info('$repo', timeout=10)
    total = sum(s.size for s in info.siblings if s.size)
    print(f'{total/1e9:.1f}')
except Exception as e:
    print('N/A')
" 2>/dev/null || echo "N/A")

    echo ""
    echo "--- Downloading: $alias ---"
    echo "  Repo:       $repo"
    echo "  Model size: ${model_size}GB"
    echo "  Est. VRAM:  ${est_vram}GB/GPU (TP=4)"
    echo "  VRAM/GPU:   ${total_vram_per_gpu}GB"
    echo "  Total VRAM: ${total_vram_all}GB"

    # VRAM fit check
    if ! will_fit "$est_vram"; then
        echo "  ✗ SKIPPED — won't fit on available VRAM (${est_vram}GB > ${vram_per_gpu}GB)"
        return 1
    fi

    echo "  ✓ Will fit"

    # Let hf handle existing files, resume, and verification
    hf download "$repo" --local-dir "$local_path" || {
        echo "  ✗ Download failed"
        return 1
    }

    # Final size check
    if [ -d "$local_path" ] && [ -f "$local_path/config.json" ]; then
        local size=$(du -sh "$local_path" | cut -f1)
        echo "  ✓ Ready ($size)"
    else
        echo "  ✗ Download failed (no config.json)"
        return 1
    fi
}

# -------------------------------------------
# Create environment config files
# -------------------------------------------
create_configs() {
    echo ""
    echo "--- Creating Environment Configs ---"

    # --- Full 4-GPU config (GPUs 0,1,2,3) ---
    write_file "$VLLM_DIR/set-env-0123-gpu.sh" \
        '#!/usr/bin/env bash' \
        '# Device Bindings: All 4 GPUs (0,1,2,3)' \
        'export ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3' \
        'export UR_L0_SYNC_MODE=BLOCKING' \
        'export TORCH_LLM_ALLREDUCE=1' \
        'export CCL_ZE_IPC_EXCHANGE=pidfd' \
        'export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0' \
        '' \
        '# GPU Affinity (binds GPUs to CPU cores for NUMA locality)' \
        'export ZE_AFFINITY_MASK=0,1,2,3' \
        '' \
        '# Graph capture with communication ops (improves multi-GPU performance)' \
        'export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1' \
        'export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1' \
        '' \
        '# vLLM Execution Parameters' \
        'export VLLM_WORKER_MULTIPROC_METHOD=spawn' \
        'export VLLM_ENGINE_ITERATION_TIMEOUT_S=300' \
        'export TRITON_CACHE_DIR="$HOME/.cache/triton"' \
        'export UVICORN_KEEP_ALIVE_TIMEOUT=300' \
        'export VLLM_XPU_ENABLE_XPU_GRAPH=1' \
        'export VLLM_TARGET_DEVICE=xpu' \
        '' \
        '# Tensor Parallelism (4 GPUs)' \
        'export TP_SIZE=4'
    chmod +x "$VLLM_DIR/set-env-0123-gpu.sh"
    echo "  ✓ set-env-0123-gpu.sh (4 GPUs, TP=4)"

    # --- GPU Pair 0,1 config ---
    write_file "$VLLM_DIR/set-env-01-gpu.sh" \
        '#!/usr/bin/env bash' \
        '# Device Bindings: GPUs 0,1' \
        'export ONEAPI_DEVICE_SELECTOR=level_zero:0,1' \
        'export UR_L0_SYNC_MODE=BLOCKING' \
        'export TORCH_LLM_ALLREDUCE=1' \
        'export CCL_ZE_IPC_EXCHANGE=pidfd' \
        'export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0' \
        '' \
        '# GPU Affinity (binds GPUs to CPU cores for NUMA locality)' \
        'export ZE_AFFINITY_MASK=0,1' \
        '' \
        '# Graph capture with communication ops (improves multi-GPU performance)' \
        'export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1' \
        'export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1' \
        '' \
        '# vLLM Execution Parameters' \
        'export VLLM_WORKER_MULTIPROC_METHOD=spawn' \
        'export VLLM_ENGINE_ITERATION_TIMEOUT_S=300' \
        'export TRITON_CACHE_DIR="$HOME/.cache/triton"' \
        'export UVICORN_KEEP_ALIVE_TIMEOUT=300' \
        'export VLLM_XPU_ENABLE_XPU_GRAPH=1' \
        'export VLLM_TARGET_DEVICE=xpu' \
        '' \
        '# Tensor Parallelism (2 GPUs)' \
        'export TP_SIZE=2'
    chmod +x "$VLLM_DIR/set-env-01-gpu.sh"
    echo "  ✓ set-env-01-gpu.sh (GPUs 0,1, TP=2)"

    # --- GPU Pair 2,3 config ---
    write_file "$VLLM_DIR/set-env-23-gpu.sh" \
        '#!/usr/bin/env bash' \
        '# Device Bindings: GPUs 2,3' \
        'export ONEAPI_DEVICE_SELECTOR=level_zero:2,3' \
        'export UR_L0_SYNC_MODE=BLOCKING' \
        'export TORCH_LLM_ALLREDUCE=1' \
        'export CCL_ZE_IPC_EXCHANGE=pidfd' \
        'export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0' \
        '' \
        '# GPU Affinity (binds GPUs to CPU cores for NUMA locality)' \
        'export ZE_AFFINITY_MASK=2,3' \
        '' \
        '# Graph capture with communication ops (improves multi-GPU performance)' \
        'export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1' \
        'export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1' \
        '' \
        '# vLLM Execution Parameters' \
        'export VLLM_WORKER_MULTIPROC_METHOD=spawn' \
        'export VLLM_ENGINE_ITERATION_TIMEOUT_S=300' \
        'export TRITON_CACHE_DIR="$HOME/.cache/triton"' \
        'export UVICORN_KEEP_ALIVE_TIMEOUT=300' \
        'export VLLM_XPU_ENABLE_XPU_GRAPH=1' \
        'export VLLM_TARGET_DEVICE=xpu' \
        '' \
        '# Tensor Parallelism (2 GPUs)' \
        'export TP_SIZE=2'
    chmod +x "$VLLM_DIR/set-env-23-gpu.sh"
    echo "  ✓ set-env-23-gpu.sh (GPUs 2,3, TP=2)"
}

# -------------------------------------------
# Create model startup scripts
# -------------------------------------------
create_startups() {
    echo ""
    echo "--- Creating Model Startup Scripts ---"

    # --- Qwen 3.6 27B INT4 (primary, full context) ---
    write_file "$VLLM_DIR/start-qwen3.6-27b.sh" \
        '#!/usr/bin/env bash' \
        '# Qwen 3.6 27B INT4 - Primary model with full context' \
        '# Uses all 4 GPUs (TP=4)' \
        "source \"$VENV_DIR/bin/activate\"" \
        "source \"$VLLM_DIR/set-env-0123-gpu.sh\"" \
        '' \
        '# Compile cache root for warm starts (faster cold launches)' \
        'export VLLM_CACHE_ROOT="$HOME/.cache/vllm"' \
        '' \
        'MODEL_PATH="$HOME/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound"' \
        '' \
        'python3 -m vllm.entrypoints.openai.api_server \' \
        '    --model "$MODEL_PATH" \' \
        '    --served-model-name qwen3.6-27b \' \
        '    --host 0.0.0.0 \' \
        '    --port 8030 \' \
        '    --tensor-parallel-size $TP_SIZE \' \
        '    --max-model-len 232144 \' \
        '    --max-num-seqs 16 \' \
        '    --max-num-batched-tokens 65536 \' \
        '    --kv-cache-dtype fp8 \' \
        '    --reasoning-parser qwen3 \' \
        '    --trust-remote-code \' \
        '    --enable-auto-tool-choice \' \
        '    --tool-call-parser qwen3_coder \' \
        '    --gpu-memory-utilization 0.80 \' \
        '    --enable-prefix-caching \' \
        '    --generation-config vllm'
    chmod +x "$VLLM_DIR/start-qwen3.6-27b.sh"
    echo "  ✓ start-qwen3.6-27b.sh (primary, full context)"

    # --- Qwen 3.6 27B INT4 (MTP enabled, speculative decoding) ---
    write_file "$VLLM_DIR/start-qwen3.6-27b-mtp.sh" \
        '#!/usr/bin/env bash' \
        '# Qwen 3.6 27B INT4 - MTP enabled with speculative decoding' \
        '# Uses all 4 GPUs (TP=4), MTP3 speculative tokens' \
        "source \"$VENV_DIR/bin/activate\"" \
        "source \"$VLLM_DIR/set-env-0123-gpu.sh\"" \
        '' \
        '# Compile cache root for warm starts (faster cold launches)' \
        'export VLLM_CACHE_ROOT="$HOME/.cache/vllm"' \
        '' \
        '# MTP (Multi-Token Prediction) Configuration' \
        'export QWEN36_27B_ENABLE_MTP=1' \
        'export NUM_SPECULATIVE_TOKENS=3' \
        'export VLLM_XPU_GDN_PROMOTE_ACCEPTED_SPEC_STATE=1' \
        'export VLLM_XPU_GDN_NONSPEC_POSTPROCESS_ACCEPTED_STATE=0' \
        'export VLLM_XPU_LM_HEAD_INT8=1' \
        'export VLLM_XPU_LM_HEAD_INT8_SCALE_DTYPE=bf16' \
        "export COMPILATION_CONFIG='{\"cudagraph_mode\":\"PIECEWISE\",\"max_cudagraph_capture_size\":8}'" \
        '' \
        'MODEL_PATH="$HOME/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound"' \
        '' \
        'python3 -m vllm.entrypoints.openai.api_server \' \
        '    --model "$MODEL_PATH" \' \
        '    --served-model-name qwen3.6-27b-mtp \' \
        '    --host 0.0.0.0 \' \
        '    --port 8004 \' \
        '    --tensor-parallel-size $TP_SIZE \' \
        '    --max-model-len 232144 \' \
        '    --max-num-seqs 256 \' \
        '    --max-num-batched-tokens 8192 \' \
        '    --kv-cache-dtype fp8 \' \
        '    --reasoning-parser qwen3 \' \
        '    --trust-remote-code \' \
        '    --enable-auto-tool-choice \' \
        '    --tool-call-parser qwen3_coder \' \
        '    --gpu-memory-utilization 0.80 \' \
        '    --enable-prefix-caching \' \
        "    --hf-overrides '{\"fix_mistral_regex\": true}'"
    chmod +x "$VLLM_DIR/start-qwen3.6-27b-mtp.sh"
    echo "  ✓ start-qwen3.6-27b-mtp.sh (MTP3 speculative decoding)"

    # --- Qwen 3.6 35B-A3B INT4 (MoE, requires patching) ---
    write_file "$VLLM_DIR/start-qwen3.6-35b-a3b.sh" \
        '#!/usr/bin/env bash' \
        '# Qwen 3.6 35B-A3B INT4 - Mixture of Experts model' \
        '# Uses 2 GPUs (TP=2), requires patching' \
        "source \"$VENV_DIR/bin/activate\"" \
        "source \"$VLLM_DIR/set-env-01-gpu.sh\"" \
        '' \
        '# Compile cache root for warm starts (faster cold launches)' \
        'export VLLM_CACHE_ROOT="$HOME/.cache/vllm"' \
        '' \
        'MODEL_PATH="$HOME/electric-sheep/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound"' \
        '' \
        'python3 -m vllm.entrypoints.openai.api_server \' \
        '    --model "$MODEL_PATH" \' \
        '    --served-model-name qwen3.6-35b-a3b \' \
        '    --host 0.0.0.0 \' \
        '    --port 8001 \' \
        '    --tensor-parallel-size $TP_SIZE \' \
        '    --max-model-len 232144 \' \
        '    --max-num-seqs 128 \' \
        '    --max-num-batched-tokens 8192 \' \
        '    --kv-cache-dtype fp8 \' \
        '    --reasoning-parser qwen3 \' \
        '    --trust-remote-code \' \
        '    --enable-auto-tool-choice \' \
        '    --tool-call-parser qwen3_coder \' \
        '    --gpu-memory-utilization 0.80 \' \
        '    --enable-prefix-caching \' \
        "    --hf-overrides '{\"fix_mistral_regex\": true}'"
    chmod +x "$VLLM_DIR/start-qwen3.6-35b-a3b.sh"
    echo "  ✓ start-qwen3.6-35b-a3b.sh (MoE, requires patch)"

    # --- Qwen 3.6 35B-A3B Quark W8A8 INT4 (TP=4, experimental) ---
    write_file "$VLLM_DIR/start-qwen3.6-35b-a3b-quark-int8.sh" \
        '#!/usr/bin/env bash' \
        '# Qwen 3.6 35B-A3B Quark W8A8 INT4 - INT8 MoE model (experimental)' \
        '# Uses all 4 GPUs (TP=4), requires Quark quantization support' \
        "source \"$VENV_DIR/bin/activate\"" \
        "source \"$VLLM_DIR/set-env-0123-gpu.sh\"" \
        '' \
        '# Compile cache root for warm starts (faster cold launches)' \
        'export VLLM_CACHE_ROOT="$HOME/.cache/vllm"' \
        '' \
        '# Graph fallback flags from 35B Quark INT8 lane (prevents decode corruption)' \
        'export VLLM_XPU_GDN_NATIVE_FALLBACK=prefill' \
        'export VLLM_XPU_GDN_PREFILL_RECURRENT_FALLBACK=1' \
        'export VLLM_XPU_DISABLE_PREFILL_CUDAGRAPH_REPLAY=1' \
        'export VLLM_XPU_GREEDY_SAMPLE_TOPK_FALLBACK=1' \
        '' \
        'MODEL_PATH="$HOME/electric-sheep/models/nameistoken-Qwen3.6-35B-A3B-Quark-W8A8-INT8"' \
        '' \
        'python3 -m vllm.entrypoints.openai.api_server \' \
        '    --model "$MODEL_PATH" \' \
        '    --served-model-name qwen3.6-35b-a3b-quark-int8 \' \
        '    --host 0.0.0.0 \' \
        '    --port 8005 \' \
        '    --tensor-parallel-size $TP_SIZE \' \
        '    --max-model-len 32768 \' \
        '    --max-num-seqs 48 \' \
        '    --max-num-batched-tokens 8192 \' \
        '    --kv-cache-dtype auto \' \
        '    --quantization quark \' \
        '    --trust-remote-code \' \
        '    --gpu-memory-utilization 0.80 \' \
        "    --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' \\" \
        '    --no-enable-prefix-caching'
    chmod +x "$VLLM_DIR/start-qwen3.6-35b-a3b-quark-int8.sh"
    echo "  ✓ start-qwen3.6-35b-a3b-quark-int8.sh (Quark W8A8 INT8, experimental)"

    # --- Gemma 4 31B INT4 (standard) ---
    write_file "$VLLM_DIR/start-gemma4-31b.sh" \
        '#!/usr/bin/env bash' \
        '# Gemma 4 31B INT4 - Standard INT4 quantized model' \
        '# Uses all 4 GPUs (TP=4)' \
        "source \"$VENV_DIR/bin/activate\"" \
        "source \"$VLLM_DIR/set-env-0123-gpu.sh\"" \
        '' \
        '# Compile cache root for warm starts (faster cold launches)' \
        'export VLLM_CACHE_ROOT="$HOME/.cache/vllm"' \
        '' \
        'MODEL_PATH="$HOME/electric-sheep/models/Intel-gemma-4-31B-it-int4-AutoRound-V2"' \
        '' \
        'python3 -m vllm.entrypoints.openai.api_server \' \
        '    --model "$MODEL_PATH" \' \
        '    --served-model-name gemma4-31b \' \
        '    --host 0.0.0.0 \' \
        '    --port 8002 \' \
        '    --tensor-parallel-size $TP_SIZE \' \
        '    --max-model-len 232144 \' \
        '    --max-num-seqs 256 \' \
        '    --max-num-batched-tokens 8192 \' \
        '    --kv-cache-dtype fp8 \' \
        '    --trust-remote-code \' \
        '    --gpu-memory-utilization 0.80 \' \
        '    --enable-prefix-caching'
    chmod +x "$VLLM_DIR/start-gemma4-31b.sh"
    echo "  ✓ start-gemma4-31b.sh (standard INT4)"

    # --- Gemma 4 26B-A4B INT4 (MoE) ---
    write_file "$VLLM_DIR/start-gemma4-26b-a4b.sh" \
        '#!/usr/bin/env bash' \
        '# Gemma 4 26B-A4B INT4 - Mixture of Experts model' \
        '# Uses 2 GPUs (TP=2)' \
        "source \"$VENV_DIR/bin/activate\"" \
        "source \"$VLLM_DIR/set-env-01-gpu.sh\"" \
        '' \
        '# Compile cache root for warm starts (faster cold launches)' \
        'export VLLM_CACHE_ROOT="$HOME/.cache/vllm"' \
        '' \
        'MODEL_PATH="$HOME/electric-sheep/models/Intel-gemma-4-26B-A4B-it-int4-AutoRound"' \
        '' \
        'python3 -m vllm.entrypoints.openai.api_server \' \
        '    --model "$MODEL_PATH" \' \
        '    --served-model-name gemma4-26b-a4b \' \
        '    --host 0.0.0.0 \' \
        '    --port 8003 \' \
        '    --tensor-parallel-size $TP_SIZE \' \
        '    --max-model-len 232144 \' \
        '    --max-num-seqs 128 \' \
        '    --max-num-batched-tokens 8192 \' \
        '    --kv-cache-dtype fp8 \' \
        '    --trust-remote-code \' \
        '    --gpu-memory-utilization 0.80 \' \
        '    --enable-prefix-caching'
    chmod +x "$VLLM_DIR/start-gemma4-26b-a4b.sh"
    echo "  ✓ start-gemma4-26b-a4b.sh (MoE)"
}

# -------------------------------------------
# List models with fit status
# -------------------------------------------
if [ "${1:-}" = "--list" ]; then
    print_header "Available Models"
    printf "  %-20s %-55s %8s %8s %s\n" "Alias" "HuggingFace Repo" "VRAM/TP4" "VRAM/TP2" "Notes"
    printf "  %-20s %-55s %8s %8s %s\n" "-----" "--------------" "------" "------" "-----"

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r alias repo est_vram_tp4 est_vram_tp2 tp notes <<< "$entry"
        fit="✓"
        if ! will_fit "$est_vram_tp4"; then
            fit="✗ too large"
        fi
        printf "  %-20s %-55s %7sGB %7sGB [%s]\n" "$alias" "$repo" "$est_vram_tp4" "$est_vram_tp2" "$fit"
    done

    echo ""
    echo "  VRAM: ${vram_per_gpu}GB/GPU usable, ${usable_vram_all}GB total usable (${GPU_MEMORY_UTIL} utilization)"
    echo "  GPUs: $gpu_count"
    exit 0
fi

# -------------------------------------------
# Status check
# -------------------------------------------
if [ "${1:-}" = "--status" ]; then
    print_header "Downloaded Models"
    echo "  Model directory: $MODEL_DIR"
    echo ""

    if [ ! -d "$MODEL_DIR" ]; then
        echo "  No models directory found."
        exit 0
    fi

    model_count=0
    for model_path in "$MODEL_DIR"/*/; do
        if [ -d "$model_path" ]; then
            model_name=$(basename "$model_path")
            size=$(du -sh "$model_path" 2>/dev/null | cut -f1)
            has_safetensors=$(find "$model_path" -name "*.safetensors" 2>/dev/null | head -1)
            has_gguf=$(find "$model_path" -name "*.gguf" 2>/dev/null | head -1)

            format="unknown"
            [ -n "$has_safetensors" ] && format="safetensors"
            [ -n "$has_gguf" ] && format="GGUF"

            printf "  %-40s %6s  %s\n" "$model_name" "$size" "$format"
            model_count=$((model_count + 1))
        fi
    done

    [ "$model_count" -eq 0 ] && echo "  No models downloaded yet."
    echo ""
    echo "  Total: $model_count model(s)"
    exit 0
fi

# -------------------------------------------
# Configs only mode
# -------------------------------------------
if [ "${1:-}" = "--configs" ]; then
    print_header "Creating Configs & Startup Scripts"
    create_configs
    create_startups
    echo ""
    print_header "Configs Created!"
    echo "  Environment configs:"
    echo "    set-env-0123-gpu.sh  (4 GPUs, TP=4)"
    echo "    set-env-01-gpu.sh    (GPUs 0,1, TP=2)"
    echo "    set-env-23-gpu.sh    (GPUs 2,3, TP=2)"
    echo ""
    echo "  Model startup scripts:"
    echo "    start-qwen3.6-27b.sh              (primary, full context)"
    echo "    start-qwen3.6-27b-mtp.sh          (MTP3 speculative decoding)"
    echo "    start-qwen3.6-35b-a3b.sh          (MoE, requires patch)"
    echo "    start-qwen3.6-35b-a3b-quark-int8.sh (Quark W8A8 INT8)"
    echo "    start-gemma4-31b.sh               (standard INT4)"
    echo "    start-gemma4-26b-a4b.sh           (MoE)"
    exit 0
fi

# -------------------------------------------
# Download specific model or all
# -------------------------------------------
if [ -n "${1:-}" ] && [ "${1:-}" != "--list" ] && [ "${1:-}" != "--status" ] && [ "${1:-}" != "--configs" ]; then
    # Download specific model
    found=0
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r alias repo est_vram_tp4 est_vram_tp2 tp notes <<< "$entry"
        if [ "$alias" = "$1" ]; then
            found=1
            download_model "$alias" "$repo" "$est_vram_tp4" || true
            break
        fi
    done
    [ "$found" -eq 0 ] && fail "Unknown model: $1 (use --list to see available models)"
else
    # Create configs + download all models that fit
    print_header "Creating Configs & Downloading Models"
    create_configs
    create_startups

    echo ""
    echo "  Checking which models fit on ${vram_per_gpu}GB usable VRAM per GPU..."
    echo ""

    downloaded=0
    skipped=0

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r alias repo est_vram_tp4 est_vram_tp2 tp notes <<< "$entry"
        if download_model "$alias" "$repo" "$est_vram_tp4"; then
            downloaded=$((downloaded + 1))
        else
            skipped=$((skipped + 1))
        fi
    done

    echo ""
    print_header "Setup Complete!"
    echo "  Downloaded: $downloaded"
    echo "  Skipped (too large): $skipped"
    echo ""
    echo "  To serve a model:"
    echo "    source $VENV_DIR/bin/activate"
    echo "    source $VLLM_DIR/set-env-0123-gpu.sh"
    echo "    vllm serve $MODEL_DIR/<model-name> --tensor-parallel-size 4 --port 8030"
    echo ""
    echo "  Or use the startup scripts:"
    echo "    bash $VLLM_DIR/start-qwen3.6-27b.sh"
    echo "    bash $VLLM_DIR/start-gemma4-31b.sh"
    echo ""
fi
