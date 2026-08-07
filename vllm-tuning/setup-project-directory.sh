#!/usr/bin/env bash
set -e

# ============================================
# Project Directory Setup Script
# ============================================
# Creates the vLLM project structure, virtual environment,
# environment configuration, and downloads the target model.
# ============================================

VLLM_DIR="$HOME/electric-sheep/vllm"
VENV_DIR="$VLLM_DIR/.venv"
MODELS_DIR="$VLLM_DIR/models"

# Graceful error handler — keeps terminal open for investigation
fail() {
    echo ""
    echo "=========================================="
    echo "  ERROR: $1"
    echo "=========================================="
    echo ""
    echo "Terminal will stay open for investigation."
    echo "Press Enter to close..."
    read -r
    exit 1
}

# Pre-flight checks
echo "=========================================="
echo "  Pre-flight Checks"
echo "=========================================="
command -v python3.12 >/dev/null 2>&1 || fail "python3.12 not found"
echo "✓ Python 3.12 available"

command -v hf >/dev/null 2>&1 || fail "huggingface-cli (hf) not installed"
echo "✓ HuggingFace CLI available"

# Check for Intel Battlemage GPUs
lspci | grep -q "Battlemage" || fail "No Intel Battlemage GPUs detected"
gpu_count=$(lspci | grep -c "Battlemage")
echo "✓ $gpu_count Intel Battlemage GPU(s) detected"

# Check disk space (need ~200GB for models)
available=$(df --output=avail -BM "$HOME" | tail -1 | awk '{printf "%d", $1/1024}')
[ "$available" -lt 200 ] && fail "Need 200GB free, have ${available}GB"
echo "✓ Sufficient disk space (${available}GB available)"

echo ""
echo "  Setting up vLLM Project Directory"
echo "=========================================="

# 1. Create directory structure
echo "[1/5] Creating directory structure..."
mkdir -p "$VLLM_DIR"
mkdir -p "$MODELS_DIR"

# 2. Create Python 3.12 virtual environment
echo "[2/5] Creating Python 3.12 virtual environment..."
python3.12 -m venv "$VENV_DIR"

# Helper function to write files reliably (each argument becomes one line)
write_file() {
    local filepath="$1"
    shift
    printf '%s\n' "$@" > "$filepath"
}

# 3. Create environment configuration files
echo "[3/5] Writing environment configurations..."

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

# 4. Create model startup scripts
echo "[4/5] Writing model startup scripts..."

# --- Qwen 3.6 27B INT4 (primary, full context) ---
write_file "$VLLM_DIR/start-qwen3.6-27b.sh" \
    '#!/usr/bin/env bash' \
    '# Qwen 3.6 27B INT4 - Primary model with full context' \
    '# Uses all 4 GPUs (TP=4)' \
    "source \"$VENV_DIR/bin/activate\"" \
    "source \"$VLLM_DIR/set-env-0123-gpu.sh\"" \
    '' \
    'MODEL_PATH="$HOME/electric-sheep/vllm/models/Intel-Qwen3.6-27B-int4-AutoRound"' \
    '' \
    'python3 -m vllm.entrypoints.openai.api_server \' \
    '    --model "$MODEL_PATH" \' \
    '    --served-model-name qwen3.6-27b \' \
    '    --host 0.0.0.0 \' \
    '    --port 8000 \' \
    '    --tensor-parallel-size $TP_SIZE \' \
    '    --max-model-len 232144 \' \
    '    --max-num-seqs 256 \' \
    '    --max-num-batched-tokens 8192 \' \
    '    --kv-cache-dtype fp8 \' \
    '    --reasoning-parser qwen3 \' \
    '    --trust-remote-code \' \
    '    --enable-auto-tool-choice \' \
    '    --tool-call-parser qwen3_coder \' \
    '    --gpu-memory-utilization 0.85 \' \
    '    --enable-prefix-caching \' \
    "    --hf-overrides '{\"fix_mistral_regex\": true}'"
chmod +x "$VLLM_DIR/start-qwen3.6-27b.sh"

# --- Qwen 3.6 27B INT4 (MTP enabled, speculative decoding) ---
write_file "$VLLM_DIR/start-qwen3.6-27b-mtp.sh" \
    '#!/usr/bin/env bash' \
    '# Qwen 3.6 27B INT4 - MTP enabled with speculative decoding' \
    '# Uses all 4 GPUs (TP=4), MTP3 speculative tokens' \
    "source \"$VENV_DIR/bin/activate\"" \
    "source \"$VLLM_DIR/set-env-0123-gpu.sh\"" \
    '' \
    '# MTP (Multi-Token Prediction) Configuration' \
    'export QWEN36_27B_ENABLE_MTP=1' \
    'export NUM_SPECULATIVE_TOKENS=3' \
    'export VLLM_XPU_GDN_PROMOTE_ACCEPTED_SPEC_STATE=1' \
    'export VLLM_XPU_GDN_NONSPEC_POSTPROCESS_ACCEPTED_STATE=0' \
    'export VLLM_XPU_LM_HEAD_INT8=1' \
    'export VLLM_XPU_LM_HEAD_INT8_SCALE_DTYPE=bf16' \
    'export COMPILATION_CONFIG=\'{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":8}\'' \
    '' \
    'MODEL_PATH="$HOME/electric-sheep/vllm/models/Intel-Qwen3.6-27B-int4-AutoRound"' \
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
    '    --gpu-memory-utilization 0.85 \' \
    '    --enable-prefix-caching \' \
    "    --hf-overrides '{\"fix_mistral_regex\": true}'"
chmod +x "$VLLM_DIR/start-qwen3.6-27b-mtp.sh"

# --- Qwen 3.6 35B-A3B INT4 (MoE, requires patching) ---
write_file "$VLLM_DIR/start-qwen3.6-35b-a3b.sh" \
    '#!/usr/bin/env bash' \
    '# Qwen 3.6 35B-A3B INT4 - Mixture of Experts model' \
    '# Uses 2 GPUs (TP=2), requires patching' \
    "source \"$VENV_DIR/bin/activate\"" \
    "source \"$VLLM_DIR/set-env-01-gpu.sh\"" \
    '' \
    'MODEL_PATH="$HOME/electric-sheep/vllm/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound"' \
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
    '    --gpu-memory-utilization 0.85 \' \
    '    --enable-prefix-caching \' \
    "    --hf-overrides '{\"fix_mistral_regex\": true}'"
chmod +x "$VLLM_DIR/start-qwen3.6-35b-a3b.sh"

# --- Gemma 4 31B INT4 (standard) ---
write_file "$VLLM_DIR/start-gemma4-31b.sh" \
    '#!/usr/bin/env bash' \
    '# Gemma 4 31B INT4 - Standard INT4 quantized model' \
    '# Uses all 4 GPUs (TP=4)' \
    "source \"$VENV_DIR/bin/activate\"" \
    "source \"$VLLM_DIR/set-env-0123-gpu.sh\"" \
    '' \
    'MODEL_PATH="$HOME/electric-sheep/vllm/models/Intel-gemma-4-31B-it-int4-AutoRound-V2"' \
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
    '    --gpu-memory-utilization 0.85 \' \
    '    --enable-prefix-caching'
chmod +x "$VLLM_DIR/start-gemma4-31b.sh"

# --- Gemma 4 26B-A4B INT4 (MoE) ---
write_file "$VLLM_DIR/start-gemma4-26b-a4b.sh" \
    '#!/usr/bin/env bash' \
    '# Gemma 4 26B-A4B INT4 - Mixture of Experts model' \
    '# Uses 2 GPUs (TP=2)' \
    "source \"$VENV_DIR/bin/activate\"" \
    "source \"$VLLM_DIR/set-env-01-gpu.sh\"" \
    '' \
    'MODEL_PATH="$HOME/electric-sheep/vllm/models/Intel-gemma-4-26B-A4B-it-int4-AutoRound"' \
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
    '    --gpu-memory-utilization 0.85 \' \
    '    --enable-prefix-caching'
chmod +x "$VLLM_DIR/start-gemma4-26b-a4b.sh"

# 5. Download model weights
echo "[5/5] Downloading models..."
echo "      This may take a while depending on your connection..."
cd "$MODELS_DIR"

download_model() {
    local repo="$1"
    local dir="$2"
    
    if [ -d "./$dir" ] && [ -f "./$dir/config.json" ]; then
        echo "  -> Skipping $repo (already exists)"
        return 0
    fi
    
    echo "  -> Downloading $repo..."
    hf download "$repo" --local-dir "./$dir" || fail "Failed to download $repo"
    
    # Verify model integrity
    [ ! -f "./$dir/config.json" ] && fail "Model verification failed for $repo (missing config.json)"
    
    echo "  -> ✓ Downloaded and verified $repo"
}

download_model Intel/Qwen3.6-27B-int4-AutoRound \
    ./Intel-Qwen3.6-27B-int4-AutoRound

download_model Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound \
    ./Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound

download_model Intel/gemma-4-31B-it-int4-AutoRound-V2 \
    ./Intel-gemma-4-31B-it-int4-AutoRound-V2

download_model Intel/gemma-4-26B-A4B-it-int4-AutoRound \
    ./Intel-gemma-4-26B-A4B-it-int4-AutoRound

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To activate your environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "For 4-GPU deployment (full context):"
echo "  source $VLLM_DIR/set-env-0123-gpu.sh"
echo ""
echo "For dual-model deployment (2 GPUs each):"
echo "  Terminal 1: source $VLLM_DIR/set-env-01-gpu.sh"
echo "  Terminal 2: source $VLLM_DIR/set-env-23-gpu.sh"
echo ""
echo "Model startup scripts:"
echo "  start-qwen3.6-27b.sh      - Qwen 3.6 27B INT4 (primary, full context)"
echo "  start-qwen3.6-27b-mtp.sh  - Qwen 3.6 27B INT4 (MTP3 speculative decoding)"
echo "  start-qwen3.6-35b-a3b.sh  - Qwen 3.6 35B-A3B INT4 (MoE, requires patching)"
echo "  start-gemma4-31b.sh        - Gemma 4 31B INT4 (standard)"
echo "  start-gemma4-26b-a4b.sh   - Gemma 4 26B-A4B INT4 (MoE)"
echo ""
echo "Project structure:"
echo "  $VLLM_DIR/"
echo "  ├── .venv/              <-- Python virtual environment"
echo "  ├── set-env-0123-gpu.sh <-- All 4 GPUs (TP=4, full context)"
echo "  ├── set-env-01-gpu.sh   <-- GPUs 0,1 (TP=2, reduced context)"
echo "  ├── set-env-23-gpu.sh   <-- GPUs 2,3 (TP=2, reduced context)"
echo "  ├── start-qwen3.6-27b.sh      - Qwen 3.6 27B INT4"
echo "  ├── start-qwen3.6-35b-a3b.sh  - Qwen 3.6 35B-A3B INT4 (MoE)"
echo "  ├── start-gemma4-31b.sh        - Gemma 4 31B INT4"
echo "  ├── start-gemma4-26b-a4b.sh   - Gemma 4 26B-A4B INT4 (MoE)"
echo "  └── models/"
echo "      ├── Intel-Qwen3.6-27B-int4-AutoRound/"
echo "      ├── Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound/"
echo "      ├── Intel-gemma-4-31B-it-int4-AutoRound-V2/"
echo "      └── Intel-gemma-4-26B-A4B-it-int4-AutoRound/"
echo ""
