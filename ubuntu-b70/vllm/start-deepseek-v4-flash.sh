#!/usr/bin/env bash
# ============================================
# Start DeepSeek-V4-Flash (UD-IQ3_XXS GGUF) — vLLM
# ============================================
# Sharded GGUF model spread across 4× Intel Arc B70
# via vLLM gguf-plugin (experimental)
#
# Model location: ~/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF/
# Quantization: UD-IQ3_XXS (~3-bit, very aggressive)
# Shards: 4 GGUF files (00001-of-00004)
#
# IMPORTANT: GGUF support in vLLM is experimental and under-optimized.
# Requires vllm-gguf-plugin: pip install --no-deps vllm-gguf-plugin
# Always pass --tokenizer with the base model repo (GGUF tokenizer is slow/buggy).
#
# Usage:
#   ./start-deepseek-v4-flash.sh          # Start server
#   ./start-deepseek-v4-flash.sh --status # Check GPU availability
#   ./start-deepseek-v4-flash.sh --test   # Quick smoke test
# ============================================

set -e

# ============================================
# Configuration
# ============================================
MODEL_DIR="$HOME/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS/UD-IQ3_XXS"
MODEL_FILE="${MODEL_DIR}/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
VENV_DIR="${HOME}/electric-sheep/vllm/.venv"
ENV_SCRIPT="${HOME}/electric-sheep/vllm/set-env-0123-gpu.sh"

# Server settings
PORT=8030
TP_SIZE=4
MAX_MODEL_LEN=65536
KV_CACHE_DTYPE="fp8"
GPU_MEMORY_UTIL=0.80

# GGUF-specific settings (required per vLLM docs)
# Always pass --tokenizer with base model repo (GGUF tokenizer is slow/buggy)
# Use --hf-config-path if model is not supported by HuggingFace
BASE_TOKENIZER_REPO="deepseek-ai/DeepSeek-V4-Flash-0731"
HF_CONFIG_REPO="deepseek-ai/DeepSeek-V4-Flash-0731"

# ============================================
# Pre-flight Checks
# ============================================
check_status() {
    echo "=========================================="
    echo "  DeepSeek-V4-Flash Pre-flight Check"
    echo "=========================================="
    echo ""

    # Model file exists?
    echo "--- Model Files ---"
    if [ -f "$MODEL_FILE" ]; then
        echo "  ✓ Primary shard found: $(basename "$MODEL_FILE")"
        shard_count=$(ls -1 "${MODEL_DIR}/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-"*.gguf 2>/dev/null | wc -l)
        echo "  ✓ Total shards found: $shard_count"
        total_size=$(du -sh "$MODEL_DIR" | cut -f1)
        echo "  ✓ Total size: $total_size"
    else
        echo "  ✗ Model file not found: $MODEL_FILE"
        echo "  Model directory: $MODEL_DIR"
        ls -la "$MODEL_DIR" 2>/dev/null || echo "  (directory not accessible)"
        exit 1
    fi
    echo ""

    # vllm-gguf-plugin installed?
    echo "--- GGUF Plugin Check ---"
    source "$VENV_DIR/bin/activate" 2>/dev/null
    if python3 -c "import vllm_gguf_plugin" 2>/dev/null; then
        echo "  ✓ vllm-gguf-plugin is installed"
    else
        echo "  ✗ vllm-gguf-plugin not found"
        echo "  Install: pip install vllm-gguf-plugin"
        exit 1
    fi
    echo ""

    # GPU availability
    echo "--- GPU Availability ---"
    source "$VENV_DIR/bin/activate" 2>/dev/null
    gpu_count=$(python3 -c "import torch; print(torch.xpu.device_count() if torch.xpu.is_available() else 0)" 2>/dev/null || echo "0")
    if [ "$gpu_count" -ge "$TP_SIZE" ]; then
        echo "  ✓ $gpu_count XPU devices available (need $TP_SIZE)"
    else
        echo "  ✗ Only $gpu_count XPU devices available (need $TP_SIZE)"
        exit 1
    fi
    echo ""

    # Port availability
    echo "--- Port Check ---"
    if command -v ss &> /dev/null; then
        port_in_use=$(ss -tlnp | grep ":$PORT " || true)
        if [ -n "$port_in_use" ]; then
            echo "  ⚠ Port $PORT is already in use:"
            echo "    $port_in_use"
        else
            echo "  ✓ Port $PORT is available"
        fi
    fi
    echo ""

    # Disk space
    echo "--- Disk Space ---"
    df -h "$MODEL_DIR" | tail -1 | awk '{print "  Available: " $4 " (on " $6 ")"}'
    echo ""
}

# ============================================
# Smoke Test
# ============================================
run_test() {
    check_status
    echo "Running quick smoke test..."
    echo ""

    source "$VENV_DIR/bin/activate"
    source "$ENV_SCRIPT" 2>/dev/null || true

    export VLLM_TARGET_DEVICE=xpu

    # Quick test: load model and generate one token
    python3 -c "
from vllm import LLM, SamplingParams
import time

print('Loading model...')
start = time.time()
llm = LLM(
    model='$MODEL_FILE',
    device='xpu',
    tensor_parallel_size=$TP_SIZE,
    max_model_len=2048,
    kv_cache_dtype='$KV_CACHE_DTYPE',
    gpu_memory_utilization=$GPU_MEMORY_UTIL,
    trust_remote_code=True,
)
load_time = time.time() - start
print(f'Model loaded in {load_time:.1f}s')

print('Generating test output...')
start = time.time()
outputs = llm.generate(
    'Hello, world!',
    SamplingParams(max_tokens=32, temperature=0.7)
)
gen_time = time.time() - start
print(f'Generated in {gen_time:.1f}s')
print(f'Output: {outputs[0].outputs[0].text[:100]}...')
print()
print('Smoke test passed!')
"
}

# ============================================
# Start Server
# ============================================
start_server() {
    check_status

    echo "=========================================="
    echo "  Starting DeepSeek-V4-Flash Server"
    echo "=========================================="
    echo ""
    echo "  Model:    DeepSeek-V4-Flash UD-IQ3_XXS (GGUF, 4 shards)"
    echo "  Location: $MODEL_FILE"
    echo "  GPUs:     $TP_SIZE× Intel Arc Pro B70 (tensor parallel)"
    echo "  Context:  64K tokens"
    echo "  KV Cache: $KV_CACHE_DTYPE"
    echo "  Port:     $PORT"
    echo ""

    # Activate environment
    source "$VENV_DIR/bin/activate"
    source "$ENV_SCRIPT" 2>/dev/null || true

    export VLLM_TARGET_DEVICE=xpu

    echo "Starting vLLM server..."
    echo ""

    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_FILE" \
        --tokenizer "$BASE_TOKENIZER_REPO" \
        --hf-config-path "$HF_CONFIG_REPO" \
        --served-model-name deepseek-v4-flash \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens 65536 \
        --max-num-seqs 8 \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --enable-prefix-caching \
        --trust-remote-code \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL"
}

# ============================================
# Main
# ============================================
case "${1:-}" in
    --status)
        check_status
        ;;
    --test)
        run_test
        ;;
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --status   Run pre-flight checks only"
        echo "  --test     Run smoke test (load model, generate one response)"
        echo "  --help     Show this help"
        echo ""
        echo "  (no args)  Start the vLLM server"
        echo ""
        echo "Configuration:"
        echo "  Model:      $MODEL_DIR"
        echo "  TP Size:    $TP_SIZE"
        echo "  Context:    ${MAX_MODEL_LEN} tokens"
        echo "  KV Cache:   $KV_CACHE_DTYPE"
        echo "  Port:       $PORT"
        ;;
    "")
        start_server
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage."
        exit 1
        ;;
esac
