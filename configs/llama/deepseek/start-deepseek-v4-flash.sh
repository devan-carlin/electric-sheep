#!/usr/bin/env bash
# ============================================
# Start DeepSeek V4-Flash (UD-IQ3_XXS GGUF)
# ============================================
# Sharded GGUF model across 4x Intel Arc Pro B70
# llama.cpp SYCL backend, layer split mode
#
# Model: unsloth/DeepSeek-V4-Flash-0731-GGUF
# Quant: UD-IQ3_XXS (~3-bit, very aggressive)
# Shards: 4 GGUF files (~98 GB total)
#
# Features:
#   - Tool calling enabled (Unsloth improved chat template)
#   - DSpark speculative decoding (optional, ~1.9x faster)
#   - Reasoning/thinking mode (on by default)
#   - Interactive context selection when DSpark is enabled
#
# Usage:
#   ./start-deepseek-v4-flash.sh          # Interactive start
#   ./start-deepseek-v4-flash.sh --status # Pre-flight checks only
#   ./start-deepseek-v4-flash.sh --test   # Quick smoke test
#   ./start-deepseek-v4-flash.sh --no-dspark  # Skip DSpark prompt
#   ./start-deepseek-v4-flash.sh --dspark     # Force DSpark if available
# ============================================

set -e

# ============================================
# Configuration
# ============================================
MODEL_DIR="$HOME/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS"
MODEL_FILE="$MODEL_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
DRAFTER_FILE="$HOME/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
LLAMA_DIR="$HOME/electric-sheep/llama"
LLAMA_SERVER="$LLAMA_DIR/llama.cpp/build/bin/llama-server"

# Server settings
PORT=8080
HOST="0.0.0.0"
PROMPT_BATCH_SIZE=4096
CPU_THREADS=6          # Minimal threads for orchestration (token sampling, graph scheduling)
CPU_THREADS_BATCH=6    # Threads for batch/prompt processing

# Context sizes (tokens)
CTX_STANDARD=98304    # 96K — default without DSpark (GPU 2: ~88%)
CTX_DSPARK=65536      # 64K — with DSpark (~10GB overhead, keeps GPU 2 < 95%)

# Inference params (from Unsloth recommended settings)
TEMPERATURE=1.0
TOP_P=1.0
MIN_P=0.01

# ============================================
# Pre-flight Checks
# ============================================
check_status() {
    echo "=========================================="
    echo "  DeepSeek V4-Flash Pre-flight Check"
    echo "=========================================="
    echo ""

    local errors=0

    # oneAPI loaded?
    if ! command -v sycl-ls >/dev/null 2>&1; then
        echo "  ✗ oneAPI not loaded"
        echo "  Fix: source /opt/intel/oneapi/setvars.sh"
        errors=$((errors + 1))
    else
        echo "  ✓ oneAPI loaded"
    fi

    # SYCL devices?
    gpu_count=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu" || echo "0")
    if [ "$gpu_count" -lt 1 ]; then
        echo "  ✗ No SYCL GPU devices detected"
        errors=$((errors + 1))
    else
        echo "  ✓ $gpu_count SYCL GPU(s) available"
    fi

    # Model shards (check all 4)
    local shard_count=0
    for i in 1 2 3 4; do
        padded=$(printf "%05d" "$i")
        shard="$MODEL_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-${padded}-of-00004.gguf"
        if [ -f "$shard" ]; then
            shard_count=$((shard_count + 1))
        else
            echo "  ✗ Missing shard $i: $(basename "$shard")"
            errors=$((errors + 1))
        fi
    done
    if [ "$shard_count" -eq 4 ]; then
        model_size=$(du -sh "$MODEL_DIR" | cut -f1)
        echo "  ✓ All 4 model shards found ($model_size total)"
    else
        echo ""
        echo "  Model shards are missing. Auto-download?"
        echo "  Repo: unsloth/DeepSeek-V4-Flash-0731-GGUF (~98 GB)"
        echo ""
        read -p "  Download now? (Y/n): " answer
        if [[ ! "$answer" =~ ^[Nn]$ ]]; then
            echo ""
            echo "  Downloading UD-IQ3_XXS shards..."
            echo "  (This may take a while depending on your connection)"
            echo ""
            mkdir -p "$MODEL_DIR"
            if hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
                --include 'UD-IQ3_XXS/*' \
                --local-dir "$HOME/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF"; then
                echo ""
                echo "  ✓ Download complete, re-checking shards..."
                errors=$((errors - 4))  # Remove the 4 shard errors
                shard_count=4
                model_size=$(du -sh "$MODEL_DIR" | cut -f1)
                echo "  ✓ All 4 model shards found ($model_size total)"
            else
                echo ""
                echo "  ✗ Download failed — check network or disk space"
            fi
        fi
    fi

    # DSpark drafter file
    if [ -f "$DRAFTER_FILE" ]; then
        drafter_size=$(du -sh "$DRAFTER_FILE" | cut -f1)
        echo "  ✓ DSpark drafter found ($drafter_size)"
        DRAFTER_AVAILABLE=1
    else
        echo "  ⚠ DSpark drafter not found (speculative decoding unavailable)"
        echo ""
        read -p "  Download DSpark drafter (~11 GB)? (y/N): " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            echo ""
            echo "  Downloading DSpark drafter..."
            echo ""
            if hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
                --include '*dspark-DeepSeek-V4-Flash-0731-Q8_0*' \
                --local-dir "$HOME/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF"; then
                echo ""
                drafter_size=$(du -sh "$DRAFTER_FILE" | cut -f1)
                echo "  ✓ DSpark drafter downloaded ($drafter_size)"
                DRAFTER_AVAILABLE=1
            else
                echo ""
                echo "  ✗ DSpark download failed"
                DRAFTER_AVAILABLE=0
            fi
        else
            echo "  → Skipping DSpark download"
            DRAFTER_AVAILABLE=0
        fi
    fi

    # llama-server binary?
    if [ ! -f "$LLAMA_SERVER" ]; then
        echo "  ✗ llama-server not found: $LLAMA_SERVER"
        echo "  Fix: run ~/electric-sheep/scripts/ubuntu/05-build-llama-cpp.sh"
        errors=$((errors + 1))
    else
        echo "  ✓ llama-server found"
    fi

    echo ""
    if [ "$errors" -gt 0 ]; then
        echo "  ✗ $errors error(s) — fix before starting"
        exit 1
    else
        echo "  All checks passed — ready to start"
    fi
}

# Quick smoke test
run_test() {
    check_status
    echo ""
    echo "Running quick inference test..."
    echo ""

    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    export ONEAPI_DEVICE_SELECTOR="level_zero:0,1,2,3"
    export ZES_ENABLE_SYSMAN=1

    "$LLAMA_DIR/llama.cpp/build/bin/llama-cli" \
        -m "$MODEL_FILE" \
        -ngl 99 \
        -n 64 \
        -c 2048 \
        -sm layer \
        -t "$CPU_THREADS" \
        --temp "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --min-p "$MIN_P" \
        --reasoning on \
        -p "What is 2+2?" \
        --load-mode mmap 2>&1 | tail -15
}

# ============================================
# Interactive DSpark & Context Selection
# ============================================
ask_dspark() {
    # Check if user forced a mode
    case "${1:-}" in
        --no-dspark)
            USE_DSPARK=0
            return
            ;;
        --dspark)
            if [ "$DRAFTER_AVAILABLE" -eq 1 ]; then
                USE_DSPARK=1
            else
                echo "⚠ DSpark requested but drafter file not found."
                echo "  Download with:"
                echo "  hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \\"
                echo "    --include '*dspark-DeepSeek-V4-Flash-0731-Q8_0*' \\"
                echo "    --local-dir ~/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF/"
                USE_DSPARK=0
            fi
            return
            ;;
    esac

    # Interactive prompt
    if [ "$DRAFTER_AVAILABLE" -eq 1 ]; then
        echo ""
        echo "  DSpark drafter is available (~1.9x faster decoding, +10GB VRAM)"
        echo ""
        read -p "  Enable DSpark? (Y/n): " answer
        if [[ "$answer" =~ ^[Nn]$ ]]; then
            USE_DSPARK=0
            echo "  → DSpark disabled"
        else
            USE_DSPARK=1
            echo "  → DSpark enabled"
        fi
    else
        USE_DSPARK=0
    fi
}

# ============================================
# Build Server Command
# ============================================
build_command() {
    local ctx_size="$1"
    local use_dspark="$2"

    echo ""
    echo "=========================================="
    echo "  Starting DeepSeek V4-Flash Server"
    echo "=========================================="
    echo ""
    echo "  Model:      DeepSeek-V4-Flash-0731 (UD-IQ3_XXS)"
    echo "  GPUs:       4 (layer split)"
    echo "  Context:    $ctx_size tokens"
    echo "  DSpark:     $([ "$use_dspark" -eq 1 ] && echo "ON (~1.9x faster)" || echo "OFF")"
    echo "  Reasoning:  ON (high effort, default)"
    echo "  Tool call:  ON"
    echo "  Temp/TopP:  $TEMPERATURE / $TOP_P"
    echo "  Port:       $PORT"
    echo "  Host:       $HOST"
    echo ""

    # Build the command
    CMD=("$LLAMA_SERVER")
    CMD+=(-m "$MODEL_FILE")
    CMD+=(-ngl 99)
    CMD+=(-c "$ctx_size")
    CMD+=(--batch-size "$PROMPT_BATCH_SIZE")
    CMD+=(-t "$CPU_THREADS")
    CMD+=(--threads-batch "$CPU_THREADS_BATCH")
    CMD+=(-sm layer)
    CMD+=(--port "$PORT")
    CMD+=(--host "$HOST")
    CMD+=(--load-mode mmap)
    CMD+=(--flash-attn auto)
    CMD+=(--temp "$TEMPERATURE")
    CMD+=(--top-p "$TOP_P")
    CMD+=(--min-p "$MIN_P")
    CMD+=(--reasoning on)

    if [ "$use_dspark" -eq 1 ]; then
        CMD+=(-md "$DRAFTER_FILE")
        CMD+=(--spec-type draft-dspark)
        CMD+=(--spec-draft-n-max 3)
    fi

    # Print the full command (single line)
    echo "Command:"
    echo "  ${CMD[*]}"
    echo ""

    # Execute
    exec "${CMD[@]}"
}

# ============================================
# Main
# ============================================
case "${1:-}" in
    --status)
        check_status
        exit 0
        ;;
    --test)
        run_test
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "  (no args)          Interactive start (asks about DSpark)"
        echo "  --status           Run pre-flight checks only"
        echo "  --test             Run quick inference test"
        echo "  --dspark           Force DSpark speculative decoding"
        echo "  --no-dspark        Start without DSpark"
        echo "  --help             Show this help"
        echo ""
        echo "Context windows:"
        echo "  Standard (no DSpark):  96K tokens (GPU 2: ~88% VRAM)"
        echo "  With DSpark:           64K tokens (GPU 2: ~93% VRAM)"
        echo ""
        echo "DSpark adds ~10GB VRAM but gives ~1.9x faster decoding."
        exit 0
        ;;
esac

# Pre-flight
check_status

# Ask about DSpark
ask_dspark "$2"

# Select context based on DSpark
if [ "$USE_DSPARK" -eq 1 ]; then
    CONTEXT_SIZE=$CTX_DSPARK
else
    CONTEXT_SIZE=$CTX_STANDARD
fi

# Start server
build_command "$CONTEXT_SIZE" "$USE_DSPARK"
