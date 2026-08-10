#!/usr/bin/env bash
# ============================================
# Start DeepSeek V4-Flash with Balanced Split + DSpark
# ============================================
# Launches DeepSeek V4-Flash on 4x Intel Arc Pro B70 GPUs
# with balanced split mode, DSpark speculative decoding,
# and 128K context window.
#
# Usage:
#   ./start-deepseek-v4-balanced.sh          # Full start
#   ./start-deepseek-v4-balanced.sh --check  # Pre-flight checks only
#   ./start-deepseek-v4-balanced.sh --test   # Quick inference test
# ============================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "${CYAN}>>>${NC} $*"; }

# ============================================
# Configuration
# ============================================
LLAMA_DIR="$HOME/llama.cpp"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
MODEL_FILE="$HOME/electric-sheep/models/DeepSeek-V4-Flash-0731-Abliterated-DS4-Quality128/DeepSeek-V4-Flash-0731-Abliterated-DS4-Quality128.gguf"
DRAFTER_FILE="$HOME/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"

# Server settings (prefer 8080, fall back to 8081)
PREFERRED_PORT=8080
FALLBACK_PORT=8081
HOST="0.0.0.0"
CTX_SIZE=131072          # 128K context
BATCH_SIZE=4096           # Prompt processing batch (larger = faster ingestion)
UBATCH_SIZE=128           # Unified batch (smaller avoids OOM on warmup)
PARALLEL=4
GPU_LAYERS=999
SPLIT_MODE="balanced"
TENSOR_SPLIT="1,1,1,1"
DSPARK_N_MAX=2            # Sweet spot: faster than n_max=5, higher acceptance

# ============================================
# Pre-flight Checks
# ============================================
check_patch() {
    step "Checking balanced split mode patch..."
    local errors=0

    # Check llama.h enum
    if grep -q "LLAMA_SPLIT_MODE_BALANCED = 4" "$LLAMA_DIR/include/llama.h" 2>/dev/null; then
        info "  ✓ llama.h: LLAMA_SPLIT_MODE_BALANCED enum present"
    else
        error "  ✗ llama.h: LLAMA_SPLIT_MODE_BALANCED enum NOT found"
        errors=$((errors + 1))
    fi

    # Check arg.cpp handler
    if grep -q 'LLAMA_SPLIT_MODE_BALANCED' "$LLAMA_DIR/common/arg.cpp" 2>/dev/null; then
        info "  ✓ arg.cpp: balanced CLI handler present"
    else
        error "  ✗ arg.cpp: balanced CLI handler NOT found"
        errors=$((errors + 1))
    fi

    # Check fit.cpp exception
    if grep -q 'SPLIT_MODE_BALANCED.*skipping fit check' "$LLAMA_DIR/common/fit.cpp" 2>/dev/null; then
        info "  ✓ fit.cpp: balanced mode fit check exception present"
    else
        error "  ✗ fit.cpp: balanced mode fit check exception NOT found"
        errors=$((errors + 1))
    fi

    # Check llama-model.cpp balanced assignment
    if grep -q 'balanced layer assignment.*quantization-aware' "$LLAMA_DIR/src/llama-model.cpp" 2>/dev/null; then
        info "  ✓ llama-model.cpp: balanced assignment code present"
    else
        error "  ✗ llama-model.cpp: balanced assignment code NOT found"
        errors=$((errors + 1))
    fi

    # Check layer_to_gpu lookup
    if grep -q 'layer_to_gpu\[il\]' "$LLAMA_DIR/src/llama-model.cpp" 2>/dev/null; then
        info "  ✓ llama-model.cpp: layer_to_gpu lookup in get_layer_buft_list"
    else
        error "  ✗ llama-model.cpp: layer_to_gpu lookup NOT found"
        errors=$((errors + 1))
    fi

    if [[ $errors -gt 0 ]]; then
        error ""
        error "  $errors patch(es) missing. Apply the patch:"
        error "  cd ~/llama.cpp && bash ~/electric-sheep/scripts/common/apply-balanced-split-mode.sh"
        return 1
    fi

    info "  All patches verified ✓"
    return 0
}

check_env() {
    step "Checking environment..."
    local errors=0

    # oneAPI loaded?
    if command -v icpx >/dev/null 2>&1; then
        info "  ✓ Intel oneAPI compiler available"
    else
        error "  ✗ Intel oneAPI not loaded"
        error "  Fix: source /opt/intel/oneapi/setvars.sh"
        errors=$((errors + 1))
    fi

    # SYCL devices?
    if command -v sycl-ls >/dev/null 2>&1; then
        local gpu_count
        gpu_count=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu" || echo "0")
        if [[ "$gpu_count" -ge 4 ]]; then
            info "  ✓ $gpu_count SYCL GPU(s) available"
        else
            warn "  ⚠ Only $gpu_count SYCL GPU(s) detected (expected 4)"
        fi
    else
        warn "  ⚠ sycl-ls not available (oneAPI may not be fully loaded)"
    fi

    # Model file?
    if [[ -f "$MODEL_FILE" ]]; then
        local model_size
        model_size=$(du -sh "$MODEL_FILE" | cut -f1)
        info "  ✓ Model file found ($model_size)"
    else
        error "  ✗ Model file not found: $MODEL_FILE"
        errors=$((errors + 1))
    fi

    # DSpark drafter?
    if [[ -f "$DRAFTER_FILE" ]]; then
        local drafter_size
        drafter_size=$(du -sh "$DRAFTER_FILE" | cut -f1)
        info "  ✓ DSpark drafter found ($drafter_size)"
    else
        warn "  ⚠ DSpark drafter not found (speculative decoding will be disabled)"
        warn "  Download: hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include '*dspark*' --local-dir ~/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF/"
        DRAFTER_FILE=""
    fi

    # llama-server binary?
    if [[ -f "$LLAMA_SERVER" ]]; then
        info "  ✓ llama-server binary found"
    else
        error "  ✗ llama-server not found: $LLAMA_SERVER"
        error "  Fix: cd ~/llama.cpp/build && cmake --build . --target llama-server -j $(nproc)"
        errors=$((errors + 1))
    fi

    # Port selection (prefer 8080, fall back to 8081)
    PORT="$PREFERRED_PORT"
    if command -v ss >/dev/null 2>&1; then
        if ss -tlnp | grep -q ":${PORT} "; then
            warn "  ⚠ Port $PORT in use, trying $FALLBACK_PORT..."
            PORT="$FALLBACK_PORT"
            if ss -tlnp | grep -q ":${PORT} "; then
                error "  ✗ Both ports $PREFERRED_PORT and $FALLBACK_PORT are in use"
                error "  Kill existing: pkill -f llama-server"
                errors=$((errors + 1))
            else
                info "  ✓ Port $PORT available (fallback)"
            fi
        else
            info "  ✓ Port $PORT available"
        fi
    else
        info "  ✓ Port $PORT (ss not available, skipping port check)"
    fi

    return $errors
}

# ============================================
# Quick Inference Test
# ============================================
run_test() {
    step "Running quick inference test on port $PORT..."

    local response
    response=$(curl -s --max-time 120 "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "test",
            "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
            "max_tokens": 20
        }' 2>/dev/null) || {
        error "  ✗ Failed to connect to server on port $PORT"
        error "  Is the server running? Check: curl http://localhost:$PORT/v1/models"
        return 1
    }

    local tokens finish reasoning
    tokens=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null)
    finish=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['finish_reason'])" 2>/dev/null)
    reasoning=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0]['message']; print(m.get('reasoning_content', m.get('content', 'N/A'))[:100])" 2>/dev/null)

    if [[ -n "$tokens" && "$tokens" != "0" ]]; then
        info "  ✓ Inference test passed"
        info "  Tokens: $tokens, Finish: $finish"
        info "  Output: $reasoning..."
    else
        error "  ✗ Inference test failed (no tokens generated)"
        return 1
    fi
}

# ============================================
# Build Server Command & Start
# ============================================
start_server() {
    step "Starting DeepSeek V4-Flash Server"
    echo ""
    echo -e "  ${CYAN}Configuration:${NC}"
    echo "    Model:       DeepSeek-V4-Flash Abliterated DS4-Quality128"
    echo "    Context:     $CTX_SIZE tokens (128K)"
    echo "    Split Mode:  $SPLIT_MODE (quantization-aware)"
    echo "    DSpark:      ON (n_max=$DSPARK_N_MAX)"
    echo "    KV Cache:    f16 (default)"
    echo "    Port:        $PORT"
    echo "    GPUs:        4 (balanced layer assignment)"
    echo ""
    info "Starting server (logs to /tmp/deepseek-v4-balanced.log)..."
    info "Press Ctrl+C to stop"
    echo ""

    # Build the command
    local CMD=("$LLAMA_SERVER")
    CMD+=(-m "$MODEL_FILE")
    CMD+=(--host "$HOST" --port "$PORT")
    CMD+=(--gpu-layers "$GPU_LAYERS")
    CMD+=(--split-mode "$SPLIT_MODE")
    CMD+=(--tensor-split "$TENSOR_SPLIT")
    CMD+=(--flash-attn on)
    CMD+=(--ctx-size "$CTX_SIZE")
    CMD+=(--batch-size "$BATCH_SIZE")
    CMD+=(--ubatch-size "$UBATCH_SIZE")
    CMD+=(--parallel "$PARALLEL")

    if [[ -n "$DRAFTER_FILE" ]]; then
        CMD+=(-md "$DRAFTER_FILE")
        CMD+=(--spec-type draft-dspark)
        CMD+=(--spec-draft-n-max "$DSPARK_N_MAX")
    fi

    CMD+=(--verbose)

    # Print the full command
    info "Command:"
    echo "  ${CMD[*]}"
    echo ""

    # Execute (logs to file and stdout)
    exec "${CMD[@]}" 2>&1 | tee /tmp/deepseek-v4-balanced.log
}

# ============================================
# Main
# ============================================
case "${1:-}" in
    --check)
        check_patch
        check_env
        exit $?
        ;;
    --test)
        run_test
        exit $?
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "  (no args)          Run pre-flight checks, then start server"
        echo "  --check            Run pre-flight checks only (patch + env)"
        echo "  --test             Run quick inference test (server must be running)"
        echo "  --help             Show this help"
        echo ""
        echo "Configuration:"
        echo "  Port:       $PORT"
        echo "  Context:    $CTX_SIZE tokens"
        echo "  Split:      $SPLIT_MODE"
        echo "  DSpark:     n_max=$DSPARK_N_MAX"
        echo ""
        exit 0
        ;;
    *)
        echo "=========================================="
        echo "  DeepSeek V4-Flash Startup"
        echo "=========================================="
        echo ""

        # Run checks
        check_patch || exit 1
        check_env || exit 1

        echo ""
        info "All checks passed — starting server..."
        echo ""

        # Source oneAPI env if not loaded
        if ! command -v icpx >/dev/null 2>&1; then
            source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
        fi

        # Set GPU selector
        export ONEAPI_DEVICE_SELECTOR="level_zero:0,1,2,3"
        export ZES_ENABLE_SYSMAN=1
        export GGML_SYCL_ENABLE_FLASH_ATTN=1
        export GGML_SYCL_ENABLE_OPT=1
        export GGML_SYCL_ENABLE_DNN=1
        export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

        # Start server
        start_server
        ;;
esac
