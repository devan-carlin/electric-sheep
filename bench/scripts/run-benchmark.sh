#!/usr/bin/env bash
# run-benchmark.sh — Benchmark Intel Qwen3.6-35B-A3B with different expert routing (top-k)
#
# Usage:
#   ./run-benchmark.sh [topk_values]
#   ./run-benchmark.sh          # Run all variants (8, 16, 32, 64)
#   ./run-benchmark.sh 8 32    # Run only top-8 and top-32
#
# Prerequisites:
#   - vLLM installed and accessible
#   - Model configs in ../configs/top{8,16,32,64}/
#   - Test prompts in ../prompts/

set -euo pipefail

# Activate venv with vLLM
VENV_DIR="/home/dc/vllm-fresh-venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Source GPU environment for 4-GPU tensor parallelism
source "/home/dc/electric-sheep/vllm/env/set-env-0123-gpu.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIGS_DIR="$ROOT_DIR/configs"
PROMPTS_DIR="$ROOT_DIR/prompts"
RESULTS_DIR="$ROOT_DIR/results"
OUTPUTS_DIR="$RESULTS_DIR/outputs"
TIMINGS_DIR="$RESULTS_DIR/timings"

# Default vLLM settings
VLLM_PORT=8000
VLLM_HOST="localhost"
MAX_TOKENS=512
TEMPERATURE=0.2
TOP_P=0.9
# Intel Qwen3.6-35B-A3B INT4 model is ~21 GiB across 4x34GB GPUs with TP=4
GPU_MEMORY_UTILIZATION=0.8
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=4

# Default all variants
DEFAULT_TOPKS=(8 16 32 64)
TOPKS=("${@:-${DEFAULT_TOPKS[@]}}")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok() { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err() { echo -e "${RED}[✗]${NC} $*"; }

# Check for existing vLLM on port and clean up
check_port_clear() {
    if lsof -i:"$VLLM_PORT" > /dev/null 2>&1; then
        warn "Port $VLLM_PORT in use, killing existing process..."
        lsof -ti:"$VLLM_PORT" | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
}

mkdir -p "$OUTPUTS_DIR" "$TIMINGS_DIR"

# ── Start vLLM server for a given variant ──
start_vllm() {
    local topk=$1
    local model_dir="$CONFIGS_DIR/top${topk}"

    check_port_clear
    log "Starting vLLM for top-${topk} (TP=${TENSOR_PARALLEL_SIZE}, GPUs=0,1,2,3)..."
    vllm serve "$model_dir" \
        --served-model-name "intel-qwen3.6-35b-a3b" \
        --port "$VLLM_PORT" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --dtype auto \
        2>&1 | tee "$TIMINGS_DIR/vllm-top${topk}.log" &
    VLLM_PID=$!

    # Wait for server to be ready (model loading takes ~90s)
    local retries=0
    while [ $retries -lt 120 ]; do
        if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
            ok "vLLM ready for top-${topk} (PID: $VLLM_PID, waited ${retries}s)"
            return 0
        fi
        sleep 2
        retries=$((retries + 1))
        if [ $((retries % 10)) -eq 0 ]; then
            log "  Still waiting for vLLM... (${retries}s)"
        fi
    done
    err "vLLM failed to start for top-${topk}"
    kill $VLLM_PID 2>/dev/null || true
    return 1
}

# ── Stop vLLM server ──
stop_vllm() {
    log "Stopping vLLM..."
    
    # Kill by port (most reliable — catches the actual server process)
    local port_pids=$(lsof -ti:"$VLLM_PORT" 2>/dev/null || true)
    if [ -n "$port_pids" ]; then
        log "  Killing processes on port $VLLM_PORT: $port_pids"
        echo "$port_pids" | xargs -r kill -9 2>/dev/null || true
    fi
    
    # Kill any lingering vLLM/Python worker processes
    pkill -9 -f "vllm.*serve" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "WorkerProc" 2>/dev/null || true
    sleep 3
    
    # Wait for all vLLM-related processes to fully exit
    log "Waiting for vLLM processes to exit..."
    local wait_retries=0
    while [ $wait_retries -lt 30 ]; do
        local remaining=$(ps aux | grep -E "vllm.*serve|EngineCore|WorkerProc" | grep -v grep | wc -l)
        if [ "$remaining" -eq 0 ]; then
            break
        fi
        wait_retries=$((wait_retries + 1))
        if [ $((wait_retries % 5)) -eq 0 ]; then
            log "  ${remaining} processes still running... (${wait_retries}s)"
        fi
        sleep 1
    done
    
    # Intel XPU driver needs time to release VRAM after process exit
    # This is a known issue - the driver doesn't immediately free memory
    log "Waiting for XPU driver to release VRAM (15s)..."
    sleep 15
    
    ok "vLLM stopped"
}

# ── Run a single prompt against the running server ──
run_prompt() {
    local topk=$1
    local prompt_file=$2
    local prompt_name=$(basename "$prompt_file" .md)

    local output_file="$OUTPUTS_DIR/top${topk}_${prompt_name}.md"
    local timing_file="$TIMINGS_DIR/top${topk}_${prompt_name}.json"

    local prompt=$(cat "$prompt_file")

    log "  Running top-${topk} × ${prompt_name}..."

    local start_time=$(date +%s%N)

    # Send request to vLLM
    local response
    response=$(curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg prompt "$prompt" \
            --argjson max_tokens "$MAX_TOKENS" \
            --argjson temperature "$TEMPERATURE" \
            --argjson top_p "$TOP_P" \
            '{
                model: "intel-qwen3.6-35b-a3b",
                messages: [{role: "user", content: $prompt}],
                max_tokens: $max_tokens,
                temperature: $temperature,
                top_p: $top_p,
                stream: false
            }')" 2>/dev/null) || {
        err "  Request failed for top-${topk} × ${prompt_name}"
        echo "FAILED" > "$output_file"
        return 1
    }

    local end_time=$(date +%s%N)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    # Extract output and timing info
    local content=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "NO OUTPUT")
    local prompt_tokens=$(echo "$response" | jq '.usage.prompt_tokens // 0' 2>/dev/null || echo "0")
    local completion_tokens=$(echo "$response" | jq '.usage.completion_tokens // 0' 2>/dev/null || echo "0")
    local total_tokens=$((prompt_tokens + completion_tokens))
    local prompt_tps=$(echo "scale=1; $prompt_tokens * 1000 / $elapsed_ms" | bc 2>/dev/null || echo "0")
    local gen_tps=$(echo "scale=1; $completion_tokens * 1000 / $elapsed_ms" | bc 2>/dev/null || echo "0")

    # Save output
    cat > "$output_file" <<EOF
# Output: top-${topk} × ${prompt_name}

## Prompt
${prompt}

## Response
${content}

## Metadata
- **Top-k**: ${topk}
- **Elapsed**: ${elapsed_ms}ms
- **Prompt tokens**: ${prompt_tokens}
- **Completion tokens**: ${completion_tokens}
- **Total tokens**: ${total_tokens}
EOF

    # Save timing JSON
    jq -n \
        --arg topk "$topk" \
        --arg prompt "$prompt_name" \
        --argjson elapsed_ms "$elapsed_ms" \
        --argjson prompt_tokens "$prompt_tokens" \
        --argjson completion_tokens "$completion_tokens" \
        --argjson total_tokens "$total_tokens" \
        --arg prompt_tps "$prompt_tps" \
        --arg gen_tps "$gen_tps" \
        '{
            topk: $topk,
            prompt: $prompt,
            elapsed_ms: $elapsed_ms,
            prompt_tokens: $prompt_tokens,
            completion_tokens: $completion_tokens,
            total_tokens: $total_tokens,
            prompt_tok_per_sec: ($prompt_tps | tonumber),
            gen_tok_per_sec: ($gen_tps | tonumber)
        }' > "$timing_file"

    ok "  top-${topk} × ${prompt_name}: ${total_tokens} tokens in ${elapsed_ms}ms"
    return 0
}

# ── Generate summary ──
generate_summary() {
    local summary_file="$RESULTS_DIR/SUMMARY.md"

    cat > "$summary_file" <<'HEADER'
# Intel Qwen3.6-35B-A3B MoE Expert Routing Benchmark

## Model
- **Name**: Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound
- **Architecture**: Qwen3.5 MoE (256 experts, 40 layers)
- **Total params**: ~35B (INT4 quantized, ~21 GB)
- **Test date**: $(date)

## Test Matrix

| Variant | `num_experts_per_tok` | Est Active Params |
|---|---|---|
| top-8 | 8 | ~3B |
| top-16 | 16 | ~6B |
| top-32 | 32 | ~12B |
| top-64 | 64 | ~24B |

## Performance Summary

HEADER

    # Aggregate timing data
    echo "| Variant | Prompt | Tokens | Time (ms) | Gen tok/s |" >> "$summary_file"
    echo "|---|---|---|---|---|" >> "$summary_file"

    for timing_file in "$TIMINGS_DIR"/top*_*.json; do
        [ -f "$timing_file" ] || continue
        local topk=$(jq -r '.topk' "$timing_file")
        local prompt=$(jq -r '.prompt' "$timing_file")
        local total=$(jq -r '.total_tokens' "$timing_file")
        local elapsed=$(jq -r '.elapsed_ms' "$timing_file")
        local gen_tps=$(jq -r '.gen_tok_per_sec' "$timing_file")
        echo "| top-${topk} | ${prompt} | ${total} | ${elapsed} | ${gen_tps} |" >> "$summary_file"
    done

    # Add output comparison section
    cat >> "$summary_file" <<'FOOTER'

## Output Comparison

See individual output files in `results/outputs/` for full responses.

### Files
- `top{8,16,32,64}_{prompt}.md` — full output for each variant × prompt combination

## Conclusions

(To be filled after manual review of outputs)
FOOTER

    ok "Summary written to $summary_file"
}

# ── Main ──
main() {
    log "=== Huihui- MoE Benchmark ==="
    log "Variants: top-${TOPKS[*]}"
    log "Prompts: $(ls "$PROMPTS_DIR"/*.md 2>/dev/null | wc -l) test files"
    log "Results: $RESULTS_DIR"
    echo ""

    for topk in "${TOPKS[@]}"; do
        log "═══════════════════════════════════════"
        log "Variant: top-${topk} (num_experts_per_tok=${topk})"
        log "═══════════════════════════════════════"

        # Start vLLM
        if ! start_vllm "$topk"; then
            err "Failed to start vLLM for top-${topk}, skipping..."
            continue
        fi

        # Run all prompts
        local success=0
        local failed=0
        for prompt_file in "$PROMPTS_DIR"/*.md; do
            if run_prompt "$topk" "$prompt_file"; then
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi
        done

        ok "top-${topk}: ${success} passed, ${failed} failed"
        echo ""

        # Stop vLLM before next variant (includes GPU memory wait)
        stop_vllm
    done

    # Generate summary
    log "Generating summary..."
    generate_summary

    log "=== Benchmark complete ==="
    log "Results: $RESULTS_DIR"
}

# Trap to clean up on exit
trap stop_vllm EXIT

main "$@"
