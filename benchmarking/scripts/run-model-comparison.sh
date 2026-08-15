#!/usr/bin/env bash
# run-model-comparison.sh — Run benchmark prompts against a single model and capture results
# Usage: ./scripts/run-model-comparison.sh <model-name> [port] [max-tokens]

set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model-name> [port] [max-tokens]}"
PORT="${2:-8000}"
MAX_TOKENS="${3:-512}"
HOST="localhost"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PROMPTS_DIR="$ROOT_DIR/prompts"
RESULTS_DIR="$ROOT_DIR/results/model-comparison/$MODEL_NAME"

mkdir -p "$RESULTS_DIR/outputs" "$RESULTS_DIR/timings"

TEMPERATURE=0.2
TOP_P=0.9

log() { echo "[$(date +%H:%M:%S)] $*"; }
ok() { log "[✓] $*"; }
err() { log "[✗] $*"; }

# Wait for vLLM to be ready
log "Waiting for vLLM on port $PORT..."
for i in $(seq 1 120); do
    if curl -s "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
        ok "vLLM ready on port $PORT"
        break
    fi
    if [ $i -eq 120 ]; then
        err "vLLM not ready after 120s"
        exit 1
    fi
    sleep 2
done

# Auto-detect served model name
VLLM_MODEL=$(curl -s "http://${HOST}:${PORT}/v1/models" | jq -r '.data[0].id // empty' 2>/dev/null)
if [ -z "$VLLM_MODEL" ]; then
    err "Could not detect model name from vLLM API"
    exit 1
fi
log "Using served model: $VLLM_MODEL"

# Run each prompt
for prompt_file in "$PROMPTS_DIR"/*.md; do
    [ -f "$prompt_file" ] || continue
    prompt_name=$(basename "$prompt_file" .md)
    prompt=$(cat "$prompt_file")

    output_file="$RESULTS_DIR/outputs/${prompt_name}.md"
    timing_file="$RESULTS_DIR/timings/${prompt_name}.json"

    log "Running ${prompt_name}..."

    start_time=$(date +%s%N)

    response=$(curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg model "$VLLM_MODEL" \
            --arg prompt "$prompt" \
            --argjson max_tokens "$MAX_TOKENS" \
            --argjson temperature "$TEMPERATURE" \
            --argjson top_p "$TOP_P" \
            '{
                model: $model,
                messages: [{role: "user", content: $prompt}],
                max_tokens: $max_tokens,
                temperature: $temperature,
                top_p: $top_p,
                stream: false
            }')" 2>/dev/null) || {
        err "Request failed for ${prompt_name}"
        echo "FAILED" > "$output_file"
        continue
    }

    end_time=$(date +%s%N)
    elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    content=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "NO OUTPUT")
    reasoning=$(echo "$response" | jq -r '.choices[0].message.reasoning // empty' 2>/dev/null || echo "")
    prompt_tokens=$(echo "$response" | jq '.usage.prompt_tokens // 0' 2>/dev/null || echo "0")
    completion_tokens=$(echo "$response" | jq '.usage.completion_tokens // 0' 2>/dev/null || echo "0")
    total_tokens=$((prompt_tokens + completion_tokens))
    gen_tps=$(echo "scale=1; $completion_tokens * 1000 / $elapsed_ms" | bc 2>/dev/null || echo "0")

    # Save output
    cat > "$output_file" <<EOF
# Output: ${MODEL_NAME} × ${prompt_name}

## Prompt
${prompt}

## Reasoning
${reasoning}

## Response
${content}

## Metadata
- **Model**: ${MODEL_NAME}
- **Elapsed**: ${elapsed_ms}ms
- **Prompt tokens**: ${prompt_tokens}
- **Completion tokens**: ${completion_tokens}
- **Total tokens**: ${total_tokens}
- **Gen tok/s**: ${gen_tps}
EOF

    # Save timing JSON
    jq -n \
        --arg prompt "$prompt_name" \
        --argjson elapsed_ms "$elapsed_ms" \
        --argjson prompt_tokens "$prompt_tokens" \
        --argjson completion_tokens "$completion_tokens" \
        --argjson total_tokens "$total_tokens" \
        --arg gen_tps "$gen_tps" \
        '{
            prompt: $prompt,
            elapsed_ms: $elapsed_ms,
            prompt_tokens: $prompt_tokens,
            completion_tokens: $completion_tokens,
            total_tokens: $total_tokens,
            gen_tok_per_sec: ($gen_tps | tonumber)
        }' > "$timing_file"

    ok "${prompt_name}: ${completion_tokens} tokens in ${elapsed_ms}ms (${gen_tps} tok/s)"
done

log "Results saved to $RESULTS_DIR"
