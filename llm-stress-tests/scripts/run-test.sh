#!/usr/bin/env bash
# run-test.sh — Run a stress test against a local LLM endpoint
#
# Usage:
#   ./run-test.sh <test_file.md> <model_name> [base_url]
#
# Examples:
#   ./run-test.sh game-prompts/01-typing-speed-test.md qwen3.6-27b-int4
#   ./run-test.sh security/01-vulnerability-audit.md qwen3.6-27b-int4 http://localhost:8000/v1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/results"

# --- Args ---
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <test_file.md> <model_name> [base_url]"
    exit 1
fi

TEST_FILE="$1"
MODEL_NAME="$2"
BASE_URL="${3:-http://localhost:8000/v1}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TEST_NAME="$(basename "$TEST_FILE" .md)"

# --- Read prompt ---
PROMPT="$(cat "$TEST_FILE")"

# --- Save output ---
OUTPUT_FILE="$RESULTS_DIR/${MODEL_NAME}_${TEST_NAME}_${TIMESTAMP}.md"
mkdir -p "$RESULTS_DIR"

echo "Running: $TEST_NAME"
echo "Model:   $MODEL_NAME"
echo "Output:  $OUTPUT_FILE"
echo "---"

# --- Send to LLM (OpenAI-compatible API) ---
curl -s "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
        --arg prompt "$PROMPT" \
        '{
            model: "local",
            messages: [
                {role: "user", content: $prompt}
            ],
            temperature: 0.1,
            max_tokens: 16384
        }')" \
    | jq -r '.choices[0].message.content' \
    | tee "$OUTPUT_FILE"

echo ""
echo "--- Done. Output saved to $OUTPUT_FILE ---"
