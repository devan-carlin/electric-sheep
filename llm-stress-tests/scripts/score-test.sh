#!/usr/bin/env bash
# score-test.sh — Interactive scoring helper for stress test results
#
# Usage:
#   ./score-test.sh <output_file.md>
#
# Prompts the user to rate each criterion, then appends the score
# to a results summary file.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <output_file.md>"
    exit 1
fi

OUTPUT_FILE="$1"
RESULTS_DIR="$(dirname "$(dirname "$OUTPUT_FILE")")/results"
SUMMARY="$RESULTS_DIR/SCORES.csv"

# --- CSV header if new ---
if [[ ! -f "$SUMMARY" ]]; then
    echo "timestamp,model,test,criterion,score,notes" > "$SUMMARY"
fi

# --- Extract metadata from filename ---
BASENAME="$(basename "$OUTPUT_FILE" .md)"
# Expected format: model_name_test_name_timestamp
IFS='_' read -ra PARTS <<< "$BASENAME"
MODEL="${PARTS[0]}"
TEST="test"  # simplified; could parse better with known test names
TIMESTAMP="$(date +%Y-%m-%dT%H:%M:%S)"

echo "Scoring: $OUTPUT_FILE"
echo "Model:   $MODEL"
echo ""

# --- Criteria (generic; customize per category) ---
CRITERIA=("completeness" "correctness" "runnable" "restraint")

for criterion in "${CRITERIA[@]}"; do
    while true; do
        read -p "$criterion (1-5): " score
        if [[ "$score" =~ ^[1-5]$ ]]; then
            break
        fi
        echo "Enter a number between 1 and 5."
    done
    read -p "Notes: " notes
    echo "$TIMESTAMP,$MODEL,$TEST,$criterion,$score,\"$notes\"" >> "$SUMMARY"
done

echo ""
echo "Score appended to $SUMMARY"
