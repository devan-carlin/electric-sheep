#!/usr/bin/env bash
# compare-models.sh — Compare scores between two models
#
# Usage:
#   ./compare-models.sh <model_a> <model_b>
#
# Reads from results/SCORES.csv and prints a side-by-side comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SUMMARY="$ROOT_DIR/results/SCORES.csv"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <model_a> <model_b>"
    exit 1
fi

if [[ ! -f "$SUMMARY" ]]; then
    echo "No scores found. Run score-test.sh first."
    exit 1
fi

MODEL_A="$1"
MODEL_B="$2"

echo "=============================================="
echo "  Comparison: $MODEL_A vs $MODEL_B"
echo "=============================================="
echo ""

# Parse CSV (skip header), compute averages per model
declare -A SCORES_A
declare -A SCORES_B
COUNT_A=0
COUNT_B=0

while IFS=',' read -r timestamp model test criterion score notes; do
    [[ "$timestamp" == "timestamp" ]] && continue  # skip header
    score="${score// /}"  # trim spaces

    if [[ "$model" == "$MODEL_A" ]]; then
        SCORES_A["$criterion"]="${SCORES_A[$criterion]:-0}"
        SCORES_A["$criterion"]=$(echo "${SCORES_A[$criterion]} + $score" | bc)
        COUNT_A=$((COUNT_A + 1))
    elif [[ "$model" == "$MODEL_B" ]]; then
        SCORES_B["$criterion"]="${SCORES_B[$criterion]:-0}"
        SCORES_B["$criterion"]=$(echo "${SCORES_B[$criterion]} + $score" | bc)
        COUNT_B=$((COUNT_B + 1))
    fi
done < "$SUMMARY"

printf "%-15s %10s %10s %10s\n" "Criterion" "$MODEL_A" "$MODEL_B" "Delta"
echo "--------------------------------------------------------------"

for criterion in completeness correctness runnable restraint; do
    total_a="${SCORES_A[$criterion]:-0}"
    total_b="${SCORES_B[$criterion]:-0}"

    if [[ "$total_a" != "0" && "$total_b" != "0" ]]; then
        avg_a=$(echo "scale=1; $total_a / 1" | bc)
        avg_b=$(echo "scale=1; $total_b / 1" | bc)
        delta=$(echo "scale=1; $avg_a - $avg_b" | bc)
        printf "%-15s %10s %10s %10s\n" "$criterion" "$avg_a" "$avg_b" "$delta"
    fi
done

echo ""
echo "Samples: $MODEL_A = $COUNT_A, $MODEL_B = $COUNT_B"
