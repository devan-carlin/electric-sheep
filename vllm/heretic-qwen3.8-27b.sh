#!/usr/bin/env bash
# =============================================================================
# Phase 2 — MAIN: Heretic abliteration on Qwen3.8-27B (the target)
#
# ONLY run after the Phase 1 probe (heretic-probe-qwen3.5-4b.sh) has PASSED.
#
# The 52GB BF16 model does not fit on one 34GB GPU, so it is spread across all
# 4 Arc Pro B70 GPUs via Accelerate device_map="auto" (the same mechanism
# AutoRound used successfully on this XPU stack).
#
# Usage:  bash heretic-qwen3.8-27b.sh
# Env:
#   N_TRIALS        (default 200)  # full quality; 100 halves time
#   N_STARTUP       (default 60)
#   SEED            (default 42)
#   CHECKPOINT_ACTION (default restart)  # use "continue" to resume an interrupted run
#   MAX_MEMORY      (default empty)      # e.g. '{"0":"30GB","1":"30GB","2":"30GB","3":"30GB"}'
#                                        # only set if device_map auto mis-distributes
# =============================================================================
set -euo pipefail
cd /home/dc/electric-sheep/vllm

VENV=/home/dc/electric-sheep/vllm/heretic-venv
MODEL=/home/dc/electric-sheep/models/Qwen3.8-27B
OUT=/home/dc/electric-sheep/models/Qwen3.8-27B-heretic
LOG=/tmp/heretic-27b.log
N_TRIALS="${N_TRIALS:-200}"
N_STARTUP="${N_STARTUP:-60}"
SEED="${SEED:-42}"
CHECKPOINT_ACTION="${CHECKPOINT_ACTION:-restart}"
MAX_MEMORY="${MAX_MEMORY:-}"

# --- 0. Prereqs -------------------------------------------------------------
[[ -d "$VENV" ]] || { echo "ERROR: heretic venv missing. Run: bash heretic-setup-venv.sh"; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model missing at $MODEL"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# shellcheck disable=SC1091
source set-env-0123-gpu.sh

# Make sure no vLLM server / quant is holding the GPUs.
if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
  echo "ERROR: a vLLM server is running. Kill it first (see HERETIC-PROCESS.md §8)."
  exit 1
fi
if pgrep -f "auto-round quantize" >/dev/null 2>&1; then
  echo "ERROR: an AutoRound quantization is still running. Wait for it to finish."
  exit 1
fi

if command -v heretic >/dev/null 2>&1; then HERETIC=(heretic); else HERETIC=(python -m heretic); fi

# --- 1. Build the (optional) max-memory flag --------------------------------
MAXMEM_ARGS=()
if [[ -n "$MAX_MEMORY" ]]; then
  MAXMEM_ARGS=(--max-memory "$MAX_MEMORY")
fi

# --- 2. Run Heretic (headless) ----------------------------------------------
echo "==> Running Heretic on Qwen3.8-27B (n_trials=$N_TRIALS, seed=$SEED, checkpoint=$CHECKPOINT_ACTION)"
echo "    Output: $OUT"
echo "    Log:    $LOG"
# shellcheck disable=SC2086
"${HERETIC[@]}" \
  --model "$MODEL" \
  --device-map auto \
  "${MAXMEM_ARGS[@]}" \
  --quantization none \
  --n-trials "$N_TRIALS" \
  --n-startup-trials "$N_STARTUP" \
  --seed "$SEED" \
  --checkpoint-action "$CHECKPOINT_ACTION" \
  --trial-index 0 \
  --model-action save \
  --save-directory "$OUT" \
  --export-strategy merge \
  2>&1 | tee "$LOG"

# --- 3. Result check ---------------------------------------------------------
echo
echo "==> Main run result check"
if [[ -f "$OUT/config.json" ]] && ls "$OUT"/*.safetensors >/dev/null 2>&1; then
  echo "[PASS] Abliterated model saved to $OUT"
  echo "       -> Verify per HERETIC-PROCESS.md §7 (load in vLLM, A/B behavior)."
  echo "       -> Optional INT4: edit quantize-qwen3.8-27b-int4.sh MODEL= to $OUT and re-run."
else
  echo "[FAIL] No merged model in $OUT — inspect $LOG."
  echo "       If interrupted, resume with:  CHECKPOINT_ACTION=continue bash heretic-qwen3.8-27b.sh"
  exit 1
fi
