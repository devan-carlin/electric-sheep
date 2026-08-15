#!/usr/bin/env bash
# =============================================================================
# Phase 1 — PROBE: Heretic abliteration on Qwen3.5-4B (small, same architecture)
#
# Purpose: de-risk the two unknowns BEFORE the 6-10h 27B run:
#   1. XPU compatibility of Heretic (PyTorch XPU backend)
#   2. qwen3_5 GDN-hybrid architecture support (same arch as Qwen3.8-27B)
#
# Qwen3.5-4B is the ideal probe: same qwen3_5 Gated-DeltaNet hybrid architecture,
# ~8GB BF16 (fits 1 GPU), fast, and a thinking model (exercises CoT-skip handling).
#
# GATE: only run heretic-qwen3.8-27b.sh if this completes and saves a model.
#
# Usage:  bash heretic-probe-qwen3.5-4b.sh
# Env:    N_TRIALS (default 20), SEED (default 42)
# =============================================================================
set -euo pipefail
cd /home/dc/electric-sheep/vllm

VENV=/home/dc/electric-sheep/vllm/heretic-venv
MODEL=/home/dc/electric-sheep/models/Qwen3.5-4B
OUT=/home/dc/electric-sheep/models/Qwen3.5-4B-heretic-probe
LOG=/tmp/heretic-probe.log
N_TRIALS="${N_TRIALS:-20}"
SEED="${SEED:-42}"

# --- 0. Prereqs -------------------------------------------------------------
[[ -d "$VENV" ]] || { echo "ERROR: heretic venv missing. Run: bash heretic-setup-venv.sh"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# XPU device env (ONEAPI_DEVICE_SELECTOR, ZE_AFFINITY_MASK, ...).
# vLLM-specific vars in this file are harmless for Heretic.
# shellcheck disable=SC1091
source set-env-0123-gpu.sh

# Pick the heretic entrypoint (console script or module).
if command -v heretic >/dev/null 2>&1; then HERETIC=(heretic); else HERETIC=(python -m heretic); fi

# --- 1. Download probe model (if absent) ------------------------------------
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "==> Downloading Qwen/Qwen3.5-4B -> $MODEL"
  hf download Qwen/Qwen3.5-4B --local-dir "$MODEL"
else
  echo "==> Probe model already present: $MODEL"
fi

# --- 2. Run Heretic (headless) ----------------------------------------------
# All interactive prompts are pre-answered via CLI flags (highest config priority):
#   --checkpoint-action restart  : fresh run, no "continue/restart" prompt
#   --trial-index 0              : pick best (Pareto) trial, no "which trial" prompt
#   --model-action save          : save and exit, no action menu
#   --save-directory + --export-strategy merge : no path/export prompts
echo "==> Running Heretic probe (n_trials=$N_TRIALS, seed=$SEED)"
echo "    Log: $LOG"
# shellcheck disable=SC2086
"${HERETIC[@]}" \
  --model "$MODEL" \
  --device-map auto \
  --quantization none \
  --n-trials "$N_TRIALS" \
  --n-startup-trials 10 \
  --seed "$SEED" \
  --checkpoint-action restart \
  --trial-index 0 \
  --model-action save \
  --save-directory "$OUT" \
  --export-strategy merge \
  2>&1 | tee "$LOG"

# --- 3. Pass/fail check ------------------------------------------------------
echo
echo "==> Probe result check"
if [[ -f "$OUT/config.json" ]] && ls "$OUT"/*.safetensors >/dev/null 2>&1; then
  echo "[PASS] Merged model saved to $OUT"
  echo "       -> Cleared to run:  bash heretic-qwen3.8-27b.sh"
else
  echo "[FAIL] No merged model in $OUT — inspect $LOG before attempting the 27B."
  exit 1
fi
