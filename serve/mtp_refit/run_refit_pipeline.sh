#!/usr/bin/env bash
# Post-datagen MTP re-fit pipeline. Safe to launch while datagen is still
# running: it polls until datagen exits, then parses + fidelity-checks + trains.
#
# Stages:
#   1. wait for datagen (gen_data.py) to exit and gen_meta.json to exist
#   2. parse trace.bin -> ./data shards
#   3. fidelity check (shipped weights) on a free GPU; GATE on loss/match
#   4. train the dense core (only if the gate passes)
#
# Env knobs (all optional):
#   DATAGEN_PID      pid of the datagen process to wait on (default: auto-detect)
#   TRACE            trace file (default: $PWD/trace.bin)
#   DATA_DIR         shard out dir (default: $PWD/data)
#   CKPT             checkpoint dir (default: the heretic-2 W4A16 model)
#   GPU              xpu device for fidelity+train (default: xpu:0)
#   MIN_LEN          min sequence length to keep (default: 64)
#   EPOCHS           training epochs (default: 3)
#   LR               learning rate (default: 1e-4)
#   OUT_DIR          checkpoint out dir (default: $PWD/checkpoints)
#   SKIP_TRAIN       =1 to stop after the fidelity check
set -uo pipefail

cd "$(dirname "$0")"
PY=/home/dc/vllm-fresh-venv/bin/python
TRACE="${TRACE:-$PWD/trace.bin}"
DATA_DIR="${DATA_DIR:-$PWD/data}"
CKPT="${CKPT:-/mnt/data/models/devancarlin-Qwen3.8-Flash-Next-heretic-2-W4A16-BF16src}"
GPU="${GPU:-xpu:0}"
MIN_LEN="${MIN_LEN:-64}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
OUT_DIR="${OUT_DIR:-$PWD/checkpoints}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
LOG="$PWD/refit_pipeline.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== MTP re-fit pipeline start ==="
log "trace=$TRACE data=$DATA_DIR gpu=$GPU ckpt=$CKPT"

# --- 1. wait for datagen --------------------------------------------------- #
if [[ -n "${DATAGEN_PID:-}" ]]; then
  log "waiting for datagen pid=$DATAGEN_PID ..."
  while kill -0 "$DATAGEN_PID" 2>/dev/null; do sleep 60; done
else
  log "auto-waiting for any gen_data.py process to exit ..."
  while pgrep -f "gen_data.py" >/dev/null 2>&1; do
    log "  datagen still running ($(pgrep -cf 'gen_data.py') procs); sleeping 60s"
    sleep 60
  done
fi
# give workers a moment to release the GPUs
sleep 30
log "datagen finished; verifying artifacts"
[[ -s "$TRACE" ]] || { log "ERROR: trace missing/empty: $TRACE"; exit 1; }
[[ -f gen_meta.json ]] && cat gen_meta.json | tee -a "$LOG"
log "trace size: $(du -h "$TRACE" | cut -f1)"

# --- 2. parse trace into shards ------------------------------------------- #
log "parsing trace -> $DATA_DIR (min_len=$MIN_LEN)"
"$PY" parse_trace.py --trace "$TRACE" --out-dir "$DATA_DIR" \
  --min-len "$MIN_LEN" --shard-tokens 200000 2>&1 | tee -a "$LOG"
[[ -n "$(ls "$DATA_DIR"/shard-*.pt 2>/dev/null)" ]] || { log "ERROR: no shards produced"; exit 1; }

# --- 3. fidelity check (gate) --------------------------------------------- #
log "fidelity check on $GPU (shipped weights, expect low loss / decent match)"
FID_OUT="$("$PY" train_mtp.py --data-dir "$DATA_DIR" --ckpt "$CKPT" \
  --device "$GPU" --mode fidelity --eval-n 16 --min-len "$MIN_LEN" 2>&1 | tee -a "$LOG")"
echo "$FID_OUT" | grep -a "FIDELITY" || true

MATCH=$(echo "$FID_OUT" | grep -aoP 'match=\K[0-9.]+' | head -1)
LOSS=$(echo "$FID_OUT" | grep -aoP 'loss=\K[0-9.]+' | head -1)
log "parsed fidelity: match=${MATCH:-NA} loss=${LOSS:-NA}"

# Gate on MATCH (the robust discriminator). A broken reimplementation (wrong
# RoPE / MoE orientation / norm convention) yields near-random logits:
#   match ~ 1/vocab ~ 0.004,  loss ~ ln(vocab) ~ 12.4
# A faithful head reproduces the shipped draft's predictions:
#   match ~ 0.3+,  loss ~ 3-7 (CE vs the *sampled* next token, not the mode).
# Threshold 0.15 sits far inside that gap -> no false negatives, catches bugs.
GATE_OK=1
if [[ -z "${MATCH:-}" ]]; then
  log "GATE FAIL: could not parse match from fidelity output"; GATE_OK=0
elif ! awk "BEGIN{exit !($MATCH > 0.15)}"; then
  log "GATE FAIL: match=$MATCH <= 0.15 (reimplementation likely wrong)"; GATE_OK=0
fi
# loss is informational only (sampled-token CE is inherently higher than mode CE)
[[ -n "${LOSS:-}" ]] && log "fidelity loss=$LOSS (informational; uniform baseline ~12.4)"
if [[ "$GATE_OK" != "1" ]]; then
  log "stopping before training (fidelity gate failed). Inspect $LOG."
  exit 2
fi
log "fidelity gate PASSED (match=$MATCH) -> proceeding to training"

# --- 4. train -------------------------------------------------------------- #
if [[ "$SKIP_TRAIN" == "1" ]]; then
  log "SKIP_TRAIN=1 -> stopping after fidelity check"
  exit 0
fi
log "training dense core: epochs=$EPOCHS lr=$LR out=$OUT_DIR"
"$PY" train_mtp.py --data-dir "$DATA_DIR" --ckpt "$CKPT" --device "$GPU" \
  --mode train --epochs "$EPOCHS" --lr "$LR" --min-len "$MIN_LEN" \
  --eval-n 16 --eval-every 50 --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

log "=== pipeline complete ==="
log "best checkpoint: $OUT_DIR/best.pt (if present)"
ls -la "$OUT_DIR" 2>/dev/null | tee -a "$LOG"
