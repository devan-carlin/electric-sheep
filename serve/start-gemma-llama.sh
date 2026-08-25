#!/usr/bin/env bash
# =============================================================================
# start-gemma-llama.sh - serve Gemma4 26B-A4B ( Balanced) via llama.cpp
#
# GPU 3 / port 8089 - the Open WebUI "gemma" slot (alias gemma, version-agnostic).
#
# Chosen over the 31B QAT MTP after a blind prose A/B (2026-08-25):
#   26B-A4B won 4/5 prompts AND decodes ~2.5-3x faster (64 vs 24 t/s).
#   See /home/dc/electric-sheep/ab-test/ for the test artifacts.
#
# Config:
#   - Q4_K_P weights + f16 mmproj ( Balanced, )
#   - q4_0 KV cache (q8_0/iq4_nl segfault on this SYCL build)
#   - 256K context (model native max), 1 slot. KV ~16.9 GiB q4_0 — just over
#     the on-GPU budget, so the top few K spill to CPU. np=1 keeps it mostly on-GPU.
#   - no MTP (26B-A4B has no MTP head in this release)
#   - thinking ON by default (--reasoning on, deepseek format, 2048-token
#     budget). Thoughts -> message.reasoning_content, answer -> message.content.
#
# Env overrides: GEMMA_LLAMA_GPU, GEMMA_LLAMA_PORT, GEMMA_LLAMA_CTX,
#                GEMMA_LLAMA_ALIAS, GEMMA_LLAMA_REASONING (on|off|auto),
#                GEMMA_LLAMA_REASONING_FORMAT (none|deepseek|deepseek-legacy),
#                GEMMA_LLAMA_REASONING_BUDGET (tokens; -1 unlimited, 0 none)
#
# Usage:
#   bash start-gemma-llama.sh          # start (kills any existing 8089 first)
#   bash start-gemma-llama.sh stop     # stop
#   bash start-gemma-llama.sh status   # check
# =============================================================================
set -uo pipefail

GPU="${GEMMA_LLAMA_GPU:-3}"
PORT="${GEMMA_LLAMA_PORT:-8089}"
B="$HOME/electric-sheep/llama/llama.cpp/build/bin"
D="$HOME/electric-sheep/models/-Gemma4-26B-A4B---Balanced"
M="$D/Gemma4-26B-A4B---Balanced-Q4_K_P.gguf"
MM="$D/mmproj-Gemma4-26B-A4B---Balanced-f16.gguf"
LOG="$HOME/electric-sheep/serve/logs/llama_8089.log"
CTX="${GEMMA_LLAMA_CTX:-262144}"
ALIAS="${GEMMA_LLAMA_ALIAS:-gemma}"
REASONING="${GEMMA_LLAMA_REASONING:-on}"
REASONING_FORMAT="${GEMMA_LLAMA_REASONING_FORMAT:-deepseek}"
REASONING_BUDGET="${GEMMA_LLAMA_REASONING_BUDGET:-2048}"

stop() {
  pkill -f "llama-server.*--port $PORT" 2>/dev/null && sleep 3
  pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null
  pgrep -af "llama-server.*--port $PORT" >/dev/null && echo "still running" || echo "stopped"
}

status() {
  if pgrep -af "llama-server.*--port $PORT" >/dev/null; then
    echo "RUNNING: $(pgrep -af "llama-server.*--port $PORT" | head -1)"
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" | head -c 200; echo
  else
    echo "not running"
  fi
}

case "${1:-start}" in
  stop)   stop ;;
  status) status ;;
  start)
    stop
    source "$HOME/electric-sheep/llama/set-env.sh" "$GPU" 2>/dev/null
    nohup "$B/llama-server" \
      -m "$M" --mmproj "$MM" \
      -ctk q4_0 -ctv q4_0 \
      -ngl 99 -sm layer -fa on \
      -c "$CTX" -np 1 -b 2048 -ub 512 \
      --jinja \
      --reasoning "$REASONING" --reasoning-format "$REASONING_FORMAT" --reasoning-budget "$REASONING_BUDGET" \
      --alias "$ALIAS" \
      --host 0.0.0.0 --port "$PORT" \
      > "$LOG" 2>&1 &
    echo "llama-server (26B-A4B, q4_0 KV, ${CTX} ctx, np=1, reasoning ${REASONING}/${REASONING_FORMAT}/budget ${REASONING_BUDGET}, alias ${ALIAS}) on GPU $GPU :$PORT, pid $!"
    for i in $(seq 1 36); do
      grep -q "listening on" "$LOG" 2>/dev/null && { echo "ready ~$((i*5))s"; break; }
      sleep 5
    done
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac