#!/usr/bin/env bash
# =============================================================================
# start-qwen38-llama.sh - serve Qwen3.8-27B ( Aggressive) via llama.cpp
#
# GPU 2 / port 8088 - the Open WebUI "qwen3.8" slot (alias qwen3.8).
#
# Replaces the vLLM -ara instance (2026-08-25). Why llama.cpp:
#   - 256K context fully on-GPU (q4_0 KV = 4.8 GiB, ~7 GB spare on one B70).
#     vLLM capped at 128K on one card (256K needs gpu-mem-util 0.99+).
#   - Single-user (np=1) is llama.cpp's sweet spot. Lost vs vLLM: prefix
#     caching, tool-call parsers, concurrency.
#
# Config (from the 8-run matrix, /home/dc/electric-sheep/bench/ab/):
#   - Q4_K_P weights + BF16 mmproj ( Aggressive, )
#   - q4_0 KV cache: fastest + most stable (q5_0 costs ~12-15%)
#   - built-in MTP depth 1 (~1.35x: 23.3 vs 17.2 t/s). The model has a single
#     NextN head, so depth >1 silently clamps to 1.
#   - 256K context (model native max), 1 slot, thinking OFF by default
#   - chat template: the GGUF has an EMBEDDED template (enable_thinking
#     toggle) - use it. The external jinja (devan-carlin vLLM dir) is only a
#     fallback if a future GGUF ships without one.
#
# Usage:
#   bash start-qwen38-llama.sh          # start (kills any existing 8088 first)
#   bash start-qwen38-llama.sh stop     # stop
#   bash start-qwen38-llama.sh status   # check
#
# Env overrides: QWEN_LLAMA_GPU, QWEN_LLAMA_PORT, QWEN_LLAMA_CTX,
#                QWEN_LLAMA_ALIAS, QWEN_LLAMA_MTP, QWEN_LLAMA_THINKING
# =============================================================================
set -uo pipefail

GPU="${QWEN_LLAMA_GPU:-2}"
PORT="${QWEN_LLAMA_PORT:-8088}"
B="$HOME/electric-sheep/llama/llama.cpp/build/bin"
D="$HOME/electric-sheep/models/-Qwen3.8-27B---Aggressive"
M="$D/-Q4_K_P.gguf"
MM="$D/mmproj-Qwen3.8-27B---Aggressive-BF16.gguf"
TPL="$HOME/electric-sheep/models/devan-carlin-Qwen3.8-27B--ara-int4-AutoRound/Qwen3.8-27B--ara-w4g128/chat_template.jinja"
LOG="$HOME/electric-sheep/serve/logs/llama_${PORT}.log"
CTX="${QWEN_LLAMA_CTX:-262144}"
ALIAS="${QWEN_LLAMA_ALIAS:-qwen3.8}"
MTP="${QWEN_LLAMA_MTP:-1}"
THINKING="${QWEN_LLAMA_THINKING:-false}"

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
    # Chat template: prefer the GGUF's embedded template (has the
    # enable_thinking toggle). Only fall back to the external jinja if the
    # GGUF ships without one. Passing an external template when an embedded
    # one exists overrides it and breaks rendering (model echoes template src).
    # Probe only the header (first 16 MB) with grep -c: `strings | grep -q`
    # under pipefail dies with SIGPIPE (exit 141) and inverts the test.
    if head -c 16M "$M" | grep -c "enable_thinking" >/dev/null; then
      TPL_ARGS=(--jinja)
    else
      TPL_ARGS=(--jinja --chat-template "$TPL")
    fi
    mtp_args=()
    [ "$MTP" -gt 0 ] && mtp_args=(--spec-type draft-mtp --spec-draft-n-max "$MTP")
    # --reasoning on/off is the modern replacement for
    # --chat-template-kwargs '{"enable_thinking":...}' (deprecated in b10355).
    reasoning_flag="on"
    [ "$THINKING" = "false" ] && reasoning_flag="off"
    nohup "$B/llama-server" \
      -m "$M" --mmproj "$MM" \
      -ctk q4_0 -ctv q4_0 \
      -ngl 99 -fa on \
      -c "$CTX" -np 1 -b 2048 -ub 512 \
      ${mtp_args[@]+"${mtp_args[@]}"} \
      "${TPL_ARGS[@]}" \
      --reasoning "$reasoning_flag" \
      --alias "$ALIAS" \
      --host 0.0.0.0 --port "$PORT" \
      > "$LOG" 2>&1 &
    echo "llama-server (Qwen3.8-27B Aggressive, q4_0 KV, MTP=$MTP, ${CTX} ctx, np=1, alias ${ALIAS}) on GPU $GPU :$PORT, pid $!"
    for i in $(seq 1 36); do
      grep -q "listening on" "$LOG" 2>/dev/null && { echo "ready ~$((i*5))s"; break; }
      sleep 5
    done
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac