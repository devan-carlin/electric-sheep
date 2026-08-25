#!/usr/bin/env bash
# =============================================================================
# start-qwen-llama.sh - serve the Qwen slot via llama.cpp SYCL
#
# GPU 2 / port 8088 - the Open WebUI "qwen" slot (alias qwen).
#
# Version-agnostic launcher: the model is pinned below; to swap Qwen versions,
# change MODEL_DIR/MODEL_FILE (or use the env overrides) - the alias stays
# "qwen" so WebUI and clients never need updating.
#
# Current model (2026-08-25):  Qwen3.6-35B-A3B Aggressive (Q4_K_P).
#   - MoE (35B total / 3B active) -> much faster decode than the dense
#     Qwen3.8-27B it replaces; user prefers the throughput.
#   - 256K context fully on-GPU (q4_0 KV ~4.8 GiB, ~5 GB spare on one B70).
#   - q4_0 KV: fastest + most stable on this SYCL build.
#   - thinking ON by default (--reasoning on, deepseek format, 2048-token
#     budget). Thoughts -> message.reasoning_content, answer -> message.content.
#   - chat template: prefer the GGUF's embedded template; external jinja
#     (devan-carlin vLLM dir) only as fallback.
#
# Usage:
#   bash start-qwen-llama.sh          # start (kills any existing 8088 first)
#   bash start-qwen-llama.sh stop     # stop
#   bash start-qwen-llama.sh status   # check
#
# Env overrides: QWEN_LLAMA_GPU, QWEN_LLAMA_PORT, QWEN_LLAMA_CTX,
#                QWEN_LLAMA_ALIAS, QWEN_LLAMA_MODEL_DIR, QWEN_LLAMA_MODEL_FILE,
#                QWEN_LLAMA_MMPROJ, QWEN_LLAMA_REASONING (on|off|auto),
#                QWEN_LLAMA_REASONING_FORMAT (none|deepseek|deepseek-legacy),
#                QWEN_LLAMA_REASONING_BUDGET (tokens; -1 unlimited, 0 none)
# =============================================================================
set -uo pipefail

GPU="${QWEN_LLAMA_GPU:-2}"
PORT="${QWEN_LLAMA_PORT:-8088}"
B="$HOME/electric-sheep/llama/llama.cpp/build/bin"
D="${QWEN_LLAMA_MODEL_DIR:-$HOME/electric-sheep/models/-Qwen3.6-35B-A3B---Aggressive}"
M="${QWEN_LLAMA_MODEL_FILE:-$D/Qwen3.6-35B-A3B---Aggressive-Q4_K_P.gguf}"
MM="${QWEN_LLAMA_MMPROJ:-$D/mmproj-Qwen3.6-35B-A3B---Aggressive-f16.gguf}"
TPL="$HOME/electric-sheep/models/devan-carlin-Qwen3.8-27B--ara-int4-AutoRound/Qwen3.8-27B--ara-w4g128/chat_template.jinja"
LOG="$HOME/electric-sheep/serve/logs/llama_${PORT}.log"
CTX="${QWEN_LLAMA_CTX:-262144}"
ALIAS="${QWEN_LLAMA_ALIAS:-qwen}"
REASONING="${QWEN_LLAMA_REASONING:-on}"
REASONING_FORMAT="${QWEN_LLAMA_REASONING_FORMAT:-deepseek}"
REASONING_BUDGET="${QWEN_LLAMA_REASONING_BUDGET:-2048}"

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
    [[ -f "$M" ]] || { echo "ERROR: model file not found: $M"; exit 1; }
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
    # --reasoning on/off is the modern replacement for
    # --chat-template-kwargs '{"enable_thinking":...}' (deprecated in b10355).
    # Only pass it when the template actually has the toggle.
    # --reasoning-format deepseek: thoughts -> message.reasoning_content,
    #   answer -> message.content (uniform across qwen/gemma; the jinja
    #   auto-parser splits each template's own wire format).
    # --reasoning-budget N: token cap on thinking (the "medium" lever;
    #   -1 = unrestricted, 0 = immediate end).
    reasoning_args=()
    if head -c 16M "$M" | grep -c "enable_thinking" >/dev/null; then
      reasoning_args=(--reasoning "$REASONING" --reasoning-format "$REASONING_FORMAT" --reasoning-budget "$REASONING_BUDGET")
    fi
    nohup "$B/llama-server" \
      -m "$M" --mmproj "$MM" \
      -ctk q4_0 -ctv q4_0 \
      -ngl 99 -fa on \
      -c "$CTX" -np 1 -b 2048 -ub 512 \
      "${TPL_ARGS[@]}" \
      ${reasoning_args[@]+"${reasoning_args[@]}"} \
      --alias "$ALIAS" \
      --host 0.0.0.0 --port "$PORT" \
      > "$LOG" 2>&1 &
    echo "llama-server ($(basename "$M"), q4_0 KV, ${CTX} ctx, np=1, reasoning ${REASONING}/${REASONING_FORMAT}/budget ${REASONING_BUDGET}, alias ${ALIAS}) on GPU $GPU :$PORT, pid $!"
    for i in $(seq 1 36); do
      grep -q "listening on" "$LOG" 2>/dev/null && { echo "ready ~$((i*5))s"; break; }
      sleep 5
    done
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac