#!/usr/bin/env bash
# =============================================================================
# start-flashnext-llama.sh - serve Qwen3.8-Flash-Next (UD-Q4_K_XL) via llama.cpp
#
# All 4 GPUs (0,1,2,3) / port 8090. Open WebUI "flash-next" slot.
#
# Model: unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL (125B MoE, 6B active,
#        + 51B n-gram embedding, + 4B MTP). ~104 GB across 4 shards.
#
# Requires the llama.cpp PR #27742 build (qwen4exp arch). The tree at
# ~/electric-sheep/llama/llama.cpp is checked out at pr-27742 (head 035e22731).
# Rebuild after any branch switch:
#   cd ~/electric-sheep/llama/llama.cpp && cmake --build build -j$(nproc)
#
# Notes vs the 27B/35B servers:
#   - NO MTP: the PR sets no_mtp=True for qwen4exp (4B MTP head not wired).
#   - NO mmproj: GGUF repo ships text-only; vision tower (Qwen3-VL ViT) is
#     present in the PR but no projector file is published yet.
#   - Hybrid attention (Gated DeltaNet + QSA): KV cache is far smaller than a
#     pure-attention model, so 262K context fits with HOST_MEM_FALLBACK to RAM.
#   - KV cache is q8_0 (-ctk/-ctv): halves the 12 full-attention layers' K/V
#     footprint (~9.7 GB f16 -> ~4.9 GB) with negligible quality loss.
#     Quantized KV enables a Hadamard K-rotation input that qwen4exp's QSA
#     graph rejects (GGML_ASSERT self_k_rot==nullptr), so the launch sets
#     LLAMA_ATTN_ROT_DISABLE=1 to satisfy it.
#   - PLE n-gram table (28.7 GB, single indivisible tensor) placement is a
#     tradeoff (FLASHNEXT_PLE_DEV):
#       CPU   (default, FAST)  table in host RAM, GPU reads it zero-copy.
#                              ~29 t/s, ~28 GB RAM page cache.
#       SYCL3 (all-VRAM)       table on GPU 3, GPUs 0-2 take all layers
#                              (--tensor-split 1,1,1,0). Model fully in VRAM,
#                              but the PLE gather (layer 0 on GPU 0) copies
#                              GPU3->RAM->GPU0 per token (no P2P on discrete
#                              Arc B70) => ~22 t/s, more CPU.
#   - Sampling (Unsloth recs): thinking temp 1.0 / top_p 0.95 / top_k 20;
#     instruct temp 0.7 / top_p 0.80 / presence_penalty 1.5. Set per-request.
#
# Usage:
#   bash start-flashnext-llama.sh          # start (kills any existing 8090 first)
#   bash start-flashnext-llama.sh stop     # stop
#   bash start-flashnext-llama.sh status   # check
#
# Env overrides: FLASHNEXT_PORT, FLASHNEXT_CTX, FLASHNEXT_ALIAS, FLASHNEXT_GPUS,
#                FLASHNEXT_PLE_DEV (CPU|SYCL0..3, default CPU), FLASHNEXT_TSPLIT
# =============================================================================
set -uo pipefail

GPUS="${FLASHNEXT_GPUS:-0,1,2,3}"
PORT="${FLASHNEXT_PORT:-8090}"
B="$HOME/electric-sheep/llama/llama.cpp/build/bin"
D="$HOME/electric-sheep/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL"
M="$D/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
LOG="$HOME/electric-sheep/serve/logs/llama_${PORT}.log"
CTX="${FLASHNEXT_CTX:-262144}"
ALIAS="${FLASHNEXT_ALIAS:-flash-next}"
# PLE n-gram table (28 GB) placement:
#   CPU   (default, FAST)  - table stays in host RAM, GPU reads it zero-copy.
#                            No cross-device copy. ~29 t/s. Uses ~28 GB RAM.
#   SYCL3 (all-VRAM)       - table pinned to GPU 3, GPUs 0-2 take all layers.
#                            Model fully in VRAM, but the PLE gather (layer 0 on
#                            GPU 0) must copy GPU3->RAM->GPU0 per token (no P2P
#                            on discrete Arc B70) => ~22 t/s, more CPU.
PLE_DEV="${FLASHNEXT_PLE_DEV:-CPU}"
if [ "$PLE_DEV" = "CPU" ]; then
  TSPLIT="${FLASHNEXT_TSPLIT:-}"          # empty = default free-memory split, all GPUs
else
  TSPLIT="${FLASHNEXT_TSPLIT:-1,1,1,0}"   # GPU 3 holds only the PLE table
fi

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
    source "$HOME/electric-sheep/llama/set-env.sh" "$GPUS" 2>/dev/null
    # q8_0 KV is quantized, so the KV cache enables a Hadamard K-rotation input
    # (attn_rot_k). qwen4exp's QSA path applies RoPE to K before caching and
    # asserts self_k_rot==nullptr, so it crashes on quantized KV. Disabling the
    # rotation satisfies the assert; negligible cost at q8_0 precision.
    export LLAMA_ATTN_ROT_DISABLE=1
    # Map GPU indices (0,1,2,3) to SYCL device names (SYCL0,SYCL1,...) for --device
    DEV_NAMES=$(echo "$GPUS" | tr ',' '\n' | sed 's/^/SYCL/' | paste -sd, -)
    # Chat template: prefer the GGUF's embedded template. Fall back to none
    # (llama.cpp default) if absent. No external jinja for this model.
    if head -c 16M "$M" | grep -c "enable_thinking" >/dev/null 2>&1; then
      TPL_ARGS=(--jinja)
    else
      TPL_ARGS=(--jinja)
    fi
    # PLE placement controls which layout flags are passed:
    #   CPU   -> default split across all GPUs, no --device/--tensor-split/-ot
    #   SYCLn -> pin PLE to that GPU, GPUs 0-2 take all layers
    if [ "$PLE_DEV" = "CPU" ]; then
      LAYOUT_ARGS=()
      LAYOUT_DESC="PLE->CPU(RAM)"
    else
      LAYOUT_ARGS=(--device "$DEV_NAMES" --tensor-split "$TSPLIT" -ot "per_layer_token_embd=$PLE_DEV")
      LAYOUT_DESC="PLE->$PLE_DEV"
    fi
    nohup "$B/llama-server" \
      -m "$M" \
      -ngl 99 -fa on \
      -c "$CTX" -np 1 -b 2048 -ub 512 \
      "${LAYOUT_ARGS[@]}" \
      -ctk q8_0 -ctv q8_0 \
      "${TPL_ARGS[@]}" \
      --reasoning on \
      --alias "$ALIAS" \
      --host 0.0.0.0 --port "$PORT" \
      > "$LOG" 2>&1 &
    echo "llama-server (Qwen3.8-Flash-Next UD-Q4_K_XL, ${CTX} ctx, np=1, KV q8_0, ${LAYOUT_DESC}, alias ${ALIAS}) on GPUs $GPUS :$PORT, pid $!"
    # Large model: allow up to ~5 min for load across 4 GPUs
    for i in $(seq 1 60); do
      grep -q "listening on" "$LOG" 2>/dev/null && { echo "ready ~$((i*5))s"; break; }
      grep -qE "error|Error|abort|exited" "$LOG" 2>/dev/null && { echo "LOAD ERROR - see $LOG"; tail -20 "$LOG"; break; }
      sleep 5
    done
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac