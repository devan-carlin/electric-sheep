#!/usr/bin/env bash
# =============================================================================
# switch-config.sh - one command to move the 4x B70 rig between the two configs.
#
# Both configs serve the SAME handle: alias qwen-256k on :8000. Suze and Open
# WebUI point at qwen-256k:8000 and do not care which config is up. The configs
# are mutually exclusive on the GPUs, so switching is: stop everything, start one.
#
#   CONFIG A  all-4:  125B Flash-Next W4A16 on GPUs 0,1,2,3 (TP4) -> :8000 qwen-256k
#                     no ComfyUI. The big-context / max-throughput config.
#   CONFIG B  split:  ComfyUI on GPU 0 (:8188) + GPU 1 (:8189)
#                     27B on GPUs 2,3 (TP2) -> :8000 qwen-256k
#                     Image gen AND an LLM at once. The 27B is the FP8 checkpoint
#                     (higher quality); falls back to the int4 AutoRound if FP8 is absent.
#
# Usage:
#   bash switch-config.sh a          # bring up config A (all-4 125B)
#   bash switch-config.sh b          # bring up config B (comfy 0,1 + 27B 2,3)
#   bash switch-config.sh stop       # stop all inference + comfy, free all GPUs
#   bash switch-config.sh status     # what is up right now
#
# Config B model override (defaults to the FP8 checkpoint):
#   QWEN12_MODEL_DIR=/path QWEN12_QUANT='' bash switch-config.sh b
#   QWEN12_QUANT='' means "the checkpoint is already quantized, do not requant".
#   To force the proven int4 AutoRound instead of FP8:
#   FORCE_INT4=1 bash switch-config.sh b
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"

# The two LLM launchers + the ComfyUI launcher. All three already exist and are proven.
A_LAUNCHER="$HERE/start-qwen-256k-intel-autoround-vllm.sh"   # 125B, TP4, :8000, alias qwen-256k
B27_LAUNCHER="$HERE/start-qwen3.8-27b-int4-gpu12.sh"          # 27B, TP2, :8000, alias qwen-256k
COMFY_LAUNCHER="$HOME/neon-demon/serve/start-comfyui.sh"     # comfy A/B on GPU 0/1

# Config B's 27B: FP8 by default (higher quality), int4 AutoRound as the proven fallback.
FP8_DIR="/mnt/data/models/Qwen3.8-27B-FP8"
INT4_DIR="$HOME/electric-sheep/models/devan-carlin-Qwen3.8-27B-int4-AutoRound/Qwen3.8-27B-w4g128"

port_up() { curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/v1/models"; }
comfy_up() { curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/"; }

# --- stop everything that could hold a GPU or :8000 --------------------------
stop_all() {
  echo "=== stopping all inference + comfy ==="
  # The 125B front door (both engines) and any 27B on :8000. Both vLLM variants share the
  # api_server module + the :8000 port, so one pkill pattern catches both.
  bash "$A_LAUNCHER" stop 2>/dev/null || true
  pkill -f "vllm.entrypoints.openai.api_server.*--port 8000" 2>/dev/null && echo "  killed a :8000 vLLM" || echo "  no :8000 vLLM"
  # ComfyUI on 8188/8189 (its launcher knows its own pidfile + pkill patterns).
  bash "$COMFY_LAUNCHER" stop 2>/dev/null || true
  # Wait for :8000 to actually free so the next start does not hit 'port in use'.
  for _ in $(seq 1 30); do port_up 8000 || break; sleep 2; done
  echo "  :8000 $([ port_up 8000 ] && echo 'STILL UP' || echo 'free')"
}

# --- config A: all four GPUs to the 125B -------------------------------------
start_a() {
  echo "=== CONFIG A: 125B Flash-Next on GPUs 0,1,2,3 -> :8000 qwen-256k ==="
  bash "$A_LAUNCHER" start
  echo "  (125B graph compile takes ~2-4 min before :8000 answers)"
}

# --- config B: comfy on 0,1 + 27B on 2,3 --------------------------------------
start_b() {
  # Pick the 27B checkpoint. FP8 unless forced or absent. A missing FP8 dir falls back to the
  # proven int4 AutoRound LOUDLY, so config B is always up and the fallback is never silent.
  local model quant
  if [ "${FORCE_INT4:-0}" = "1" ]; then
    model="$INT4_DIR"; quant="auto-round"; echo "  27B: int4 AutoRound (forced)"
  elif [ -d "$FP8_DIR" ] && [ -f "$FP8_DIR/config.json" ] && [ "$(find "$FP8_DIR" -name '*.safetensors' 2>/dev/null | wc -l)" -ge 1 ]; then
    model="$FP8_DIR"; quant=""; echo "  27B: FP8 (pre-quantized, no requant) at $FP8_DIR"
  else
    model="$INT4_DIR"; quant="auto-round"
    echo "  27B: FP8 NOT PRESENT at $FP8_DIR -> falling back to int4 AutoRound. Suze keeps a backend."
  fi

  echo "=== CONFIG B: ComfyUI on GPU 0,1 + 27B on GPU 2,3 -> :8000 qwen-256k ==="
  # Comfy first (fast to come up), then the 27B (slow to load). Comfy owns 0,1; the 27B owns 2,3.
  COMFY_GPU_A=0 COMFY_GPU_B=1 bash "$COMFY_LAUNCHER" start
  QWEN12_MODEL_DIR="$model" QWEN12_QUANT="$quant" \
  QWEN12_GPU="2,3" QWEN12_PORT=8000 QWEN12_SERVED="qwen-256k" \
    bash "$B27_LAUNCHER" start
  echo "  (27B loads, then :8000 answers as qwen-256k)"
}

status() {
  echo "=== :8000 (the qwen-256k handle) ==="
  if port_up 8000; then
    curl -s --max-time 3 "http://127.0.0.1:8000/v1/models" | python3 -c "import json,sys;d=json.load(sys.stdin);print('  UP, models:',[m['id'] for m in d.get('data',[])])" 2>/dev/null || echo "  UP"
  else echo "  DOWN"; fi
  echo "=== ComfyUI ==="
  for p in 8188 8189; do printf "  :%s " "$p"; comfy_up "$p" && echo "up" || echo "down"; done
  echo "=== vLLM procs on :8000 ==="
  pgrep -af "vllm.entrypoints.openai.api_server.*--port 8000" | sed 's/^/  /' || echo "  none"
}

case "${1:-status}" in
  a) stop_all; start_a ;;
  b) stop_all; start_b ;;
  stop) stop_all ;;
  status) status ;;
  *) echo "usage: $0 {a|b|stop|status}"; exit 1 ;;
esac
status
