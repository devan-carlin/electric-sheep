#!/usr/bin/env bash
# Interactive front door for the guide docker stack (vllm/vllm-openai-xpu).
# Wraps run-guide-docker-gpu1.sh with numbered choices + defaults.
#
# Usage:
#   bash start-qwen-128k.sh            # interactive menu
#   bash start-qwen-128k.sh start      # launch with current defaults
#   bash start-qwen-128k.sh logs|health|stop
#
# Defaults: heretic int4, temp 0.5, 128k ctx, id qwen-128k, text-only,
#           GPU0, port 8000, MTP spec on, draft BF16, no API key.
set -euo pipefail
cd "$(dirname "$0")"

HERETIC_DIR="$HOME/electric-sheep/models/heretic-org-Qwen3.8-27B-heretic-ara-int4-gptq"
BASE_DIR="/mnt/data/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
API_KEY_FILE="$HOME/electric-sheep/serve/.api_key"

# --- defaults -------------------------------------------------------------
MODEL="heretic"          # heretic | base
GPU="0"                  # 0..3
PORT="8000"
SERVED_NAME="qwen-128k"
CTX="131072"             # 131072 | 65536 | 32768
MAX_SEQS="64"
MEM_UTIL="0.88"
VISION="0"               # 0 = text only (--language-model-only)
SPEC="1"                 # 1 = MTP x4 speculative decoding
DRAFT_INT4="0"           # 1 = S+M1 INT4 draft overlay (base model only)
TEMP="0.5"
TOP_P="0.8"
TOP_K="20"
MIN_P="0.0"
PRESENCE="1.5"
REPEAT="1.0"
USE_API_KEY="0"

# --- helpers --------------------------------------------------------------
ask() { # ask <prompt> <varname>
  local prompt="$1" var="$2" cur="${!2}" reply
  read -r -p "$prompt [${cur}]: " reply || reply=""
  [ -n "$reply" ] && printf -v "$var" '%s' "$reply"
}
choose() { # choose <prompt> <varname> <default> <label...>
  local prompt="$1" var="$2" def="$3"; shift 3
  echo "$prompt"
  local i=1
  for label in "$@"; do printf "  %d) %s\n" "$i" "$label"; i=$((i+1)); done
  local n=$((i-1)) pick
  read -r -p "Choice [${def}]: " pick || pick=""
  case "${pick:-$def}" in
    ([0-9]*) if [ "$pick" -ge 1 ] && [ "$pick" -le "$n" ]; then
               printf -v "$var" '%s' "${@:$pick}"; else
               printf -v "$var" '%s' "$def"; fi ;;
    (*) printf -v "$var" '%s' "$def" ;;
  esac
}
gpu_label() { # render node + affinity mask for GPU N
  case "$1" in
    0) echo "/dev/dri/renderD130 0" ;;
    1) echo "/dev/dri/renderD131 1" ;;
    2) echo "/dev/dri/renderD128 2" ;;
    3) echo "/dev/dri/renderD129 3" ;;
  esac
}
model_dir() { [ "$MODEL" = "heretic" ] && echo "$HERETIC_DIR" || echo "$BASE_DIR"; }
container_name() { echo "qw38_${SERVED_NAME}_${PORT}"; }

gen_config_json() {
  python3 -c 'import json,sys; print(json.dumps({
    "temperature": float(sys.argv[1]), "top_p": float(sys.argv[2]),
    "top_k": int(sys.argv[3]), "min_p": float(sys.argv[4]),
    "presence_penalty": float(sys.argv[5]), "repetition_penalty": float(sys.argv[6])},
    separators=(",", ":")))' "$TEMP" "$TOP_P" "$TOP_K" "$MIN_P" "$PRESENCE" "$REPEAT"
}

# --- interactive menu -----------------------------------------------------
menu() {
  clear
  echo "=== Qwen 128k launcher (guide docker stack) ==="
  echo
  echo "Current configuration:"
  echo "  Model:        $MODEL"
  echo "  GPU:          $GPU"
  echo "  Port:         $PORT"
  echo "  Served name:  $SERVED_NAME"
  echo "  Context:      $CTX"
  echo "  Max seqs:     $MAX_SEQS"
  echo "  Mem util:     $MEM_UTIL"
  echo "  Vision:       $( [ "$VISION" = 1 ] && echo on || echo off )"
  echo "  Spec decode:  $( [ "$SPEC" = 1 ] && echo "on (MTP x4)" || echo off )"
  echo "  Draft int4:   $( [ "$DRAFT_INT4" = 1 ] && echo on || echo off )"
  echo "  Sampling:     temp=$TEMP top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P presence=$PRESENCE repeat=$REPEAT"
  echo "  API key:      $( [ "$USE_API_KEY" = 1 ] && echo on || echo off )"
  echo
  echo "Options:"
  echo "  1) Model checkpoint"
  echo "  2) GPU"
  echo "  3) Host port"
  echo "  4) Served model name"
  echo "  5) Context length"
  echo "  6) Max concurrent sequences"
  echo "  7) GPU memory utilization"
  echo "  8) Vision support"
  echo "  9) Speculative decoding (MTP)"
  echo "  10) INT4 draft overlay (base model only)"
  echo "  11) Sampling parameters"
  echo "  12) API key"
  echo "  l) Launch"
  echo "  s) Stop container"
  echo "  h) Health check"
  echo "  o) Tail logs"
  echo "  q) Quit"
  while true; do
    read -r -p "> " sel || break
    case "$sel" in
      1) choose "Model checkpoint" MODEL "heretic" \
             "heretic (abliterated, de-refusal)" "base (GPTQ int4 sym G128)" ;;
      2) choose "GPU" GPU "0" "GPU0 (renderD130)" "GPU1 (renderD131)" \
             "GPU2 (renderD128)" "GPU3 (renderD129)" ;;
      3) ask "Host port" PORT ;;
      4) ask "Served model name" SERVED_NAME ;;
      5) choose "Context length" CTX "131072" \
             "131072 (128k)" "65536 (64k)" "32768 (32k)" ;;
      6) ask "Max concurrent sequences" MAX_SEQS ;;
      7) ask "GPU memory utilization" MEM_UTIL ;;
      8) choose "Vision support" VISION "0" \
             "off (text only, --language-model-only)" "on (image inputs, --limit-mm-per-prompt image=4)" ;;
      9) choose "Speculative decoding" SPEC "1" "on (MTP x4)" "off" ;;
      10) choose "INT4 draft overlay (base only)" DRAFT_INT4 "0" "off" "on (S+M1 patches)" ;;
      11) sampling_menu ;;
      12) choose "API key" USE_API_KEY "0" "off" "on (from serve/.api_key)" ;;
      l) return 0 ;;
      s) stop_container; pause ;;
      h) health_check; pause ;;
      o) tail_logs; pause ;;
      q|"") return 1 ;;
      *) echo "unknown option" ;;
    esac
  done
}
sampling_menu() {
  echo "--- Sampling parameters ---"
  ask "Temperature" TEMP
  ask "Top P" TOP_P
  ask "Top K" TOP_K
  ask "Min P" MIN_P
  ask "Presence penalty" PRESENCE
  ask "Repetition penalty" REPEAT
}
pause() { read -r -p "Press Enter to continue..." _; }

# --- actions --------------------------------------------------------------
launch() {
  local rd am
  read -r rd am <<<"$(gpu_label "$GPU")"
  local gen_cfg api_key
  gen_cfg="$(gen_config_json)"
  if [ "$USE_API_KEY" = "1" ]; then
    [ -f "$API_KEY_FILE" ] || { echo "MISSING $API_KEY_FILE"; exit 1; }
    api_key="$(tr -d '\n' < "$API_KEY_FILE")"
  else
    api_key=""
  fi
  echo "Launching: model=$MODEL gpu=$GPU port=$PORT name=$SERVED_NAME"
  echo "  ctx=$CTX seqs=$MAX_SEQS mem=$MEM_UTIL vision=$VISION spec=$SPEC draft_int4=$DRAFT_INT4"
  echo "  sampling: temp=$TEMP top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P presence=$PRESENCE repeat=$REPEAT"
  B70_MODEL_DIR="$(model_dir)" \
  B70_RENDER_NODE="$rd" B70_AFFINITY_MASK="$am" \
  GUIDE_HOST_PORT="$PORT" GUIDE_SERVED_NAME="$SERVED_NAME" \
  GUIDE_CONTAINER_NAME="$(container_name)" \
  GUIDE_MAX_MODEL_LEN="$CTX" GUIDE_MAX_NUM_SEQS="$MAX_SEQS" \
  GUIDE_GPU_MEM_UTIL="$MEM_UTIL" B70_VISION="$VISION" B70_SPEC="$SPEC" \
  DRAFT_INT4="$DRAFT_INT4" \
  GUIDE_GEN_CONFIG="$gen_cfg" GUIDE_API_KEY="$api_key" \
  bash run-guide-docker-gpu1.sh start
  echo
  echo "Container: $(container_name)  ->  http://localhost:$PORT (served as $SERVED_NAME)"
  echo "Health:    bash $0 health"
  echo "Logs:      bash $0 logs"
}
stop_container() {
  GUIDE_CONTAINER_NAME="$(container_name)" bash run-guide-docker-gpu1.sh stop
}
health_check() {
  GUIDE_HOST_PORT="$PORT" bash run-guide-docker-gpu1.sh health
}
tail_logs() {
  GUIDE_CONTAINER_NAME="$(container_name)" bash run-guide-docker-gpu1.sh logs 200
}

# --- entry ----------------------------------------------------------------
cmd="${1:-menu}"
case "$cmd" in
  menu) menu && launch ;;
  start) launch ;;
  logs) tail_logs ;;
  health) health_check ;;
  stop) stop_container ;;
  *) echo "usage: $0 {menu|start|logs|health|stop}"; exit 1 ;;
esac
