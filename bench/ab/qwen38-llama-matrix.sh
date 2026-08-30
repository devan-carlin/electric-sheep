#!/usr/bin/env bash
# =============================================================================
# qwen38-llama-matrix.sh
# Throughput + stability matrix for  Qwen3.8-27B Aggressive (Q4_K_P)
# on GPU 2 via llama.cpp SYCL. 256K ctx, np=1, thinking OFF, built-in MTP.
#
# Matrix:
#   q4_0/q4_0  MTP 0,1,2,3   (MTP sweep; built-in head clamps depth to 1)
#   q4_0K/q5_0V, q5_0K/q4_0V, q5_0/q5_0  at MTP 1  (mixed + same quant)
#   q4_0/q4_0 MTP 1 without mmproj       (control)
#
# Per run: start server -> warm-up -> 500-token sustained decode -> record
# decode t/s, draft acceptance, reasoning/content lengths, alive/crashed.
# =============================================================================
set -uo pipefail

source "$HOME/electric-sheep/llama/set-env.sh" 2 2>/dev/null
B="$HOME/electric-sheep/llama/llama.cpp/build/bin"
D="$HOME/electric-sheep/models/-Qwen3.8-27B---Aggressive"
M="$D/-Q4_K_P.gguf"
MM="$D/mmproj-Qwen3.8-27B---Aggressive-BF16.gguf"
TPL="$HOME/electric-sheep/models/devan-carlin-Qwen3.8-27B--ara-int4-AutoRound/Qwen3.8-27B--ara-w4g128/chat_template.jinja"
PORT=8091
LOG="$HOME/electric-sheep/serve/logs/llama_${PORT}.log"
RESULTS="$HOME/electric-sheep/bench/ab/qwen38_matrix_results.csv"
CTX=262144

# Chat template: embedded in GGUF, or fall back to the vLLM-dir jinja.
# Probe the header with grep -c (NOT grep -q): under pipefail, grep -q exits
# early and SIGPIPEs strings (exit 141), inverting the test. NOTE: the 8-run
# matrix (2026-08-25) hit this bug and ran on the external template; the
# throughput numbers are still valid (decode speed is content-independent).
if head -c 16M "$M" | grep -c "enable_thinking" >/dev/null; then
  TPL_ARGS=(--jinja)
  echo "chat template: embedded in GGUF"
else
  TPL_ARGS=(--jinja --chat-template "$TPL")
  echo "chat template: external ($TPL)"
fi

stop_srv() {
  pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3
  pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null; sleep 2
}

run() { # label ctk ctv mtp mmproj(1/0)
  local label="$1" ctk="$2" ctv="$3" mtp="$4" mm="$5"
  stop_srv
  local spec_args=()
  if [ "$mtp" -gt 0 ]; then
    spec_args=(--spec-type draft-mtp --spec-draft-n-max "$mtp")
  fi
  local mm_args=()
  [ "$mm" = "1" ] && mm_args=(--mmproj "$MM")

  nohup "$B/llama-server" \
    -m "$M" ${mm_args[@]+"${mm_args[@]}"} \
    ${spec_args[@]+"${spec_args[@]}"} \
    -ctk "$ctk" -ctv "$ctv" \
    -ngl 99 -fa on \
    -c "$CTX" -np 1 -b 2048 -ub 512 \
    "${TPL_ARGS[@]}" \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --alias qwen38-test \
    --host 0.0.0.0 --port "$PORT" \
    > "$LOG" 2>&1 &
  local pid=$!
  local ready=0
  for i in $(seq 1 48); do
    grep -q "listening on" "$LOG" 2>/dev/null && { ready=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  if [ "$ready" != "1" ]; then
    echo "$label,START_FAIL,,,," >> "$RESULTS"
    echo "[$label] START FAILED (pid $pid)"
    tail -8 "$LOG"
    stop_srv
    return
  fi
  local mtp_line
  mtp_line=$(grep -E "MTP draft context|loading draft model" "$LOG" | head -1)
  echo "[$label] up. ${mtp_line:-no MTP ctx line}"

  # warm-up
  curl -s --max-time 60 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen38-test","messages":[{"role":"user","content":"Say ONLINE in exactly one word."}],"max_tokens":20,"temperature":0}' \
    > /dev/null 2>&1

  # sustained decode: 500 tokens
  curl -s --max-time 600 http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen38-test","messages":[{"role":"user","content":"Write a detailed technical essay about how gated delta networks work in hybrid LLM architectures, covering the state update rule, the short convolution, and why linear attention is memory efficient. Keep going with concrete detail."}],"max_tokens":500,"temperature":0.7}' \
    | python3 -c "import json,sys
try:
  d=json.load(sys.stdin)
  m=d['choices'][0]['message']
  print('REASONING_LEN', len((m.get('reasoning_content') or '').strip()))
  print('CONTENT_LEN', len((m.get('content') or '').strip()))
except Exception as e:
  print('PARSE_FAIL', e)" > /tmp/q38_out.txt 2>&1

  local alive="ALIVE"
  pgrep -f "llama-server.*--port $PORT" >/dev/null || alive="CRASHED"
  local tps acc rlen clen
  tps=$(grep "eval time" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+ tokens per second" | grep -oE "^[0-9]+\.[0-9]+" | tail -1)
  acc=$(grep "draft acceptance" "$LOG" | tail -1 | grep -oE "draft acceptance = [0-9.]+" | grep -oE "[0-9.]+$")
  rlen=$(grep REASONING_LEN /tmp/q38_out.txt 2>/dev/null | awk '{print $2}')
  clen=$(grep CONTENT_LEN /tmp/q38_out.txt 2>/dev/null | awk '{print $2}')
  echo "$label,${tps:-},${acc:-},${rlen:-},${clen:-},$alive" >> "$RESULTS"
  echo "[$label] tps=${tps:-?} acc=${acc:-none} reasoning=${rlen:-?} content=${clen:-?} $alive"
  stop_srv
}

: > "$RESULTS"
echo "label,decode_tps,draft_acc,reasoning_len,content_len,status" > "$RESULTS"

run "q4_0_mtp0"     q4_0 q4_0 0 1
run "q4_0_mtp1"     q4_0 q4_0 1 1
run "q4_0_mtp2"     q4_0 q4_0 2 1
run "q4_0_mtp3"     q4_0 q4_0 3 1
run "q4k_q5v_mtp1"  q4_0 q5_0 1 1
run "q5k_q4v_mtp1"  q5_0 q4_0 1 1
run "q5_0_mtp1"     q5_0 q5_0 1 1
run "q4_0_mtp1_nomm" q4_0 q4_0 1 0

echo
echo "=== RESULTS ==="
column -s, -t < "$RESULTS"