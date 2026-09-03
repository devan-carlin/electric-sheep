#!/usr/bin/env bash
# bench-flashnext.sh <split-mode>  — start flash-next with a given split mode,
# run a fixed 400-token generation, report t/s, stop.
set -uo pipefail
SM="${1:-layer}"
GPUS="${FLASHNEXT_GPUS:-0,1,2,3}"
PORT="${FLASHNEXT_PORT:-8090}"
B="$HOME/electric-sheep/llama/llama.cpp/build/bin"
D="$HOME/electric-sheep/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL"
M="$D/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
LOG="/tmp/fn_bench_${SM}.log"

# stop any existing
pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 3
pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null

source "$HOME/electric-sheep/llama/set-env.sh" "$GPUS" >/dev/null 2>&1
nohup "$B/llama-server" \
  -m "$M" -sm "$SM" \
  -ngl 99 -fa on \
  -c 262144 -np 1 -b 2048 -ub 512 \
  --jinja --reasoning on --alias flash-next \
  --host 0.0.0.0 --port "$PORT" \
  > "$LOG" 2>&1 &
PID=$!
echo "[$SM] starting pid $PID ..."
for i in $(seq 1 90); do
  grep -q "listening on" "$LOG" 2>/dev/null && { echo "[$SM] ready ~$((i*5))s"; break; }
  grep -qE "error|Error|abort|exited" "$LOG" 2>/dev/null && { echo "[$SM] LOAD ERROR"; tail -15 "$LOG"; exit 1; }
  sleep 5
done

# warm-up (loads graphs / page cache)
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"hi"}],"max_tokens":20,"temperature":1.0,"top_p":0.95,"top_k":20}' -o /dev/null

# timed 400-token generation
MARK=$(wc -l < "$LOG")
START=$(date +%s.%N)
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"Write a detailed 300-word explanation of how mixture-of-experts routing works, including load balancing and expert capacity."}],"max_tokens":400,"temperature":1.0,"top_p":0.95,"top_k":20}' -o /tmp/fn_bench_out.json
END=$(date +%s.%N)
WALL=$(echo "$END $START" | awk '{print $1-$2}')
TOK=$(python3 -c "import json;print(json.load(open('/tmp/fn_bench_out.json'))['usage']['completion_tokens'])" 2>/dev/null || echo 0)
echo "[$SM] wall=${WALL}s tokens=${TOK} => $(echo "$TOK $WALL" | awk '{printf "%.2f", $1/$2}') t/s (wall)"
echo "[$SM] log timing:"
tail -n +$((MARK+1)) "$LOG" | grep -E "prompt eval|eval time" | tail -2

# stop
kill $PID 2>/dev/null; sleep 3; pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null
echo "[$SM] done"