#!/bin/bash
# env_bench.sh <label> [ENV=VAL ...]
# Restart llama-server (no MTP) with the given env vars, warm up, then time a 64-token gen.
set -u
LABEL="$1"; shift
cd /home/dc/electric-sheep/ninfer-sycl
LOG="/tmp/llama_server_env_${LABEL}.log"

pkill -f "llama-server.*Qwen3.8-27B" 2>/dev/null
sleep 3

env "$@" ./llama-src/build-sycl/bin/llama-server \
  -m models/unsloth/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -dev SYCL0 -sm none -mg 0 -ngl 99 \
  -c 262144 -b 512 -ub 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8081 \
  > "$LOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 90); do
  curl -s --max-time 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q ok && break
  sleep 2
done
if ! curl -s --max-time 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q ok; then
  echo "RESULT $LABEL SERVER_FAILED"
  tail -20 "$LOG"
  kill $SRV_PID 2>/dev/null
  exit 1
fi

# warmup (absorbs graph capture / first-decode cost)
curl -s --max-time 120 http://127.0.0.1:8081/completion -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":16,"temperature":0.0}' > /dev/null

# timed run
RESP=$(curl -s --max-time 240 http://127.0.0.1:8081/completion -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":64,"temperature":0.0}')
echo "$RESP" | python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d.get('timings',{})
print('RESULT $LABEL decode_tps=%.2f prompt_tps=%.2f' % (
    t.get('predicted_per_second',0), t.get('prompt_per_second',0)))
"

kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null
echo "DONE $LABEL"