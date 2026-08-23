#!/bin/bash
# mtp_bench.sh <n_max>  - restart llama-server with MTP n_max, run a timed 64-token gen, report.
set -u
NMAX="$1"
LOG="/tmp/llama_server_mtp_${NMAX}.log"
cd /home/dc/electric-sheep/ninfer-sycl

# stop any running server
pkill -f "llama-server.*Qwen3.8-27B" 2>/dev/null
sleep 3

./llama-src/build-sycl/bin/llama-server \
  -m models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp \
  --spec-draft-model models/gguf/mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-draft-n-max "$NMAX" \
  --spec-draft-ngl 99 \
  -dev SYCL0 -sm none -mg 0 -ngl 99 \
  -c 262144 -b 512 -ub 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8081 \
  > "$LOG" 2>&1 &
SRV_PID=$!

# wait for health (up to 180s)
for i in $(seq 1 90); do
  if curl -s --max-time 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q ok; then
    break
  fi
  sleep 2
done

if ! curl -s --max-time 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q ok; then
  echo "RESULT n_max=$NMAX SERVER_FAILED"
  tail -20 "$LOG"
  kill $SRV_PID 2>/dev/null
  exit 1
fi

# timed generation
RESP=$(curl -s --max-time 240 http://127.0.0.1:8081/completion -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":64,"temperature":0.0}')

echo "$RESP" | python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d.get('timings',{})
print('RESULT n_max=$NMAX tok/s=%.2f accepted=%s mean_len=%s' % (
    t.get('predicted_per_second',0),
    d.get('draft_acceptance','?'),
    d.get('draft_mean_len','?')))
"

# acceptance from server log
grep -iE "draft acceptance" "$LOG" | tail -1

kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null
echo "DONE n_max=$NMAX"