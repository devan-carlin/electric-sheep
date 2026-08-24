#!/bin/bash
# fastmtp_matched_bench.sh <embedded|sidecar> <n_max>
# TRUE FastMTP test: SAME  Q4_K_P target, two drafts.
#   embedded = built-in MTP (in-file, full-vocab)  -> baseline
#   sidecar  = FastMTP-32K (d2t reduced-vocab)     -> the technique
# Same target, two drafts => clean A/B on acceptance + tok/s.
set -u
MODE="$1"; NMAX="$2"
LOG="/tmp/llama_server_matched_${MODE}_${NMAX}.log"
cd /home/dc/electric-sheep/ninfer-sycl
TARGET=models//Qwen3.8-27B---Aggressive/-Q4_K_P.gguf

pkill -f "llama-server.*Qwen3.8-27B" 2>/dev/null
sleep 3

if [ "$MODE" = "sidecar" ]; then
  DRAFT_ARGS="--spec-draft-model models//Qwen3.8-27B---Aggressive/FastMTP-32K.gguf --spec-draft-p-min 0"
else
  DRAFT_ARGS=""
fi

./llama-src/build-sycl/bin/llama-server \
  -m "$TARGET" \
  --spec-type draft-mtp \
  $DRAFT_ARGS \
  --spec-draft-n-max "$NMAX" \
  --spec-draft-ngl 99 \
  -dev SYCL0 -sm none -mg 0 -ngl 99 \
  -c 262144 -b 512 -ub 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8081 \
  > "$LOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 120); do
  if curl -s --max-time 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q ok; then break; fi
  sleep 2
done

if ! curl -s --max-time 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q ok; then
  echo "RESULT matched $MODE n_max=$NMAX SERVER_FAILED"
  tail -30 "$LOG"; kill $SRV_PID 2>/dev/null; exit 1
fi

# warmup
curl -s --max-time 240 http://127.0.0.1:8081/completion -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":16,"temperature":0.0}' > /dev/null

RESP=$(curl -s --max-time 240 http://127.0.0.1:8081/completion -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":64,"temperature":0.0}')

echo "$RESP" | python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d.get('timings',{})
print('RESULT matched $MODE n_max=$NMAX tok/s=%.2f' % t.get('predicted_per_second',0))
"
grep -iE "draft acceptance|d2t" "$LOG" | tail -3

kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null
echo "DONE matched $MODE n_max=$NMAX"