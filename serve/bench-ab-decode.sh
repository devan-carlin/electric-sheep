#!/usr/bin/env bash
# =============================================================================
# bench-ab-decode.sh — compare decode throughput between two vLLM endpoints
#
# Fires an identical workload (fixed ~512-token prompt + N generated tokens,
# greedy) at each given port and reports:
#   - completion_tokens / wall  (end-to-end effective t/s incl. prefill)
#   - vLLM's own peak "generation throughput" from its engine log (pure decode)
#
# Usage:
#   bash bench-ab-decode.sh                 # benches 8000 and 8001
#   bash bench-ab-decode.sh 8000 8001 8088  # any set of ports
#
# Env overrides: N_GEN (default 300), ROUNDS (default 3), PROMPT_WORDS (400)
# =============================================================================
set -uo pipefail

PORTS=("$@"); [[ ${#PORTS[@]} -eq 0 ]] && PORTS=(8000 8001)
N_GEN="${N_GEN:-300}"
ROUNDS="${ROUNDS:-3}"
WORDS="${PROMPT_WORDS:-400}"

LOGDIR="$HOME/electric-sheep/serve/logs"

# Deterministic ~N-word prompt (same for every port => fair A/B).
PROMPT="$(python3 - "$WORDS" <<'PY'
import sys
words=int(sys.argv[1])
base=("The quick brown fox jumps over the lazy dog near the river bank while the "
      "sun sets slowly behind the distant mountains and the evening light fades "
      "across the quiet valley floor below the winding path home.")
parts=[]; n=0
while n < words:
    parts.append(base); n += len(base.split())
print(" ".join(parts))
PY
)"
PROMPT_JSON="$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT")"

echo "workload: ~${WORDS}-word prompt + ${N_GEN} FORCED tokens (greedy, ignore_eos), ${ROUNDS} rounds"
echo
for port in "${PORTS[@]}"; do
  log="$LOGDIR/vllm_${port}.log"
  # warmup (primes JIT/graphs/prefix-cache so timed rounds are steady-state)
  curl -s -m 60 "http://127.0.0.1:${port}/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen-256k","prompt":"Hello","max_tokens":8,"temperature":0}' >/dev/null 2>&1
  echo "--- port $port ---"
  effs=""; peaks=""
  for r in $(seq 1 "$ROUNDS"); do
    mark=$(wc -l < "$log" 2>/dev/null || echo 0)
    t0=$(date +%s.%N)
    body=$(curl -s -m 300 "http://127.0.0.1:${port}/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"qwen-256k\",\"prompt\":${PROMPT_JSON},\"max_tokens\":${N_GEN},\"temperature\":0,\"ignore_eos\":true}" 2>/dev/null)
    t1=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
    toks=$(printf '%s' "$body" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["usage"]["completion_tokens"])' 2>/dev/null || echo 0)
    eff=$(awk "BEGIN{printf \"%.1f\", $toks/$wall}")
    peak=$(tail -n +"$((mark+1))" "$log" 2>/dev/null | grep -oE "generation throughput: [0-9.]+" | awk '{print $3}' | sort -rn | head -1)
    effs+="$eff "; peaks+="${peak:-na} "
    printf "  round %d: %4s tok  wall=%6ss  eff=%6s t/s  vllm_peak=%s t/s\n" "$r" "$toks" "$wall" "$eff" "${peak:-na}"
  done
  best=$(echo $effs | tr ' ' '\n' | sort -rn | head -1)
  pbest=$(echo $peaks | tr ' ' '\n' | grep -E '^[0-9]' | sort -rn | head -1)
  echo "  => best effective: ${best} t/s | best vLLM-reported decode: ${pbest:-na} t/s"
  echo
done
echo "(eff = completion_tokens/wall incl. prefill; vllm_peak = highest 10s-window"
echo " 'Avg generation throughput' the engine logged during the rounds.)"
