#!/usr/bin/env bash
# =============================================================================
# bench-throughput.sh — Online serving throughput benchmark for the start-all
# llama.cpp instances (qwen @ 8088, gemma @ 8089).
#
# Three workloads, each swept over concurrency 2 and 4 (a single B70 saturates
# well before 8, so higher concurrency is pointless):
#   1. qwen  thinking OFF  (per-request chat_template_kwargs enable_thinking=false;
#                           the server default is ON)
#   2. qwen  thinking ON   (server default)
#   3. gemma thinking ON   (server default)
#
# Per workload: random 1024-token input, 4096-token output, ignore-eos,
# temperature 0, seed 42, 8 prompts. Qwen and Gemma run on separate GPUs, so
# the 'all' target runs them as parallel streams to halve wall-clock time.
# Note: with thinking ON, output tokens include the reasoning trace (capped at
# the server's --reasoning-budget), so 'on' runs are not directly comparable
# to 'off' runs.
#
# Usage:
#   bash bench-throughput.sh            # run all three (qwen+gemma in parallel)
#   bash bench-throughput.sh qwen-off   # run a single workload
#   bash bench-throughput.sh qwen-on
#   bash bench-throughput.sh gemma
#
# Results: bench_results/<name>_c<conc>.json  (+ per-run .log, stream_*.log)
# The vLLM venv is only used for the `vllm bench serve` CLI; the requests go
# to the llama.cpp servers.
# =============================================================================
set -uo pipefail

VENV="$HOME/electric-sheep/vllm/.venv"
LOG_DIR="$HOME/electric-sheep/serve/bench_results"
mkdir -p "$LOG_DIR"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Tokenizer for client-side random prompt generation. The served models are
# GGUFs (the  repos ship no tokenizer files), so point at the base HF
# repos - only the small tokenizer files are fetched and cached.
QWEN_TOK="Qwen/Qwen3.6-35B-A3B"
GEMMA_TOK="google/gemma-4-26B-A4B-it"

# Workload: 4K output for a valid decode-throughput measurement.
# Concurrency is swept over CONC (single B70 saturates well before 8).
# --endpoint must be /v1/chat/completions for the openai-chat backend
# (the CLI default is /v1/completions, which the chat backend rejects).
NUM_PROMPTS=8
CONC=(2 4)

# run_one <name> <port> <model> <tokenizer> <extra_body_json_or_empty>
# Sweeps CONC, writing bench_results/<name>_c<conc>.{json,log} per level.
run_one() {
  local name="$1" port="$2" model="$3" tok="$4" extra="$5" c
  for c in "${CONC[@]}"; do
    echo "===== RUN: ${name}  concurrency=${c}  (port ${port}) ====="
    local cmd=(
      vllm bench serve --backend openai-chat --host 127.0.0.1 --port "$port"
      --model "$model" --tokenizer "$tok"
      --endpoint /v1/chat/completions
      --dataset-name random --random-input-len 1024 --random-output-len 4096
      --num-prompts "$NUM_PROMPTS" --max-concurrency "$c"
      --temperature 0 --ignore-eos --seed 42
      --save-result --save-detailed
      --result-filename "$LOG_DIR/${name}_c${c}.json"
    )
    [[ -n "$extra" ]] && cmd+=(--extra-body "$extra")
    "${cmd[@]}" 2>&1 | tee "$LOG_DIR/${name}_c${c}.log"
  done
}

run_qwen_off() { run_one qwen_off 8088 qwen "$QWEN_TOK" '{"chat_template_kwargs":{"enable_thinking":false}}'; }
run_qwen_on()  { run_one qwen_on  8088 qwen "$QWEN_TOK" ""; }
run_gemma()    { run_one gemma    8089 gemma "$GEMMA_TOK" ""; }

case "${1:-all}" in
  qwen-off) run_qwen_off ;;
  qwen-on)  run_qwen_on ;;
  gemma)    run_gemma ;;
  all)
    # Qwen (GPU 2) and Gemma (GPU 3) run on separate GPUs -> parallel streams.
    ( run_qwen_off; run_qwen_on ) > "$LOG_DIR/stream_qwen.log" 2>&1 &
    qa=$!
    ( run_gemma ) > "$LOG_DIR/stream_gemma.log" 2>&1 &
    gb=$!
    wait "$qa" "$gb"
    echo "===== ALL BENCHMARKS DONE ====="
    ;;
  *) echo "unknown: $1 (use: all|qwen-off|qwen-on|gemma)"; exit 1 ;;
esac