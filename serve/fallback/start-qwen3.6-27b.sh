#!/usr/bin/env bash
# Qwen 3.6 27B INT4 - Primary model
# Uses all 4 GPUs (TP=4), 192K context
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/env/set-env-0123-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

MODEL_PATH="$HOME/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen-192k \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len 196608 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching