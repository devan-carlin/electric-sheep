#!/usr/bin/env bash
# Gemma 4 26B-A4B INT4 - Mixture of Experts model
# Uses 2 GPUs (TP=2)
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/set-env-01-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

MODEL_PATH="$HOME/electric-sheep/models/Intel-gemma-4-26B-A4B-it-int4-AutoRound"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name gemma4-26b-a4b \
    --host 0.0.0.0 \
    --port 8003 \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len 232144 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching
