#!/usr/bin/env bash
# Qwen 3.6 27B INT4 - Primary model with full context
# Uses all 4 GPUs (TP=4)
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/set-env-0123-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

MODEL_PATH="$HOME/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b \
    --host 0.0.0.0 \
    --port 8030 \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len 232144 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 65536 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching \
    --generation-config vllm
