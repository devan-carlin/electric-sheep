#!/usr/bin/env bash
# Qwen 3.8 27B INT4 (AutoRound W4A16) - quantized model
# Uses all 4 GPUs (TP=4), 256K context
# Mirrors the BF16 launch (start-qwen3.8-27b) + --quantization auto-round
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/env/set-env-0123-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

MODEL_PATH="$HOME/electric-sheep/models/Qwen3.8-27B-int4-AutoRound/Qwen3.8-27B-w4g128"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen-256k \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --quantization auto-round
