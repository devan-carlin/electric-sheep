#!/usr/bin/env bash
# Qwen 3.8 27B BF16 (unquantized) - Primary model
# Uses all 4 GPUs (TP=4), 256K context
# Config matches the validated long-running instance (stable for hours):
#   fp8 KV, 0.85 util, block 32, batched 16384, 2 seqs, text-only
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/env/set-env-0123-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

MODEL_PATH="$HOME/electric-sheep/models/Qwen3.8-27B"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen-256k \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len 262144 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8 \
    --block-size 32 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --generation-config vllm \
    --language-model-only \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching
