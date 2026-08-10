#!/usr/bin/env bash
# Qwen 3.6 35B-A3B INT4 - Mixture of Experts model
# Uses 2 GPUs (TP=2), requires patching
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/set-env-01-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

MODEL_PATH="$HOME/electric-sheep/models/Intel-Qwen3.6-35B-A3B-int4-mixed-AutoRound"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-35b-a3b \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len 232144 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching \
    --hf-overrides '{"fix_mistral_regex": true}'
