#!/usr/bin/env bash
# Gemma 4 31B INT4 - Standard INT4 quantized model
# Uses all 4 GPUs (TP=4)
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/env/set-env-0123-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

MODEL_PATH="$HOME/electric-sheep/models/Intel-gemma-4-31B-it-int4-AutoRound-V2"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name gemma4-31b \
    --host 0.0.0.0 \
    --port 8002 \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len 232144 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching
