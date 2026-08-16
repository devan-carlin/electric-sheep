#!/usr/bin/env bash
# Qwen 3.8 27B ARA INT4 (AutoRound W4A16) - official ARA  model, quantized
# Uses GPUs 0,1 (TP=2), 256K context (model's full native window).
# Leaves GPUs 2,3 free for another model.
# NO MTP / speculative decoding (measured slower on XPU: 34.9 vs 52.2 tok/s at TP=4).
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/set-env-01-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

MODEL_PATH="$HOME/electric-sheep/models/Qwen3.8-27B--ara-int4-AutoRound/Qwen3.8-27B--ara-w4g128"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen-256k \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
