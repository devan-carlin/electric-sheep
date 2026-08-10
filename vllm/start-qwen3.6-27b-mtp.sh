#!/usr/bin/env bash
# Qwen 3.6 27B INT4 - MTP enabled with speculative decoding
# Uses all 4 GPUs (TP=4), MTP3 speculative tokens
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/set-env-0123-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

# MTP (Multi-Token Prediction) Configuration
export QWEN36_27B_ENABLE_MTP=1
export NUM_SPECULATIVE_TOKENS=3
export VLLM_XPU_GDN_PROMOTE_ACCEPTED_SPEC_STATE=1
export VLLM_XPU_GDN_NONSPEC_POSTPROCESS_ACCEPTED_STATE=0
export VLLM_XPU_LM_HEAD_INT8=1
export VLLM_XPU_LM_HEAD_INT8_SCALE_DTYPE=bf16
export COMPILATION_CONFIG='{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":8}'

MODEL_PATH="$HOME/electric-sheep/models/Intel-Qwen3.6-27B-int4-AutoRound"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen3.6-27b-mtp \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len 232144 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching \
    --hf-overrides '{"fix_mistral_regex": true}'
