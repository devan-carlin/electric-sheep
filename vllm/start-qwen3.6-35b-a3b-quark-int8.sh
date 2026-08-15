#!/usr/bin/env bash
# Qwen 3.6 35B-A3B Quark W8A8 INT4 - INT8 MoE model (experimental)
# Uses all 4 GPUs (TP=4), 192K context, requires Quark quantization support
source "/home/dc/electric-sheep/vllm/.venv/bin/activate"
source "/home/dc/electric-sheep/vllm/set-env-0123-gpu.sh"

# Compile cache root for warm starts (faster cold launches)
export VLLM_CACHE_ROOT="$HOME/.cache/vllm"

# Graph fallback flags from 35B Quark INT8 lane (prevents decode corruption)
export VLLM_XPU_GDN_NATIVE_FALLBACK=prefill
export VLLM_XPU_GDN_PREFILL_RECURRENT_FALLBACK=1
export VLLM_XPU_DISABLE_PREFILL_CUDAGRAPH_REPLAY=1
export VLLM_XPU_GREEDY_SAMPLE_TOPK_FALLBACK=1

MODEL_PATH="$HOME/electric-sheep/models/nameistoken-Qwen3.6-35B-A3B-Quark-W8A8-INT8"

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
    --quantization quark \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --no-enable-prefix-caching
