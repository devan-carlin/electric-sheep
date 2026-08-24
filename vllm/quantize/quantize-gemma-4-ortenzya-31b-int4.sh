#!/usr/bin/env bash
# =============================================================================
# Quantize gemma-4-ortenzya-31b- (BF16, ~59GB) -> INT4 AutoRound
# (W4A16, group_size 128, sym). vLLM-loadable on a single 32GB B70.
#
# Base recipe mirrors the proven Qwen int4 runs (quantize-qwen3.8-27b-*-int4.sh):
#   - bits=4, group_size=128, sym, format=auto_round:auto_gptq (vLLM-loadable)
#   - --nsamples 256, --iters 10, --enable_quanted_input, --scale_dtype fp32
#   - lm_head NOT quantized
#
# Gemma4 is a standard transformer + MoE (NO GDN linear-attention), so — unlike
# the Qwen runs — no text layers need to be excluded. The MoE router and PLE
# projections quantize cleanly (the Intel-gemma-4-31B-it-int4 reference kept
# only vision_tower + multi_modal_projector at 16-bit and works in vLLM).
#
# Vision + audio towers stay 16-bit automatically (quant_nontext_module is off
# by default), matching the Intel reference extra_config.
#
# Hardware: 4x Arc Pro B70 (32GB each, 128GB total). Model is ~59GB BF16, so
# --device_map auto distributes it across all 4 XPU devices.
#
# PREREQUISITE: No other vLLM/ComfyUI server running (needs all 4 GPUs free).
#   pkill -9 -f "vllm.entrypoints.openai.api_server"; pkill -9 -f "VLLM::"
#   pkill -9 -f "comfyui/venv/bin/python main.py"
#
# Estimated runtime: several hours (overnight is safe).
# =============================================================================
set -euo pipefail

cd /home/dc/electric-sheep/vllm
source .venv/bin/activate
source env/set-env-0123-gpu.sh   # ZE_AFFINITY_MASK=0,1,2,3 (all 4 GPUs visible)

MODEL=/home/dc/electric-sheep/models/gemma-4-ortenzya-31b-
OUT=/home/dc/electric-sheep/models/gemma-4-ortenzya-31b--int4-AutoRound
LOG=/tmp/autoround-gemma-4-ortenzya-31b.log

auto-round quantize \
  --model_name "$MODEL" \
  --bits 4 \
  --group_size 128 \
  --format auto_round:auto_gptq \
  --device_map auto \
  --dataset NeelNanda/pile-10k \
  --nsamples 256 \
  --seqlen 2048 \
  --batch_size 8 \
  --iters 10 \
  --enable_quanted_input \
  --scale_dtype fp32 \
  --no-quant_lm_head \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"

echo ""
echo "=== Quantization complete ==="
echo "Output: $OUT"
echo "Verify config:"
python3 -c "
import json
with open('$OUT/config.json') as f:
    qc = json.load(f).get('quantization_config', {})
print('  bits:', qc.get('bits'), '| group_size:', qc.get('group_size'),
      '| sym:', qc.get('sym'), '| format:', qc.get('packing_format'))
ec = qc.get('extra_config', {})
print('  excluded (16-bit) entries:', len(ec))
"
echo "Output size:"
du -sh "$OUT"
