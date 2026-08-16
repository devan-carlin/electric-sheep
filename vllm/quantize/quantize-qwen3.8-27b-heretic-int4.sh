#!/usr/bin/env bash
# =============================================================================
# Quantize Qwen3.8-27B-heretic (BF16, OUR directional-ablation model) -> INT4
# AutoRound (W4A16, group_size 128, sym).
#
# Recipe is IDENTICAL to the proven base-model run
# (quantize-qwen3.8-27b-int4.sh -> Qwen3.8-27B-int4-AutoRound), same qwen3_5
# hybrid GDN architecture:
#   - bits=4, group_size=128, sym, format=auto_round:auto_gptq (vLLM-loadable)
#   - GDN gate projections kept 16-bit (numerically sensitive, tiny [48,5120]):
#       model.language_model.layers.N.linear_attn.in_proj_a  (48 layers)
#       model.language_model.layers.N.linear_attn.in_proj_b  (48 layers)
#   - mtp.fc kept 16-bit
#   - lm_head NOT quantized
#
# Hardware: 4x Arc Pro B70 (34.2GB each). Model is 51GB BF16, so
# --device_map auto distributes it across all 4 XPU devices.
#
# PREREQUISITE: No other vLLM server running (needs all 4 GPUs free).
#   pkill -9 -f "vllm.entrypoints.openai.api_server"; pkill -9 -f "VLLM::"; fuser -k 8000/tcp
#
# Estimated runtime: 1-3 hours (calibration + per-layer GPTQ tuning).
# =============================================================================
set -euo pipefail

cd /home/dc/electric-sheep/vllm
source .venv/bin/activate
source env/set-env-0123-gpu.sh   # ZE_AFFINITY_MASK=0,1,2,3 (all 4 GPUs visible)

MODEL=/home/dc/electric-sheep/models/Qwen3.8-27B-heretic
OUT=/home/dc/electric-sheep/models/Qwen3.8-27B-heretic-int4-AutoRound
LOG=/tmp/autoround-qwen3.8-27b-heretic.log

# GDN gate projections + mtp.fc excluded from quantization (16-bit).
# ignore_layers uses substring matching, so these patterns cover all 48 layers.
IGNORE_LAYERS="linear_attn.in_proj_a,linear_attn.in_proj_b,mtp.fc"

auto-round quantize \
  --model_name "$MODEL" \
  --bits 4 \
  --group_size 128 \
  --format auto_round:auto_gptq \
  --device_map auto \
  --dataset NeelNanda/pile-10k \
  --nsamples 128 \
  --seqlen 2048 \
  --batch_size 8 \
  --ignore_layers "$IGNORE_LAYERS" \
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
print('  excluded layers:', len(ec), 'entries')
"
