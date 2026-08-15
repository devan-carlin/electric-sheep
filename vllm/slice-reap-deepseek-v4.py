#!/usr/bin/env python3
"""Slice REAP-pruned DeepSeek-V4-Flash model from 256 -> 192 experts.

REAP zeroed pruned experts but left tensor shapes at 256.
This script identifies kept experts (by gate weight norms) and slices all tensors.
"""

import os
import sys
import json
import time
import torch
import numpy as np

print("=" * 60)
print("REAP Tensor Slicing — DeepSeek-V4-Flash 256 -> 192 experts")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print()

# ============================================================
# Configuration
# ============================================================
REAP_MODEL_PATH = "/mnt/data/models/DeepSeek-V4-Flash-0731-Abliterated-REAP-192experts"
OUTPUT_DIR = "/mnt/data/models/DeepSeek-V4-Flash-0731-Abliterated-REAP-192experts-sliced"
ORIGINAL_EXPERTS = 256
PRUNED_EXPERTS = 192
SPARSITY = 0.25

print(f"REAP model: {REAP_MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Experts: {ORIGINAL_EXPERTS} -> {PRUNED_EXPERTS}")
print()

# ============================================================
# Load REAP model on CPU
# ============================================================
print("[MODEL] Loading REAP model on CPU...")
start_time = time.time()

# Apply patches before loading
import auto_round.modeling.fused_moe.replace_modules as replace_mod
replace_mod.apply_replacements = lambda model, *a, **k: model

import transformers.models.deepseek_v4.modeling_deepseek_v4 as dv4_module
_orig_attn_forward = dv4_module.DeepseekV4Attention.forward
def patched_attn_forward(self, hidden_states, **kwargs):
    pe = kwargs.get("position_embeddings")
    if isinstance(pe, tuple):
        kwargs["position_embeddings"] = {self.rope_layer_type: pe}
    elif pe is None or (isinstance(pe, dict) and len(pe) == 0):
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        head_dim = self.config.head_dim if hasattr(self.config, 'head_dim') else 512
        partial_rotary_factor = getattr(self.config, 'partial_rotary_factor', 64/512)
        qk_rope_head_dim = int(head_dim * partial_rotary_factor)
        rope_half_dim = qk_rope_head_dim // 2
        cos = torch.ones(batch_size, seq_len, rope_half_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        sin = torch.zeros(batch_size, seq_len, rope_half_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        kwargs["position_embeddings"] = {self.rope_layer_type: (cos, sin)}
    return _orig_attn_forward(self, hidden_states, **kwargs)
dv4_module.DeepseekV4Attention.forward = patched_attn_forward

dv4_module.DeepseekV4TopKRouter.forward = lambda self, hs, input_ids=None: dv4_module.DeepseekV4TopKRouter.forward.fget(f"__orig__")(self, hs) if hasattr(dv4_module.DeepseekV4TopKRouter.forward, 'fget') else _orig_topk_forward(self, hs)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    REAP_MODEL_PATH,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
load_time = time.time() - start_time
print(f"  Model loaded in {load_time:.0f}s ({load_time/60:.1f} min)")

num_layers = model.config.num_hidden_layers
print(f"  Layers: {num_layers}")
print()

# ============================================================
# Determine which experts to keep (same logic as REAP)
# ============================================================
print("[SLICE] Determining kept experts per layer...")
pruning_mask = []
for layer_idx in range(num_layers):
    model_layer = model.model.layers[layer_idx]
    gate = model_layer.mlp.gate
    
    if hasattr(gate, 'weight') and gate.weight is not None:
        w = gate.weight.float()
        norms = torch.norm(w, dim=1).detach().cpu().numpy()
        # Keep experts with largest norms (non-zeroed ones)
        sorted_indices = np.argsort(norms)[::-1]
        keep_indices = set(sorted_indices[:PRUNED_EXPERTS].tolist())
        layer_mask = [idx in keep_indices for idx in range(ORIGINAL_EXPERTS)]
        pruning_mask.append(layer_mask)
    else:
        # Fallback: keep first 192
        layer_mask = [i < PRUNED_EXPERTS for i in range(ORIGINAL_EXPERTS)]
        pruning_mask.append(layer_mask)

# Verify: kept experts should have non-zero norms, pruned should be near zero
for layer_idx in [0, 10, 20, num_layers-1]:
    mask = pruning_mask[layer_idx]
    kept = sum(mask)
    model_layer = model.model.layers[layer_idx]
    gate = model_layer.mlp.gate
    w = gate.weight.float()
    norms = torch.norm(w, dim=1).detach().cpu().numpy()
    kept_norms = norms[[i for i, k in enumerate(mask) if k]]
    pruned_norms = norms[[i for i, k in enumerate(mask) if not k]]
    print(f"  Layer {layer_idx}: {kept} kept (avg_norm={kept_norms.mean():.4f}), "
          f"{ORIGINAL_EXPERTS-kept} pruned (avg_norm={pruned_norms.mean():.6f})")

print()

# ============================================================
# Slice expert tensors per layer
# ============================================================
print("[SLICE] Slicing expert tensors...")
for layer_idx in range(num_layers):
    model_layer = model.model.layers[layer_idx]
    mlp = model_layer.mlp
    experts = mlp.experts
    gate = mlp.gate
    
    keep_mask = pruning_mask[layer_idx]
    keep_indices = [i for i, k in enumerate(keep_mask) if k]
    
    # Slice gate weight: [num_experts, hidden] -> [192, hidden]
    if hasattr(gate, 'weight') and gate.weight is not None:
        gate.weight = torch.nn.Parameter(gate.weight[keep_indices].clone())
    
    # Slice e_score_correction_bias: [num_experts] -> [192]
    if hasattr(gate, 'e_score_correction_bias') and gate.e_score_correction_bias is not None:
        gate.e_score_correction_bias = gate.e_score_correction_bias[keep_indices].clone()
    
    # Slice expert weight tensors (3D: [num_experts, in, out])
    for weight_attr in ['gate_up_proj', 'down_proj']:
        if hasattr(experts, weight_attr) and getattr(experts, weight_attr) is not None:
            w = getattr(experts, weight_attr)
            setattr(experts, weight_attr, w[keep_indices].clone())
    
    # Slice FP8 scale tensors
    for scale_attr in ['gate_up_proj_scale_inv', 'down_proj_scale_inv']:
        if hasattr(experts, scale_attr) and getattr(experts, scale_attr) is not None:
            s = getattr(experts, scale_attr)
            setattr(experts, scale_attr, s[keep_indices].clone())
    
    if (layer_idx + 1) % 10 == 0 or layer_idx == 0:
        print(f"  Sliced layer {layer_idx + 1}/{num_layers}")

print("  All layers sliced.")
print()

# ============================================================
# Update config
# ============================================================
print("[CONFIG] Updating model config...")
model.config.num_local_experts = PRUNED_EXPERTS
model.config.n_routed_experts = PRUNED_EXPERTS
model.config.reap_sliced = True
print(f"  num_local_experts: {PRUNED_EXPERTS}")
print(f"  n_routed_experts: {PRUNED_EXPERTS}")
print()

# ============================================================
# Save sliced model
# ============================================================
print(f"[SAVE] Saving sliced model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
print(f"  Saved.")

# Copy tokenizer files
import shutil
for f in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json']:
    src = os.path.join(REAP_MODEL_PATH, f)
    dst = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# Save metadata
metadata = {
    "reap_pruned": True,
    "reap_sliced": True,
    "original_experts": ORIGINAL_EXPERTS,
    "pruned_experts": PRUNED_EXPERTS,
    "sparsity": SPARSITY,
    "method": "gate_weight_norms",
    "sliced_from": REAP_MODEL_PATH,
}
with open(os.path.join(OUTPUT_DIR, "reap_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

total_time = time.time() - start_time
print()
print("=" * 60)
print(f"DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Output: {OUTPUT_DIR}")
print(f"Experts: {ORIGINAL_EXPERTS} -> {PRUNED_EXPERTS} (sliced, not just zeroed)")
print("=" * 60)
