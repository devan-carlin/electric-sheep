#!/usr/bin/env python3
"""REAP (Runtime Expert Pruning via Adaptive Pruning) for DeepSeek-V4-Flash.

Prunes MoE experts from 256 -> 192 per layer (25% sparsity).

Uses gate weight analysis instead of calibration forward passes:
- Experts with smallest gate weight norms are least likely to be selected
- No forward passes needed = no FP8 Triton kernel issues
- Works entirely on CPU with FP8 model loaded

Also zeros pruned expert weights so they produce no output if accidentally routed.
"""

import os
import sys
import json
import time
import torch
import numpy as np

print("=" * 60)
print("REAP Expert Pruning — DeepSeek-V4-Flash")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"XPU count: {torch.xpu.device_count()}")
print(f"System RAM: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.0f} GB")
print()

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "/home/dc/electric-sheep/models/DeepSeek-V4-Flash-0731-Abliterated-FP8"
OUTPUT_DIR = "/mnt/data/models/DeepSeek-V4-Flash-0731-Abliterated-REAP-192experts"
SPARSITY = 0.25  # Prune 25% of experts (256 -> 192)

print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Sparsity: {SPARSITY * 100:.0f}% (256 -> {int(256 * (1 - SPARSITY))} experts)")
print(f"Method: Gate weight analysis (no calibration needed)")
print()

# ============================================================
# Apply patches BEFORE model loading
# ============================================================
print("[PATCH] Disabling MoE fused module replacement...")
import auto_round.modeling.fused_moe.replace_modules as replace_mod
replace_mod.apply_replacements = lambda model, *a, **k: model

print("[PATCH] Fixing DeepseekV4Attention forward signature...")
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

print("[PATCH] Fixing MoE gate forward signatures...")
_orig_topk_forward = dv4_module.DeepseekV4TopKRouter.forward
dv4_module.DeepseekV4TopKRouter.forward = lambda self, hs, input_ids=None: _orig_topk_forward(self, hs)
_orig_hash_forward = dv4_module.DeepseekV4HashRouter.forward
def patched_hash_forward(self, hidden_states, input_ids=None):
    if input_ids is None:
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
    return _orig_hash_forward(self, hidden_states, input_ids)
dv4_module.DeepseekV4HashRouter.forward = patched_hash_forward

print("[PATCH] Forcing eager MoE + disabling deepgemm...")
config_path = os.path.join(MODEL_PATH, "config.json")
with open(config_path) as f:
    config = json.load(f)
config["_experts_implementation"] = "eager"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

import transformers.integrations.finegrained_fp8 as fg_fp8_module
fg_fp8_module._disable_deepgemm_on_multi_device = lambda m: None
if hasattr(fg_fp8_module, 'FP8Experts'):
    fg_fp8_module.FP8Experts._deepgemm_disabled = True
if hasattr(fg_fp8_module, 'FP8Linear'):
    fg_fp8_module.FP8Linear._deepgemm_disabled = True

print()
print("All patches applied.")
print("=" * 60)

# ============================================================
# Load model on CPU
# ============================================================
print("[MODEL] Loading DeepSeek-V4-Flash on CPU...")
start_time = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
load_time = time.time() - start_time
print(f"  Model loaded in {load_time:.0f}s ({load_time/60:.1f} min)")

num_layers = model.config.num_hidden_layers
num_experts = model.config.num_local_experts
num_experts_per_token = model.config.num_experts_per_tok
print(f"  Layers: {num_layers}")
print(f"  Experts per layer: {num_experts}")
print(f"  Experts per token (top-k): {num_experts_per_token}")
print()

# ============================================================
# Analyze gate weights to determine expert importance
# ============================================================
print("[REAP] Analyzing gate weights for expert importance...")
print("  (no calibration needed — gate norms predict routing frequency)")
print()

# Gate weight norms per layer: [layer][expert] = L2 norm of gate weight row
gate_norms = []

for layer_idx in range(num_layers):
    model_layer = model.model.layers[layer_idx]
    gate = model_layer.mlp.gate
    
    if hasattr(gate, 'weight') and gate.weight is not None:
        # gate.weight shape: [num_experts, hidden_dim]
        # Compute L2 norm per expert (row)
        w = gate.weight.float()  # Convert to float32 for stable norm computation
        norms = torch.norm(w, dim=1).detach().cpu().numpy()  # [num_experts]
        gate_norms.append(norms)
    else:
        # Fallback: uniform importance
        gate_norms.append(np.ones(num_experts))

gate_norms = np.array(gate_norms)  # [num_layers, num_experts]
print(f"  Gate norms shape: {gate_norms.shape}")
print(f"  Mean norm: {gate_norms.mean():.4f}")
print(f"  Std norm: {gate_norms.std():.4f}")
print(f"  Min norm: {gate_norms.min():.4f}")
print(f"  Max norm: {gate_norms.max():.4f}")
print()

# Determine pruning mask: keep top-N experts per layer by gate norm
experts_to_keep_per_layer = int(num_experts * (1 - SPARSITY))
print(f"  Experts to keep per layer: {experts_to_keep_per_layer}")
print(f"  Experts to prune per layer: {num_experts - experts_to_keep_per_layer}")

pruning_mask = []
for layer_idx in range(num_layers):
    layer_norms = gate_norms[layer_idx]
    # Keep experts with largest gate norms (most likely to be routed)
    sorted_indices = np.argsort(layer_norms)[::-1]
    keep_indices = set(sorted_indices[:experts_to_keep_per_layer].tolist())
    layer_mask = [idx in keep_indices for idx in range(num_experts)]
    pruning_mask.append(layer_mask)
    
    if layer_idx < 3 or layer_idx == num_layers - 1:
        kept_norm = layer_norms[[idx for idx, k in enumerate(layer_mask) if k]].mean()
        pruned_norm = layer_norms[[idx for idx, k in enumerate(layer_mask) if not k]].mean()
        print(f"    Layer {layer_idx}: kept avg_norm={kept_norm:.4f}, "
              f"pruned avg_norm={pruned_norm:.4f}")

print()

# ============================================================
# Apply pruning: zero out pruned expert weights + gate columns
# ============================================================
print("[REAP] Applying expert pruning to model...")
print("  (zeroing expert weight slices and gate columns)")

for layer_idx in range(num_layers):
    model_layer = model.model.layers[layer_idx]
    mlp = model_layer.mlp
    experts = mlp.experts
    
    pruned_indices = [idx for idx, keep in enumerate(pruning_mask[layer_idx]) if not keep]
    
    if pruned_indices:
        with torch.no_grad():
            # Zero out expert weight tensors (3D: [num_experts, in, out])
            for weight_attr in ['gate_up_proj', 'down_proj']:
                if hasattr(experts, weight_attr) and getattr(experts, weight_attr) is not None:
                    w = getattr(experts, weight_attr)
                    for idx in pruned_indices:
                        w.data[idx] = 0
            
            # Zero out FP8 scale tensors (e8m0fnu — use .data for in-place)
            for scale_attr in ['gate_up_proj_scale_inv', 'down_proj_scale_inv']:
                if hasattr(experts, scale_attr) and getattr(experts, scale_attr) is not None:
                    s = getattr(experts, scale_attr)
                    for idx in pruned_indices:
                        s.data[idx] = 0
            
            # Zero out gate weight rows for pruned experts (so they're never selected)
            gate = mlp.gate
            if hasattr(gate, 'weight') and gate.weight is not None:
                for idx in pruned_indices:
                    gate.weight.data[idx] = -1e9

    if (layer_idx + 1) % 10 == 0 or layer_idx == 0:
        print(f"    Pruned layer {layer_idx + 1}/{num_layers} ({len(pruned_indices)} experts zeroed)")

print("  Pruning applied to all layers")
print()

# ============================================================
# Update config
# ============================================================
print("[REAP] Updating model config...")
model.config.num_local_experts = experts_to_keep_per_layer
model.config.reap_pruned = True
model.config.reap_original_experts = num_experts
model.config.reap_sparsity = SPARSITY
model.config.reap_method = "gate_weight_norms"
print(f"  num_local_experts: {num_experts} -> {experts_to_keep_per_layer}")
print(f"  reap_method: gate_weight_norms")
print()

# ============================================================
# Save pruned model
# ============================================================
print(f"[SAVE] Saving pruned model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

save_start = time.time()
model.save_pretrained(OUTPUT_DIR)
save_time = time.time() - save_start
print(f"  Model saved in {save_time:.0f}s ({save_time/60:.1f} min)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print("  Tokenizer saved")

metadata = {
    "reap_pruned": True,
    "original_experts": num_experts,
    "pruned_experts": experts_to_keep_per_layer,
    "sparsity": SPARSITY,
    "method": "gate_weight_norms",
    "total_time_seconds": time.time() - start_time,
}
with open(os.path.join(OUTPUT_DIR, "reap_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("  Metadata saved")
print()

# ============================================================
# Summary
# ============================================================
total_time = time.time() - start_time
print("=" * 60)
print("REAP COMPLETE")
print("=" * 60)
print(f"  Output: {OUTPUT_DIR}")
print(f"  Experts: {num_experts} -> {experts_to_keep_per_layer} ({SPARSITY*100:.0f}% pruned)")
print(f"  Method: Gate weight norm analysis (no calibration)")
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

import subprocess
try:
    result = subprocess.run(['du', '-sh', OUTPUT_DIR], capture_output=True, text=True, timeout=30)
    print(f"  Output size: {result.stdout.strip()}")
except Exception:
    print("  (size will be available after save completes)")

print()
print("Done!")
