#!/usr/bin/env python3
"""Slice REAP-pruned DeepSeek-V4-Flash model from 256 -> 192 experts.

Direct safetensors approach - no model loading needed.
Reads tensors, determines kept experts via gate norms, slices, writes new safetensors.
"""

import os
import json
import time
import torch
import numpy as np
from safetensors.torch import load_file, save_file

print("=" * 60)
print("REAP Tensor Slicing — DeepSeek-V4-Flash 256 -> 192 (direct)")
print("=" * 60)

# ============================================================
# Configuration
# ============================================================
REAP_DIR = "/mnt/data/models/DeepSeek-V4-Flash-0731-Abliterated-REAP-192experts"
OUTPUT_DIR = "/mnt/data/models/DeepSeek-V4-Flash-0731-Abliterated-REAP-192experts-sliced"
ORIGINAL_EXPERTS = 256
PRUNED_EXPERTS = 192

print(f"REAP model: {REAP_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

# ============================================================
# Load index
# ============================================================
with open(os.path.join(REAP_DIR, "model.safetensors.index.json")) as f:
    index = json.load(f)

# Load config
with open(os.path.join(REAP_DIR, "config.json")) as f:
    config = json.load(f)

num_layers = config["num_hidden_layers"]
print(f"Layers: {num_layers}")
print(f"Shards: {len(set(index['weight_map'][t] for t in index['weight_map']))}")
print()

# ============================================================
# Load all shards into memory
# ============================================================
print("[LOAD] Loading safetensors shards...")
start_time = time.time()
shards = {}
shard_files = sorted(set(index['weight_map'].values()))
for sf in shard_files:
    print(f"  Loading {sf}...")
    shards[sf] = load_file(os.path.join(REAP_DIR, sf))
load_time = time.time() - start_time
print(f"  Loaded {len(shard_files)} shards in {load_time:.0f}s")
print()

# ============================================================
# Collect all tensors into one dict
# ============================================================
print("[MERGE] Collecting all tensors...")
all_tensors = {}
for sf, tensors in shards.items():
    for name, tensor in tensors.items():
        all_tensors[name] = tensor
print(f"  Total tensors: {len(all_tensors)}")
print()

# ============================================================
# Determine kept experts per layer from gate weight norms
# ============================================================
print("[ANALYZE] Determining kept experts per layer...")
keep_indices_per_layer = {}

for layer_idx in range(num_layers):
    gate_name = f"model.layers.{layer_idx}.mlp.gate.weight"
    if gate_name not in all_tensors:
        gate_name = f"model.layers.{layer_idx}.ffn.gate.weight"
    
    if gate_name in all_tensors:
        gate_w = all_tensors[gate_name].float()  # [256, hidden]
        norms = torch.norm(gate_w, dim=1).cpu().numpy()
        # Keep experts with largest norms (non-zeroed)
        sorted_idx = np.argsort(norms)[::-1]
        keep_idx = sorted(sorted_idx[:PRUNED_EXPERTS].tolist())
        keep_indices_per_layer[layer_idx] = keep_idx
        
        kept_norms = norms[keep_idx]
        pruned_norms = norms[[i for i in range(ORIGINAL_EXPERTS) if i not in set(keep_idx)]]
        if layer_idx < 3 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx}: kept={len(keep_idx)}, avg_kept_norm={kept_norms.mean():.4f}, avg_pruned_norm={pruned_norms.mean():.6f}")
    else:
        print(f"  WARNING: Layer {layer_idx} gate not found, keeping first {PRUNED_EXPERTS}")
        keep_indices_per_layer[layer_idx] = list(range(PRUNED_EXPERTS))

print()

# ============================================================
# Slice tensors
# ============================================================
print("[SLICE] Slicing tensors...")
sliced_count = 0

for layer_idx in range(num_layers):
    keep_idx = keep_indices_per_layer[layer_idx]
    keep_tensor = torch.tensor(keep_idx)
    
    # Gate weight: [256, hidden] -> [192, hidden]
    gate_name = f"model.layers.{layer_idx}.mlp.gate.weight"
    if gate_name in all_tensors:
        all_tensors[gate_name] = all_tensors[gate_name][keep_idx].clone()
        sliced_count += 1
    
    # e_score_correction_bias: [256] -> [192]
    bias_name = f"model.layers.{layer_idx}.ffn.gate.e_score_correction_bias"
    if bias_name in all_tensors:
        all_tensors[bias_name] = all_tensors[bias_name][keep_idx].clone()
        sliced_count += 1
    
    # Expert tensors: [256, in, out] -> [192, in, out]
    for expert_tensor_suffix in ['w1.weight', 'w1.scale', 'w2.weight', 'w2.scale', 'w3.weight', 'w3.scale']:
        for expert_base in [f"model.layers.{layer_idx}.mlp.experts", f"model.layers.{layer_idx}.ffn.experts"]:
            exp_name = f"{expert_base}.{expert_tensor_suffix}"
            if exp_name in all_tensors:
                t = all_tensors[exp_name]
                if t.dim() == 3 and t.shape[0] == ORIGINAL_EXPERTS:
                    all_tensors[exp_name] = t[keep_idx].clone()
                    sliced_count += 1
    
    if (layer_idx + 1) % 10 == 0 or layer_idx == 0:
        print(f"  Sliced layer {layer_idx + 1}/{num_layers}")

print(f"  Sliced {sliced_count} tensors total")
print()

# ============================================================
# Update config
# ============================================================
print("[CONFIG] Updating config...")
config["num_local_experts"] = PRUNED_EXPERTS
config["n_routed_experts"] = PRUNED_EXPERTS
config["reap_sliced"] = True
print(f"  num_local_experts: {PRUNED_EXPERTS}")
print(f"  n_routed_experts: {PRUNED_EXPERTS}")
print()

# ============================================================
# Save sliced model
# ============================================================
print(f"[SAVE] Saving sliced model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save config
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Save tensors as single large safetensors file (or shard if needed)
print("  Saving tensors...")
save_start = time.time()
save_file(all_tensors, os.path.join(OUTPUT_DIR, "model.safetensors"))
save_time = time.time() - save_start
print(f"  Saved in {save_time:.0f}s ({save_time/60:.1f} min)")

# Create index
new_index = {"metadata": {"format": "safeensors"}, "weight_map": {}}
for name in all_tensors:
    new_index["weight_map"][name] = "model.safetensors"
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(new_index, f, indent=2)

# Copy tokenizer files
import shutil
for f_name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json']:
    src = os.path.join(REAP_DIR, f_name)
    dst = os.path.join(OUTPUT_DIR, f_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# Save metadata
metadata = {
    "reap_pruned": True,
    "reap_sliced": True,
    "original_experts": ORIGINAL_EXPERTS,
    "pruned_experts": PRUNED_EXPERTS,
    "sparsity": 0.25,
    "method": "gate_weight_norms",
}
with open(os.path.join(OUTPUT_DIR, "reap_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

total_time = time.time() - start_time
print()
print("=" * 60)
print(f"DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Output: {OUTPUT_DIR}")
print(f"Tensors sliced: {sliced_count}")
print(f"Experts: {ORIGINAL_EXPERTS} -> {PRUNED_EXPERTS}")
print("=" * 60)
