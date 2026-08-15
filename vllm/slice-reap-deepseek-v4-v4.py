#!/usr/bin/env python3
"""Slice REAP-pruned DeepSeek-V4-Flash model from 256 -> 192 experts.

Direct safetensors approach - per-expert tensor removal + renumbering.
Expert tensors renamed: model.layers.{L}.mlp.experts.{E} -> sequential 0-191.
"""

import os
import json
import time
import re
import torch
import numpy as np
from safetensors.torch import load_file, save_file

print("=" * 60)
print("REAP Tensor Slicing — DeepSeek-V4-Flash 256 -> 192 (renumbered)")
print("=" * 60)

# ============================================================
# Configuration
# ============================================================
REAP_DIR = "/mnt/data/models/DeepSeek-V4-Flash-0731--REAP-192experts"
OUTPUT_DIR = "/mnt/data/models/DeepSeek-V4-Flash-0731--REAP-192experts-sliced"
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

with open(os.path.join(REAP_DIR, "config.json")) as f:
    config = json.load(f)

num_layers = config["num_hidden_layers"]
print(f"Layers: {num_layers}")
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
keep_indices_per_layer = {}  # layer -> set of original expert indices

for layer_idx in range(num_layers):
    gate_name = f"model.layers.{layer_idx}.ffn.gate.weight"
    if gate_name not in all_tensors:
        gate_name = f"model.layers.{layer_idx}.mlp.gate.weight"
    
    if gate_name in all_tensors:
        gate_w = all_tensors[gate_name].float()
        norms = torch.norm(gate_w, dim=1).cpu().numpy()
        sorted_idx = np.argsort(norms)[::-1]
        keep_idx = set(sorted_idx[:PRUNED_EXPERTS].tolist())
        keep_indices_per_layer[layer_idx] = keep_idx
        
        kept_norms = norms[[i for i in keep_idx]]
        pruned_norms = norms[[i for i in range(ORIGINAL_EXPERTS) if i not in keep_idx]]
        if layer_idx < 3 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx}: kept={len(keep_idx)}, avg_kept_norm={kept_norms.mean():.4f}, avg_pruned_norm={pruned_norms.mean():.6f}")
    else:
        print(f"  WARNING: Layer {layer_idx} gate not found, keeping first {PRUNED_EXPERTS}")
        keep_indices_per_layer[layer_idx] = set(range(PRUNED_EXPERTS))

print()

# ============================================================
# Build mapping: (layer, old_expert) -> new_expert (sequential 0-191)
# ============================================================
print("[MAP] Building expert index mappings...")
expert_map = {}  # (layer, old_expert) -> new_expert

for layer_idx in range(num_layers):
    keep_set = keep_indices_per_layer[layer_idx]
    sorted_kept = sorted(keep_set)
    for new_idx, old_idx in enumerate(sorted_kept):
        expert_map[(layer_idx, old_idx)] = new_idx

print(f"  Mapped {len(expert_map)} expert slots across {num_layers} layers")
print()

# ============================================================
# Build remap table: (layer, old_expert) -> new_expert as lookup array
# ============================================================
print("[REMAP] Building expert remap arrays for tid2eid routing tables...")
remap_arrays = {}  # layer -> tensor of shape [256] mapping old->new
for layer_idx in range(num_layers):
    sorted_kept = sorted(keep_indices_per_layer[layer_idx])
    # Build remap: old_expert -> new_expert (or 0 for pruned)
    remap = torch.full((ORIGINAL_EXPERTS,), 0, dtype=torch.long)
    for new_idx, old_idx in enumerate(sorted_kept):
        remap[old_idx] = new_idx
    remap_arrays[layer_idx] = remap

# ============================================================
# Slice gate tensors + rename expert tensors + remap tid2eid
# ============================================================
print("[SLICE] Slicing gate tensors + renaming expert tensors + remapping tid2eid...")

# Regex for per-expert tensors
expert_re = re.compile(r'^(model\.layers\.(\d+)\.mlp\.experts\.)((\d+)\.(w[123]\.(?:weight|scale)))$')

new_tensors = {}
sliced_count = 0
renamed_count = 0
removed_count = 0
remapped_count = 0

for name, tensor in all_tensors.items():
    m = expert_re.match(name)
    if m:
        layer_idx = int(m.group(2))
        old_expert = int(m.group(4))
        keep_set = keep_indices_per_layer[layer_idx]
        
        if old_expert not in keep_set:
            removed_count += 1
            continue  # Skip pruned expert
        
        # Rename to new sequential index
        new_expert = expert_map[(layer_idx, old_expert)]
        new_name = f"{m.group(1)}{new_expert}.{m.group(5)}"
        new_tensors[new_name] = tensor
        renamed_count += 1
    elif 'tid2eid' in name:
        # Remap routing table: old expert IDs -> new sequential IDs
        layer_match = re.search(r'model\.layers\.(\d+)', name)
        if layer_match:
            layer_idx = int(layer_match.group(1))
            remap = remap_arrays[layer_idx]
            # tensor shape: [num_tasks, 6] - each value is an expert ID
            remapped = remap[tensor.long()]
            new_tensors[name] = remapped
            remapped_count += 1
    else:
        # Check if it's a gate weight to slice
        if name.endswith('.ffn.gate.weight') or name.endswith('.mlp.gate.weight'):
            layer_match = re.search(r'model\.layers\.(\d+)', name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                keep_set = keep_indices_per_layer[layer_idx]
                sorted_kept = sorted(keep_set)
                keep_tensor = torch.tensor(sorted_kept)
                new_tensors[name] = tensor[keep_tensor].clone()
                sliced_count += 1
        # Check if it's e_score_correction_bias
        elif 'e_score_correction_bias' in name:
            layer_match = re.search(r'model\.layers\.(\d+)', name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                keep_set = keep_indices_per_layer[layer_idx]
                sorted_kept = sorted(keep_set)
                keep_tensor = torch.tensor(sorted_kept)
                new_tensors[name] = tensor[keep_tensor].clone()
                sliced_count += 1
        else:
            # Keep all other tensors as-is
            new_tensors[name] = tensor

all_tensors = new_tensors

print(f"  Sliced {sliced_count} gate tensors (256 -> 192)")
print(f"  Renamed {renamed_count} expert tensors (old indices -> 0-191)")
print(f"  Remapped {remapped_count} tid2eid routing tables")
print(f"  Removed {removed_count} pruned expert tensors")
print(f"  Remaining tensors: {len(all_tensors)}")
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

# Save tensors
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
print(f"Gate tensors sliced: {sliced_count}")
print(f"Expert tensors renamed: {renamed_count}")
print(f"Expert tensors removed: {removed_count}")
print(f"Remaining tensors: {len(all_tensors)}")
print(f"Experts: {ORIGINAL_EXPERTS} -> {PRUNED_EXPERTS}")
print("=" * 60)
