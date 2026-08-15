#!/usr/bin/env python3
"""
Fix tensor name prefixes in safetensors index.

Some HF models have bare tensor names (layers.0...) instead of
model-prefixed names (model.layers.0...). llama.cpp expects the prefix.

This script rewrites model.safetensors.index.json to add the 'model.' prefix
to all tensor names that start with 'layers.' or 'head.' or 'embed.'.
"""

import json
import sys
import os
import shutil

def fix_prefix(model_dir: str, backup: bool = True):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    
    if not os.path.exists(index_path):
        print(f"ERROR: No model.safetensors.index.json found in {model_dir}")
        sys.exit(1)
    
    print(f"Loading {index_path}...")
    with open(index_path) as f:
        data = json.load(f)
    
    weight_map = data["weight_map"]
    print(f"Original tensor count: {len(weight_map)}")
    
    # Backup original
    if backup:
        backup_path = index_path + ".bak"
        shutil.copy2(index_path, backup_path)
        print(f"Backed up to {backup_path}")
    
    # Remap tensor names
    new_weight_map = {}
    prefix_map = {}  # old -> new
    
    for old_name in weight_map:
        new_name = old_name
        if old_name.startswith("layers."):
            new_name = f"model.{old_name}"
        elif old_name == "head.weight":
            new_name = "model.head.weight"
        elif old_name == "embed.weight":
            new_name = "model.embed.weight"
        elif old_name.startswith("hc_"):
            # Hash cache tensors - also prefix
            new_name = f"model.{old_name}"
        
        if new_name != old_name:
            prefix_map[old_name] = new_name
        
        new_weight_map[new_name] = weight_map[old_name]
    
    print(f"Renamed {len(prefix_map)} tensors")
    
    # Show some examples
    for old, new in list(prefix_map.items())[:5]:
        print(f"  {old} -> {new}")
    
    data["weight_map"] = new_weight_map
    
    # Write back
    with open(index_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nUpdated {index_path}")
    print(f"New tensor count: {len(new_weight_map)}")
    
    # Also fix the individual shard files if they have internal name references
    # (safetensors shards use the tensor names as keys internally)
    print("\nNote: You'll also need to fix the individual shard files.")
    print("Use fix-shards.py to rewrite shard headers with new tensor names.")
    
    return prefix_map

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir>")
        sys.exit(1)
    
    fix_prefix(sys.argv[1])
