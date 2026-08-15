#!/usr/bin/env python3
"""
Convert DeepSeek V4 FP8 model to FP16 for INT4 AutoRound quantization.

The FP8 model uses block-wise quantization:
  - Expert weights: int8 + float8_e8m0fnu scales (block size 16)
  - Attention weights: float8_e4m3fn + float8_e8m0fnu scales (block size 16)

This script dequantizes all FP8 weights to FP16, preserving non-FP8 weights as-is.
"""

import glob
import json
import os
import sys
import time
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequantize_block_fp8(weight_int8, scale_fp8):
    """
    Dequantize 2D block-wise FP8 weights to FP16.
    
    weight_int8: [out, in] int8 weights
    scale_fp8: [out_blocks, in_blocks] float8_e8m0fnu scales
    Block sizes are inferred: block_h = out / out_blocks, block_w = in / in_blocks
    Returns: [out, in] FP16 weights
    """
    out_ch, in_ch = weight_int8.shape
    out_blocks, in_blocks = scale_fp8.shape
    
    block_h = out_ch // out_blocks
    block_w = in_ch // in_blocks
    
    # Reshape to [out_blocks, block_h, in_blocks, block_w]
    weight_blocks = weight_int8.reshape(out_blocks, block_h, in_blocks, block_w)
    
    # Convert scale from float8 to float32 for multiplication
    scale_f32 = scale_fp8.to(torch.float32)  # [out_blocks, in_blocks]
    
    # Broadcast scale: [out_blocks, 1, in_blocks, 1] * [out_blocks, block_h, in_blocks, block_w]
    dequantized = weight_blocks.to(torch.float32) * scale_f32.unsqueeze(1).unsqueeze(3)
    
    # Flatten back
    dequantized = dequantized.reshape(out_ch, in_ch)
    
    return dequantized.to(torch.float16)


def dequantize_block_fp8_attn(weight_fp8, scale_fp8):
    """
    Dequantize 2D block-wise FP8 attention weights to FP16.
    
    weight_fp8: [out, in] float8_e4m3fn weights
    scale_fp8: [out_blocks, in_blocks] float8_e8m0fnu scales
    Block sizes are inferred: block_h = out / out_blocks, block_w = in / in_blocks
    Returns: [out, in] FP16 weights
    """
    out_ch, in_ch = weight_fp8.shape
    out_blocks, in_blocks = scale_fp8.shape
    
    block_h = out_ch // out_blocks
    block_w = in_ch // in_blocks
    
    # Reshape to [out_blocks, block_h, in_blocks, block_w]
    weight_blocks = weight_fp8.reshape(out_blocks, block_h, in_blocks, block_w)
    
    # Convert scale from float8 to float32 for multiplication
    scale_f32 = scale_fp8.to(torch.float32)  # [out_blocks, in_blocks]
    
    # Broadcast scale: [out_blocks, 1, in_blocks, 1] * [out_blocks, block_h, in_blocks, block_w]
    dequantized = weight_blocks.to(torch.float32) * scale_f32.unsqueeze(1).unsqueeze(3)
    
    # Flatten back
    dequantized = dequantized.reshape(out_ch, in_ch)
    
    return dequantized.to(torch.float16)


def process_file(src_path, dst_path, file_idx, total_files):
    """Process a single safetensor file."""
    print(f"  [{file_idx}/{total_files}] {os.path.basename(src_path)}", flush=True)
    
    new_tensors = {}
    dequantized_weights = set()  # Track which weights were already dequantized
    
    with safe_open(src_path, framework='pt') as sf:
        all_keys = list(sf.keys())
        for key in all_keys:
            # Skip if this weight was already dequantized via its scale
            if key in dequantized_weights:
                continue
            
            tensor = sf.get_tensor(key)
            
            # Check if this is a scale tensor (indicates FP8 weight pair)
            if key.endswith('.scale') and tensor.dtype == torch.float8_e8m0fnu:
                # Find the corresponding weight key
                weight_key = key.replace('.scale', '.weight')
                
                if weight_key not in sf.keys():
                    print(f"    WARNING: scale {key} has no matching weight {weight_key}", flush=True)
                    new_tensors[key] = tensor
                    continue
                
                weight = sf.get_tensor(weight_key)
                
                # Determine dequantization method based on weight dtype
                if weight.dtype == torch.int8:
                    # Expert weights: int8 + fp8 scale
                    dequantized = dequantize_block_fp8(weight, tensor)
                elif weight.dtype == torch.float8_e4m3fn:
                    # Attention weights: fp8 + fp8 scale
                    dequantized = dequantize_block_fp8_attn(weight, tensor)
                else:
                    # Unknown format, keep as-is
                    print(f"    WARNING: unexpected weight dtype {weight.dtype} for {weight_key}", flush=True)
                    new_tensors[key] = tensor
                    new_tensors[weight_key] = weight
                    dequantized_weights.add(weight_key)
                    continue
                
                # Store dequantized weight, drop the scale
                new_tensors[weight_key] = dequantized
                dequantized_weights.add(weight_key)
                # Skip the scale tensor (don't store it)
                continue
            
            # Non-scale tensor, keep as-is
            new_tensors[key] = tensor
    
    # Save
    save_file(new_tensors, dst_path)
    return True


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <output_dir>")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    start_time = time.time()
    
    # Find source files
    src_files = sorted(glob.glob(os.path.join(source_dir, "*.safetensors")))
    if not src_files:
        print(f"ERROR: No safetensor files found in {source_dir}")
        sys.exit(1)
    
    print(f"Converting {len(src_files)} FP8 safetensor files to FP16")
    print(f"Source:  {source_dir}")
    print(f"Output:  {output_dir}")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for i, src_path in enumerate(src_files, 1):
        dst_path = os.path.join(output_dir, os.path.basename(src_path))
        process_file(src_path, dst_path, i, len(src_files))
    
    # Copy non-safetensor files
    for fname in os.listdir(source_dir):
        src_path = os.path.join(source_dir, fname)
        if not os.path.isfile(src_path):
            continue
        if fname.endswith('.safetensors'):
            continue
        dst_path = os.path.join(output_dir, fname)
        if not os.path.exists(dst_path):
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"  Copied {fname}")
    
    # Update config.json
    config_src = os.path.join(output_dir, "config.json")
    if os.path.exists(config_src):
        with open(config_src) as f:
            config = json.load(f)
        
        # Remove expert_dtype to prevent FP8 dispatch
        if 'expert_dtype' in config:
            del config['expert_dtype']
            print(f"  Removed expert_dtype from config.json")
        
        # Force eager expert implementation
        config['_experts_implementation'] = 'eager'
        print(f"  Set _experts_implementation='eager' in config.json")
        
        with open(config_src, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Calculate sizes
    src_size = sum(os.path.getsize(f) for f in src_files)
    dst_files = sorted(glob.glob(os.path.join(output_dir, "*.safetensors")))
    dst_size = sum(os.path.getsize(f) for f in dst_files)
    
    elapsed = time.time() - start_time
    
    print()
    print(f"Conversion complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Source size:  {src_size/1e9:.1f}GB (FP8)")
    print(f"Output size:  {dst_size/1e9:.1f}GB (FP16)")
    print(f"Output dir:   {output_dir}")


if __name__ == '__main__':
    main()
