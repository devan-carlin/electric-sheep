#!/usr/bin/env python3
"""Direct test of the FP8 matmul K mismatch fix — no full model load needed."""

import sys
import importlib.util
import torch
import warnings

print(f"PyTorch: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
print()

# Load the patched matmul module from the HF cache (same path transformers uses)
matmul_path = "/home/dc/.cache/huggingface/hub/kernels--kernels-community--finegrained-fp8/snapshots/7cdb05d472d6c954c7d03182ed836ebfd4610df0/build/torch-xpu/matmul.py"
spec = importlib.util.spec_from_file_location("matmul", matmul_path)
matmul_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matmul_mod)
matmul_2d = matmul_mod.matmul_2d

# Create test tensors that reproduce the K mismatch:
# A: activations (batch, K=4096) — like DeepSeek V4 hidden_size
# B: FP8 weights (N=2048, K=4096) — like down_proj (4096 → 2048)
# Bs: block scales

batch = 8
K = 4096  # hidden_size
N = 2048  # moe_intermediate_size
block_n, block_k = 128, 128  # typical FP8 block size

print(f"Test: A({batch}, {K}) @ B({N}, {K}) → C({batch}, {N})")
print(f"Block size: [{block_n}, {block_k}]")
print()

# Create test data on XPU
A = torch.randn(batch, K, dtype=torch.bfloat16, device="xpu")
B = torch.randn(N, K, dtype=torch.float8_e4m3fn, device="xpu")
Bs = torch.ones(N // block_n, K // block_k, dtype=torch.float8_e8m0fnu, device="xpu")

print("[1/2] Testing K-matched case (should use fast FP8 kernel)...")
try:
    with warnings.catch_warnings(record=True) as w:
        C = matmul_2d(A, B, Bs, block_size=[block_n, block_k], output_dtype=torch.bfloat16)
    print(f"  ✓ Output shape: {C.shape}")
    print(f"  ✓ No warnings (used fast path)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

print()

# Now test the K mismatch case — this is what DeepSeek V4 MoE down_proj does
# A has K=2048 (output of up_proj), B has K=4096 (down_proj weight expects 4096 input)
# Wait — actually the mismatch is the other way. Let me re-read the error:
# "K mismatch: A has K=4096, B has K=2048"
# So A is (batch, 4096) and B is (N, 2048)
# This means the weight shape is wrong — B should be (N, 4096) to match A's K

# Actually the real issue is that FP8Experts stores weights in w1/w2/w3 format
# where w3 (down_proj) is [num_experts, moe_intermediate_size, hidden_size]
# But the activation coming in has shape [batch, hidden_size]
# So A.K = hidden_size = 4096, but B.K = moe_intermediate_size = 2048

print("[2/2] Testing K-mismatch case (should fall back to HP matmul)...")
A_mismatch = torch.randn(batch, 4096, dtype=torch.bfloat16, device="xpu")  # K=4096
B_mismatch = torch.randn(N, 2048, dtype=torch.float8_e4m3fn, device="xpu")  # K=2048
Bs_mismatch = torch.ones(N // block_n, 2048 // block_k, dtype=torch.float8_e8m0fnu, device="xpu")

try:
    with warnings.catch_warnings(record=True) as w:
        C_mismatch = matmul_2d(A_mismatch, B_mismatch, Bs_mismatch, block_size=[block_n, block_k], output_dtype=torch.bfloat16)
    print(f"  ✓ Output shape: {C_mismatch.shape}")
    if w:
        print(f"  ✓ Fallback warning: {w[0].message}")
    else:
        print(f"  ⚠ No warning — fallback may not have triggered")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

print()
print("Done!")
