"""Minimal standalone repro of the int8_gemm_w8a16 oneDNN descriptor failure.
Runs a tiny GEMM directly through the op with ONEDNN_VERBOSE to capture the
exact reason oneDNN rejects the primitive descriptor. No model load needed.
"""
import os

os.environ["ONEDNN_VERBOSE"] = "all"
os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"

import torch
import vllm_xpu_kernels._xpu_C  # registers ops

torch.xpu.set_device(0)

# Mirror the GDN in_proj_qkvz shape: N=10240, K=5120, group=128, m=1 (decode)
m, k, n, g = 1, 5120, 10240, 128
A = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
# weight [N, K] int8 contiguous, passed as .t() -> [K, N] k-contiguous
W = torch.randint(-128, 127, (n, k), device="xpu", dtype=torch.int8)
# scale [K/group, N] bf16
S = torch.randn(k // g, n, device="xpu", dtype=torch.bfloat16)

print("=== calling int8_gemm_w8a16 (m=%d k=%d n=%d g=%d) ===" % (m, k, n, g), flush=True)
try:
    out = torch.ops._xpu_C.int8_gemm_w8a16(A, W.t(), None, S, g)
    torch.xpu.synchronize()
    print("=== SUCCESS, out shape:", tuple(out.shape), "===")
except Exception as e:
    print("=== FAILED:", type(e).__name__, str(e)[:200], "===")
