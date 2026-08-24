"""Ground-truth check: does the int4 kernel (which sets fpmath_mode) work on
this hardware, and which oneDNN engine handles it? Same shapes as the int8
repro so the comparison is apples-to-apples.
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
# int4 weight packed 8-per-byte: [N, K/8] int8, passed as .t() -> [K/8, N]
W = torch.randint(-128, 127, (n, k // 8), device="xpu", dtype=torch.int8)
# scale [K/group, N] bf16
S = torch.randn(k // g, n, device="xpu", dtype=torch.bfloat16)
# zero point: production uses 1D symmetric zp = [8] (int8)
ZP = torch.tensor([8], device="xpu", dtype=torch.int8)

print("=== calling int4_gemm_w4a16 (m=%d k=%d n=%d g=%d) ===" % (m, k, n, g), flush=True)
try:
    out = torch.ops._xpu_C.int4_gemm_w4a16(A, W.t(), None, S, ZP, g, None)
    torch.xpu.synchronize()
    print("=== SUCCESS, out shape:", tuple(out.shape), "===")
except Exception as e:
    print("=== FAILED:", type(e).__name__, str(e)[:200], "===")
