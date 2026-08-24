"""Numerical correctness check for int8_gemm_w8a16.
Computes a CPU reference (dequantize weights, group-scaled matmul) and compares
against the XPU kernel output. Catches wrong scale application, wrong group
layout, or sign errors that a shape-only smoke test would miss.
"""
import os

os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"

import torch
import vllm_xpu_kernels._xpu_C  # registers ops

torch.manual_seed(0)
torch.xpu.set_device(0)

# Mirror the GDN in_proj_qkvz shape: N=10240, K=5120, group=128
m, k, n, g = 4, 5120, 10240, 128
A = torch.randn(m, k, dtype=torch.bfloat16)
# weight [N, K] int8 contiguous
W = torch.randint(-128, 127, (n, k), dtype=torch.int8)
# scale [K/group, N] bf16, POSITIVE (real quant scales are positive)
S = (torch.rand(k // g, n, dtype=torch.bfloat16) * 0.01 + 0.001)

out = torch.ops._xpu_C.int8_gemm_w8a16(A.xpu(), W.t().xpu(), None, S.xpu(), g)
torch.xpu.synchronize()

# CPU reference: out[m, n] = sum_g ( sum_{k in g} A[m,k] * W[n,k] ) * S[g, n]
Af = A.float()
Wf = W.float()
# group the k dim: Ag [m, ng, gi], Wg [n, ng, gi] -> sum over gi -> [m, ng, n]
Ag = Af.reshape(m, k // g, g)
Wg = Wf.reshape(n, k // g, g)
partial = torch.einsum("abc,dbc->abd", Ag, Wg)  # [m, ng, n]
ref = (partial * S.float().unsqueeze(0)).sum(dim=1)  # [m, n]

out_f = out.float().cpu()
diff = (out_f - ref).abs()
rel = diff / (ref.abs() + 1e-3)
print("out shape:", tuple(out.shape), "ref shape:", tuple(ref.shape))
print("out  mean/std:", float(out_f.mean()), float(out_f.std()))
print("ref  mean/std:", float(ref.mean()), float(ref.std()))
print("max abs diff:", float(diff.max()))
print("mean abs diff:", float(diff.mean()))
print("max rel diff:", float(rel.max()))
print("mean rel diff:", float(rel.mean()))
# correlation as a sanity signal
corr = torch.corrcoef(torch.stack([out_f.flatten(), ref.flatten()]))[0, 1]
print("corr(out, ref):", float(corr))
ok = float(rel.mean()) < 0.05 and float(corr) > 0.999
print("=== NUMERICAL", "PASS" if ok else "FAIL", "===")
