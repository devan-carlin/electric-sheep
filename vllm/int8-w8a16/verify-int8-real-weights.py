"""Decisive kernel check on REAL checkpoint weights.

Loads the actual in_proj_qkv weight_packed + weight_scale from the INT8
checkpoint, runs the XPU int8_gemm_w8a16 kernel, and compares against a CPU
dequantize+matmul reference. If this passes, the kernel is fully correct on
real data and the empty-output loop is a model/quantization/load issue, not a
kernel bug.
"""
import os

os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"

import torch
from safetensors import safe_open
import vllm_xpu_kernels._xpu_C  # registers ops

MODEL = "/mnt/data/models/lued-Qwen3.8-27B-INT8-W8A16-MTP"
F = "model-00001-of-00006.safetensors"
PQ = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_packed"
SQ = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale"
GROUP = 128

torch.manual_seed(0)
torch.xpu.set_device(0)

with safe_open(os.path.join(MODEL, F), framework="pt") as st:
    w_packed = st.get_tensor(PQ)  # [N, K/4] int32
    w_scale = st.get_tensor(SQ)   # [N, K/g] bf16

N, K4 = w_packed.shape
K = K4 * 4
print(f"weight_packed {tuple(w_packed.shape)} int32 -> int8 [N={N}, K={K}]")
print(f"weight_scale  {tuple(w_scale.shape)} bf16 (N, K/g)")

# Mirror model process_weights_after_loading exactly.
w_i8 = w_packed.view(torch.uint8).view(torch.int8)  # [N, K] k-contiguous
w_s = w_scale.t().contiguous()                       # [K/g, N]

m = 4
A = torch.randn(m, K, dtype=torch.bfloat16)

out = torch.ops._xpu_C.int8_gemm_w8a16(A.xpu(), w_i8.t().xpu(), None, w_s.xpu(), GROUP)
torch.xpu.synchronize()

# CPU reference: dequantize W per group, then matmul.
Wf = w_i8.float()                       # [N, K]
Sf = w_s.float()                        # [K/g, N]
# scale each k-group: W_deq[n, k] = Wf[n, k] * Sf[k//g, n]
S_exp = Sf.repeat_interleave(GROUP, dim=0).t()  # [N, K]
W_deq = Wf * S_exp                        # [N, K]
ref = A.float() @ W_deq.t()               # [m, N]

out_f = out.float().cpu()
diff = (out_f - ref).abs()
rel = diff / (ref.abs() + 1e-3)
corr = torch.corrcoef(torch.stack([out_f.flatten(), ref.flatten()]))[0, 1]
print("out shape:", tuple(out.shape), "ref shape:", tuple(ref.shape))
print("out  mean/std:", float(out_f.mean()), float(out_f.std()))
print("ref  mean/std:", float(ref.mean()), float(ref.std()))
print("max abs diff:", float(diff.max()))
print("mean abs diff:", float(diff.mean()))
print("max rel diff:", float(rel.max()))
print("mean rel diff:", float(rel.mean()))
print("corr(out, ref):", float(corr))
ok = float(rel.mean()) < 0.05 and float(corr) > 0.999
print("=== REAL-WEIGHT NUMERICAL", "PASS" if ok else "FAIL", "===")
