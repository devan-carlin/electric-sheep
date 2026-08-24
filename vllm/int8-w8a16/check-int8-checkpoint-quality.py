"""Checkpoint quality check: dequantize INT8 weights and compare to base bf16.

If dequantized INT8 ~= bf16 (small rel err), the checkpoint is a faithful
quantization and the loop is an XPU-path issue. If they diverge badly, the
checkpoint itself is broken (bad scales / bad quantization).
"""
import os
import torch
from safetensors import safe_open

BASE = "/mnt/data/models/Qwen3.8-27B"
INT8 = "/mnt/data/models/lued-Qwen3.8-27B-INT8-W8A16-MTP"
GROUP = 128

# Find the base bf16 in_proj_qkv tensor name + shard
import json

def find_tensor(root, name_substr):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    matches = [k for k in wm if name_substr in k]
    return matches, wm

bm, bwm = find_tensor(BASE, "layers.0.linear_attn.in_proj_qkv")
im, iwm = find_tensor(INT8, "layers.0.linear_attn.in_proj_qkv")
print("base matches:", bm)
print("int8 matches:", im)

base_name = [k for k in bm if k.endswith("in_proj_qkv.weight")][0]
int8_packed = [k for k in im if k.endswith("in_proj_qkv.weight_packed")][0]
int8_scale = [k for k in im if k.endswith("in_proj_qkv.weight_scale")][0]

with safe_open(os.path.join(BASE, bwm[base_name]), framework="pt") as st:
    W_bf16 = st.get_tensor(base_name)  # [N, K] bf16
with safe_open(os.path.join(INT8, iwm[int8_packed]), framework="pt") as st:
    W_packed = st.get_tensor(int8_packed)  # [N, K/4] int32
    S = st.get_tensor(int8_scale)          # [N, K/g] bf16

print("bf16 weight:", tuple(W_bf16.shape), W_bf16.dtype)
print("int8 packed:", tuple(W_packed.shape), W_packed.dtype)
print("int8 scale: ", tuple(S.shape), S.dtype)

N, K = W_bf16.shape
w_i8 = W_packed.view(torch.uint8).view(torch.int8)  # [N, K]
Sf = S.float()                                       # [N, K/g]
S_exp = Sf.repeat_interleave(GROUP, dim=1)           # [N, K]
W_deq = (w_i8.float() * S_exp)                        # [N, K]

Wb = W_bf16.float()
diff = (W_deq - Wb).abs()
rel = diff / (Wb.abs() + 1e-6)
corr = torch.corrcoef(torch.stack([W_deq.flatten(), Wb.flatten()]))[0, 1]
print()
print("bf16  mean/std:", float(Wb.mean()), float(Wb.std()))
print("deq   mean/std:", float(W_deq.mean()), float(W_deq.std()))
print("max abs diff:", float(diff.max()))
print("mean abs diff:", float(diff.mean()))
print("mean rel diff:", float(rel.mean()))
print("p99 rel diff:", float(rel.flatten().kthvalue(int(0.99 * rel.numel())).values))
print("corr(deq, bf16):", float(corr))
# scale sanity
print()
print("scale min/max/mean:", float(S.min()), float(S.max()), float(S.float().mean()))
print("bf16 per-group absmax sample (first 4 groups, row 0):")
Wg = Wb[0].reshape(K // GROUP, GROUP)
print("  ", [float(x) for x in Wg[:4].abs().max(dim=1).values])
print("int8 scale row 0 first 4:", [float(x) for x in S[0, :4]])
print("  (scale should ~= group_absmax / 127)")
