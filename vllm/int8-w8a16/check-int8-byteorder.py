"""Determine the correct int8 unpacking for weight_packed [N, K/4] int32.

Scales match base bf16 per-group absmax, so the quantization was done on this
weight. Find which unpacking reconstructs bf16 best. Tests:
  (a) 4 byte orders within a word (consecutive k)
  (b) strided packing (word j holds k = j, j+K/4, j+2K/4, j+3K/4)
"""
import os
import json
import torch
from safetensors import safe_open

BASE = "/mnt/data/models/Qwen3.8-27B"
INT8 = "/mnt/data/models/lued-Qwen3.8-27B-INT8-W8A16-MTP"
GROUP = 128

def find_tensor(root, name_substr):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    return [k for k in wm if name_substr in k], wm

bm, bwm = find_tensor(BASE, "layers.0.linear_attn.in_proj_qkv")
im, iwm = find_tensor(INT8, "layers.0.linear_attn.in_proj_qkv")
base_name = [k for k in bm if k.endswith("in_proj_qkv.weight")][0]
int8_packed = [k for k in im if k.endswith("in_proj_qkv.weight_packed")][0]
int8_scale = [k for k in im if k.endswith("in_proj_qkv.weight_scale")][0]

with safe_open(os.path.join(BASE, bwm[base_name]), framework="pt") as st:
    W_bf16 = st.get_tensor(base_name)
with safe_open(os.path.join(INT8, iwm[int8_packed]), framework="pt") as st:
    W_packed = st.get_tensor(int8_packed)
    S = st.get_tensor(int8_scale)

N, K = W_bf16.shape
R = 256
Wb_s = W_bf16[:R].float()
S_s = S[:R].float()
P_s = W_packed[:R].contiguous()  # [R, K/4] int32
K4 = K // 4

S_exp = S_s.repeat_interleave(GROUP, dim=1)  # [R, K]
expected = torch.round(Wb_s / (S_exp + 1e-12)).clamp(-128, 127)  # [R, K]

P_u8 = P_s.view(torch.uint8).view(R, K4, 4)  # [R, K/4, 4]
print("expected int8 row0 k0..7:", [int(x) for x in expected[0, :8]])
print("packed u8 row0 word0 (b0..b3):", P_u8[0, 0].tolist())
print("packed u8 row0 word1 (b0..b3):", P_u8[0, 1].tolist())
print()

def score(w, label):
    deq = w.float() * S_exp
    diff = (deq - Wb_s).abs()
    rel = diff / (Wb_s.abs() + 1e-6)
    corr = torch.corrcoef(torch.stack([deq.flatten(), Wb_s.flatten()]))[0, 1]
    print(f"{label:34s} corr={float(corr):+.4f}  mean_rel={float(rel.mean()):.4f}")
    return float(corr)

print("--- (a) byte orders, consecutive k (word j -> k=4j..4j+3) ---")
orders = {
    "b0,b1,b2,b3": [0, 1, 2, 3],
    "b3,b2,b1,b0": [3, 2, 1, 0],
    "b1,b0,b3,b2": [1, 0, 3, 2],
    "b2,b3,b0,b1": [2, 3, 0, 1],
}
for name, o in orders.items():
    idx = torch.tensor(o).view(1, 1, 4)
    bytes_ord = P_u8.gather(2, idx.expand(R, K4, 4))  # [R, K/4, 4]
    w = bytes_ord.view(torch.int8).reshape(R, K)
    score(w, name)

print()
print("--- (b) strided: word j byte b -> k = j + b*(K/4) ---")
for bname, b in [("byte0", 0), ("byte1", 1), ("byte2", 2), ("byte3", 3)]:
    plane = P_u8[:, :, b].view(torch.int8)  # [R, K4]
    ks = torch.arange(K4) + b * K4
    deq_plane = plane.float() * S_exp[:, ks]
    corr = torch.corrcoef(torch.stack([deq_plane.flatten(), Wb_s[:, ks].flatten()]))[0, 1]
    print(f"  {bname} -> k=j+{b}*K4  corr={float(corr):+.4f}")
