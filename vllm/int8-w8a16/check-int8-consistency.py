"""Internal consistency check: is the packed int8 a valid quantization?

If scale = group_absmax/127 (symmetric), then a valid quantization has
max(|int8|) ~= 127 per group. Check the distribution of per-group max(|int8|).
Also re-verify the scale-vs-bf16-absmax match precisely.
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

im, iwm = find_tensor(INT8, "layers.0.linear_attn.in_proj_qkv")
int8_packed = [k for k in im if k.endswith("in_proj_qkv.weight_packed")][0]
int8_scale = [k for k in im if k.endswith("in_proj_qkv.weight_scale")][0]

with safe_open(os.path.join(INT8, iwm[int8_packed]), framework="pt") as st:
    W_packed = st.get_tensor(int8_packed)  # [N, K/4] int32
    S = st.get_tensor(int8_scale)          # [N, K/g] bf16

N, K4 = W_packed.shape
K = K4 * 4
w_i8 = W_packed.view(torch.uint8).view(torch.int8).reshape(N, K)  # [N, K] current order
Sf = S.float()  # [N, K/g]

# per-group max |int8|
ng = K // GROUP
wg = w_i8.reshape(N, ng, GROUP)
gmax = wg.abs().amax(dim=2)  # [N, ng]
print("per-group max(|int8|) stats:")
print("  min:", float(gmax.min()), " max:", float(gmax.max()))
print("  mean:", float(gmax.mean()))
print("  frac groups with max>=120:", float((gmax >= 120).float().mean()))
print("  frac groups with max>=100:", float((gmax >= 100).float().mean()))
print("  frac groups with max<50:  ", float((gmax < 50).float().mean()))
print("  histogram of gmax (bins of 16):")
hist, _ = torch.histogram(gmax.flatten(), bins=[0,16,32,48,64,80,96,112,120,128])
for b, c in zip([0,16,32,48,64,80,96,112,120], hist.tolist()):
    print(f"    {b:3d}-{b+16:3d}: {c}")

# Now compare to base bf16 absmax precisely (first 8 groups, row 0)
bm, bwm = find_tensor(BASE, "layers.0.linear_attn.in_proj_qkv")
base_name = [k for k in bm if k.endswith("in_proj_qkv.weight")][0]
with safe_open(os.path.join(BASE, bwm[base_name]), framework="pt") as st:
    W_bf16 = st.get_tensor(base_name)
Wb = W_bf16[:1].float()  # row 0
Wbg = Wb.reshape(1, ng, GROUP)
bf16_gmax = Wbg.abs().amax(dim=2)  # [1, ng]
print()
print("row0: bf16 group absmax vs int8 scale*127 (first 8 groups):")
for g in range(8):
    print(f"  g{g}: bf16_absmax={float(bf16_gmax[0,g]):.6f}  scale*127={float(Sf[0,g]*127):.6f}  ratio={float(bf16_gmax[0,g]/(Sf[0,g]*127+1e-12)):.4f}")

# overall: does bf16_absmax == scale*127 across all groups row 0?
ratio = bf16_gmax[0] / (Sf[0]*127 + 1e-12)
print(f"  ratio mean={float(ratio.mean()):.4f} std={float(ratio.std()):.4f} min={float(ratio.min()):.4f} max={float(ratio.max()):.4f}")
