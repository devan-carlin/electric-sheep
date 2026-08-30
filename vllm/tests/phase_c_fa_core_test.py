"""Phase C: verify the full-attention CORE at layer 3 (real weights, real data).

The qkv_proj (W4A16) is already verified. This verifies the rest of the core:
  1. attention softmax: q,k,v (post norm+RoPE) -> attn_before_gate
  2. gate: attn_before_gate * sigmoid(gate) -> attn_after_gate
  3. o_proj (W4A16): attn_after_gate @ o_proj.T -> output
If all match -> the full-attention core is correct. If a stage mismatches ->
that stage is the bug.
"""
import glob, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
LAYER = 3
GROUP = 128
TP = 4
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 24, 2, 256
Q_PER_RANK = NUM_Q_HEADS // TP          # 6 q-heads
KV_PER_RANK = 1                          # 1 kv-head (replicated)

def load(name):
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()

def dequant_w4a16(packed, scale):
    out, in8 = packed.shape
    inp = in8 * 8
    packed = packed.to(torch.int64)
    shifts = torch.arange(0, 32, 4, device=packed.device)
    q = (packed.unsqueeze(-1) >> shifts).to(torch.int32) & 0xF
    q = q.reshape(out, inp).to(torch.float32)
    s = scale.to(torch.float32).repeat_interleave(GROUP, dim=1)
    return (q - 8.0) * s

def main():
    base = f"model.language_model.layers.{LAYER}.self_attn"
    o_packed = load(f"{base}.o_proj.weight_packed")
    o_scale = load(f"{base}.o_proj.weight_scale")
    o_w = dequant_w4a16(o_packed, o_scale)  # [2560, 6144]
    print(f"o_proj {tuple(o_w.shape)} absmax {o_w.abs().max().item():.4f}")

    # Load all ranks first (needed to reassemble the full attn_after_gate for o_proj).
    caps = [torch.load(f"/tmp/qwen4exp_fa_capture_L3_R{r}.pt", map_location="cpu")
            for r in range(TP)]
    T = caps[0]["q"].shape[0]

    for r in range(TP):
        d = caps[r]
        q = d["q"].float().reshape(-1, Q_PER_RANK, HEAD_DIM)   # (T, 6, 256)
        k = d["k"].float().reshape(-1, KV_PER_RANK, HEAD_DIM)  # (T, 1, 256)
        v = d["v"].float().reshape(-1, KV_PER_RANK, HEAD_DIM)  # (T, 1, 256)
        gate = d["gate"].float()                                # (T, 1536)
        attn_cap = d["attn_before_gate"].float()                # (T, 1536)
        attn_after_cap = d["attn_after_gate"].float()           # (T, 1536)
        out_cap = d["output"].float()                           # (T, 2560) all-reduced

        # GQA: all 6 q-heads on this rank share the single kv-head.
        k2 = k.squeeze(1)  # (T, D)
        v2 = v.squeeze(1)  # (T, D)
        scores = torch.einsum("thd,id->thi", q, k2) / (HEAD_DIM ** 0.5)  # (T, 6, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.einsum("thi,id->thd", torch.softmax(scores, dim=-1), v2)  # (T, 6, D)
        attn_ref = attn.reshape(T, -1)  # (T, 1536)

        # gate
        attn_after_ref = attn_ref * torch.sigmoid(gate)

        print(f"\n=== RANK {r} ===")
        print(f"  attn softmax cos = {cos(attn_cap, attn_ref):.6f}  "
              f"absmax cap {attn_cap.abs().max().item():.4f} ref {attn_ref.abs().max().item():.4f}")
        print(f"  gate cos = {cos(attn_after_cap, attn_after_ref):.6f}")

    # o_proj: reassemble the full attn_after_gate [T, 6144] from all ranks,
    # then the full o_proj (all-reduced output is identical on every rank).
    full_after = torch.cat([caps[r]["attn_after_gate"].float() for r in range(TP)], dim=1)  # (T, 6144)
    out_ref = full_after @ o_w.T  # (T, 2560)
    out_cap = caps[0]["output"].float()
    print(f"\n=== O_PROJ (full, all-reduced) ===")
    print(f"  cos = {cos(out_cap, out_ref):.6f}")
    print(f"  absmax cap {out_cap.abs().max().item():.4f} ref {out_ref.abs().max().item():.4f}")
    print(f"  norm cap {out_cap.norm().item():.4f} ref {out_ref.norm().item():.4f}")

if __name__ == "__main__":
    main()