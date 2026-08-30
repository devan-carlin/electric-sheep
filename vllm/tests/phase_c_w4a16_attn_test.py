"""Phase C: verify the W4A16 full-attention projection (layer 3) on CPU.

The full-attention q/k/v/o_proj are W4A16 (compressed-tensors uint4b8,
symmetric, group_size 128) -- a DIFFERENT dequant path than the MoE (which
uses grouped GEMM). This is the prime suspect for the token-soup bug.

Dequantizes q_proj/k_proj/v_proj on CPU, computes the full qkv projection,
and compares the per-rank slices to the captured qkv. If it matches -> the
W4A16 attention dequant is correct. If it mismatches -> the int4_gemm_w4a16
kernel (or its weight prep) is the bug.
"""
import glob, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
LAYER = 3
GROUP = 128
TP = 4
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 24, 2, 256
Q_PER_RANK = NUM_Q_HEADS // TP * HEAD_DIM * 2   # 6*256*2 = 3072 (q+gate)
KV_PER_RANK = HEAD_DIM                          # 256 (1 kv head, replicated)

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
    """packed: [out, in/8] int32 (8 uint4 per int32). scale: [out, in/group].
    Symmetric uint4b8, zp=8. Returns [out, in] f32."""
    out, in8 = packed.shape
    inp = in8 * 8
    # unpack 8 uint4 per int32: nibble j at bits [4j, 4j+4]
    packed = packed.to(torch.int64)
    shifts = torch.arange(0, 32, 4, device=packed.device)  # [8]
    q = (packed.unsqueeze(-1) >> shifts).to(torch.int32) & 0xF  # [out, in8, 8]
    q = q.reshape(out, inp).to(torch.float32)  # [out, in]
    # scale: [out, in/group] -> [out, in]
    s = scale.to(torch.float32).repeat_interleave(GROUP, dim=1)  # [out, in]
    return (q - 8.0) * s

def main():
    base = f"model.language_model.layers.{LAYER}.self_attn"
    q_packed = load(f"{base}.q_proj.weight_packed")
    q_scale = load(f"{base}.q_proj.weight_scale")
    k_packed = load(f"{base}.k_proj.weight_packed")
    k_scale = load(f"{base}.k_proj.weight_scale")
    v_packed = load(f"{base}.v_proj.weight_packed")
    v_scale = load(f"{base}.v_proj.weight_scale")
    print(f"q_proj packed {tuple(q_packed.shape)} scale {tuple(q_scale.shape)}")
    print(f"k_proj packed {tuple(k_packed.shape)} scale {tuple(k_scale.shape)}")
    print(f"v_proj packed {tuple(v_packed.shape)} scale {tuple(v_scale.shape)}")

    q_w = dequant_w4a16(q_packed, q_scale)  # [12288, 2560]
    k_w = dequant_w4a16(k_packed, k_scale)  # [512, 2560]
    v_w = dequant_w4a16(v_packed, v_scale)  # [512, 2560]
    print(f"q_w {tuple(q_w.shape)} absmax {q_w.abs().max().item():.4f}")
    print(f"k_w {tuple(k_w.shape)} absmax {k_w.abs().max().item():.4f}")
    print(f"v_w {tuple(v_w.shape)} absmax {v_w.abs().max().item():.4f}")

    # Full qkv projection: [q_gate(12288); k(512); v(512)]
    w_full = torch.cat([q_w, k_w, v_w], dim=0)  # [13312, 2560]

    for r in range(TP):
        d = torch.load(f"/tmp/qwen4exp_fa_capture_L3_R{r}.pt", map_location="cpu")
        hidden = d["hidden_states"].float()  # (T, 2560)
        qkv_cap = d["qkv"].float()           # (T, 3584)
        T = hidden.shape[0]
        qkv_ref = hidden @ w_full.T  # (T, 13312)
        # per-rank slice: q_gate rows [r*3072, (r+1)*3072), k/v replicated.
        # GQA: 24 q-heads / 2 kv-heads = 12 q-heads per kv-head. TP=4 -> 6
        # q-heads/rank. rank r holds q[r*6,(r+1)*6), which all fall in
        # kv_head = r // 2 (block replication).
        q_gate_r = qkv_ref[:, r*Q_PER_RANK:(r+1)*Q_PER_RANK]
        # GQA: 24 q-heads / 2 kv-heads = 12 q-heads per kv-head. TP=4 -> 6
        # q-heads/rank. 2 ranks share each kv-head -> kv_head = r // 2.
        ranks_per_kv = (NUM_Q_HEADS // NUM_KV_HEADS) // (NUM_Q_HEADS // TP)  # 2
        kvh = r // ranks_per_kv
        k_r = qkv_ref[:, 12288 + kvh*HEAD_DIM:12288 + (kvh+1)*HEAD_DIM]
        v_r = qkv_ref[:, 12288 + 512 + kvh*HEAD_DIM:12288 + 512 + (kvh+1)*HEAD_DIM]
        qkv_ref_r = torch.cat([q_gate_r, k_r, v_r], dim=1)  # (T, 3584)

        print(f"\n=== RANK {r} (kv_head={kvh}) ===")
        print(f"  qkv cos = {cos(qkv_cap, qkv_ref_r):.6f}")
        print(f"  q_gate cos = {cos(qkv_cap[:, :Q_PER_RANK], q_gate_r):.6f}")
        print(f"  k cos = {cos(qkv_cap[:, Q_PER_RANK:Q_PER_RANK+KV_PER_RANK], k_r):.6f}")
        print(f"  v cos = {cos(qkv_cap[:, Q_PER_RANK+KV_PER_RANK:], v_r):.6f}")
        print(f"  absmax: cap {qkv_cap.abs().max().item():.4f} ref {qkv_ref_r.abs().max().item():.4f}")
        print(f"  norm:   cap {qkv_cap.norm().item():.4f} ref {qkv_ref_r.norm().item():.4f}")

if __name__ == "__main__":
    main()