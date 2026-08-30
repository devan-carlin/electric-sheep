"""Probe: verify the interleaved MRoPE (manual CPU) matches the captured L3 q/k.

For text-only input all 3 mrope modality positions are identical, so the
interleaved reassembly is a no-op: cos/sin are just the standard NeoX RoPE
over rotary_dim=64 (partial_rotary_factor=0.25 of head_dim=256).
This de-risks the full per-layer CPU reference.
"""
import glob, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
HEAD_DIM, ROT_DIM, BASE = 256, 64, 1e7
EPS = 1e-6

def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()

def load(name):
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def dequant_w4a16(packed, scale, group=128):
    out, in8 = packed.shape
    inp = in8 * 8
    p = packed.to(torch.int64)
    shifts = torch.arange(0, 32, 4)
    q = (p.unsqueeze(-1) >> shifts).to(torch.int32) & 0xF
    q = q.reshape(out, inp).to(torch.float32)
    s = scale.to(torch.float32).repeat_interleave(group, dim=1)
    return (q - 8.0) * s

def rmsnorm(x, w):
    # GemmaRMSNorm: x * (1 + w), not x * w
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + EPS) * (1.0 + w)

def manual_rope(x, positions, cos_cache, sin_cache):
    """x: (T, n_heads, head_dim). NeoX RoPE over first ROT_DIM dims."""
    c = cos_cache[positions].unsqueeze(1)  # (T,1,32)
    s = sin_cache[positions].unsqueeze(1)
    x1 = x[..., :ROT_DIM // 2]
    x2 = x[..., ROT_DIM // 2:ROT_DIM]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out = x.clone()
    out[..., :ROT_DIM // 2] = o1
    out[..., ROT_DIM // 2:ROT_DIM] = o2
    return out

def main():
    inv_freq = 1.0 / (BASE ** (torch.arange(0, ROT_DIM, 2, dtype=torch.float) / ROT_DIM))
    t = torch.arange(262144, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_cache, sin_cache = freqs.cos(), freqs.sin()  # (maxpos, 32)

    base3 = "model.language_model.layers.3.self_attn"
    q_w = dequant_w4a16(load(f"{base3}.q_proj.weight_packed"), load(f"{base3}.q_proj.weight_scale"))
    k_w = dequant_w4a16(load(f"{base3}.k_proj.weight_packed"), load(f"{base3}.k_proj.weight_scale"))
    v_w = dequant_w4a16(load(f"{base3}.v_proj.weight_packed"), load(f"{base3}.v_proj.weight_scale"))
    w_full = torch.cat([q_w, k_w, v_w], dim=0)  # [13312, 2560]

    cap = torch.load("/tmp/qwen4exp_fa_capture_L3_R0.pt", map_location="cpu")
    hidden = cap["hidden_states"].float()       # (T, 2560)
    q_cap = cap["q"].float()                    # (T, 1536) post-norm+rope (6 heads)
    k_cap = cap["k"].float()                    # (T, 256)  post-norm+rope (1 kv head)
    positions = cap["positions"]
    if positions.ndim == 2:
        positions = positions[0]  # T row (text: all 3 rows identical)
    positions = positions.flatten().long()
    T = hidden.shape[0]
    print(f"T={T} positions {positions.tolist()}")

    qkv = hidden @ w_full.T                     # (T, 13312)
    q_gate = qkv[:, :12288].view(T, 24, 512)
    k = qkv[:, 12288:12800].view(T, 2, 256)
    q_pre, gate = torch.chunk(q_gate, 2, dim=-1)   # (T,24,256)
    qn_w = load(f"{base3}.q_norm.weight").float()
    kn_w = load(f"{base3}.k_norm.weight").float()
    q_n = rmsnorm(q_pre, qn_w)                  # (T,24,256)
    k_n = rmsnorm(k, kn_w)                      # (T,2,256)
    q_rope = manual_rope(q_n, positions, cos_cache, sin_cache)
    k_rope = manual_rope(k_n, positions, cos_cache, sin_cache)
    q_rope_r0 = q_rope[:, :6].reshape(T, 6 * 256)
    k_rope_r0 = k_rope[:, :1].reshape(T, 256)
    print(f"[b] cos(q_rope_r0, q_cap) = {cos(q_rope_r0, q_cap):.6f}")
    print(f"[b] cos(k_rope_r0, k_cap) = {cos(k_rope_r0, k_cap):.6f}")
    print(f"    q absmax: ref {q_rope_r0.abs().max().item():.4f} cap {q_cap.abs().max().item():.4f}")
    print(f"    k absmax: ref {k_rope_r0.abs().max().item():.4f} cap {k_cap.abs().max().item():.4f}")

if __name__ == "__main__":
    main()