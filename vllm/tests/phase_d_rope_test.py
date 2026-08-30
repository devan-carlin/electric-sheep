"""Non-circular RoPE test for the full-attention layer (L3).

Take the captured RAW qkv (pre-norm, pre-RoPE), apply:
  1. q/k split (per-head [q|gate] interleave)
  2. GemmaRMSNorm (zero-centered: x*rsqrt(var+eps)*(1+w))
  3. IMROPE (llama.cpp convention) -- for text-only this is standard NeoX RoPE
     on the first rotary_dim=64 dims, pairing (x[j], x[j+32]), freq=base^(-j/32)
and compare to vLLM's captured post-RoPE q/k.

If they match  -> RoPE + norm are correct.
If they diverge -> RoPE (or norm) is the bug.

This is NON-circular: we use the raw qkv + an independent IMROPE, not vLLM's
post-RoPE q/k as input.
"""
import json
import torch
from safetensors import safe_open

CKPT = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8"
D = 256                 # head_dim
ROT = 64                # rotary_dim = D * 0.25
BASE = 1e7
EPS = 1e-6
N_Q_HEADS = 6           # per rank (24/4)
N_KV_HEADS = 1          # per rank (2/4, replicated)


def load_tensor(key):
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    shard = idx[key]
    with safe_open(f"{CKPT}/{shard}", framework="pt") as st:
        return st.get_tensor(key)


def gemma_rmsnorm(x, w, eps=EPS):
    """x: (..., D), w: (D,) zero-centered. Returns x*rsqrt(var+eps)*(1+w)."""
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * (1.0 + w.float()))


def imrope_textonly(x, positions, base=BASE, rot=ROT, head_dim=D):
    """x: (T, H, D) post-norm. positions: (T,) token positions.
    Text-only IMROPE == standard NeoX RoPE on first `rot` dims.
    Pair (x[j], x[j+rot/2]) for j in 0..rot/2-1, freq = base^(-j/(rot/2))."""
    T, H, _ = x.shape
    x = x.float()
    half = rot // 2  # 32
    # freq for pair j: base^(-j/half)
    j = torch.arange(half, dtype=torch.float32, device=x.device)
    freqs = base ** (-j / half)  # (32,)
    theta = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (T, 32)
    cos = theta.cos()  # (T, 32)
    sin = theta.sin()  # (T, 32)
    # x0 = x[..., :32], x1 = x[..., 32:64]
    x0 = x[..., :half]          # (T, H, 32)
    x1 = x[..., half:rot]       # (T, H, 32)
    cos = cos[:, None, :]       # (T, 1, 32)
    sin = sin[:, None, :]
    out = x.clone()
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:rot] = x0 * sin + x1 * cos
    # rest (64:D) passes through unchanged
    return out


def main():
    q_norm_w = load_tensor("model.language_model.layers.3.self_attn.q_norm.weight")
    k_norm_w = load_tensor("model.language_model.layers.3.self_attn.k_norm.weight")
    print(f"q_norm {tuple(q_norm_w.shape)} k_norm {tuple(k_norm_w.shape)}")

    for r in range(4):
        c = torch.load(f"/tmp/qwen4exp_fa_capture_L3_R{r}.pt", map_location="cpu")
        qkv = c["qkv"].float()          # (T, 3584)
        q_ref = c["q"].float()          # (T, 1536) post-RoPE
        k_ref = c["k"].float()          # (T, 256)  post-RoPE
        pos = c["positions"]            # (3, T)
        T = qkv.shape[0]
        # text-only: all 3 position rows equal; use row 0
        pos_t = pos[0] if pos.ndim == 2 else pos
        # verify rows equal
        rows_equal = torch.equal(pos[0], pos[1]) and torch.equal(pos[1], pos[2]) if pos.ndim == 2 else True

        # split qkv: [q_gate (3072) | k (256) | v (256)]
        q_gate = qkv[:, : N_Q_HEADS * D * 2]   # (T, 3072)
        k_raw = qkv[:, N_Q_HEADS * D * 2: N_Q_HEADS * D * 2 + N_KV_HEADS * D]  # (T, 256)

        # q_gate -> per head [q | gate], chunk
        q_gate = q_gate.view(T, N_Q_HEADS, 2 * D)
        q_raw, gate = torch.chunk(q_gate, 2, dim=-1)  # each (T, 6, 256)
        q_raw = q_raw.reshape(T, -1)                  # (T, 1536)

        # norm (per head)
        q_n = gemma_rmsnorm(q_raw.view(T, N_Q_HEADS, D), q_norm_w).view(T, -1)
        k_n = gemma_rmsnorm(k_raw.view(T, N_KV_HEADS, D), k_norm_w).view(T, -1)

        # RoPE
        q_rope = imrope_textonly(q_n.view(T, N_Q_HEADS, D), pos_t).view(T, -1)
        k_rope = imrope_textonly(k_n.view(T, N_KV_HEADS, D), pos_t).view(T, -1)

        # compare
        for name, mine, ref in [("q", q_rope, q_ref), ("k", k_rope, k_ref)]:
            diff = (mine - ref).abs()
            cos = torch.nn.functional.cosine_similarity(mine.flatten(), ref.flatten(), dim=0).item()
            # per-head cos
            print(f"rank{r} {name}: pos_rows_equal={rows_equal} cos={cos:.6f} "
                  f"maxdiff={diff.max().item():.5f} meandiff={diff.mean().item():.6f}")


if __name__ == "__main__":
    main()