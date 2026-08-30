"""Recompute the full-attention CORE (layer 3) from the captured post-RoPE
q/k/v and compare to vLLM's captured attn_before_gate.

This isolates the attention kernel math + intra-rank GQA head layout, using
ONLY the captured q/k/v (no weights, no TP ambiguity). If it matches, the
attention kernel is correct and the bug is upstream (projection/norm/RoPE).

Per rank (TP=4): 6 Q heads, 1 KV head (replicated), head_dim 256.
Causal attention, scale = 1/sqrt(256).
"""
import torch

D = 256
SCALE = D ** -0.5


def causal_attn(q, k, v):
    """q: (T, Hq, D), k: (T, Hkv, D), v: (T, Hkv, D) -> (T, Hq, D).
    GQA: each Q head uses its KV head. Here Hkv=1 so all Q heads share it."""
    T, Hq, _ = q.shape
    Hkv = k.shape[1]
    rep = Hq // Hkv
    # Expand KV to match Q heads: Q head h -> KV head h // rep
    k_exp = k.repeat_interleave(rep, dim=1)  # (T, Hq, D)
    v_exp = v.repeat_interleave(rep, dim=1)  # (T, Hq, D)
    # scores: (T, Hq, T)
    scores = torch.einsum("thd,shd->ths", q, k_exp) * SCALE
    # causal mask: s <= t
    idx = torch.arange(T, device=q.device)
    mask = idx[None, :] <= idx[:, None]  # (T, T) True where s<=t
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("ths,shd->thd", probs, v_exp)  # (T, Hq, D)
    return out


def main():
    for r in range(4):
        c = torch.load(f"/tmp/qwen4exp_fa_capture_L3_R{r}.pt", map_location="cpu")
        q = c["q"].float()          # (T, Hq*D)
        k = c["k"].float()          # (T, Hkv*D)
        v = c["v"].float()          # (T, Hkv*D)
        ref = c["attn_before_gate"].float()  # (T, Hq*D)
        T = q.shape[0]
        Hq = q.shape[1] // D
        Hkv = k.shape[1] // D
        q = q.view(T, Hq, D)
        k = k.view(T, Hkv, D)
        v = v.view(T, Hkv, D)
        out = causal_attn(q, k, v).reshape(T, -1)
        diff = (out - ref).abs()
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
        tok_cos = torch.nn.functional.cosine_similarity(out, ref, dim=-1)
        print(f"rank{r}: T={T} Hq={Hq} Hkv={Hkv}  cos={cos:.6f}  "
              f"maxdiff={diff.max().item():.5f}  meandiff={diff.mean().item():.6f}  "
              f"per-tok cos min={tok_cos.min().item():.4f} mean={tok_cos.mean().item():.4f}")


if __name__ == "__main__":
    main()