"""Phase C: verify the PLE FULL forward at layer 1 (real weights, real data).

Recomputes the entire PLE forward on CPU from the checkpoint weights + the
mmap'd n-gram table + the captured rows/metadata, and compares to the
captured `out` (the updated wide residual). Stages:
  emb   = table[rows]
  key   = grouped_norm(emb @ key_proj.T)
  value = emb @ value_proj.T
  query = grouped_norm(resid_in)
  s     = (key*query).sum(-1)/sqrt(n_embd)
  gate  = sigmoid(sign(s)*sqrt(|s|))
  gated = value.unsqueeze(1)*gate.unsqueeze(-1)
  norm  = grouped_norm(gated)
  conv  = dilated_causal_conv(norm)   (kernel 4, dilation 3, zero init state)
  out   = resid_in + gated + conv
"""
import glob, math, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
TABLE = "/mnt/data/ple_cache/ple_table_qwen4exp.pt"
RANK = 0
HC, N_EMBD = 4, 2560
HC_DIM = HC * N_EMBD
KERN, DIL = 4, 3
HIST = (KERN - 1) * DIL
EPS = 1e-6

def load(name):
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()

def grouped_norm(x, w):
    # x: [T, hc, n_embd]; per-stream RMSNorm over n_embd + affine over hc_dim
    T = x.shape[0]
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + EPS).to(x.dtype)
    return (xn.reshape(T, HC_DIM) * w).reshape(T, HC, N_EMBD)

def main():
    d = torch.load(f"/tmp/qwen4exp_ple_capture_rank{RANK}.pt", map_location="cpu")
    resid_in = d["resid_in"].float()   # (T, hc, n_embd)
    out_cap = d["out"].float()         # (T, hc, n_embd)
    rows = d["rows"]                   # (T, 16) int64
    T = resid_in.shape[0]
    print(f"T={T} resid_in {tuple(resid_in.shape)} out {tuple(out_cap.shape)} rows {tuple(rows.shape)}")

    base = "model.language_model.layers.1.ple"
    key_proj = load(f"{base}.key_proj.weight").float()    # (hc_dim, n_embd)
    value_proj = load(f"{base}.value_proj.weight").float()  # (n_embd, n_embd)
    norm_key = load(f"{base}.norm_key.weight").float()    # (hc_dim,)
    norm_query = load(f"{base}.norm_query.weight").float()
    norm_conv = load(f"{base}.norm_conv.weight").float()
    conv_w = load(f"{base}.conv1d.weight").float()        # (hc_dim, 1, 4)

    # table lookup
    table = torch.load(TABLE, mmap=True, weights_only=True)["table"]
    emb = table[rows.reshape(-1)].reshape(T, -1).float()  # (T, 2560)
    print(f"emb {tuple(emb.shape)} absmax {emb.abs().max().item():.4f} "
          f"cos(emb, cap)={cos(emb, d['emb']):.6f}")

    key = grouped_norm(emb @ key_proj.T, norm_key)
    value = emb @ value_proj.T
    query = grouped_norm(resid_in, norm_query)
    s = (key * query).sum(dim=-1) / math.sqrt(N_EMBD)
    gate = torch.sigmoid(torch.sign(s) * torch.sqrt(torch.clamp(s.abs(), min=1e-6)))
    gated = value.unsqueeze(1) * gate.unsqueeze(-1)
    normalized = grouped_norm(gated, norm_conv)

    # dilated causal conv, zero init state (fresh prefill)
    x = normalized.reshape(T, HC_DIM)
    w = conv_w.squeeze(1)  # (hc_dim, 4)
    state = torch.zeros(HIST, HC_DIM)
    padded = torch.cat([state, x], dim=0)  # (HIST+T, hc_dim)
    acc = torch.zeros(T, HC_DIM)
    for k in range(KERN):
        offset = HIST - (KERN - 1 - k) * DIL
        acc = acc + padded[offset:offset + T].float() * w[:, k].float()
    conv_out = F.silu(acc)

    out_ref = resid_in + gated + conv_out.reshape(T, HC, N_EMBD)

    print("\n=== PLE FULL FORWARD (captured out vs CPU ref) ===")
    print(f"cosine = {cos(out_cap, out_ref):.6f}")
    print(f"relerr = {((out_cap-out_ref).norm()/(out_ref.norm()+1e-9)).item():.4f}")
    print(f"max|cap-ref| = {(out_cap-out_ref).abs().max().item():.5f}")
    print(f"absmax: cap {out_cap.abs().max().item():.4f} ref {out_ref.abs().max().item():.4f}")
    print(f"norm:   cap {out_cap.norm().item():.4f} ref {out_ref.norm().item():.4f}")

    # stage-by-stage vs captured intermediates
    print("\n=== stage checks (CPU ref vs captured) ===")
    print(f"key   cos={cos(key, d['key']):.6f}")
    print(f"value cos={cos(value, d['value']):.6f}")
    print(f"s     cos={cos(s, d['s']):.6f}  maxdiff={(s-d['s']).abs().max().item():.5f}")
    print(f"gate  cos={cos(gate, d['gate']):.6f}  maxdiff={(gate-d['gate']).abs().max().item():.5f}")
    print(f"gated cos={cos(gated, d['gated']):.6f}")
    print(f"conv  cos={cos(conv_out, d['conv_out']):.6f}  maxdiff={(conv_out-d['conv_out']).abs().max().item():.5f}")

if __name__ == "__main__":
    main()