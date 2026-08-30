"""Phase C: element-wise ratio of pure-Python GDN core vs XPU core.

The direct test showed cos=1.0 (collinear) but norm ratio 11.31 (=sqrt(128)).
RMSNorm should cancel a UNIFORM scale, yet post_kernel(PP core) != post_kernel(XPU core).
This test computes the element-wise ratio r = core_pp / core_xpu to see whether
the scale is uniform (constant) or varies over the HV dim (within a head).
Also re-runs the RMSNorm invariance test with full diagnostics.
"""
import glob, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
EPS = 1e-6
NUM_K, NUM_V, HK, HV = 16, 48, 128, 128
TP = 4
V_PER_RANK = NUM_V // TP

_KEY2FILE = {}
def _build_index():
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                _KEY2FILE[k] = f
_build_index()
_WCACHE = {}
def load(name):
    if name in _WCACHE:
        return _WCACHE[name]
    f = _KEY2FILE.get(name)
    if f is None:
        raise KeyError(name)
    with safe_open(f, framework="pt") as sf:
        t = sf.get_tensor(name)
    _WCACHE[name] = t
    return t

def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()

def pure_python_gdn_core(x, base):
    in_qkv = load(f"{base}.in_proj_qkv.weight").float()
    in_z = load(f"{base}.in_proj_z.weight").float()
    in_b = load(f"{base}.in_proj_b.weight").float()
    in_a = load(f"{base}.in_proj_a.weight").float()
    A_log = load(f"{base}.A_log").float()
    dt_bias = load(f"{base}.dt_bias").float()
    conv_w = load(f"{base}.conv1d.weight").float()
    T = x.shape[0]
    qkvz = x @ torch.cat([in_qkv, in_z], dim=0).T
    ba = x @ torch.cat([in_b, in_a], dim=0).T
    q_dim = NUM_K * HK
    v_dim = NUM_V * HV
    z = qkvz[:, q_dim * 2 + v_dim:].reshape(T, NUM_V, HV)
    qkv = qkvz[:, :q_dim * 2 + v_dim]
    w2d = conv_w.view(conv_w.shape[0], conv_w.shape[2])
    xx = qkv.transpose(0, 1).unsqueeze(0)
    xp = F.pad(xx, (3, 0))
    conv = F.conv1d(xp, w2d.unsqueeze(1), groups=w2d.shape[0])
    conv = F.silu(conv).transpose(1, 2).squeeze(0)
    q = conv[:, :q_dim].view(T, NUM_K, HK)
    k = conv[:, q_dim:q_dim * 2].view(T, NUM_K, HK)
    v = conv[:, q_dim * 2:].view(T, NUM_V, HV)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    b = ba[:, :NUM_V]
    a = ba[:, NUM_V:]
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)
    beta = torch.sigmoid(b)
    rep = NUM_V // NUM_K
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)
    state = torch.zeros(NUM_V, HV, HK, dtype=torch.float32)
    outs = []
    for t in range(T):
        qt, kt, vt = q[t], k[t], v[t]
        gt, bt = g[t], beta[t].unsqueeze(-1)
        decay = torch.exp(gt).unsqueeze(-1).unsqueeze(-1)
        state = state * decay
        pred = torch.bmm(state, kt.unsqueeze(-1)).squeeze(-1)
        delta = bt * (vt - pred)
        state = state + torch.bmm(delta.unsqueeze(-1), kt.unsqueeze(1))
        out_t = torch.bmm(state, qt.unsqueeze(-1)).squeeze(-1)
        outs.append(out_t)
    core = torch.stack(outs, dim=0)
    return core, z

def main():
    base = "model.language_model.layers.0.linear_attn"
    hc = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    x = hc["mix_out"].float()
    T = x.shape[0]

    core_xpu = torch.zeros(T, NUM_V, HV)
    z_xpu = torch.zeros(T, NUM_V, HV)
    for r in range(TP):
        d = torch.load(f"/tmp/qwen4exp_gdn_capture_L0_R{r}.pt", map_location="cpu")
        core_xpu[:, r*V_PER_RANK:(r+1)*V_PER_RANK] = d["core_attn_out"].float()
        z_xpu[:, r*V_PER_RANK:(r+1)*V_PER_RANK] = d["z"].float()
    core_pp, z_pp = pure_python_gdn_core(x, base)

    # element-wise ratio (guard against ~0)
    r = core_pp / core_xpu.clamp(min=1e-6)
    print("=== element-wise ratio core_pp / core_xpu ===")
    print(f"  mean {r.mean().item():.4f}  std {r.std().item():.4f}")
    print(f"  min {r.min().item():.4f}  max {r.max().item():.4f}")
    print(f"  sqrt(128) = {128**0.5:.4f}")
    # is it constant? check coefficient of variation
    print(f"  coeff of variation (std/mean) = {r.std().item()/r.mean().item():.4f}")

    # per-HV-element mean ratio (average over T and V) -> shows if it varies over HV
    r_hv = r.mean(dim=(0, 1))  # (HV,)
    print(f"\n=== per-HV-element mean ratio (over T,V) ===")
    print(f"  first 8: {r_hv[:8].tolist()}")
    print(f"  last 8:  {r_hv[-8:].tolist()}")
    print(f"  min {r_hv.min().item():.4f} max {r_hv.max().item():.4f} "
          f"cv {r_hv.std().item()/r_hv.mean().item():.4f}")

    # per-head mean ratio (average over T, HV) -> shows if it varies over V
    r_v = r.mean(dim=(0, 2))  # (NUM_V,)
    print(f"\n=== per-head mean ratio (over T,HV) ===")
    print(f"  first 8: {[f'{v:.3f}' for v in r_v[:8].tolist()]}")
    print(f"  min {r_v.min().item():.4f} max {r_v.max().item():.4f} "
          f"cv {r_v.std().item()/r_v.mean().item():.4f}")

    # ---- RMSNorm invariance diagnostics ----
    print("\n=== RMSNorm invariance diagnostics ===")
    c2 = core_xpu.reshape(-1, HV)
    print(f"  c2 dtype {c2.dtype} shape {tuple(c2.shape)}")
    print(f"  c2 absmax {c2.abs().max().item():.6f} min {c2.min().item():.6f} max {c2.max().item():.6f}")
    m1 = c2.pow(2).mean(-1, keepdim=True)
    m2 = (c2 * 11.31).pow(2).mean(-1, keepdim=True)
    print(f"  mean(c2^2) absmax {m1.abs().max().item():.6f}")
    print(f"  mean((c2*11.31)^2) absmax {m2.abs().max().item():.6f}")
    print(f"  m2/m1 ratio: min {(m2/m1).min().item():.4f} max {(m2/m1).max().item():.4f} (expect 127.9)")
    xn = c2 * torch.rsqrt(m1 + EPS)
    xn2 = (c2 * 11.31) * torch.rsqrt(m2 + EPS)
    print(f"  cos(xn, xn2) = {cos(xn, xn2):.6f}")
    # relative diff
    rel = (xn - xn2).abs() / (xn.abs() + 1e-9)
    print(f"  rel diff: mean {rel.mean().item():.6f} max {rel.max().item():.6f}")

if __name__ == "__main__":
    main()