"""Phase C: direct comparison of pure-Python GDN core vs ground-truth XPU core.

The sub-block test showed my pure-Python GDN (core + post-kernel) gives attn_out
cos 0.624 (3x too big), but the stage test claimed core cos 0.999997. This test
resolves the contradiction by directly comparing:
  1. my pure-Python core  vs  ground-truth XPU core (reassembled from 4 ranks)
  2. per-head magnitude ratios
  3. the post-kernel applied to BOTH cores
If the cores match, the post-kernel is the issue. If not, the core is the issue.
"""
import glob, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
EPS = 1e-6
NUM_K, NUM_V, HK, HV = 16, 48, 128, 128
TP = 4
V_PER_RANK = NUM_V // TP  # 12

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
    """Returns (core [T,NUM_V,HV], z [T,NUM_V,HV]) f32, NO post-kernel."""
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
    # FLA kernel: after l2norm, q is scaled by 1/sqrt(K) (K = head_k_dim).
    # The output (state @ q) is therefore scaled by 1/sqrt(128).
    q = q * (1.0 / (HK ** 0.5))
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

def post_kernel(core, z, base):
    norm_w = load(f"{base}.norm.weight").float()
    out_w = load(f"{base}.out_proj.weight").float()
    T = core.shape[0]
    core2 = core.reshape(-1, HV)
    z2 = z.reshape(-1, HV)
    var = core2.pow(2).mean(dim=-1, keepdim=True)
    xn = core2 * torch.rsqrt(var + EPS)
    zgated = xn * norm_w * torch.sigmoid(z2)
    zgated = zgated.reshape(T, NUM_V * HV)
    return zgated @ out_w.T

def main():
    base = "model.language_model.layers.0.linear_attn"
    hc = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    x = hc["mix_out"].float()  # (T, 2560) GDN input
    T = x.shape[0]
    print(f"T={T} GDN input {tuple(x.shape)}")

    # ground-truth XPU core + z (reassembled from 4 ranks)
    core_xpu = torch.zeros(T, NUM_V, HV)
    z_xpu = torch.zeros(T, NUM_V, HV)
    for r in range(TP):
        d = torch.load(f"/tmp/qwen4exp_gdn_capture_L0_R{r}.pt", map_location="cpu")
        core_xpu[:, r*V_PER_RANK:(r+1)*V_PER_RANK] = d["core_attn_out"].float()
        z_xpu[:, r*V_PER_RANK:(r+1)*V_PER_RANK] = d["z"].float()
    print(f"XPU core {tuple(core_xpu.shape)} absmax {core_xpu.abs().max().item():.5f} norm {core_xpu.norm().item():.4f}")

    # my pure-Python core
    core_pp, z_pp = pure_python_gdn_core(x, base)
    print(f"PP  core {tuple(core_pp.shape)} absmax {core_pp.abs().max().item():.5f} norm {core_pp.norm().item():.4f}")

    print(f"\n=== CORE comparison (PP vs XPU) ===")
    print(f"  cos = {cos(core_pp, core_xpu):.6f}")
    print(f"  norm ratio (pp/xpu) = {core_pp.norm().item()/core_xpu.norm().item():.4f}")
    print(f"  absmax ratio (pp/xpu) = {core_pp.abs().max().item()/core_xpu.abs().max().item():.4f}")

    # per-head magnitude ratio
    print(f"\n=== per-head norm ratio (pp/xpu) ===")
    for h in range(0, NUM_V, 4):
        r_pp = core_pp[:, h].norm().item()
        r_xpu = core_xpu[:, h].norm().item()
        print(f"  head {h:2d}: {r_pp/r_xpu if r_xpu>0 else float('nan'):.4f}")

    # z comparison
    print(f"\n=== Z comparison (PP vs XPU) ===")
    print(f"  cos = {cos(z_pp, z_xpu):.6f}")
    print(f"  norm ratio (pp/xpu) = {z_pp.norm().item()/z_xpu.norm().item():.4f}")

    # post-kernel applied to BOTH
    attn_xpu = post_kernel(core_xpu, z_xpu, base)
    attn_pp = post_kernel(core_pp, z_pp, base)
    attn_cap = hc["attn_out"].float()
    print(f"\n=== POST-KERNEL (both cores) vs captured attn_out ===")
    print(f"  XPU core -> post_kernel: cos = {cos(attn_xpu, attn_cap):.6f}  norm {attn_xpu.norm().item():.4f}")
    print(f"  PP  core -> post_kernel: cos = {cos(attn_pp, attn_cap):.6f}  norm {attn_pp.norm().item():.4f}")
    print(f"  captured attn_out norm = {attn_cap.norm().item():.4f}")

if __name__ == "__main__":
    main()