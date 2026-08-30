"""Phase C: CPU reference for the GDN layer-0 FULL forward (in_proj -> kernel -> z-gate -> out_proj).

Uses the HC capture (mix_out = GDN input, attn_out = GDN output) and the real
checkpoint GDN weights (all BF16). Computes the full GDN forward on CPU:
  projected_qkvz = x @ in_proj_qkvz.T   ([q;k;v;z])
  projected_ba   = x @ in_proj_ba.T     ([b;a])
  core_attn_out, z = pure-Python GDN (conv1d + delta-rule)   [from phase_a]
  zgated = RMSNormGated(core_attn_out, z)
  out = zgated @ out_proj.T
Compares to the captured attn_out. If it matches -> GDN forward is correct.
If it mismatches -> the GDN forward (in_proj/kernel/z-gate/out_proj) is the bug.
"""
import glob, math, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
LAYER = 0
EPS = 1e-6

def load(name):
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def pure_python_gdn(projected_qkvz, projected_ba, num_k, num_v, hk, hv,
                    conv_w, A_log, dt_bias):
    """Pure-Python GDN reference (conv1d + delta-rule). Returns (core, z) f32."""
    T = projected_qkvz.shape[0]
    qkvz = projected_qkvz.float()
    ba = projected_ba.float()
    q_dim = num_k * hk
    v_dim = num_v * hv
    qkv = qkvz[:, : q_dim * 2 + v_dim]
    z = qkvz[:, q_dim * 2 + v_dim:]
    # conv1d (causal, kernel 4, silu) on qkv
    w2d = conv_w.view(conv_w.shape[0], conv_w.shape[2]).float()  # [10240, 4]
    x = qkv.transpose(0, 1).unsqueeze(0)
    xp = F.pad(x, (3, 0))
    conv = F.conv1d(xp, w2d.unsqueeze(1), groups=w2d.shape[0])
    conv = F.silu(conv).transpose(1, 2).squeeze(0)
    q = conv[:, :q_dim].view(T, num_k, hk)
    k = conv[:, q_dim:q_dim * 2].view(T, num_k, hk)
    v = conv[:, q_dim * 2:].view(T, num_v, hv)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    b = ba[:, :num_v]
    a = ba[:, num_v:]
    g = -torch.exp(A_log.float()) * F.softplus(a + dt_bias.float())
    beta = torch.sigmoid(b)
    rep = num_v // num_k
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)
    state = torch.zeros(num_v, hv, hk, dtype=torch.float32)
    outs = []
    for t in range(T):
        qt, kt, vt = q[t], k[t], v[t]
        gt, bt = g[t], beta[t].unsqueeze(-1)
        decay = torch.exp(gt).unsqueeze(-1).unsqueeze(-1)
        # FLA order: decay state FIRST, then predict from the DECAYED state.
        state = state * decay
        pred = torch.bmm(state, kt.unsqueeze(-1)).squeeze(-1)
        delta = bt * (vt - pred)
        state = state + torch.bmm(delta.unsqueeze(-1), kt.unsqueeze(1))
        out_t = torch.bmm(state, qt.unsqueeze(-1)).squeeze(-1)
        outs.append(out_t)
    core = torch.stack(outs, dim=0)
    return core, z

def main():
    cap = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    x = cap["mix_out"].float()        # (T, 2560) GDN input
    attn_out_cap = cap["attn_out"].float()  # (T, 2560) GDN output
    T = x.shape[0]
    print(f"T={T} GDN input {tuple(x.shape)}")

    base = f"model.language_model.layers.{LAYER}.linear_attn"
    in_qkv = load(f"{base}.in_proj_qkv.weight").float()   # (10240, 2560)
    in_z = load(f"{base}.in_proj_z.weight").float()       # (6144, 2560)
    in_b = load(f"{base}.in_proj_b.weight").float()       # (48, 2560)
    in_a = load(f"{base}.in_proj_a.weight").float()       # (48, 2560)
    out_w = load(f"{base}.out_proj.weight").float()       # (2560, 6144)
    norm_w = load(f"{base}.norm.weight").float()          # (128,)
    A_log = load(f"{base}.A_log").float()                 # (48,)
    dt_bias = load(f"{base}.dt_bias").float()             # (48,)
    conv_w = load(f"{base}.conv1d.weight").float()        # (10240, 1, 4)

    num_k, num_v, hk, hv = 16, 48, 128, 128

    # in_proj (fused qkvz = [qkv; z], ba = [b; a])
    projected_qkvz = x @ torch.cat([in_qkv, in_z], dim=0).T   # (T, 16384)
    projected_ba = x @ torch.cat([in_b, in_a], dim=0).T       # (T, 96)
    print(f"projected_qkvz {tuple(projected_qkvz.shape)} absmax {projected_qkvz.abs().max().item():.4f}")
    print(f"projected_ba {tuple(projected_ba.shape)} absmax {projected_ba.abs().max().item():.4f}")

    core, z = pure_python_gdn(projected_qkvz, projected_ba, num_k, num_v, hk, hv,
                              conv_w, A_log, dt_bias)
    print(f"core {tuple(core.shape)} absmax {core.abs().max().item():.4f}  z absmax {z.abs().max().item():.4f}")

    # z-gate: RMSNormGated(core, z). core [T, num_v, hv], z [T, num_v, hv].
    # RMSNorm over last dim (hv), gated by sigmoid(z).
    core2 = core.reshape(-1, hv)
    z2 = z.reshape(-1, hv)
    var = core2.pow(2).mean(dim=-1, keepdim=True)
    xn = core2 * torch.rsqrt(var + EPS)
    xn = xn * norm_w
    zgated = xn * torch.sigmoid(z2)
    zgated = zgated.reshape(T, num_v * hv)
    out = zgated @ out_w.T   # (T, 2560)

    def cos(a, b):
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    def relerr(a, b):
        return ((a - b).norm() / (b.norm() + 1e-9)).item()

    print("\n=== GDN FULL FORWARD (captured attn_out vs CPU reference) ===")
    print(f"cosine(attn_out_cap, ref) = {cos(attn_out_cap, out):.6f}")
    print(f"relerr(attn_out_cap, ref) = {relerr(attn_out_cap, out):.4f}")
    print(f"max|cap - ref|            = {(attn_out_cap-out).abs().max().item():.5f}")
    print(f"absmax: cap {attn_out_cap.abs().max().item():.4f}  ref {out.abs().max().item():.4f}")
    print(f"norm:   cap {attn_out_cap.norm().item():.4f}  ref {out.norm().item():.4f}")

if __name__ == "__main__":
    main()