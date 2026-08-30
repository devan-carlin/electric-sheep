"""Phase C: GDN stage-by-stage comparison (XPU kernel intermediates vs CPU ref).

Loads the per-rank GDN captures (projected_qkvz, projected_ba, core_attn_out, z)
and the HC capture (mix_out = GDN input x). Computes the FULL GDN on CPU from
true weights. Empirically determines the per-rank head mapping by matching z
(a pure projection), then compares each stage:
  1. in_proj  : per-rank projected_qkvz/ba  vs  CPU full projection slices
  2. z-slice  : per-rank z                  vs  CPU z slice
  3. core     : per-rank core_attn_out      vs  CPU delta-rule core slice
Pinpoints exactly where the XPU kernel diverges from the CPU reference.
"""
import glob, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
LAYER = 0
EPS = 1e-6
NUM_K, NUM_V, HK, HV = 16, 48, 128, 128
TP = 4
K_PER_RANK = NUM_K // TP   # 4
V_PER_RANK = NUM_V // TP   # 12

def load(name):
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()

def pure_python_gdn(projected_qkvz, projected_ba, conv_w, A_log, dt_bias):
    """Full GDN (all heads). Returns (core [T,NUM_V,HV], z [T,NUM_V,HV]) f32."""
    T = projected_qkvz.shape[0]
    qkvz = projected_qkvz.float()
    ba = projected_ba.float()
    q_dim = NUM_K * HK
    v_dim = NUM_V * HV
    qkv = qkvz[:, : q_dim * 2 + v_dim]
    z = qkvz[:, q_dim * 2 + v_dim:].reshape(T, NUM_V, HV)
    w2d = conv_w.view(conv_w.shape[0], conv_w.shape[2]).float()
    x = qkv.transpose(0, 1).unsqueeze(0)
    xp = F.pad(x, (3, 0))
    conv = F.conv1d(xp, w2d.unsqueeze(1), groups=w2d.shape[0])
    conv = F.silu(conv).transpose(1, 2).squeeze(0)
    q = conv[:, :q_dim].view(T, NUM_K, HK)
    k = conv[:, q_dim:q_dim * 2].view(T, NUM_K, HK)
    v = conv[:, q_dim * 2:].view(T, NUM_V, HV)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    b = ba[:, :NUM_V]
    a = ba[:, NUM_V:]
    g = -torch.exp(A_log.float()) * F.softplus(a + dt_bias.float())
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
    hc = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    x = hc["mix_out"].float()   # (T, 2560) GDN input
    T = x.shape[0]
    print(f"T={T} GDN input {tuple(x.shape)}")

    base = f"model.language_model.layers.{LAYER}.linear_attn"
    in_qkv = load(f"{base}.in_proj_qkv.weight").float()
    in_z = load(f"{base}.in_proj_z.weight").float()
    in_b = load(f"{base}.in_proj_b.weight").float()
    in_a = load(f"{base}.in_proj_a.weight").float()
    A_log = load(f"{base}.A_log").float()
    dt_bias = load(f"{base}.dt_bias").float()
    conv_w = load(f"{base}.conv1d.weight").float()

    # Full CPU projections
    qkvz_full = x @ torch.cat([in_qkv, in_z], dim=0).T   # (T, 16384) [q;k;v;z]
    ba_full = x @ torch.cat([in_b, in_a], dim=0).T       # (T, 96) [b;a]
    q_dim = NUM_K * HK
    v_dim = NUM_V * HV
    z_full = qkvz_full[:, q_dim * 2 + v_dim:].reshape(T, NUM_V, HV)  # (T,48,128)
    b_full = ba_full[:, :NUM_V]
    a_full = ba_full[:, NUM_V:]

    core_full, _ = pure_python_gdn(qkvz_full, ba_full, conv_w, A_log, dt_bias)
    print(f"z_full {tuple(z_full.shape)} absmax {z_full.abs().max().item():.4f}")
    print(f"core_full {tuple(core_full.shape)} absmax {core_full.abs().max().item():.4f}")

    # Per-rank qkvz layout: [q(512);k(512);v(1536);z(1536)]
    qk = K_PER_RANK * HK   # 512
    vv = V_PER_RANK * HV   # 1536
    for r in range(TP):
        d = torch.load(f"/tmp/qwen4exp_gdn_capture_L0_R{r}.pt", map_location="cpu")
        qkvz_r = d["projected_qkvz"].float()   # (T, 4096)
        ba_r = d["projected_ba"].float()       # (T, 24)
        core_r = d["core_attn_out"].float()    # (T, 12, 128)
        z_r = d["z"].float()                   # (T, 12, 128)
        # z slice within per-rank qkvz: offset qk+qk+vv = 512+512+1536=2560
        z_r_from_qkvz = qkvz_r[:, 2 * qk + vv: 2 * qk + vv + vv].reshape(T, V_PER_RANK, HV)

        # --- find v-head mapping by matching z_r against z_full slices ---
        best = None
        for h0 in range(0, NUM_V - V_PER_RANK + 1):
            c = cos(z_r, z_full[:, h0:h0 + V_PER_RANK])
            if best is None or c > best[1]:
                best = (h0, c)
        h0, cz = best
        # also try strided (interleaved) mapping: heads h0 + r*TP pattern
        print(f"\n=== RANK {r} ===")
        print(f"  z contiguous-head match: h0={h0} cos={cz:.6f}")

        # --- in_proj q/k/v check: compare per-rank qkv to CPU slices ---
        # per-rank q: qkvz_r[:, :qk] -> (T, 4 k-heads, 128)
        q_r = qkvz_r[:, :qk].reshape(T, K_PER_RANK, HK)
        k_r = qkvz_r[:, qk:2*qk].reshape(T, K_PER_RANK, HK)
        v_r = qkvz_r[:, 2*qk:2*qk+vv].reshape(T, V_PER_RANK, HV)
        # CPU full q/k/v (pre-conv, pre-l2norm)
        q_full = qkvz_full[:, :q_dim].reshape(T, NUM_K, HK)
        k_full = qkvz_full[:, q_dim:2*q_dim].reshape(T, NUM_K, HK)
        v_full = qkvz_full[:, 2*q_dim:2*q_dim+v_dim].reshape(T, NUM_V, HV)
        # find k-head mapping
        bestk = None
        for h0k in range(0, NUM_K - K_PER_RANK + 1):
            c = cos(q_r, q_full[:, h0k:h0k+K_PER_RANK])
            if bestk is None or c > bestk[1]:
                bestk = (h0k, c)
        print(f"  q contiguous-head match: h0={bestk[0]} cos={bestk[1]:.6f}")
        print(f"  v contiguous-head match: h0={h0} cos={cos(v_r, v_full[:, h0:h0+V_PER_RANK]):.6f}")
        print(f"  z_r vs z_r_from_qkvz (internal consistency) cos={cos(z_r, z_r_from_qkvz):.6f}")

        # --- core comparison (delta-rule) using the found v-head mapping ---
        core_ref_slice = core_full[:, h0:h0+V_PER_RANK]
        print(f"  CORE cos(core_r, ref_slice[h0={h0}]) = {cos(core_r, core_ref_slice):.6f}")
        print(f"     core_r absmax={core_r.abs().max().item():.5f} ref absmax={core_ref_slice.abs().max().item():.5f}")
        print(f"     core_r norm={core_r.norm().item():.4f} ref norm={core_ref_slice.norm().item():.4f}")

if __name__ == "__main__":
    main()