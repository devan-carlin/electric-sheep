"""Phase C: layer-0 sub-block isolation.

Layer 0's attention sub-block was verified correct (GDN cos 0.999992). But the
per-layer reference diverges at layer 0 (cos 0.63, norm 2.85x). This test
isolates WHICH sub-block of layer 0 diverges:
  1. attn_out   : CPU GDN vs captured attn_out (HC capture)
  2. resid_attn : CPU HC-combine vs captured resid_after_attn
  3. mlp_out    : CPU MoE vs GPU-derived mlp_out
The GPU's mlp_out is derived from the captured resid_after_attn + resid_out:
  resid_out = resid_after_attn + mlp_out * w_mlp,  w_mlp = 2*sigmoid(inject2/HC)
Compares cosine AND magnitude (cosine is scale-invariant; magnitude catches
the 5x norm growth).
"""
import glob, math, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
HC, N_EMBD = 4, 2560
HC_DIM = HC * N_EMBD
LOWRANK = 320
EPS = 1e-6
NUM_K, NUM_V, HK, HV = 16, 48, 128, 128
N_EXPERTS, TOPK = 512, 10
GROUP = 128

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

def dequant_w4a16(packed, scale, group=GROUP):
    out, in8 = packed.shape
    inp = in8 * 8
    p = packed.to(torch.int64)
    shifts = torch.arange(0, 32, 4)
    q = (p.unsqueeze(-1) >> shifts).to(torch.int32) & 0xF
    q = q.reshape(out, inp).to(torch.float32)
    s = scale.to(torch.float32).repeat_interleave(group, dim=1)
    return (q - 8.0) * s

def grouped_rmsnorm(x):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + EPS)

def hc_mix(resid, hc_norm, w_down, w_up, w_inject):
    T = resid.shape[0]
    xn = grouped_rmsnorm(resid)
    xn_flat = xn.reshape(T, HC_DIM) * hc_norm
    lo = xn_flat @ w_down.T
    lo = F.silu(lo / HC)
    gate = torch.sigmoid(lo @ w_up.T)
    gated = (xn_flat * gate).reshape(T, HC, N_EMBD)
    mix_out = gated.mean(dim=1)
    inject = xn_flat @ w_inject.T
    return mix_out, inject

def hc_combine(resid, block_out, inject):
    w = 2.0 * torch.sigmoid(inject / HC)
    return resid + block_out.unsqueeze(1) * w.unsqueeze(-1)

def pure_python_gdn(x, base):
    in_qkv = load(f"{base}.in_proj_qkv.weight").float()
    in_z = load(f"{base}.in_proj_z.weight").float()
    in_b = load(f"{base}.in_proj_b.weight").float()
    in_a = load(f"{base}.in_proj_a.weight").float()
    A_log = load(f"{base}.A_log").float()
    dt_bias = load(f"{base}.dt_bias").float()
    conv_w = load(f"{base}.conv1d.weight").float()
    norm_w = load(f"{base}.norm.weight").float()
    out_w = load(f"{base}.out_proj.weight").float()
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
    core2 = core.reshape(-1, HV)
    z2 = z.reshape(-1, HV)
    var = core2.pow(2).mean(dim=-1, keepdim=True)
    xn = core2 * torch.rsqrt(var + EPS)
    zgated = xn * norm_w * torch.sigmoid(z2)
    zgated = zgated.reshape(T, NUM_V * HV)
    return zgated @ out_w.T

def moe(x, base):
    T = x.shape[0]
    gate_w = load(f"{base}.gate.weight").float()
    se_gate = load(f"{base}.shared_expert.gate_proj.weight").float()
    se_up = load(f"{base}.shared_expert.up_proj.weight").float()
    se_down = load(f"{base}.shared_expert.down_proj.weight").float()
    se_gate_scalar = load(f"{base}.shared_expert_gate.weight").float()
    logits = x @ gate_w.T
    probs = torch.softmax(logits, dim=-1)
    topk_w, topk_idx = torch.topk(probs, TOPK, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    shared = F.silu(x @ se_gate.T) * (x @ se_up.T)
    shared = shared @ se_down.T
    shared = torch.sigmoid(x @ se_gate_scalar.T) * shared
    used = torch.unique(topk_idx)
    routed = torch.zeros_like(x)
    for e in used.tolist():
        eb = f"{base}.experts.{e}"
        wg = dequant_w4a16(load(f"{eb}.gate_proj.weight_packed"), load(f"{eb}.gate_proj.weight_scale"))
        wu = dequant_w4a16(load(f"{eb}.up_proj.weight_packed"), load(f"{eb}.up_proj.weight_scale"))
        wd = dequant_w4a16(load(f"{eb}.down_proj.weight_packed"), load(f"{eb}.down_proj.weight_scale"))
        h = F.silu(x @ wg.T) * (x @ wu.T)
        eout = h @ wd.T
        mask = (topk_idx == e)
        w = (topk_w * mask).sum(dim=-1, keepdim=True)
        routed = routed + w * eout
    return routed + shared, shared, routed

def main():
    hc = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    resid_in = hc["resid_in"].float()          # (T, hc, n_embd)
    attn_out_cap = hc["attn_out"].float()      # (T, n_embd)
    resid_after_cap = hc["resid_after_attn"].float()  # (T, hc, n_embd)
    l0 = torch.load("/tmp/qwen4exp_layer00_resid.pt", map_location="cpu")
    resid_out_cap = l0["resid_out"].float()    # (T, hc, n_embd)
    T = resid_in.shape[0]
    print(f"T={T}")

    base = "model.language_model.layers.0"
    a_base = f"{base}.attn_hyper_connection"
    m_base = f"{base}.mlp_hyper_connection"

    # ---- attention sub-block ----
    cur, inject = hc_mix(resid_in, load(f"{a_base}.hc_norm.weight").float(),
                         load(f"{a_base}.input_mix_weight_down.weight").float(),
                         load(f"{a_base}.input_mix_weight_up.weight").float(),
                         load(f"{a_base}.block_inject_weight.weight").float())
    # KEY CHECK: is the computed hc_mix output == the captured mix_out?
    mix_out_cap = hc["mix_out"].float()
    print("\n=== 0. hc_mix input check ===")
    print(f"  cos(cur, mix_out_cap) = {cos(cur, mix_out_cap):.6f}")
    print(f"  norm: cur {cur.norm().item():.4f} cap {mix_out_cap.norm().item():.4f} "
          f"ratio {cur.norm().item()/mix_out_cap.norm().item():.3f}")
    attn_out_cpu = pure_python_gdn(cur, f"{base}.linear_attn")
    resid_after_cpu = hc_combine(resid_in, attn_out_cpu, inject)
    print("\n=== 1. attn_out (GDN) ===")
    print(f"  cos = {cos(attn_out_cpu, attn_out_cap):.6f}")
    print(f"  norm: cpu {attn_out_cpu.norm().item():.4f} cap {attn_out_cap.norm().item():.4f} "
          f"ratio {attn_out_cpu.norm().item()/attn_out_cap.norm().item():.3f}")
    print(f"  absmax: cpu {attn_out_cpu.abs().max().item():.4f} cap {attn_out_cap.abs().max().item():.4f}")
    print("\n=== 2. resid_after_attn (HC combine) ===")
    print(f"  cos = {cos(resid_after_cpu, resid_after_cap):.6f}")
    print(f"  norm: cpu {resid_after_cpu.norm().item():.4f} cap {resid_after_cap.norm().item():.4f} "
          f"ratio {resid_after_cpu.norm().item()/resid_after_cap.norm().item():.3f}")

    # ---- MLP sub-block ----
    # Use the GPU's resid_after_attn as the MoE input (to isolate the MoE).
    cur2, inject2 = hc_mix(resid_after_cap, load(f"{m_base}.hc_norm.weight").float(),
                           load(f"{m_base}.input_mix_weight_down.weight").float(),
                           load(f"{m_base}.input_mix_weight_up.weight").float(),
                           load(f"{m_base}.block_inject_weight.weight").float())
    mlp_out_cpu, shared_cpu, routed_cpu = moe(cur2, f"{base}.mlp")
    # GPU-derived mlp_out: resid_out = resid_after + mlp_out * w_mlp
    # block_out is (T, n_embd), broadcast across hc streams. Take stream 0.
    w_mlp = 2.0 * torch.sigmoid(inject2 / HC)  # (T, hc)
    mlp_out_gpu = (resid_out_cap[:, 0] - resid_after_cap[:, 0]) / w_mlp[:, 0].unsqueeze(-1)
    print("\n=== 3. mlp_out (MoE) ===")
    print(f"  cos = {cos(mlp_out_cpu, mlp_out_gpu):.6f}")
    print(f"  norm: cpu {mlp_out_cpu.norm().item():.4f} gpu {mlp_out_gpu.norm().item():.4f} "
          f"ratio {mlp_out_cpu.norm().item()/mlp_out_gpu.norm().item():.3f}")
    print(f"  absmax: cpu {mlp_out_cpu.abs().max().item():.4f} gpu {mlp_out_gpu.abs().max().item():.4f}")
    print(f"  shared norm {shared_cpu.norm().item():.4f}  routed norm {routed_cpu.norm().item():.4f}")
    print(f"  w_mlp absmax {w_mlp.abs().max().item():.4f} mean {w_mlp.mean().item():.4f}")

    # ---- full layer-0 resid_out ----
    resid_out_cpu = hc_combine(resid_after_cap, mlp_out_cpu, inject2)
    print("\n=== 4. resid_out (full layer 0) ===")
    print(f"  cos = {cos(resid_out_cpu, resid_out_cap):.6f}")
    print(f"  norm: cpu {resid_out_cpu.norm().item():.4f} cap {resid_out_cap.norm().item():.4f} "
          f"ratio {resid_out_cpu.norm().item()/resid_out_cap.norm().item():.3f}")

if __name__ == "__main__":
    main()