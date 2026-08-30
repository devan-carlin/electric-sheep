"""Phase C: isolate the POST-kernel GDN part (RMSNormGated + out_proj).

The stage test proved the XPU kernel (in_proj+conv+l2norm+gating+delta-rule)
produces the correct core and z (cos ~1.0). This test reassembles the
ground-truth XPU core + z from the 4 ranks, applies RMSNormGated + out_proj,
and compares to the captured attn_out. If it matches -> post-kernel is correct
and the full-forward mismatch was a CPU-core scale artifact. If it mismatches
-> RMSNormGated or out_proj is the bug.
"""
import glob, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
LAYER = 0
EPS = 1e-6
NUM_V, HV, TP = 48, 128, 4
V_PER_RANK = NUM_V // TP  # 12

def load(name):
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()

def main():
    hc = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    attn_out_cap = hc["attn_out"].float()  # (T, 2560)
    T = attn_out_cap.shape[0]

    # Reassemble full core + z from 4 ranks (rank r holds v-heads [12r,12r+12))
    core_full = torch.zeros(T, NUM_V, HV)
    z_full = torch.zeros(T, NUM_V, HV)
    for r in range(TP):
        d = torch.load(f"/tmp/qwen4exp_gdn_capture_L0_R{r}.pt", map_location="cpu")
        core_full[:, r*V_PER_RANK:(r+1)*V_PER_RANK] = d["core_attn_out"].float()
        z_full[:, r*V_PER_RANK:(r+1)*V_PER_RANK] = d["z"].float()
    print(f"reassembled core {tuple(core_full.shape)} absmax {core_full.abs().max().item():.5f}")
    print(f"reassembled z    {tuple(z_full.shape)} absmax {z_full.abs().max().item():.5f}")

    base = f"model.language_model.layers.{LAYER}.linear_attn"
    norm_w = load(f"{base}.norm.weight").float()   # (128,)
    out_w = load(f"{base}.out_proj.weight").float()  # (2560, 6144)
    print(f"norm_w {tuple(norm_w.shape)} absmax {norm_w.abs().max().item():.4f} mean {norm_w.mean().item():.4f}")
    print(f"out_w  {tuple(out_w.shape)} absmax {out_w.abs().max().item():.4f}")

    # RMSNormGated: out = (core / rms(core)) * weight * sigmoid(z), norm over last dim
    core2 = core_full.reshape(-1, HV)
    z2 = z_full.reshape(-1, HV)
    var = core2.pow(2).mean(dim=-1, keepdim=True)
    xn = core2 * torch.rsqrt(var + EPS)
    zgated = xn * norm_w * torch.sigmoid(z2)
    zgated = zgated.reshape(T, NUM_V * HV)
    out = zgated @ out_w.T  # (T, 2560)

    print("\n=== POST-KERNEL (ground-truth XPU core -> RMSNormGated -> out_proj) vs attn_out ===")
    print(f"cosine = {cos(attn_out_cap, out):.6f}")
    print(f"relerr = {((attn_out_cap-out).norm()/(out.norm()+1e-9)).item():.4f}")
    print(f"max|cap-ref| = {(attn_out_cap-out).abs().max().item():.5f}")
    print(f"absmax: cap {attn_out_cap.abs().max().item():.4f} ref {out.abs().max().item():.4f}")
    print(f"norm:   cap {attn_out_cap.norm().item():.4f} ref {out.norm().item():.4f}")

    # Also: what if the gate is silu instead of sigmoid?
    zgated_silu = xn * norm_w * F.silu(z2)
    out_silu = zgated_silu.reshape(T, NUM_V*HV) @ out_w.T
    print(f"\n[gate=silu] cosine = {cos(attn_out_cap, out_silu):.6f}")

    # What if norm is applied AFTER gate (norm_before_gate=False)?
    xg = core2 * torch.sigmoid(z2)
    var2 = xg.pow(2).mean(dim=-1, keepdim=True)
    xn2 = xg * torch.rsqrt(var2 + EPS) * norm_w
    out_nb = xn2.reshape(T, NUM_V*HV) @ out_w.T
    print(f"[norm_after_gate] cosine = {cos(attn_out_cap, out_nb):.6f}")

if __name__ == "__main__":
    main()