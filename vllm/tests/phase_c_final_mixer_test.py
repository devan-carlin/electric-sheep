"""Phase C: verify the top-level final mixer (hyper_connection_mixer).

The final mixer is the final output norm: resid_before [T,hc,n_embd] ->
grouped-RMSNorm -> hc_norm -> down -> silu(x/hc) -> up -> sigmoid gate ->
x*gate -> mean over hc streams -> final_hidden [T,n_embd].

Compares the captured final_hidden against a CPU recomputation using the
CHECKPOINT weights (also confirms the weights loaded correctly).
"""
import glob, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
RANK = 0

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
    d = torch.load(f"/tmp/qwen4exp_final_capture_R{RANK}.pt", map_location="cpu")
    resid = d["resid_before"].float()      # (T, hc, n_embd)
    final_cap = d["final_hidden"].float()  # (T, n_embd)
    T, hc, n_embd = resid.shape
    print(f"resid {tuple(resid.shape)} final {tuple(final_cap.shape)}")

    base = "model.language_model.hyper_connection_mixer"
    hc_norm = load(f"{base}.hc_norm.weight").float()        # (hc_dim,)
    down = load(f"{base}.input_mix_weight_down.weight").float()  # (lowrank, hc_dim)
    up = load(f"{base}.input_mix_weight_up.weight").float()      # (hc_dim, lowrank)
    print(f"hc_norm {tuple(hc_norm.shape)} down {tuple(down.shape)} up {tuple(up.shape)}")

    # Compare captured (loaded) weights vs checkpoint weights
    print(f"\n[weight load check] hc_norm cos={cos(d['hc_norm'], hc_norm):.6f} "
          f"down cos={cos(d['mix_down'], down):.6f} up cos={cos(d['mix_up'], up):.6f}")

    hc_dim = hc * n_embd
    eps = d["eps"]
    # grouped RMSNorm per stream (over n_embd)
    var = resid.pow(2).mean(dim=-1, keepdim=True)
    xn = resid * torch.rsqrt(var + eps)
    xn_flat = xn.reshape(T, hc_dim) * hc_norm
    lo = F.linear(xn_flat, down)
    lo = F.silu(lo / hc)
    gate = torch.sigmoid(F.linear(lo, up))
    gated = (xn_flat * gate).reshape(T, hc, n_embd)
    final_ref = gated.mean(dim=1)

    print("\n=== FINAL MIXER (captured final_hidden vs CPU ref from checkpoint weights) ===")
    print(f"cosine = {cos(final_cap, final_ref):.6f}")
    print(f"relerr = {((final_cap-final_ref).norm()/(final_ref.norm()+1e-9)).item():.4f}")
    print(f"max|cap-ref| = {(final_cap-final_ref).abs().max().item():.5f}")
    print(f"absmax: cap {final_cap.abs().max().item():.4f} ref {final_ref.abs().max().item():.4f}")
    print(f"norm:   cap {final_cap.norm().item():.4f} ref {final_ref.norm().item():.4f}")

if __name__ == "__main__":
    main()