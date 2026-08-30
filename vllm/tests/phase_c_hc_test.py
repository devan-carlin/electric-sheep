"""Phase C: CPU reference for the HC mix/combine at layer 0.

Loads the captured HC I/O (resid_in, mix_out, inject, attn_out, resid_after_attn)
and recomputes mix + combine on CPU from the W4A16 checkpoint HC weights:
  mix: grouped-RMSNorm(per stream) -> *hc_norm -> down -> silu(x/hc) -> up ->
       sigmoid gate -> xn*gate -> mean over hc streams; inject = xn @ inject_w.T
  combine: residual + block_out * 2*sigmoid(inject/hc)
Compares to the captured mix_out / inject / resid_after_attn.
"""
import glob, torch
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
LAYER = 0
HC = 4
N_EMBD = 2560
HC_DIM = HC * N_EMBD
LOWRANK = 320
EPS = 1e-6

def load(name):
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def main():
    cap = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    resid_in = cap["resid_in"].float()          # (T, hc, n_embd)
    mix_cap = cap["mix_out"].float()            # (T, n_embd)
    inject_cap = cap["inject"].float()          # (T, hc)
    attn_out = cap["attn_out"].float()          # (T, n_embd)
    resid_after_cap = cap["resid_after_attn"].float()  # (T, hc, n_embd)
    T = resid_in.shape[0]
    print(f"T={T} resid_in {tuple(resid_in.shape)}")

    base = f"model.language_model.layers.{LAYER}.attn_hyper_connection"
    hc_norm = load(f"{base}.hc_norm.weight").float()                 # (hc_dim,)
    w_down = load(f"{base}.input_mix_weight_down.weight").float()    # (lowrank, hc_dim)
    w_up = load(f"{base}.input_mix_weight_up.weight").float()        # (hc_dim, lowrank)
    w_inject = load(f"{base}.block_inject_weight.weight").float()    # (hc, hc_dim)

    # ---- mix ----
    # grouped RMSNorm: normalize each [n_embd] stream independently
    var = resid_in.pow(2).mean(dim=-1, keepdim=True)
    xn = resid_in * torch.rsqrt(var + EPS)
    xn_flat = xn.reshape(T, HC_DIM)
    xn_flat = xn_flat * hc_norm
    lo = xn_flat @ w_down.T                       # (T, lowrank)
    lo = torch.nn.functional.silu(lo / HC)
    gate = torch.sigmoid(lo @ w_up.T)             # (T, hc_dim)
    gated = (xn_flat * gate).reshape(T, HC, N_EMBD)
    mix_ref = gated.mean(dim=1)                   # (T, n_embd)
    inject_ref = xn_flat @ w_inject.T             # (T, hc)

    # ---- combine ----
    w = 2.0 * torch.sigmoid(inject_ref / HC)      # (T, hc)
    combine_ref = resid_in + attn_out.unsqueeze(1) * w.unsqueeze(-1)

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    def relerr(a, b):
        return ((a - b).norm() / (b.norm() + 1e-9)).item()

    print("\n=== MIX ===")
    print(f"cosine(mix_cap, mix_ref)   = {cos(mix_cap, mix_ref):.6f}")
    print(f"relerr(mix_cap, mix_ref)   = {relerr(mix_cap, mix_ref):.4f}")
    print(f"max|mix_cap - mix_ref|     = {(mix_cap-mix_ref).abs().max().item():.5f}")
    print(f"absmax: cap {mix_cap.abs().max().item():.4f}  ref {mix_ref.abs().max().item():.4f}")
    print("\n=== INJECT ===")
    print(f"cosine(inj_cap, inj_ref)   = {cos(inject_cap, inject_ref):.6f}")
    print(f"relerr(inj_cap, inj_ref)   = {relerr(inject_cap, inject_ref):.4f}")
    print(f"max|inj_cap - inj_ref|     = {(inject_cap-inject_ref).abs().max().item():.5f}")
    print(f"absmax: cap {inject_cap.abs().max().item():.4f}  ref {inject_ref.abs().max().item():.4f}")
    print("\n=== COMBINE (resid_after_attn) ===")
    print(f"cosine(cap, combine_ref)   = {cos(resid_after_cap, combine_ref):.6f}")
    print(f"relerr(cap, combine_ref)   = {relerr(resid_after_cap, combine_ref):.4f}")
    print(f"max|cap - combine_ref|     = {(resid_after_cap-combine_ref).abs().max().item():.5f}")
    print(f"norm: cap {resid_after_cap.norm().item():.4f}  ref {combine_ref.norm().item():.4f}")

if __name__ == "__main__":
    main()