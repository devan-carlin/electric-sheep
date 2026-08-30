"""Phase C: extend the per-layer CPU reference through the final mixer + lm_head.

The per-layer reference (phase_c_perlayer_ref.py) matches the captured vLLM run
at all 48 layers (cos > 0.997) but stops at the wide residual. This script runs
the SAME 48-layer reference, then applies the final hyper-connection mixer and
lm_head, and prints the predicted token + entropy.

Decisive test:
  - predicts "Paris"  -> reference is correct; vLLM diverges somewhere uncaught.
  - flat / soup       -> reference shares vLLM's bug (pure Python -> findable).
"""
import glob, math, os, torch
import torch.nn.functional as F
from safetensors import safe_open

# Reuse the exact reference implementation.
import phase_c_perlayer_ref as R

W4 = R.W4
HC, N_EMBD = R.HC, R.N_EMBD
HC_DIM = HC * N_EMBD
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

def final_mixer(resid):
    base = "model.language_model.hyper_connection_mixer"
    hc_norm = load(f"{base}.hc_norm.weight").float()
    down = load(f"{base}.input_mix_weight_down.weight").float()
    up = load(f"{base}.input_mix_weight_up.weight").float()
    T = resid.shape[0]
    var = resid.pow(2).mean(dim=-1, keepdim=True)
    xn = resid * torch.rsqrt(var + EPS)
    xn_flat = xn.reshape(T, HC_DIM) * hc_norm
    lo = F.linear(xn_flat, down)
    lo = F.silu(lo / HC)
    gate = torch.sigmoid(F.linear(lo, up))
    gated = (xn_flat * gate).reshape(T, HC, N_EMBD)
    return gated.mean(dim=1)

def main():
    emb = torch.load("/tmp/qwen4exp_embed_capture.pt", map_location="cpu")
    embed = emb["embed"].float()
    T = embed.shape[0]
    positions = torch.arange(T, dtype=torch.long)
    inv_freq = 1.0 / (R.BASE ** (torch.arange(0, R.ROT_DIM, 2, dtype=torch.float) / R.ROT_DIM))
    t = torch.arange(262144, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_cache, sin_cache = freqs.cos(), freqs.sin()
    ple_cap = torch.load("/tmp/qwen4exp_ple_capture_rank0.pt", map_location="cpu")
    ple_rows = ple_cap["rows"]

    resid = embed.unsqueeze(1).expand(-1, HC, -1).contiguous()
    print(f"T={T} running 48-layer CPU reference + final mixer + lm_head\n")
    for L in range(48):
        ltype = R.LAYER_TYPES[L]
        base = f"model.language_model.layers.{L}"
        if L in R.PLE_LAYERS and os.environ.get("PLE_SKIP") != "1":
            resid = R.ple_forward(resid, ple_rows, f"{base}.ple")
        a_base = f"{base}.attn_hyper_connection"
        cur, inject = R.hc_mix(resid, load(f"{a_base}.hc_norm.weight").float(),
                               load(f"{a_base}.input_mix_weight_down.weight").float(),
                               load(f"{a_base}.input_mix_weight_up.weight").float(),
                               load(f"{a_base}.block_inject_weight.weight").float())
        if ltype == "linear_attention":
            attn_out = R.pure_python_gdn(cur, f"{base}.linear_attn")
        else:
            attn_out = R.full_attention(cur, f"{base}.self_attn", positions, cos_cache, sin_cache)
        resid = R.hc_combine(resid, attn_out, inject)
        m_base = f"{base}.mlp_hyper_connection"
        cur2, inject2 = R.hc_mix(resid, load(f"{m_base}.hc_norm.weight").float(),
                                 load(f"{m_base}.input_mix_weight_down.weight").float(),
                                 load(f"{m_base}.input_mix_weight_up.weight").float(),
                                 load(f"{m_base}.block_inject_weight.weight").float())
        mlp_out = R.moe(cur2, f"{base}.mlp")
        resid = R.hc_combine(resid, mlp_out, inject2)
        if (L + 1) % 8 == 0:
            print(f"  layer {L} done")

    # compare final resid to captured
    cap47 = torch.load("/tmp/qwen4exp_layer47_resid.pt", map_location="cpu")["resid_out"].float()
    print(f"\nfinal resid cos(ref, cap47) = {cos(resid, cap47):.6f}")

    final_ref = final_mixer(resid)
    print(f"final_hidden norm {final_ref.norm().item():.4f} absmax {final_ref.abs().max().item():.4f}")

    # compare to captured final_hidden (from the buggy run)
    try:
        d = torch.load("/tmp/qwen4exp_final_capture_R0.pt", map_location="cpu")
        final_cap = d["final_hidden"].float()
        print(f"final_hidden cos(ref, cap) = {cos(final_ref, final_cap):.6f}")
    except Exception as e:
        print(f"(no captured final_hidden: {e})")

    lm_head = load("lm_head.weight").float()
    h = final_ref[-1]
    logits = h @ lm_head.T
    logprobs = F.log_softmax(logits, dim=-1)
    topv, topi = torch.topk(logprobs, 8)
    p = torch.softmax(logits, dim=-1)
    entropy = -(p * logprobs).sum().item()
    print(f"\n=== CPU reference logits (last token) top-8 ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(W4, trust_remote_code=True)
    for i in range(8):
        print(f"  {topv[i].item():.4f}  id={topi[i].item():>7}  {tok.decode([topi[i].item()])!r}")
    print(f"\nlogits absmax {logits.abs().max().item():.4f} std {logits.std().item():.4f}")
    print(f"entropy {entropy:.4f} nats (uniform = {math.log(248320):.4f})")

if __name__ == "__main__":
    main()