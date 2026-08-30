#!/usr/bin/env python3
"""
Decisive final-path test.

Take vLLM's captured final_hidden (post final-mixer, pre lm_head) and apply
the CHECKPOINT lm_head weight (ground truth). Check if logits are flat.

  - flat logits  -> final_hidden is garbage -> bug is UPSTREAM (embed/PLE/inter-layer)
  - sharp logits -> final_hidden is fine    -> bug is in lm_head itself

Also recompute the final mixer from resid_before + checkpoint weights and
compare to captured final_hidden (tests the final mixer wiring).
"""
import torch
import torch.nn.functional as F
from safetensors import safe_open

HC = 4
N_EMBD = 2560
HC_DIM = HC * N_EMBD
EPS = 1e-6
LMHEAD = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8/model-00131-of-00131.safetensors"


def grouped_rms_norm(x, eps):
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps).to(x.dtype)


def main():
    # Load final capture (rank 0; hidden states are replicated across TP)
    cap = torch.load("/tmp/qwen4exp_final_capture_R0.pt", map_location="cpu")
    resid_before = cap["resid_before"].float()   # (5, 4, 2560)
    final_hidden = cap["final_hidden"].float()   # (5, 2560)
    hc_norm = cap["hc_norm"].float()
    mix_down = cap["mix_down"].float()
    mix_up = cap["mix_up"].float()
    T = resid_before.shape[0]
    print(f"T={T}, resid_before={tuple(resid_before.shape)}, final_hidden={tuple(final_hidden.shape)}")

    # --- Recompute final mixer (has_inject=False) ---
    xn = grouped_rms_norm(resid_before, EPS)
    xn_flat = xn.reshape(T, HC_DIM) * hc_norm
    lo = F.silu(F.linear(xn_flat, mix_down) / HC)
    gate = torch.sigmoid(F.linear(lo, mix_up))
    mixed = (xn_flat * gate).reshape(T, HC, N_EMBD).mean(dim=1)
    cos_mix = F.cosine_similarity(mixed.flatten(), final_hidden.flatten(), dim=0).item()
    print(f"\n=== FINAL MIXER ===")
    print(f"  mixed vs final_hidden cos: {cos_mix:.6f}")
    print(f"  mixed absmax={mixed.abs().max().item():.4f} final_hidden absmax={final_hidden.abs().max().item():.4f}")

    # --- Apply checkpoint lm_head to final_hidden ---
    with safe_open(LMHEAD, framework="pt") as f:
        lm_w = f.get_tensor("lm_head.weight").float()  # (248320, 2560)
    print(f"\nlm_head.weight: {tuple(lm_w.shape)}")

    # logits = final_hidden @ lm_w.T  -> (T, 248320)
    logits = final_hidden @ lm_w.T  # (5, 248320)
    print(f"\n=== LM_HEAD (checkpoint) on captured final_hidden ===")
    print(f"  logits shape: {tuple(logits.shape)}")
    print(f"  logits absmax: {logits.abs().max().item():.4f}")
    print(f"  logits mean: {logits.mean().item():.4f}")

    # Per-token: top-5 + entropy
    import math
    for t in range(T):
        lg = logits[t]
        logp = F.log_softmax(lg, dim=-1)
        topk = torch.topk(lg, 5)
        ent = -(logp.exp() * logp).sum().item()
        max_ent = math.log(lg.shape[0])
        print(f"\n  token {t}: entropy={ent:.4f} ({100*ent/max_ent:.1f}% of uniform {max_ent:.4f})")
        for i in range(5):
            print(f"    logit={topk.values[i].item():8.3f}  id={topk.indices[i].item()}")

    # Also: apply lm_head to the RECOMPUTED mixed (in case final_hidden capture is stale)
    logits2 = mixed @ lm_w.T
    print(f"\n=== LM_HEAD on recomputed mixed ===")
    for t in range(T):
        lg = logits2[t]
        logp = F.log_softmax(lg, dim=-1)
        ent = -(logp.exp() * logp).sum().item()
        max_ent = math.log(lg.shape[0])
        topk = torch.topk(lg, 5)
        print(f"  token {t}: entropy={ent:.4f} ({100*ent/max_ent:.1f}%) top5_ids={[int(x) for x in topk.indices.tolist()]}")

    print("\nDone.")


if __name__ == "__main__":
    main()