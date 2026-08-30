#!/usr/bin/env python3
"""
Phase D: HC (hyper-connection) mix/combine test (non-circular).

Recomputes the HC mix + inject + combine in pure PyTorch from:
  - captured resid_in (vLLM's actual wide state entering the layer)
  - captured attn_out (vLLM's actual attention output)
  - checkpoint weights (hc_norm, input_mix_weight_down/up, block_inject_weight)
and compares to vLLM's captured mix_out / inject / resid_after_attn.

HC is pure PyTorch in vLLM (no custom kernel), so this directly tests the
wiring. If it matches, HC is exonerated.

HC mix (from qwen4_exp.py HyperConnection.mix):
  xn = grouped_rms_norm(resid_in)          # per-stream over n_embd
  xn_flat = xn.reshape(T, hc_dim) * hc_norm
  lo = silu(linear(xn_flat, down) / hc)
  gate = sigmoid(linear(lo, up))
  mixed = (xn_flat * gate).reshape(T, hc, n_embd).mean(dim=1)
  inject = linear(xn_flat, block_inject_weight)

HC combine (from HyperConnection.combine):
  w = 2 * sigmoid(inject / hc)
  resid_after = resid_in + attn_out.unsqueeze(1) * w.unsqueeze(-1)
"""
import os
import torch
import torch.nn.functional as F
from safetensors import safe_open

HC = 4
N_EMBD = 2560
HC_DIM = HC * N_EMBD  # 10240
LOWRANK = 320
EPS = 1e-6  # rms_norm_eps (verify)
CKPT = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8/model-00001-of-00131.safetensors"
PREFIX = "model.language_model.layers.0.attn_hyper_connection"


def load_hc_weights():
    with safe_open(CKPT, framework="pt") as f:
        hc_norm = f.get_tensor(f"{PREFIX}.hc_norm.weight").float()
        down = f.get_tensor(f"{PREFIX}.input_mix_weight_down.weight").float()
        up = f.get_tensor(f"{PREFIX}.input_mix_weight_up.weight").float()
        inject_w = f.get_tensor(f"{PREFIX}.block_inject_weight.weight").float()
    return hc_norm, down, up, inject_w


def grouped_rms_norm(x, eps):
    # x: [T, hc, n_embd]; normalize each stream over n_embd
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps).to(x.dtype)


def main():
    hc_norm, down, up, inject_w = load_hc_weights()
    print(f"hc_norm: {hc_norm.shape}, down: {down.shape}, up: {up.shape}, inject_w: {inject_w.shape}")

    cap = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    resid_in = cap["resid_in"].float()      # (T, 4, 2560)
    mix_out_vllm = cap["mix_out"].float()   # (T, 2560)
    inject_vllm = cap["inject"].float()     # (T, 4)
    attn_out = cap["attn_out"].float()      # (T, 2560)
    resid_after_vllm = cap["resid_after_attn"].float()  # (T, 4, 2560)
    T = resid_in.shape[0]
    print(f"T={T}, resid_in={tuple(resid_in.shape)}")

    # --- Recompute mix ---
    xn = grouped_rms_norm(resid_in, EPS)
    xn_flat = xn.reshape(T, HC_DIM) * hc_norm
    lo = F.linear(xn_flat, down)
    lo = F.silu(lo / HC)
    gate = torch.sigmoid(F.linear(lo, up))
    mixed = (xn_flat * gate).reshape(T, HC, N_EMBD).mean(dim=1)
    inject = F.linear(xn_flat, inject_w)

    # --- Recompute combine ---
    w = 2.0 * torch.sigmoid(inject / HC)  # (T, 4)
    w = w.unsqueeze(-1)                    # (T, 4, 1)
    b = attn_out.unsqueeze(1)              # (T, 1, 2560)
    resid_after = resid_in + b * w

    # --- Compare ---
    def cos(a, b):
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    print(f"\n=== MIX ===")
    print(f"  mix_out cos: {cos(mixed, mix_out_vllm):.6f}")
    print(f"  mix_out max_abs_diff: {(mixed - mix_out_vllm).abs().max().item():.6f}")
    print(f"  mix_out absmax ref={mixed.abs().max().item():.4f} vllm={mix_out_vllm.abs().max().item():.4f}")
    print(f"  inject cos: {cos(inject, inject_vllm):.6f}")
    print(f"  inject max_abs_diff: {(inject - inject_vllm).abs().max().item():.6f}")
    print(f"  inject ref={inject[0].tolist()}")
    print(f"  inject vllm={inject_vllm[0].tolist()}")

    print(f"\n=== COMBINE ===")
    print(f"  resid_after cos: {cos(resid_after, resid_after_vllm):.6f}")
    print(f"  resid_after max_abs_diff: {(resid_after - resid_after_vllm).abs().max().item():.6f}")
    print(f"  resid_after absmax ref={resid_after.abs().max().item():.4f} vllm={resid_after_vllm.abs().max().item():.4f}")

    # Per-stream cos for resid_after
    for s in range(HC):
        c = cos(resid_after[:, s, :], resid_after_vllm[:, s, :])
        print(f"  stream {s}: cos={c:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()