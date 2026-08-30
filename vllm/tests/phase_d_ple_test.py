#!/usr/bin/env python3
"""
Non-circular PLE compute test.

Recomputes the PLE forward (key/value proj, norms, gate, dilated conv) from:
  - captured emb (gathered n-gram embedding, (5, 2560))
  - captured resid_in (wide state entering PLE, (5, 4, 2560))
  - checkpoint weights (key_proj, value_proj, conv1d, norm_key/query/conv)
and compares to vLLM's captured key/value/s/gate/gated/conv_out/out.

This tests the PLE compute (NOT the n-gram table gather, which uses the
102GB FP8 table). If it matches, PLE compute is exonerated.

PLE config: ngram=3, heads_per_ngram=8, n_heads=16, head_dim=160,
kern=4, dil=3, hist=9, eos=248044, hc=4, n_embd=2560, hc_dim=10240.
"""
import math
import torch
import torch.nn.functional as F
from safetensors import safe_open

HC = 4
N_EMBD = 2560
HC_DIM = HC * N_EMBD
EPS = 1e-6
KERN = 4
DIL = 3
HIST = (KERN - 1) * DIL  # 9
CKPT1 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16/model-00001.safetensors"
CKPT17 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16/model-00017.safetensors"
PREFIX = "model.language_model.layers.1.ple"


def load_ple_weights():
    with safe_open(CKPT1, framework="pt") as f:
        key_proj = f.get_tensor(f"{PREFIX}.key_proj.weight").float()  # (10240, 2560)
        norm_key = f.get_tensor(f"{PREFIX}.norm_key.weight").float()  # (10240,)
        norm_query = f.get_tensor(f"{PREFIX}.norm_query.weight").float()  # (10240,)
        norm_conv = f.get_tensor(f"{PREFIX}.norm_conv.weight").float()  # (10240,)
    with safe_open(CKPT17, framework="pt") as f:
        value_proj = f.get_tensor(f"{PREFIX}.value_proj.weight").float()  # (2560, 2560)
        conv1d = f.get_tensor(f"{PREFIX}.conv1d.weight").float()  # (10240, 1, 4)
    return key_proj, value_proj, conv1d, norm_key, norm_query, norm_conv


def grouped_norm(x, w, eps):
    # x: [T, hc, n_embd]; w: [hc_dim]. Per-stream RMSNorm + affine.
    T = x.shape[0]
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps).to(x.dtype)
    xn_flat = xn.reshape(T, HC_DIM) * w
    return xn_flat.reshape(T, HC, N_EMBD)


def main():
    key_proj, value_proj, conv1d, norm_key, norm_query, norm_conv = load_ple_weights()
    print(f"key_proj: {tuple(key_proj.shape)}, value_proj: {tuple(value_proj.shape)}, conv1d: {tuple(conv1d.shape)}")

    cap = torch.load("/tmp/qwen4exp_ple_capture_rank0.pt", map_location="cpu")
    resid_in = cap["resid_in"].float()   # (5, 4, 2560)
    emb = cap["emb"].float()             # (5, 2560)
    key_vllm = cap["key"].float()        # (5, 4, 2560)
    value_vllm = cap["value"].float()    # (5, 2560)
    s_vllm = cap["s"].float()            # (5, 4)
    gate_vllm = cap["gate"].float()      # (5, 4)
    gated_vllm = cap["gated"].float()    # (5, 4, 2560)
    conv_out_vllm = cap["conv_out"].float()  # (5, 10240)
    out_vllm = cap["out"].float()        # (5, 4, 2560)
    has_init = cap["has_init"]           # (1,)
    T = resid_in.shape[0]
    print(f"T={T}, has_init={has_init.tolist()}")

    def cos(a, b):
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    # --- Recompute PLE compute ---
    # key = (emb @ key_proj.T).reshape(T, hc, n_embd)
    key = (emb @ key_proj.T).reshape(T, HC, N_EMBD)
    # value = emb @ value_proj.T
    value = emb @ value_proj.T
    # key = grouped_norm(key, norm_key)
    key = grouped_norm(key, norm_key, EPS)
    # query = grouped_norm(resid_in, norm_query)
    query = grouped_norm(resid_in, norm_query, EPS)
    # s = (key * query).sum(-1) / sqrt(n_embd)
    s = (key * query).sum(dim=-1) / math.sqrt(N_EMBD)
    # gate = sigmoid(sign(s) * sqrt(clamp(|s|, min=1e-6)))
    gate = torch.sigmoid(torch.sign(s) * torch.sqrt(torch.clamp(s.abs(), min=1e-6)))
    # gated = value.unsqueeze(1) * gate.unsqueeze(-1)
    gated = value.unsqueeze(1) * gate.unsqueeze(-1)
    # normalized = grouped_norm(gated, norm_conv)
    normalized = grouped_norm(gated, norm_conv, EPS)
    # conv_out = dilated_conv(normalized)  [fresh seq: zero state]
    x = normalized.reshape(T, HC_DIM)
    w = conv1d.squeeze(1)  # (10240, 4)
    conv_out = torch.zeros_like(x)
    # Fresh sequence (has_init=False): state starts zero
    state = torch.zeros(HIST, HC_DIM)
    padded = torch.cat([state, x], dim=0)  # (HIST+T, HC_DIM)
    acc = torch.zeros(T, HC_DIM, dtype=torch.float32)
    for k in range(KERN):
        offset = HIST - (KERN - 1 - k) * DIL
        acc = acc + padded[offset:offset + T].float() * w[:, k].float()
    conv_out = F.silu(acc)

    # --- Compare ---
    print(f"\n=== PLE COMPUTE ===")
    print(f"  key:      cos={cos(key, key_vllm):.6f} max_diff={(key-key_vllm).abs().max().item():.6f}")
    print(f"  value:    cos={cos(value, value_vllm):.6f} max_diff={(value-value_vllm).abs().max().item():.6f}")
    print(f"  s:        cos={cos(s, s_vllm):.6f} max_diff={(s-s_vllm).abs().max().item():.6f}")
    print(f"  gate:     cos={cos(gate, gate_vllm):.6f} max_diff={(gate-gate_vllm).abs().max().item():.6f}")
    print(f"  gated:    cos={cos(gated, gated_vllm):.6f} max_diff={(gated-gated_vllm).abs().max().item():.6f}")
    print(f"  conv_out: cos={cos(conv_out, conv_out_vllm):.6f} max_diff={(conv_out-conv_out_vllm).abs().max().item():.6f}")
    print(f"    conv_out absmax ref={conv_out.abs().max().item():.4f} vllm={conv_out_vllm.abs().max().item():.4f}")

    # out = resid_in + gated + conv_out.reshape(T, hc, n_embd)
    out = resid_in + gated + conv_out.reshape(T, HC, N_EMBD)
    print(f"  out:      cos={cos(out, out_vllm):.6f} max_diff={(out-out_vllm).abs().max().item():.6f}")
    print(f"    out absmax ref={out.abs().max().item():.4f} vllm={out_vllm.abs().max().item():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()