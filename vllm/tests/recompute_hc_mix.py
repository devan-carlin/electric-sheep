#!/usr/bin/env python
"""Recompute the layer-0 attention HC mix from checkpoint weights + the matching
input, and compare each intermediate to llama.cpp ground truth.

This breaks the circularity: instead of comparing vLLM's mix to a recompute on
vLLM's own input (self-consistent), we compare the recompute to llama.cpp's
actual hc_mixed/hc_gate/hc_inject (the working baseline).

vLLM mix math (verified identical to llama.cpp build_hc_mix):
    xn      = grouped_rms_norm(x)          # per-stream over n_embd
    xn_flat = xn.reshape(T, hc_dim) * hc_norm
    lo      = silu(linear(xn_flat, down) / hc)
    gate    = sigmoid(linear(lo, up))      # [T, hc_dim]
    mixed   = (xn_flat * gate).reshape(T, hc, n_embd).mean(1)  # [T, n_embd]
    inject  = linear(xn_flat, block_inject_weight)             # [T, hc]
"""
import json
import numpy as np
import torch
from safetensors import safe_open

W4A16 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16/model-00001.safetensors"
LLAMA_DIR = "/tmp/llama_actdump_l0"
HC = 4
N_EMBD = 2560
HC_DIM = HC * N_EMBD  # 10240
LOWRANK = 320
EPS = 1e-6


def load_llama(name, manifest):
    entry = next((e for e in manifest["tensors"] if e["name"] == name), None)
    if entry is None:
        return None
    ne = entry["ne"]
    with open(entry["file"], "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(ne[3], ne[2], ne[1], ne[0])


def cos(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main():
    m = json.load(open(f"{LLAMA_DIR}/manifest.json"))

    # checkpoint HC weights (bf16)
    with safe_open(W4A16, framework="pt") as sf:
        hc_norm = sf.get_tensor("model.language_model.layers.0.attn_hyper_connection.hc_norm.weight").float()
        down = sf.get_tensor("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight").float()
        up = sf.get_tensor("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight").float()
        inject_w = sf.get_tensor("model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight").float()
    print(f"weights: hc_norm={tuple(hc_norm.shape)} down={tuple(down.shape)} up={tuple(up.shape)} inject={tuple(inject_w.shape)}")

    # input: vLLM resid_in [T, hc, n_embd] (matches llama hc_init at cos 0.999983)
    resid = torch.load("/tmp/qwen4exp_layer00_resid.pt", map_location="cpu")
    x = resid["resid_in"].float()  # [5, 4, 2560]
    T = x.shape[0]

    # --- recompute (vLLM math) ---
    var = x.pow(2).mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + EPS)
    xn_flat = xn.reshape(T, HC_DIM) * hc_norm
    lo = torch.nn.functional.linear(xn_flat, down)
    lo = torch.nn.functional.silu(lo / HC)
    gate = torch.sigmoid(torch.nn.functional.linear(lo, up))  # [T, HC_DIM]
    gated = (xn_flat * gate).reshape(T, HC, N_EMBD)
    mixed = gated.mean(dim=1)  # [T, N_EMBD]
    inject = torch.nn.functional.linear(xn_flat, inject_w)  # [T, HC]

    # --- llama.cpp ground truth ---
    l_gate = load_llama("hc_gate-0", m)[0]    # [5, 10240]
    l_mixed = load_llama("hc_mixed-0", m)[0]  # [5, 2560]
    l_inject = load_llama("hc_inject-0", m)[0]  # [5, 4]

    print("\n=== HC mix recompute (checkpoint weights + matching input) vs llama.cpp ===\n")
    print(f"gate   cos={cos(gate.numpy(), l_gate):.6f}   (recompute [5,10240] vs llama hc_gate-0)")
    print(f"mixed  cos={cos(mixed.numpy(), l_mixed):.6f}   (recompute [5,2560] vs llama hc_mixed-0)")
    print(f"inject cos={cos(inject.numpy(), l_inject):.6f}   (recompute [5,4] vs llama hc_inject-0)")

    # also: what does vLLM's ACTUAL mix_out look like vs llama?
    hc = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    v_mix = hc["mix_out"].float().numpy()
    print(f"\nvLLM actual mix_out vs llama hc_mixed-0: cos={cos(v_mix, l_mixed):.6f}")
    print(f"recompute mixed vs vLLM actual mix_out:  cos={cos(mixed.numpy(), v_mix):.6f}")

    # norms to catch scale issues
    print(f"\nnorms (mixed): recompute={np.linalg.norm(mixed.numpy()):.4f} llama={np.linalg.norm(l_mixed):.4f} vllm={np.linalg.norm(v_mix):.4f}")
    print(f"norms (gate):  recompute={np.linalg.norm(gate.numpy()):.4f} llama={np.linalg.norm(l_gate):.4f}")


if __name__ == "__main__":
    main()