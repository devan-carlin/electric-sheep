#!/usr/bin/env python
"""Comprehensive HC-mix gate diagnostic.

For each combination of (input, down/up source, norm), compute the gate and
mixed, and report cos vs llama.cpp ground truth (hc_gate, hc_mixed) plus the
pre-sigmoid mean. This isolates what drives the gate discrepancy.
"""
import json
import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType, dequantize
from safetensors import safe_open

GGUF = "/mnt/data/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf"
W4A16 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16/model-00001.safetensors"
LLAMA_DIR = "/tmp/llama_actdump_l0"
HC, N_EMBD, HC_DIM = 4, 2560, 10240
EPS = 1e-6

m = json.load(open(f"{LLAMA_DIR}/manifest.json"))


def load_llama(name):
    e = next(x for x in m["tensors"] if x["name"] == name)
    ne = e["ne"]
    d = np.fromfile(e["file"], dtype=np.float32)
    return d.reshape(ne[3], ne[2], ne[1], ne[0])


def cos(a, b):
    a = a.ravel(); b = b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


r = GGUFReader(GGUF)


def gg(name):
    t = next(x for x in r.tensors if x.name == name)
    raw = np.frombuffer(t.data, dtype=np.uint8)
    flat = dequantize(raw, GGMLQuantizationType(t.tensor_type)).astype(np.float32)
    return flat.reshape(t.shape[1], t.shape[0])


g_down = torch.from_numpy(gg("blk.0.hc_attn_down.weight")).clone()
g_up = torch.from_numpy(gg("blk.0.hc_attn_up.weight")).clone()
g_norm = torch.from_numpy(np.frombuffer(next(t for t in r.tensors if t.name == "blk.0.hc_attn_norm.weight").data, dtype=np.float32)).clone()

with safe_open(W4A16, framework="pt") as sf:
    w4_down = torch.from_numpy(sf.get_tensor("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight").float().numpy()).clone()
    w4_up = torch.from_numpy(sf.get_tensor("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight").float().numpy()).clone()
    w4_norm = torch.from_numpy(sf.get_tensor("model.language_model.layers.0.attn_hyper_connection.hc_norm.weight").float().numpy()).clone()

# inputs
llama_init = torch.from_numpy(load_llama("hc_init")[0]).float()  # [5,4,2560]
vllm_resid_in = torch.load("/tmp/qwen4exp_layer00_resid.pt", map_location="cpu")["resid_in"].float()
l_gate = load_llama("hc_gate-0")[0]
l_mixed = load_llama("hc_mixed-0")[0]

print(f"input norms: llama hc_init={np.linalg.norm(llama_init.numpy()):.4f}  vLLM resid_in={np.linalg.norm(vllm_resid_in.numpy()):.4f}")
print(f"cos(llama hc_init, vLLM resid_in) = {cos(llama_init.numpy(), vllm_resid_in.numpy()):.6f}")
print(f"llama hc_gate: mean={l_gate.mean():.4f} norm={np.linalg.norm(l_gate):.4f}")
print()


def compute(x, down, up, gamma):
    T = x.shape[0]
    var = x.pow(2).mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + EPS)
    xn_flat = xn.reshape(T, HC_DIM) * gamma
    lo = torch.nn.functional.silu(torch.nn.functional.linear(xn_flat, down) / HC)
    pre = torch.nn.functional.linear(lo, up)
    gate = torch.sigmoid(pre)
    mixed = (xn_flat * gate).reshape(T, HC, N_EMBD).mean(1)
    return gate, mixed, pre, xn_flat, lo


combos = [
    ("llama_init + GGUF d/u + raw w", llama_init, g_down, g_up, w4_norm),
    ("llama_init + GGUF d/u + 1+w   ", llama_init, g_down, g_up, g_norm),
    ("llama_init + W4A16 d/u + raw w", llama_init, w4_down, w4_up, w4_norm),
    ("llama_init + W4A16 d/u + 1+w   ", llama_init, w4_down, w4_up, 1.0 + w4_norm),
    ("vLLM_resid + W4A16 d/u + raw w", vllm_resid_in, w4_down, w4_up, w4_norm),
]

print(f"{'combo':34s} {'gate_cos':>9s} {'mixed_cos':>9s} {'pre_mean':>9s} {'gate_mean':>9s}")
for label, x, down, up, gamma in combos:
    gate, mixed, pre, xn_flat, lo = compute(x, down, up, gamma)
    print(f"{label:34s} {cos(gate.numpy(), l_gate):9.6f} {cos(mixed.numpy(), l_mixed):9.6f} {pre.mean():9.4f} {gate.mean():9.4f}")

# reference: vLLM actual mix_out
v_mix = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")["mix_out"].float()
print(f"\nvLLM actual mix_out norm={np.linalg.norm(v_mix.numpy()):.4f}")
gate, mixed, pre, xn_flat, lo = compute(vllm_resid_in, w4_down, w4_up, w4_norm)
print(f"recompute(vLLM_resid, W4A16, raw w) mixed vs vLLM actual: cos={cos(mixed.numpy(), v_mix.numpy()):.6f}")