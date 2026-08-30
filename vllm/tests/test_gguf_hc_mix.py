#!/usr/bin/env python
"""Decisive test: recompute the layer-0 attention HC mix from the EXACT GGUF
weights (dequantized down/up + 1+w norm + inject) + llama's own hc_init, and
compare to llama.cpp's captured hc_mixed / hc_gate / hc_inject.

If this matches at ~0.999, then:
  - the ONLY weight difference is norm = 1 + w (GGUF folds the 1 in)
  - vLLM's bug is using raw w instead of (1 + w)
  - the residual gap in the W4A16 recompute was quant noise + sigmoid saturation
"""
import json
import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType, dequantize

GGUF_SHARD = "/mnt/data/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf"
LLAMA_DIR = "/tmp/llama_actdump_l0"
HC, N_EMBD, HC_DIM, LOWRANK = 4, 2560, 10240, 320

# eps: check config
import json as _json
cfg = _json.load(open("/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16/config.json"))
# find rms_norm_eps (may be nested)
def find_key(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                return v
            r = find_key(v, key)
            if r is not None:
                return r
    return None
EPS = find_key(cfg, "rms_norm_eps")
print(f"config rms_norm_eps = {EPS}")
if EPS is None:
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


r = GGUFReader(GGUF_SHARD)


def gg(name):
    t = next(x for x in r.tensors if x.name == name)
    raw = np.frombuffer(t.data, dtype=np.uint8)
    flat = dequantize(raw, GGMLQuantizationType(t.tensor_type)).astype(np.float32)
    return flat.reshape(t.shape[1], t.shape[0])  # [ne1, ne0] = [out, in]


# GGUF weights (exact, as llama.cpp uses them)
g_down = gg("blk.0.hc_attn_down.weight")      # [320, 10240]
g_up = gg("blk.0.hc_attn_up.weight")          # [10240, 320]
g_norm = np.frombuffer(next(x for x in r.tensors if x.name == "blk.0.hc_attn_norm.weight").data, dtype=np.float32)  # [10240] = 1+w
g_inject = gg("blk.0.hc_attn_inject.weight")  # [4, 10240]

# ground-truth input: llama's own hc_init [5,4,2560]
x = torch.from_numpy(load_llama("hc_init")[0]).float()
T = x.shape[0]
l_mixed = load_llama("hc_mixed-0")[0]
l_gate = load_llama("hc_gate-0")[0]
l_inject = load_llama("hc_inject-0")[0]

down = torch.from_numpy(g_down)
up = torch.from_numpy(g_up)
gamma = torch.from_numpy(g_norm)
inject_w = torch.from_numpy(g_inject)

# recompute (vLLM math) with GGUF weights
var = x.pow(2).mean(dim=-1, keepdim=True)
xn = x * torch.rsqrt(var + EPS)
xn_flat = xn.reshape(T, HC_DIM) * gamma
lo = torch.nn.functional.silu(torch.nn.functional.linear(xn_flat, down) / HC)
gate = torch.sigmoid(torch.nn.functional.linear(lo, up))
mixed = (xn_flat * gate).reshape(T, HC, N_EMBD).mean(1)
inject = torch.nn.functional.linear(xn_flat, inject_w)

print("\n=== recompute with EXACT GGUF weights + llama hc_init ===")
print(f"mixed  cos={cos(mixed.numpy(), l_mixed):.6f}")
print(f"gate   cos={cos(gate.numpy(), l_gate):.6f}")
print(f"inject cos={cos(inject.numpy(), l_inject):.6f}")
print(f"\nnorms: mixed recompute={np.linalg.norm(mixed.numpy()):.4f} llama={np.linalg.norm(l_mixed):.4f}")
print(f"       gate  recompute={np.linalg.norm(gate.numpy()):.4f} llama={np.linalg.norm(l_gate):.4f}")