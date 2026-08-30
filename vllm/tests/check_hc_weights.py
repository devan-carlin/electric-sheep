#!/usr/bin/env python
"""Compare FP8 vs W4A16 vs GGUF for layer-0 attention HC weights.

F32 weights (norm, inject) need no dequant -> reliable.
Goal: confirm FP8==W4A16 (same model) and identify GGUF transformations.
"""
import numpy as np
from gguf import GGUFReader
from safetensors import safe_open

FP8_SHARD = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8/model-00001-of-00131.safetensors"
W4_SHARD = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16/model-00001.safetensors"
GGUF_SHARD = "/mnt/data/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf"
PREFIX = "model.language_model.layers.0.attn_hyper_connection."


def cos(a, b):
    a = a.ravel(); b = b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


with safe_open(FP8_SHARD, framework="pt") as sf:
    fp8_norm = sf.get_tensor(PREFIX + "hc_norm.weight")
    fp8_inject = sf.get_tensor(PREFIX + "block_inject_weight.weight")
print("FP8 norm dtype", fp8_norm.dtype, "shape", tuple(fp8_norm.shape))
print("FP8 inject dtype", fp8_inject.dtype, "shape", tuple(fp8_inject.shape))
fp8_norm = fp8_norm.float().numpy()
fp8_inject = fp8_inject.float().numpy()

with safe_open(W4_SHARD, framework="pt") as sf:
    w4_norm = sf.get_tensor(PREFIX + "hc_norm.weight").float().numpy()
    w4_inject = sf.get_tensor(PREFIX + "block_inject_weight.weight").float().numpy()

r = GGUFReader(GGUF_SHARD)


def gg_f32(name):
    t = next(x for x in r.tensors if x.name == name)
    return np.frombuffer(t.data, dtype=np.float32).reshape(t.shape)


g_norm = gg_f32("blk.0.hc_attn_norm.weight")
g_inject = gg_f32("blk.0.hc_attn_inject.weight")

print("\n=== norm (F32, reliable) ===")
print("cos(FP8, W4A16)    =", cos(fp8_norm, w4_norm))
print("cos(GGUF, FP8)     =", cos(g_norm, fp8_norm))
print("cos(GGUF, 1+FP8)   =", cos(g_norm, 1.0 + fp8_norm))
print("cos(GGUF, 1+W4A16) =", cos(g_norm, 1.0 + w4_norm))
print("FP8 norm mean/std  :", fp8_norm.mean(), fp8_norm.std())
print("W4A16 norm mean/std:", w4_norm.mean(), w4_norm.std())
print("GGUF norm mean/std :", g_norm.mean(), g_norm.std())

print("\n=== inject (F32, reliable) ===")
print("FP8 inject shape", fp8_inject.shape, "W4A16 inject shape", w4_inject.shape, "GGUF inject shape", g_inject.shape)
print("cos(FP8, W4A16)    =", cos(fp8_inject, w4_inject))
print("cos(GGUF, W4A16.T) =", cos(g_inject, w4_inject.T))
print("cos(GGUF, FP8.T)   =", cos(g_inject, fp8_inject.T))
print("GGUF  inject[0,:4]:", g_inject[0, :4])
print("W4A16 inject.T[0,:4]:", w4_inject.T[0, :4])
print("FP8   inject.T[0,:4]:", fp8_inject.T[0, :4])