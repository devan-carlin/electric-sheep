#!/usr/bin/env python3
"""Rewrite the heretic-ara AutoRound-int4 checkpoint into a true-GPTQ layout
that is structurally identical to the working Ser gioB base model, so vLLM
routes it through the proven `gptq` (AutoGPTQConfig) backend instead of the
`inc`/auto_round_kernel backend that mis-decodes it on this XPU build.

Transformations:
  1. Re-stamp quantization_config as vanilla `gptq` (copy the base model's
     quantization_config verbatim).
  2. Dequantize the 7 MTP decoder linears (INT4 qweight/scales) back to plain
     F16 `.weight`, matching the base model's all-16bit MTP. This lets the MTP
     module build fully unquantized under B70_MTP_BF16_DRAFT=1.
  3. Keep mtp.fc.weight (already 16-bit) and all MTP norms as-is.
  4. Body linears (INT4 GPTQ) + in_proj_a/b (plain) copied tensor-for-tensor.

Uses the official `safetensors.torch.save_file` writer (guaranteed-valid file
format). Processes one source shard at a time; peak RAM ~ one shard (~3.2 GB).
No calibration, no GPU.
"""
import json
import os
import shutil

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/mnt/data/models/devan-carlin-Qwen3.8-27B-heretic-ara-int4-AutoRound/Qwen3.8-27B-heretic-ara-w4g128"
OUT = "/home/dc/electric-sheep/models/heretic-org-Qwen3.8-27B-heretic-ara-int4-gptq"
BASE = "/home/dc/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"

# MTP decoder linears stored as INT4 in the source -> dequant to F16 .weight.
MTP_DEQUANT_PREFIXES = {
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.mlp.down_proj",
}


def dequant_int4(qw, scales, group_size=128):
    """qw: int32 [K//8, N]; each int32 packs 8 consecutive K-values, nibble j
    (bits 4j..4j+3) = K-index (row*8 + j), stored value = q + 8.
    scales: [K//group_size, N]. Returns float16 [N, K] (nn.Linear [out, in])."""
    Kd, N = qw.shape
    K = Kd * 8
    Q = np.empty((K, N), dtype=np.float32)
    for j in range(8):
        Q[j::8, :] = (((qw >> (4 * j)) & 0xF).astype(np.float32) - 8.0)
    sc = np.ascontiguousarray(scales).astype(np.float32)
    sc_full = np.repeat(sc, group_size, axis=0)[:K]  # [K, N]
    W = Q * sc_full  # [K, N]
    return torch.from_numpy(W.T.astype(np.float16).copy())  # [N, K]


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    base_qc = json.load(open(os.path.join(BASE, "config.json"))).get("quantization_config", {})
    her_cfg = json.load(open(os.path.join(SRC, "config.json")))

    # Group tensors by source shard.
    by_shard = {}
    for name, shard in wm.items():
        by_shard.setdefault(shard, []).append(name)

    # Open every source shard once (read-only) for fast random access.
    src_files = {shard: safe_open(os.path.join(SRC, shard), framework="pt")
                 for shard in sorted(by_shard)}

    new_wm = {}
    n_copy = n_dequant = n_drop = 0

    for shard in sorted(by_shard):
        names = by_shard[shard]
        sf = src_files[shard]
        out_tensors = {}
        for name in names:
            prefix = name[: -len(".qweight")] if name.endswith(".qweight") else None
            if prefix is not None and prefix in MTP_DEQUANT_PREFIXES:
                # Dequant this INT4 linear to F16 .weight.
                qw_t = sf.get_tensor(name)                       # int32 [K//8, N]
                sc_t = src_files[wm[prefix + ".scales"]].get_tensor(prefix + ".scales")
                qw = qw_t.numpy().astype(np.int32)
                sc = sc_t.numpy().astype(np.float16)
                Wf = dequant_int4(qw, sc)                        # float16 [N, K]
                out_name = prefix + ".weight"
                out_tensors[out_name] = Wf
                new_wm[out_name] = shard
                n_dequant += 1
                continue

            if name.endswith(".scales") or name.endswith(".qzeros"):
                p = name.rsplit(".", 1)[0]
                if p in MTP_DEQUANT_PREFIXES:
                    n_drop += 1
                    continue  # consumed by dequant

            out_tensors[name] = sf.get_tensor(name)              # passthrough
            new_wm[name] = shard
            n_copy += 1

        out_path = os.path.join(OUT, shard)
        save_file(out_tensors, out_path)
        sz = os.path.getsize(out_path)
        print(f"wrote {shard}: {len(out_tensors)} tensors, {sz/1e6:.1f} MB", flush=True)
        del out_tensors

    # New index.
    new_idx = {"metadata": idx.get("metadata"), "weight_map": new_wm}
    json.dump(new_idx, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)

    # config.json: heretic's own config, quantization_config re-stamped to gptq.
    her_cfg["quantization_config"] = base_qc
    json.dump(her_cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)

    # Copy non-weight assets from source (skip the auto-round quantization_config.json).
    for fn in ["chat_template.jinja", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "preprocessor_config.json", "processor_config.json"]:
        s = os.path.join(SRC, fn)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(OUT, fn))

    print(f"\nDONE. copy={n_copy} dequant={n_dequant} drop={n_drop}")
    print(f"output: {OUT}")
    print("quant_method:", base_qc.get("quant_method"))
    print("dynamic:", base_qc.get("dynamic"))


if __name__ == "__main__":
    main()
