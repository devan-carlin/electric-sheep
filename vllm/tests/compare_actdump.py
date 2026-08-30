#!/usr/bin/env python
"""Compare llama.cpp actdump activations vs vLLM captured resid_out, per layer.

llama.cpp l_last-i : [n_embd, hc, T] = [2560, 4, 5]  (from actdump .bin, f32)
vLLM resid_out     : [T, hc, n_embd] = [5, 4, 2560]  (from /tmp/qwen4exp_layer{ii}_resid.pt)

Align by reshaping llama to [2560,4,5] then transpose(2,1,0) -> [5,4,2560].
Report per-layer cosine similarity. First layer where cos < ~0.999 is the bug.
"""
import json
import os
import sys
import numpy as np
import torch

LLAMA_DIR = "/tmp/llama_actdump"
VLLM_DIR = "/tmp"  # qwen4exp_layer{ii}_resid.pt


def load_llama(name, manifest):
    """Load a llama.cpp actdump tensor.

    to_f32() emits elements with i0 fastest, i3 slowest, so the flat buffer is
    row-major with shape [ne3, ne2, ne1, ne0]. We return it in that true layout.
    For l_last-i (ne=[n_embd,hc,T,1]) that is [1, T, hc, n_embd]; arr[0] -> [T,hc,n_embd].
    For model.input_embed (ne=[n_embd,T,1,1]) that is [1,1,T,n_embd]; arr[0][0] -> [T,n_embd].
    """
    entry = None
    for e in manifest["tensors"]:
        if e["name"] == name:
            entry = e
            break
    if entry is None:
        return None
    ne = entry["ne"]  # [ne0, ne1, ne2, ne3]
    with open(entry["file"], "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    # row-major [ne3, ne2, ne1, ne0]
    arr = data.reshape(ne[3], ne[2], ne[1], ne[0])
    return arr


def cos(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main():
    with open(os.path.join(LLAMA_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    names = [e["name"] for e in manifest["tensors"]]
    print(f"llama.cpp dumped {len(names)} tensors")

    # sanity: embedding
    emb = load_llama("model.input_embed", manifest)
    if emb is not None:
        print(f"  model.input_embed ne={emb.shape}")

    # embedding cross-check: llama emb [2560,5] vs vLLM embed capture [5,2560]
    ecap = os.path.join("/tmp", "qwen4exp_embed_capture.pt")
    if os.path.exists(ecap):
        ec = torch.load(ecap, map_location="cpu")
        vemb = ec["embed"].float().numpy()  # [T, 2560]
        lids = ec["input_ids"].flatten().tolist()
        print(f"  vLLM embed capture: ids={lids} shape={vemb.shape}")
        l_emb = emb[0][0]  # [T, 2560]
        print(f"  cos(llama emb, vLLM embed) = {cos(l_emb, vemb):.6f}")

    rows = []
    for il in range(48):
        lname = f"l_last-{il}"
        larr = load_llama(lname, manifest)
        if larr is None:
            print(f"layer {il:02d}: MISSING {lname}")
            continue
        # larr is [1, T, hc, n_embd] -> [T, hc, n_embd]
        l3 = larr[0]  # [5, 4, 2560]

        vpath = os.path.join(VLLM_DIR, f"qwen4exp_layer{il:02d}_resid.pt")
        if not os.path.exists(vpath):
            print(f"layer {il:02d}: MISSING vLLM {vpath}")
            continue
        v = torch.load(vpath, map_location="cpu")
        vo = v["resid_out"].float().numpy()  # [5,4,2560]
        c_out = cos(l3, vo)

        # resid_in cross-check: llama l_last-(il-1) should equal vLLM resid_in of layer il
        c_in = float("nan")
        if il > 0:
            prev = load_llama(f"l_last-{il-1}", manifest)
            if prev is not None and "resid_in" in v:
                p3 = prev[0]  # [T, hc, n_embd]
                vi = v["resid_in"].float().numpy()
                c_in = cos(p3, vi)

        rows.append((il, c_out, l3.shape, vo.shape))
        flag = ""
        if c_out < 0.999:
            flag = "  <-- DIVERGENCE"
        in_str = f"  resid_in_cos={c_in:.6f}" if c_in == c_in else ""
        print(f"layer {il:02d}: resid_out_cos={c_out:.6f}{in_str}{flag}")

    print("\n=== summary ===")
    if rows:
        first_bad = next((il for il, c, _, _ in rows if c < 0.999), None)
        print(f"first layer with cos < 0.999: {first_bad}")
        worst = min(rows, key=lambda r: r[1])
        print(f"worst layer: {worst[0]:02d} cos={worst[1]:.6f}")


if __name__ == "__main__":
    main()