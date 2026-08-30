#!/usr/bin/env python
"""Layer-0 sub-component comparison: llama.cpp (ground truth) vs vLLM.

All vLLM tensors from the SAME W4A16 run (19:22):
  - mix_out, attn_out  : /tmp/qwen4exp_hc_capture_rank0.pt
  - resid_out          : /tmp/qwen4exp_layer00_resid.pt
llama.cpp tensors from /tmp/llama_actdump_l0 (UD-Q4_K_XL).

Chain (no TP layout issues):
  hc_mixed-0  [2560,5]  -> mix_out  [5,2560]   (GDN input)
  linear_attn_out-0 [2560,5] -> attn_out [5,2560] (GDN output)
  l_last-0    [2560,4,5] -> resid_out [5,4,2560] (layer output)
"""
import json
import numpy as np
import torch

LLAMA_DIR = "/tmp/llama_actdump_l0"


def load_llama(name, manifest):
    entry = next((e for e in manifest["tensors"] if e["name"] == name), None)
    if entry is None:
        return None
    ne = entry["ne"]
    with open(entry["file"], "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    # to_f32 emits i0 fastest -> row-major [ne3, ne2, ne1, ne0]
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

    hc = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    resid = torch.load("/tmp/qwen4exp_layer00_resid.pt", map_location="cpu")

    checks = []

    # 1. GDN input: llama hc_mixed-0 [2560,5] -> [5,2560] vs vLLM mix_out [5,2560]
    l = load_llama("hc_mixed-0", m)
    l_in = l[0]  # [5, 2560]
    v_in = hc["mix_out"].float().numpy()  # [5, 2560]
    c_in = cos(l_in, v_in)
    checks.append(("GDN input  (hc_mixed vs mix_out)", c_in, l_in.shape, v_in.shape))

    # 2. GDN output: llama linear_attn_out-0 [2560,5] -> [5,2560] vs vLLM attn_out [5,2560]
    l = load_llama("linear_attn_out-0", m)
    l_out = l[0]  # [5, 2560]
    v_out = hc["attn_out"].float().numpy()  # [5, 2560]
    c_out = cos(l_out, v_out)
    checks.append(("GDN output (linear_attn_out vs attn_out)", c_out, l_out.shape, v_out.shape))

    # 3. Layer output: llama l_last-0 [2560,4,5] -> [5,4,2560] vs vLLM resid_out [5,4,2560]
    l = load_llama("l_last-0", m)
    l_last = l[0]  # [5, 4, 2560]
    v_last = resid["resid_out"].float().numpy()  # [5,4,2560]
    c_last = cos(l_last, v_last)
    checks.append(("Layer output (l_last vs resid_out)", c_last, l_last.shape, v_last.shape))

    # 4. resid_in cross-check: llama hc_init [2560,4,5] -> [5,4,2560] vs vLLM resid_in [5,4,2560]
    l = load_llama("hc_init", m)
    l_init = l[0]  # [5, 4, 2560]
    v_init = resid["resid_in"].float().numpy()  # [5,4,2560]
    c_init = cos(l_init, v_init)
    checks.append(("Layer input  (hc_init vs resid_in)", c_init, l_init.shape, v_init.shape))

    print("=== Layer-0 sub-component comparison (llama.cpp vs vLLM, W4A16) ===\n")
    for name, c, ls, vs in checks:
        flag = ""
        if c < 0.99:
            flag = "  <-- DIVERGENCE"
        print(f"{name:45s} cos={c:.6f}  llama={ls} vllm={vs}{flag}")

    print()
    # Interpretation
    c_in = checks[0][1]
    c_out = checks[1][1]
    if c_in > 0.99 and c_out < 0.99:
        print("=> GDN input matches, GDN output diverges: BUG IS INSIDE THE GDN CORE")
    elif c_in < 0.99:
        print("=> GDN input already diverges: BUG IS UPSTREAM (HC mix or PLE)")
    elif c_out > 0.99:
        print("=> GDN output matches: BUG IS DOWNSTREAM (HC combine or FFN/MoE)")


if __name__ == "__main__":
    main()