#!/usr/bin/env python3
"""Dry-run weight loading against the real checkpoint (CPU, no GPU).

Builds the full-config MTPDraftHead on CPU, runs load_weights(), and verifies:
  * every mtp.* / embed_tokens / lm_head key is mapped (no 'unmapped' warns)
  * a sample of loaded tensors is BIT-EXACT vs the checkpoint (copy_ is exact)
  * the frozen/trainable param split is as designed
Catches naming/orientation bugs before the GPU fidelity run.
"""
import os
import sys

import torch

sys.path.insert(0, "/home/dc/electric-sheep/serve/mtp_refit")
import train_mtp as tm

CKPT = os.environ.get(
    "MTP_CKPT",
    "/mnt/data/models/devancarlin-Qwen3.8-Flash-Next-heretic-2-W4A16-BF16src",
)


def main():
    tc = tm.load_text_config(CKPT)
    c = tm.Cfg(tc)
    print(f"cfg: hidden={c.hidden_size} hc={c.hc} heads={c.n_heads}/{c.n_kv} "
          f"hd={c.head_dim} rotary_dim={c.rotary_dim} experts={c.n_experts} "
          f"topk={c.top_k} vocab={c.vocab_size}")

    model = tm.MTPDraftHead(c).to(dtype=torch.bfloat16)
    print("built MTPDraftHead on CPU (bf16)")

    # capture stdout to detect 'unmapped' warnings
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        loaded = tm.load_weights(model, CKPT, device="cpu", verbose=True)
    out = buf.getvalue()
    print(out.strip())
    unmapped = [ln for ln in out.splitlines() if "unmapped" in ln]
    print(f"loaded={len(loaded)} unmapped_warnings={len(unmapped)}")
    for u in unmapped:
        print("  UNMAPPED:", u.strip())

    tm.freeze_frozen(model)
    tr, fr = tm.count_params(model)
    print(f"params: trainable={tr/1e6:.2f}M frozen={fr/1e6:.2f}M total={(tr+fr)/1e6:.2f}M")

    # ---- bit-exact spot checks vs checkpoint ----
    from safetensors import safe_open
    idx = json_load_index(CKPT)
    checks = {
        "mtp.fc_embedding.weight": model.fc_embedding.weight,
        "mtp.pre_fc_norm_hidden.weight": model.pre_fc_norm_hidden_w,
        "mtp.layers.0.self_attn.q_proj.weight": model.layer.self_attn.q_proj.weight,
        "mtp.layers.0.self_attn.o_proj.weight": model.layer.self_attn.o_proj.weight,
        "mtp.layers.0.mlp.experts.gate_up_proj": model.layer.mlp.expert_gate_up,
        "mtp.layers.0.mlp.experts.down_proj": model.layer.mlp.expert_down,
        "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": model.layer.attn_hc.inject,
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": model.mixer.up,
        "lm_head.weight": model.lm_head.weight,
        "model.language_model.embed_tokens.weight": model.embed_tokens.weight,
    }
    ok = True
    for ck_key, mod_param in checks.items():
        f = idx[ck_key]
        with safe_open(os.path.join(CKPT, f), framework="pt") as sf:
            ref = sf.get_tensor(ck_key)
        ref = ref.to(dtype=mod_param.dtype)
        same_shape = tuple(ref.shape) == tuple(mod_param.shape)
        bitexact = bool(torch.equal(ref, mod_param))
        status = "OK" if (same_shape and bitexact) else "FAIL"
        if not (same_shape and bitexact):
            ok = False
        print(f"  [{status}] {ck_key}: shape_match={same_shape} bitexact={bitexact} "
              f"{tuple(ref.shape)}")
    print("\nWEIGHT LOAD " + ("VERIFIED (all bit-exact)" if ok else "HAS FAILURES"))
    sys.exit(0 if ok and not unmapped else 1)


def json_load_index(ckpt):
    import json
    p = os.path.join(ckpt, "model.safetensors.index.json")
    return json.load(open(p))["weight_map"]


if __name__ == "__main__":
    main()
