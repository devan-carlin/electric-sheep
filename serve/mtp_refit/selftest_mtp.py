#!/usr/bin/env python3
"""CPU self-test for train_mtp.py: build a small draft head, run forward +
backward, check shapes/gradients. No GPU, no checkpoint needed. Catches
shape/orientation bugs in RoPE / GQA attention / MoE bmm / HC mix-combine.
"""
import sys
import torch

sys.path.insert(0, "/home/dc/electric-sheep/serve/mtp_refit")
import train_mtp as tm


def main():
    # Reduced config so it runs fast on CPU.
    c = tm.Cfg({
        "hidden_size": 64, "hc_count": 2, "hc_lowrank": 16, "rms_norm_eps": 1e-6,
        "vocab_size": 512, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "moe_intermediate_size": 48, "num_experts": 8,
        "num_experts_per_tok": 3, "shared_expert_intermediate_size": 48,
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
        "max_position_embeddings": 512,
    })
    print(f"test cfg: hidden={c.hidden_size} hc={c.hc} heads={c.n_heads}/{c.n_kv} "
          f"hd={c.head_dim} rotary_dim={c.rotary_dim} experts={c.n_experts} topk={c.top_k}")
    assert c.rotary_dim == 8, c.rotary_dim  # 32 * 0.25

    torch.manual_seed(0)
    model = tm.MTPDraftHead(c).to(dtype=torch.float32)
    # params are small-random-initialized in constructors; freeze the frozen set
    tm.freeze_frozen(model)
    model.eval()

    T = 17
    input_ids = torch.randint(0, c.vocab_size, (T,))
    wide_flat = torch.randn(T, c.hc * c.hidden_size)
    positions = torch.arange(T, dtype=torch.int64)

    # --- logits path ---
    lg = model.logits(input_ids, wide_flat, positions)
    assert lg.shape == (T, c.vocab_size), lg.shape
    print(f"logits OK: {tuple(lg.shape)}")

    # --- loss + backward ---
    tgt = torch.randint(0, c.vocab_size, (T - 1,))
    src = input_ids[:T - 1]
    wf = wide_flat[:T - 1]
    pos = torch.arange(T - 1, dtype=torch.int64)
    loss = torch.nn.functional.cross_entropy(model.logits(src, wf, pos), tgt)
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"loss={loss.item():.4f} grads_on={n_grad}/{n_train} trainable params")
    assert n_grad == n_train, "some trainable params got no gradient"

    # --- verify frozen params are frozen ---
    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    print(f"frozen params: {len(frozen_names)} -> {[n.split('.')[-1] for n in frozen_names]}")
    assert any("expert_gate_up" in n for n in frozen_names)
    assert any("expert_down" in n for n in frozen_names)
    assert any("embed_tokens" in n for n in frozen_names)
    assert any("lm_head" in n for n in frozen_names)

    # --- RoPE sanity: rotating by position 0 must be identity on the rotated half ---
    attn = model.layer.self_attn
    x = torch.randn(1, c.n_heads, c.head_dim)
    p0 = torch.zeros(1, dtype=torch.int64)
    r0 = attn._rope(x, p0)
    diff0 = (r0 - x).abs().max().item()
    assert diff0 < 1e-4, f"rope at pos0 not identity: {diff0}"
    p1 = torch.ones(1, dtype=torch.int64)
    r1 = attn._rope(x, p1)
    # passthrough tail (dims >= rotary_dim) must be unchanged
    tail_diff = (r1[..., c.rotary_dim:] - x[..., c.rotary_dim:]).abs().max().item()
    assert tail_diff < 1e-4, f"passthrough tail changed: {tail_diff}"
    print(f"rope OK: pos0-identity maxdiff={diff0:.2e} passthrough-tail maxdiff={tail_diff:.2e}")

    # --- MoE routed: single-token, verify output finite & shape ---
    moe = model.layer.mlp
    h = torch.randn(5, c.hidden_size)
    out = moe(h)
    assert out.shape == (5, c.hidden_size), out.shape
    assert torch.isfinite(out).all(), "MoE produced non-finite output"
    print(f"moe OK: {tuple(out.shape)} finite")

    # --- HC combine: inject=0 -> weight ~ 2*sigmoid(0)=1 -> residual + block_out ---
    hc = model.layer.attn_hc
    resid = torch.randn(T, c.hc, c.hidden_size)
    block = torch.randn(T, c.hidden_size)
    zero_inject = torch.zeros(T, c.hc)
    comb = hc.combine(resid, block, zero_inject)
    expect = resid + block.unsqueeze(1)
    d = (comb - expect).abs().max().item()
    assert d < 1e-3, f"combine(inject=0) != residual+block: {d}"
    print(f"hc combine OK: inject=0 -> residual+block maxdiff={d:.2e}")

    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
