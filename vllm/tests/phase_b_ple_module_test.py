"""Phase B4: verify the Qwen4ExpPLE module against a pure-Python reference.

The reference re-implements llama.cpp build_ple in float64 on CPU:
  rows (uint64 xor hash + EOS reset) -> table gather -> key/value proj
  -> grouped RMSNorm -> signed-sqrt sigmoid gate -> dilated causal conv
  (kernel 4, dilation 3, rolling 9-position state) -> residual + gated + conv.

Cases:
  A. prefill T=17, 1 seq, fresh state
  B. decode 1 token after the prefill (state carried)
  C. EOS mid-sequence (window reset)
  D. 2-sequence batch
"""

import os
import sys

import torch

CKPT = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
BASE = "model.language_model.layers.1.ple."
TABLE_PATH = "/mnt/data/ple_cache/ple_table_qwen4exp.pt"

HC = 4
N_EMBD = 2560
HC_DIM = HC * N_EMBD
N_HEADS = 16
HEAD_DIM = 160
NGRAM = 3
PER_GRAM = 8
KERN = 4
DIL = 3
HIST = (KERN - 1) * DIL
EOS = 248044
MASK64 = (1 << 64) - 1


def load_ple_weights():
    from safetensors import safe_open
    import json

    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    def get(key):
        with safe_open(os.path.join(CKPT, wm[key]), framework="pt") as sf:
            return sf.get_tensor(key)

    w = {
        "key_proj": get(BASE + "key_proj.weight"),
        "value_proj": get(BASE + "value_proj.weight"),
        "conv1d": get(BASE + "conv1d.weight"),
        "norm_key": get(BASE + "norm_key.weight"),
        "norm_query": get(BASE + "norm_query.weight"),
        "norm_conv": get(BASE + "norm_conv.weight"),
        "multipliers": get(BASE + "ple_embedding.layer_multipliers"),
        "offsets": get(BASE + "ple_embedding.ngram_heads_offsets"),
        "vocabs": get(BASE + "ple_embedding.ngram_heads_vocab_sizes"),
    }
    return w


def ref_rows(toks, prev1, prev2, multipliers, offsets, vocabs):
    """Pure-Python rows for a sequence. toks: list of token ids (this step).
    prev1/prev2: the two tokens before this step (EOS if none)."""
    m = multipliers.tolist()
    offs = offsets.tolist()
    vocs = vocabs.tolist()
    rows = []
    p1, p2 = prev1, prev2
    for t in toks:
        c0 = t
        c1 = p1
        c2 = EOS if p1 == EOS else p2
        mixed2 = (c0 * m[0] ^ c1 * m[1]) & MASK64
        mixed3 = (mixed2 ^ (c2 * m[2])) & MASK64
        r = []
        for h in range(PER_GRAM):
            r.append((mixed2 % vocs[h]) + offs[h])
        for h in range(PER_GRAM, N_HEADS):
            r.append((mixed3 % vocs[h]) + offs[h])
        rows.append(r)
        p2, p1 = p1, t
    return rows, p1, p2


def ref_grouped_norm(x, w, eps=1e-6):
    # x: [T, HC, N_EMBD] float64; w: [HC_DIM] (stream-major)
    var = (x**2).mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps)
    return (xn.reshape(x.shape[0], HC_DIM) * w).reshape(x.shape)


def ref_ple_step(residual, toks, prev1, prev2, table, w, conv_state):
    """One PLE step for a single sequence. residual: [T, HC, N_EMBD] f64.
    conv_state: [HIST, HC_DIM] f64 (zero for a fresh sequence).
    Returns (out_residual, new_conv_state, new_prev1, new_prev2)."""
    T = residual.shape[0]
    rows, np1, np2 = ref_rows(toks, prev1, prev2, w["multipliers"],
                              w["offsets"], w["vocabs"])
    rows_t = torch.tensor(rows, dtype=torch.int64)  # [T, 16]
    emb = table[rows_t.reshape(-1)].reshape(T, N_HEADS * HEAD_DIM).double()

    key = (emb @ w["key_proj"].double().t()).reshape(T, HC, N_EMBD)
    value = emb @ w["value_proj"].double().t()  # [T, N_EMBD]

    key = ref_grouped_norm(key, w["norm_key"].double())
    query = ref_grouped_norm(residual, w["norm_query"].double())

    s = (key * query).sum(dim=-1) / (N_EMBD ** 0.5)  # [T, HC]
    gate = torch.sigmoid(torch.sign(s) * torch.sqrt(torch.clamp(s.abs(), min=1e-6)))

    gated = value.unsqueeze(1) * gate.unsqueeze(-1)  # [T, HC, N_EMBD]
    normalized = ref_grouped_norm(gated, w["norm_conv"].double())  # [T, HC, N_EMBD]
    x = normalized.reshape(T, HC_DIM)

    # dilated causal conv with rolling state
    padded = torch.cat([conv_state, x], dim=0)  # [HIST+T, HC_DIM]
    wconv = w["conv1d"].double().squeeze(1)  # [HC_DIM, KERN]
    acc = torch.zeros(T, HC_DIM)
    for k in range(KERN):
        offset = HIST - (KERN - 1 - k) * DIL
        acc = acc + padded[offset:offset + T] * wconv[:, k]
    conv_out = torch.nn.functional.silu(acc)  # [T, HC_DIM]
    new_state = padded[-HIST:]

    out = residual + gated + conv_out.reshape(T, HC, N_EMBD)
    return out, new_state, np1, np2


def cos(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return (a @ b) / (a.norm() * b.norm())


def contrib_cos(out, residual_bf16, ref_out, residual_f64):
    """Cos of the PLE contribution (out - residual), the real signal."""
    c_mod = (out - residual_bf16).double()
    c_ref = (ref_out - residual_f64)
    return (c_mod.flatten() @ c_ref.flatten()) / (
        c_mod.norm() * c_ref.norm())


def make_module(w, device="cpu"):
    """Build a Qwen4ExpPLE with real weights, no vLLM engine."""
    from vllm.model_executor.models import qwen4_exp as M

    class Cfg:
        hc_count = HC
        hidden_size = N_EMBD
        rms_norm_eps = 1e-6
        ngram_size = NGRAM
        heads_per_ngram = PER_GRAM
        ple_embed_dim = N_EMBD
        ple_conv_kernel_size = KERN
        eos_token_id = EOS
        image_token_id = 248056

    # Real model is built under set_default_torch_dtype(bf16); mirror that so
    # the _WeightContainer params are bf16 (int64 constants stay int64).
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        ple = M.Qwen4ExpPLE(Cfg()).to(device)
    finally:
        torch.set_default_dtype(old)
    ple.key_proj.weight.data.copy_(w["key_proj"])
    ple.value_proj.weight.data.copy_(w["value_proj"])
    ple.conv1d.weight.data.copy_(w["conv1d"])
    ple.norm_key.weight.data.copy_(w["norm_key"])
    ple.norm_query.weight.data.copy_(w["norm_query"])
    ple.norm_conv.weight.data.copy_(w["norm_conv"])
    ple.ple_embedding.layer_multipliers.data.copy_(w["multipliers"])
    ple.ple_embedding.ngram_heads_offsets.data.copy_(w["offsets"])
    ple.ple_embedding.ngram_heads_vocab_sizes.data.copy_(w["vocabs"])
    return ple, M


def run_module(ple, M, residual, input_ids, qsl, state_idx, has_init, num_slots,
               prefix="model.layers.1.linear_attn"):
    """Drive ple.forward with a fake forward context."""
    import types

    meta = types.SimpleNamespace(
        non_spec_query_start_loc=qsl,
        non_spec_state_indices_tensor=state_idx,
        has_initial_state=has_init,
    )
    ctx = types.SimpleNamespace(attn_metadata={prefix: meta})
    orig = M.get_forward_context
    M.get_forward_context = lambda: ctx
    try:
        out = ple(residual, input_ids=input_ids, gdn_prefix=prefix,
                  num_slots=num_slots)
    finally:
        M.get_forward_context = orig
    return out


def main():
    torch.manual_seed(0)
    print("loading PLE weights ...", flush=True)
    w = load_ple_weights()
    print("loading table (mmap) ...", flush=True)
    table = torch.load(TABLE_PATH, mmap=True, weights_only=True)["table"]

    ple, M = make_module(w)
    NUM_SLOTS = 8

    # ---------------- Case A: prefill T=17, fresh ----------------
    T = 17
    toks = [101, 2053, 872, 321, 4096, 7, 1563, 220, 318, 4168, 1037, 502,
            262, 9693, 17799, 29, 151645]
    assert EOS not in toks
    residual = torch.randn(T, HC, N_EMBD, dtype=torch.float64)
    ref_out, ref_state, ref_p1, ref_p2 = ref_ple_step(
        residual, toks, EOS, EOS, table, w,
        torch.zeros(HIST, HC_DIM, dtype=torch.float64))

    in_ids = torch.tensor(toks, dtype=torch.int64)
    qsl = torch.tensor([0, T], dtype=torch.int32)
    sidx = torch.tensor([3], dtype=torch.int32)
    has_init = torch.tensor([False])
    # f32 residual: the PLE contribution (~0.002) is tiny vs the residual
    # (~1.0); a bf16 residual would cancel it in (out - residual). f32 keeps
    # the add precise so the math validates cleanly.
    res_bf = residual.float()
    out = run_module(ple, M, res_bf, in_ids, qsl, sidx, has_init, NUM_SLOTS)
    c = contrib_cos(out, res_bf, ref_out, residual)
    print(f"Case A (prefill T={T}, fresh): contrib cos = {c.item():.6f}")
    assert c.item() > 0.99, f"Case A failed: {c.item()}"

    # ---------------- Case B: decode 1 token, state carried ----------
    # Module state after Case A should equal ref_state (bf16 vs f64).
    mod_state = ple._conv_state[3].double()
    c_state = cos(mod_state, ref_state)
    print(f"Case B (conv state after prefill): cos = {c_state.item():.6f}")
    assert c_state.item() > 0.99, f"state mismatch: {c_state.item()}"

    t_dec = [388]
    residual_d = torch.randn(1, HC, N_EMBD, dtype=torch.float64)
    ref_out_d, ref_state_d, ref_p1, ref_p2 = ref_ple_step(
        residual_d, t_dec, ref_p1, ref_p2, table, w, ref_state)
    res_d_bf = residual_d.float()
    out_d = run_module(ple, M, res_d_bf,
                       torch.tensor(t_dec, dtype=torch.int64),
                       torch.tensor([0, 1], dtype=torch.int32),
                       torch.tensor([3], dtype=torch.int32),
                       torch.tensor([True]), NUM_SLOTS)
    c = contrib_cos(out_d, res_d_bf, ref_out_d, residual_d)
    print(f"Case B (decode 1 token, carried state): contrib cos = {c.item():.6f}")
    assert c.item() > 0.99, f"Case B failed: {c.item()}"

    # ---------------- Case C: EOS mid-sequence ----------------
    # Fresh slot 5. Sequence: [a, b, EOS, c, d]. After EOS, the window for
    # 'c' is [c, EOS, EOS] and for 'd' is [d, c, EOS].
    toks_c = [11, 22, EOS, 33, 44]
    residual_c = torch.randn(5, HC, N_EMBD, dtype=torch.float64)
    ref_out_c, _, _, _ = ref_ple_step(
        residual_c, toks_c, EOS, EOS, table, w,
        torch.zeros(HIST, HC_DIM, dtype=torch.float64))
    res_c_bf = residual_c.float()
    out_c = run_module(ple, M, res_c_bf,
                       torch.tensor(toks_c, dtype=torch.int64),
                       torch.tensor([0, 5], dtype=torch.int32),
                       torch.tensor([5], dtype=torch.int32),
                       torch.tensor([False]), NUM_SLOTS)
    c = contrib_cos(out_c, res_c_bf, ref_out_c, residual_c)
    print(f"Case C (EOS mid-sequence): contrib cos = {c.item():.6f}")
    assert c.item() > 0.99, f"Case C failed: {c.item()}"

    # ---------------- Case D: 2-sequence batch ----------------
    toks_d1 = [5, 6, 7]
    toks_d2 = [8, 9]
    Td = 5
    residual_d = torch.randn(Td, HC, N_EMBD, dtype=torch.float64)
    # reference: seq1 = first 3 tokens (slot 1), seq2 = last 2 (slot 2)
    ref1, st1, p1a, p2a = ref_ple_step(
        residual_d[:3], toks_d1, EOS, EOS, table, w,
        torch.zeros(HIST, HC_DIM, dtype=torch.float64))
    ref2, st2, _, _ = ref_ple_step(
        residual_d[3:], toks_d2, EOS, EOS, table, w,
        torch.zeros(HIST, HC_DIM, dtype=torch.float64))
    ref_out_d = torch.cat([ref1, ref2], dim=0)
    res_d_bf = residual_d.float()
    out_d = run_module(ple, M, res_d_bf,
                       torch.tensor(toks_d1 + toks_d2, dtype=torch.int64),
                       torch.tensor([0, 3, 5], dtype=torch.int32),
                       torch.tensor([1, 2], dtype=torch.int32),
                       torch.tensor([False, False]), NUM_SLOTS)
    c = contrib_cos(out_d, res_d_bf, ref_out_d, residual_d)
    print(f"Case D (2-seq batch): contrib cos = {c.item():.6f}")
    assert c.item() > 0.99, f"Case D failed: {c.item()}"

    print("\nPASS: PLE module matches the pure-Python reference (all cases).")


if __name__ == "__main__":
    main()