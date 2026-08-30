#!/usr/bin/env python3
"""
Phase D: GDN core test (non-circular).

Recomputes the GDN (Gated DeltaNet) core in pure PyTorch from:
  - captured projected_qkvz / projected_ba (vLLM's actual in_proj output)
  - checkpoint weights (conv1d, A_log, dt_bias)
and compares to vLLM's captured core_attn_out.

This tests the GDN core (conv, l2norm, delta rule, GQA mapping) but NOT
the in_proj GEMM (already exonerated via MoE/FA tests).

Layout (gqa_interleaved_layout=True, Qwen3-Next):
  projected_qkvz (T, 4096) per rank = 4 K-groups x [q(128), k(128), v(384), z(384)]
  projected_ba   (T, 24)   per rank = 4 K-groups x [b(3), a(3)]
  core_attn_out  (T, 12, 128) per rank

Per-rank dims (TP=4):
  q=512 (4 heads x 128), k=512 (4 heads x 128), v=1536 (12 heads x 128), z=1536
  b=12, a=12
  conv1d per rank = [2560, 1, 4] (qkv only, depthwise)
  A_log/dt_bias per rank = [12]
"""
import os
import sys
import math
import torch
import torch.nn.functional as F
from safetensors import safe_open

# --- Config ---
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
TP_SIZE = 4
NUM_K_PER_RANK = NUM_K_HEADS // TP_SIZE  # 4
NUM_V_PER_RANK = NUM_V_HEADS // TP_SIZE  # 12
REP = NUM_V_HEADS // NUM_K_HEADS  # 3
EPS = 1e-6
SCALE = 1.0 / math.sqrt(HEAD_K_DIM)

CKPT = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8/model-00001-of-00131.safetensors"
CAPTURE_DIR = "/tmp"


def load_ckpt_weights():
    """Load GDN layer 0 weights from checkpoint."""
    with safe_open(CKPT, framework="pt") as f:
        conv_w = f.get_tensor("model.language_model.layers.0.linear_attn.conv1d.weight").float()
        A_log = f.get_tensor("model.language_model.layers.0.linear_attn.A_log").float()
        dt_bias = f.get_tensor("model.language_model.layers.0.linear_attn.dt_bias").float()
    return conv_w, A_log, dt_bias


def extract_qkv_ba_z(projected_qkvz, projected_ba):
    """
    Extract qkv (2560), b (12), a (12), z (12, 128) from interleaved layout.
    Mirrors _extract_qkv_b_a_z with reorder_input=False.
    """
    T = projected_qkvz.shape[0]
    # Reshape to (T, num_k_per_rank, 2*head_k + 2*rep*head_v)
    per_group = 2 * HEAD_K_DIM + 2 * REP * HEAD_V_DIM  # 2*128 + 2*3*128 = 1024
    qkvz = projected_qkvz.reshape(T, NUM_K_PER_RANK, per_group)
    # Split per group: q(128), k(128), v(384), z(384)
    q_split, k_split, v_split, z_split = torch.split(
        qkvz, [HEAD_K_DIM, HEAD_K_DIM, REP * HEAD_V_DIM, REP * HEAD_V_DIM], dim=-1
    )
    # Flatten
    q = q_split.reshape(T, NUM_K_PER_RANK * HEAD_K_DIM)  # (T, 512)
    k = k_split.reshape(T, NUM_K_PER_RANK * HEAD_K_DIM)  # (T, 512)
    v = v_split.reshape(T, NUM_K_PER_RANK * REP * HEAD_V_DIM)  # (T, 1536)
    z = z_split.reshape(T, NUM_V_PER_RANK, HEAD_V_DIM)  # (T, 12, 128)
    qkv = torch.cat([q, k, v], dim=-1)  # (T, 2560)

    # ba: (T, 24) -> (T, 4, 6) -> b(3), a(3) per group
    ba = projected_ba.reshape(T, NUM_K_PER_RANK, 2 * REP)
    b_split, a_split = torch.split(ba, [REP, REP], dim=-1)
    b = b_split.reshape(T, NUM_V_PER_RANK)  # (T, 12)
    a = a_split.reshape(T, NUM_V_PER_RANK)  # (T, 12)

    return qkv, b, a, z


def gdn_core(qkv, b, a, conv_w_rank, A_log_rank, dt_bias_rank):
    """
    Recompute GDN core in pure PyTorch.
    qkv: (T, 2560) interleaved
    b, a: (T, 12)
    conv_w_rank: (2560, 1, 4)
    A_log_rank, dt_bias_rank: (12,)
    Returns: core_attn_out (T, 12, 128)
    """
    T = qkv.shape[0]
    # Conv1d (depthwise, causal, left-padded with 3 zeros)
    qkv_f = qkv.float()
    # (T, 2560) -> (1, 2560, T)
    x = qkv_f.transpose(0, 1).unsqueeze(0)
    x = F.pad(x, (3, 0))  # (1, 2560, T+3)
    conv_out = F.conv1d(x, conv_w_rank, groups=2560)  # (1, 2560, T)
    conv_out = F.silu(conv_out).transpose(1, 2).squeeze(0)  # (T, 2560)

    # Split into q, k, v
    q = conv_out[:, :512].reshape(T, NUM_K_PER_RANK, HEAD_K_DIM)  # (T, 4, 128)
    k = conv_out[:, 512:1024].reshape(T, NUM_K_PER_RANK, HEAD_K_DIM)  # (T, 4, 128)
    v = conv_out[:, 1024:].reshape(T, NUM_V_PER_RANK, HEAD_V_DIM)  # (T, 12, 128)

    # l2norm + scale
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + EPS)
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + EPS)
    q = q * SCALE

    # Gate
    A_log_exp = -torch.exp(A_log_rank)  # (12,)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)
    g = torch.exp(A_log_exp * softplus(a.float() + dt_bias_rank))  # (T, 12)
    beta = torch.sigmoid(b.float())  # (T, 12)

    # GQA: repeat_interleave q, k by REP (v-head h -> k-head h//REP)
    q = q.repeat_interleave(REP, dim=1)  # (T, 12, 128)
    k = k.repeat_interleave(REP, dim=1)  # (T, 12, 128)

    # Delta rule (sequential over T tokens, zero initial state)
    state = torch.zeros(NUM_V_PER_RANK, HEAD_V_DIM, HEAD_K_DIM, dtype=torch.float32)
    outs = []
    dbg = os.environ.get("GDN_DBG") == "1"
    if dbg:
        print(f"  [dbg] conv_out absmax={conv_out.abs().max().item():.6f}")
        print(f"  [dbg] q absmax={q.abs().max().item():.6f} k absmax={k.abs().max().item():.6f} v absmax={v.abs().max().item():.6f}")
        print(f"  [dbg] g absmax={g.abs().max().item():.6f} beta absmax={beta.abs().max().item():.6f}")
        print(f"  [dbg] q[0,0,:4]={q[0,0,:4].tolist()}")
        print(f"  [dbg] k[0,0,:4]={k[0,0,:4].tolist()}")
        print(f"  [dbg] v[0,0,:4]={v[0,0,:4].tolist()}")
    for t in range(T):
        g_t = g[t]  # (12,)
        beta_t = beta[t]  # (12,)
        q_t = q[t]  # (12, 128)
        k_t = k[t]  # (12, 128)
        v_t = v[t]  # (12, 128)

        # Decay
        state = state * g_t.unsqueeze(-1).unsqueeze(-1)
        # pred = state @ k
        pred = torch.einsum("vhk,vk->vh", state, k_t)
        # delta = (v - pred) * beta
        delta = (v_t - pred) * beta_t.unsqueeze(-1)
        # state += delta @ k^T
        state = state + torch.einsum("vh,vk->vhk", delta, k_t)
        # out = state @ q
        out_t = torch.einsum("vhk,vk->vh", state, q_t)
        outs.append(out_t)
        if dbg:
            print(f"  [dbg] t={t} state_absmax={state.abs().max().item():.6f} out_absmax={out_t.abs().max().item():.6f} out[0,:4]={out_t[0,:4].tolist()}")

    core = torch.stack(outs, dim=0)  # (T, 12, 128)
    return core


def main():
    conv_w, A_log, dt_bias = load_ckpt_weights()
    print(f"conv_w: {conv_w.shape}, A_log: {A_log.shape}, dt_bias: {dt_bias.shape}")

    # Per-rank conv weight: [2560, 1, 4]
    # Two hypotheses:
    #   (A) head-major: q[0:2048], k[2048:4096], v[4096:10240]; rank r = cat(q_r, k_r, v_r)
    #   (B) interleaved: 16 groups x [q(128),k(128),v(384)] = 16 x 640; rank r = conv_w[r*2560:(r+1)*2560]
    q_block = conv_w[0:2048]  # (2048, 1, 4)
    k_block = conv_w[2048:4096]  # (2048, 1, 4)
    v_block = conv_w[4096:10240]  # (6144, 1, 4)
    conv_layout = os.environ.get("CONV_LAYOUT", "interleaved")  # "headmajor" or "interleaved"

    for rank in range(TP_SIZE):
        cap = torch.load(f"{CAPTURE_DIR}/qwen4exp_gdn_capture_L0_R{rank}.pt", map_location="cpu")
        projected_qkvz = cap["projected_qkvz"]  # (T, 4096)
        projected_ba = cap["projected_ba"]  # (T, 24)
        core_attn_out = cap["core_attn_out"]  # (T, 12, 128)
        T = projected_qkvz.shape[0]
        print(f"\n=== Rank {rank} (T={T}) conv_layout={conv_layout} ===")

        # Per-rank conv weight
        if conv_layout == "headmajor":
            q_r = q_block[rank * 512:(rank + 1) * 512]  # (512, 1, 4)
            k_r = k_block[rank * 512:(rank + 1) * 512]  # (512, 1, 4)
            v_r = v_block[rank * 1536:(rank + 1) * 1536]  # (1536, 1, 4)
            conv_w_rank = torch.cat([q_r, k_r, v_r], dim=0)  # (2560, 1, 4)
        else:  # interleaved
            conv_w_rank = conv_w[rank * 2560:(rank + 1) * 2560]  # (2560, 1, 4)

        # Per-rank A_log, dt_bias
        A_log_rank = A_log[rank * 12:(rank + 1) * 12]  # (12,)
        dt_bias_rank = dt_bias[rank * 12:(rank + 1) * 12]  # (12,)

        # Extract qkv, b, a, z
        qkv, b, a, z = extract_qkv_ba_z(projected_qkvz, projected_ba)
        print(f"  qkv: {qkv.shape}, b: {b.shape}, a: {a.shape}, z: {z.shape}")

        # Recompute GDN core
        core_ref = gdn_core(qkv, b, a, conv_w_rank, A_log_rank, dt_bias_rank)
        print(f"  core_ref: {core_ref.shape}, core_attn_out: {core_attn_out.shape}")

        # Compare
        core_ref_bf = core_ref.to(core_attn_out.dtype)
        diff = (core_ref_bf.float() - core_attn_out.float()).abs()
        cos = F.cosine_similarity(core_ref_bf.float().flatten(), core_attn_out.float().flatten(), dim=0).item()
        rel_err = (diff.sum() / core_attn_out.float().abs().sum()).item()
        print(f"  cos_sim: {cos:.6f}")
        print(f"  rel_err: {rel_err:.6f}")
        print(f"  max_abs_diff: {diff.max().item():.6f}")
        print(f"  mean_abs_diff: {diff.mean().item():.6f}")
        print(f"  core_ref_absmax: {core_ref_bf.float().abs().max().item():.6f}")
        print(f"  core_attn_out_absmax: {core_attn_out.float().abs().max().item():.6f}")
        if os.environ.get("GDN_DBG") == "1":
            print(f"  [dbg] projected_qkvz[0,:8]={projected_qkvz[0,:8].tolist()}")
            print(f"  [dbg] projected_ba[0,:8]={projected_ba[0,:8].tolist()}")
            print(f"  [dbg] core_attn_out[0,0,:4]={core_attn_out[0,0,:4].tolist()}")
            print(f"  [dbg] core_ref[0,0,:4]={core_ref[0,0,:4].tolist()}")
            print(f"  [dbg] core_attn_out[0,1,:4]={core_attn_out[0,1,:4].tolist()}")
            print(f"  [dbg] core_ref[0,1,:4]={core_ref[0,1,:4].tolist()}")

        # Per-head cos
        for h in range(12):
            cos_h = F.cosine_similarity(
                core_ref_bf[:, h, :].float().flatten(),
                core_attn_out[:, h, :].float().flatten(),
                dim=0
            ).item()
            if abs(cos_h - 1.0) > 0.01:
                print(f"  head {h}: cos={cos_h:.6f} (MISMATCH)")

    print("\nDone.")


if __name__ == "__main__":
    main()