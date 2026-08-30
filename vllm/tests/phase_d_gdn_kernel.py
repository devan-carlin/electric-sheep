#!/usr/bin/env python3
"""
Decisive GDN kernel test.

Runs the ACTUAL vLLM kernel (torch.ops._xpu_C.gdn_attention) on the captured
projected_qkvz/projected_ba with ZERO conv_state and ZERO ssm_state, and
compares to the pure-Python reference (also zero state).

If kernel == reference  -> kernel is correct; the earlier mismatch was due to
                           non-zero initial state in the capture (warmup).
If kernel != reference  -> the kernel itself is buggy.
"""
import os
import sys
import math
import torch
import torch.nn.functional as F
from safetensors import safe_open

import vllm_xpu_kernels._xpu_C  # noqa: F401

NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
TP_SIZE = 4
NUM_K_PER_RANK = NUM_K_HEADS // TP_SIZE
NUM_V_PER_RANK = NUM_V_HEADS // TP_SIZE
REP = NUM_V_HEADS // NUM_K_HEADS
EPS = 1e-6
SCALE = 1.0 / math.sqrt(HEAD_K_DIM)
CKPT = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8/model-00001-of-00131.safetensors"
DEVICE = "xpu"


def load_ckpt_weights():
    with safe_open(CKPT, framework="pt") as f:
        conv_w = f.get_tensor("model.language_model.layers.0.linear_attn.conv1d.weight").float()
        A_log = f.get_tensor("model.language_model.layers.0.linear_attn.A_log").float()
        dt_bias = f.get_tensor("model.language_model.layers.0.linear_attn.dt_bias").float()
    return conv_w, A_log, dt_bias


def extract_qkv_ba_z(projected_qkvz, projected_ba):
    """
    Convert HEAD-MAJOR input to INTERLEAVED, then extract qkv/b/a/z.
    Mirrors the reorder_input=True path in _extract_qkv_b_a_z.
    """
    T = projected_qkvz.shape[0]
    # Step 1: head-major -> interleaved (reorder_input=True path)
    q_tmp, k_tmp, v_tmp, z_tmp = projected_qkvz.split([512, 512, 1536, 1536], dim=-1)
    q_tmp = q_tmp.reshape(T, NUM_K_PER_RANK, HEAD_K_DIM)  # (T, 4, 128)
    k_tmp = k_tmp.reshape(T, NUM_K_PER_RANK, HEAD_K_DIM)  # (T, 4, 128)
    v_tmp = v_tmp.reshape(T, NUM_K_PER_RANK, REP * HEAD_V_DIM)  # (T, 4, 384)
    z_tmp = z_tmp.reshape(T, NUM_K_PER_RANK, REP * HEAD_V_DIM)  # (T, 4, 384)
    projected_qkvz = torch.cat([q_tmp, k_tmp, v_tmp, z_tmp], dim=-1).reshape(T, -1).contiguous()
    # ba: head-major [b(12), a(12)] -> interleaved [b_g0, a_g0, b_g1, a_g1, ...]
    b_tmp, a_tmp = projected_ba.chunk(2, dim=-1)  # (T, 12) each
    b_tmp = b_tmp.reshape(T, NUM_K_PER_RANK, REP)  # (T, 4, 3)
    a_tmp = a_tmp.reshape(T, NUM_K_PER_RANK, REP)  # (T, 4, 3)
    projected_ba = torch.cat([b_tmp, a_tmp], dim=-1).reshape(T, -1).contiguous()  # (T, 24) interleaved

    # Step 2: interpret as interleaved
    per_group = 2 * HEAD_K_DIM + 2 * REP * HEAD_V_DIM  # 1024
    qkvz = projected_qkvz.reshape(T, NUM_K_PER_RANK, per_group)
    q_split, k_split, v_split, z_split = torch.split(
        qkvz, [HEAD_K_DIM, HEAD_K_DIM, REP * HEAD_V_DIM, REP * HEAD_V_DIM], dim=-1
    )
    q = q_split.reshape(T, NUM_K_PER_RANK * HEAD_K_DIM)
    k = k_split.reshape(T, NUM_K_PER_RANK * HEAD_K_DIM)
    v = v_split.reshape(T, NUM_K_PER_RANK * REP * HEAD_V_DIM)
    z = z_split.reshape(T, NUM_V_PER_RANK, HEAD_V_DIM)
    qkv = torch.cat([q, k, v], dim=-1)
    ba = projected_ba.reshape(T, NUM_K_PER_RANK, 2 * REP)
    b_split, a_split = torch.split(ba, [REP, REP], dim=-1)
    b = b_split.reshape(T, NUM_V_PER_RANK)
    a = a_split.reshape(T, NUM_V_PER_RANK)
    return qkv, b, a, z


def gdn_core_ref(qkv, b, a, conv_w_rank, A_log_rank, dt_bias_rank):
    T = qkv.shape[0]
    x = qkv.float().transpose(0, 1).unsqueeze(0)
    x = F.pad(x, (3, 0))
    conv_out = F.conv1d(x, conv_w_rank, groups=2560)
    conv_out = F.silu(conv_out).transpose(1, 2).squeeze(0)
    q = conv_out[:, :512].reshape(T, NUM_K_PER_RANK, HEAD_K_DIM)
    k = conv_out[:, 512:1024].reshape(T, NUM_K_PER_RANK, HEAD_K_DIM)
    v = conv_out[:, 1024:].reshape(T, NUM_V_PER_RANK, HEAD_V_DIM)
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + EPS)
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + EPS)
    q = q * SCALE
    A_log_exp = -torch.exp(A_log_rank)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)
    g = torch.exp(A_log_exp * softplus(a.float() + dt_bias_rank))
    beta = torch.sigmoid(b.float())
    q = q.repeat_interleave(REP, dim=1)
    k = k.repeat_interleave(REP, dim=1)
    state = torch.zeros(NUM_V_PER_RANK, HEAD_V_DIM, HEAD_K_DIM, dtype=torch.float32)
    outs = []
    for t in range(T):
        g_t, beta_t, q_t, k_t, v_t = g[t], beta[t], q[t], k[t], v[t]
        state = state * g_t.unsqueeze(-1).unsqueeze(-1)
        pred = torch.einsum("vhk,vk->vh", state, k_t)
        delta = (v_t - pred) * beta_t.unsqueeze(-1)
        state = state + torch.einsum("vh,vk->vhk", delta, k_t)
        outs.append(torch.einsum("vhk,vk->vh", state, q_t))
    return torch.stack(outs, dim=0)


def main():
    conv_w, A_log, dt_bias = load_ckpt_weights()
    q_block = conv_w[0:2048]; k_block = conv_w[2048:4096]; v_block = conv_w[4096:10240]
    conv_layout = os.environ.get("CONV_LAYOUT", "headmajor")

    for rank in range(TP_SIZE):
        cap = torch.load(f"/tmp/qwen4exp_gdn_capture_L0_R{rank}.pt", map_location="cpu")
        projected_qkvz = cap["projected_qkvz"].to(DEVICE)
        projected_ba = cap["projected_ba"].to(DEVICE)
        core_attn_out_vllm = cap["core_attn_out"]  # captured (may have non-zero state)
        T = projected_qkvz.shape[0]

        if conv_layout == "headmajor":
            q_r = q_block[rank*512:(rank+1)*512]; k_r = k_block[rank*512:(rank+1)*512]; v_r = v_block[rank*1536:(rank+1)*1536]
            conv_w_rank = torch.cat([q_r, k_r, v_r], dim=0)
        else:
            conv_w_rank = conv_w[rank*2560:(rank+1)*2560]
        A_log_rank = A_log[rank*12:(rank+1)*12]
        dt_bias_rank = dt_bias[rank*12:(rank+1)*12]

        # --- Pure-Python reference (zero state) ---
        qkv, b, a, z = extract_qkv_ba_z(projected_qkvz.cpu(), projected_ba.cpu())
        ref = gdn_core_ref(qkv, b, a, conv_w_rank, A_log_rank, dt_bias_rank).to(DEVICE)

        # --- Actual vLLM kernel (zero state) ---
        core_attn_out = torch.zeros((T, NUM_V_PER_RANK, HEAD_V_DIM), dtype=projected_qkvz.dtype, device=DEVICE)
        z_out = torch.empty_like(core_attn_out)
        conv_state = torch.zeros((1, 3, 2560), dtype=projected_qkvz.dtype, device=DEVICE)
        ssm_state = torch.zeros((1, NUM_V_PER_RANK, HEAD_V_DIM, HEAD_K_DIM), dtype=torch.float32, device=DEVICE)
        conv_weights = conv_w_rank.to(projected_qkvz.dtype).to(DEVICE)  # (2560, 1, 4) -> (2560, 4)
        conv_weights = conv_weights.view(2560, 4)
        A_log_k = A_log_rank.to(DEVICE)
        dt_bias_k = dt_bias_rank.to(projected_qkvz.dtype).to(DEVICE)
        non_spec_query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
        has_initial_state = torch.tensor([False], dtype=torch.bool, device=DEVICE)
        non_spec_state_indices_tensor = torch.tensor([0], dtype=torch.int32, device=DEVICE)

        torch.ops._xpu_C.gdn_attention(
            core_attn_out, z_out, projected_qkvz, projected_ba,
            NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM,
            conv_state=conv_state, ssm_state=ssm_state,
            conv_weights=conv_weights, conv_bias=None, activation="silu",
            A_log=A_log_k, dt_bias=dt_bias_k,
            num_prefills=1, num_decodes=0, num_spec_decodes=0,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=non_spec_query_start_loc,
            non_spec_token_indx=None, non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_query_start_loc=None, spec_token_indx=None, spec_state_indices_tensor=None,
            num_accepted_tokens=None, num_actual_tokens=T,
            tp_size=TP_SIZE, reorder_input=True,
        )
        torch.xpu.synchronize()

        # --- Compare kernel vs reference (both zero state) ---
        cos_kr = F.cosine_similarity(core_attn_out.float().flatten(), ref.float().flatten(), dim=0).item()
        diff_kr = (core_attn_out.float() - ref.float()).abs()
        print(f"\n=== Rank {rank} (T={T}) conv_layout={conv_layout} ===")
        print(f"  KERNEL vs REF (both zero state):")
        print(f"    cos_sim: {cos_kr:.6f}")
        print(f"    max_abs_diff: {diff_kr.max().item():.6f}")
        print(f"    kernel_absmax: {core_attn_out.float().abs().max().item():.6f}")
        print(f"    ref_absmax: {ref.float().abs().max().item():.6f}")
        # --- Compare kernel vs captured vLLM output ---
        cos_kc = F.cosine_similarity(core_attn_out.float().flatten(), core_attn_out_vllm.float().to(DEVICE).flatten(), dim=0).item()
        print(f"  KERNEL vs CAPTURED vLLM (zero state vs captured):")
        print(f"    cos_sim: {cos_kc:.6f}")
        print(f"    captured_absmax: {core_attn_out_vllm.float().abs().max().item():.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()