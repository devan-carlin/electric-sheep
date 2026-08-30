#!/usr/bin/env python3
"""
Non-circular GDN in_proj test.

Recompute the GDN in_proj (in_proj_qkvz) from:
  - captured mix_out (HC mix output = GDN input, (5, 2560))
  - checkpoint in_proj_qkv (10240, 2560) + in_proj_z (6144, 2560)
and compare to vLLM's captured projected_qkvz (5, 4096) per rank.

The checkpoint stores GDN weights in INTERLEAVED GQA order:
  in_proj_qkv = [g0_q(128), g0_k(128), g0_v(384), g1_q(128), ...]  (16 groups x 640)
  in_proj_z   = [g0_z(384), g1_z(384), ...]                        (16 groups x 384)
vLLM fuses into in_proj_qkvz = [g0_q, g0_k, g0_v, g0_z, g1_q, ...] (16 groups x 1024)
Per-rank shard (TP=4) = 4 groups x 1024 = 4096.
"""
import os
import torch
import torch.nn.functional as F
from safetensors import safe_open

NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
TP_SIZE = 4
REP = NUM_V_HEADS // NUM_K_HEADS  # 3
CKPT = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16/model-00001.safetensors"


def main():
    with safe_open(CKPT, framework="pt") as f:
        in_proj_qkv = f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight").float()  # (10240, 2560)
        in_proj_z = f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_z.weight").float()  # (6144, 2560)
        in_proj_a = f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_a.weight").float()  # (48, 2560)
        in_proj_b = f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_b.weight").float()  # (48, 2560)
    print(f"in_proj_qkv: {tuple(in_proj_qkv.shape)}, in_proj_z: {tuple(in_proj_z.shape)}")

    # Build fused in_proj_qkvz weight.
    # gqa_interleaved_layout=False -> HEAD-MAJOR: [q_all(2048), k_all(2048), v_all(6144), z_all(6144)]
    # in_proj_qkv (10240, 2560) = [q_all(2048), k_all(2048), v_all(6144)]
    # in_proj_z   (6144, 2560)  = [z_all(6144)]
    # MergedColumnParallelLinear output_sizes=[2048,2048,6144,6144] shards EACH
    # sub-block independently across TP. Per-rank weight = [q_r, k_r, v_r, z_r].
    q_all = in_proj_qkv[0:2048]        # (2048, 2560)
    k_all = in_proj_qkv[2048:4096]     # (2048, 2560)
    v_all = in_proj_qkv[4096:10240]    # (6144, 2560)
    z_all = in_proj_z                  # (6144, 2560)
    b_all = in_proj_b                  # (48, 2560)
    a_all = in_proj_a                  # (48, 2560)

    # Load captures
    hc = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    mix_out = hc["mix_out"].float()  # (5, 2560) - same across all ranks

    for rank in range(TP_SIZE):
        gdn = torch.load(f"/tmp/qwen4exp_gdn_capture_L0_R{rank}.pt", map_location="cpu")
        projected_qkvz = gdn["projected_qkvz"].float()  # (5, 4096)
        projected_ba = gdn["projected_ba"].float()  # (5, 24)

        # Per-rank shard: slice each sub-block independently (column-parallel)
        q_r = q_all[rank * 512:(rank + 1) * 512]      # (512, 2560)
        k_r = k_all[rank * 512:(rank + 1) * 512]      # (512, 2560)
        v_r = v_all[rank * 1536:(rank + 1) * 1536]    # (1536, 2560)
        z_r = z_all[rank * 1536:(rank + 1) * 1536]    # (1536, 2560)
        fused_qkvz_rank = torch.cat([q_r, k_r, v_r, z_r], dim=0)  # (4096, 2560)

        b_r = b_all[rank * 12:(rank + 1) * 12]        # (12, 2560)
        a_r = a_all[rank * 12:(rank + 1) * 12]        # (12, 2560)
        fused_ba_rank = torch.cat([b_r, a_r], dim=0)  # (24, 2560)

        # Recompute in_proj
        in_proj_qkvz_ref = mix_out @ fused_qkvz_rank.T  # (5, 4096)
        in_proj_ba_ref = mix_out @ fused_ba_rank.T  # (5, 24)

        # Compare
        cos_qkvz = F.cosine_similarity(in_proj_qkvz_ref.flatten(), projected_qkvz.flatten(), dim=0).item()
        diff_qkvz = (in_proj_qkvz_ref - projected_qkvz).abs().max().item()
        cos_ba = F.cosine_similarity(in_proj_ba_ref.flatten(), projected_ba.flatten(), dim=0).item()
        diff_ba = (in_proj_ba_ref - projected_ba).abs().max().item()
        print(f"\n=== Rank {rank} (headmajor, correct slicing) ===")
        print(f"  in_proj_qkvz: cos={cos_qkvz:.6f} max_diff={diff_qkvz:.6f}")
        print(f"    ref_absmax={in_proj_qkvz_ref.abs().max().item():.4f} vllm_absmax={projected_qkvz.abs().max().item():.4f}")
        print(f"  in_proj_ba:   cos={cos_ba:.6f} max_diff={diff_ba:.6f}")
        print(f"    ref_absmax={in_proj_ba_ref.abs().max().item():.4f} vllm_absmax={projected_ba.abs().max().item():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()