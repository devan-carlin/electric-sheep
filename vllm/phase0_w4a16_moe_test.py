# Phase 0: synthetic W4A16 MoE grouped-GEMM test on Arc Pro B70
# Exercises the real vLLM path: vllm_xpu_kernels.fused_moe_interface.XpuFusedMoe
# with is_int4=True (uint8, 2 nibbles/byte, symmetric u4 zp=8, group-128).
import time

import torch

import vllm  # noqa: F401  (registers _xpu_C / _moe_C ops)
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

torch.manual_seed(7)
DEV = "xpu"

# expert-shaped dims (divisible by 8, group_size divides K)
E = 8          # num experts
HIDDEN = 256   # hidden_size (K of gemm1, N of gemm2)
INTER = 128    # moe_intermediate_size (N of gemm2, half of gemm1 N)
GS = 128       # group size
TOPK = 2       # experts per token
T = 16         # tokens
DT = torch.bfloat16


def quantize_u4(W, group_size):
    """W [E,N,K] float -> packed uint8 [E,N,K//2] (hi=even, lo=odd), scales [E,N,K//gs] float."""
    En, N, K = W.shape
    G = K // group_size
    Wg = W.reshape(En, N, G, group_size)
    amax = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0  # u4 symmetric usable range -7..7 (8 == zero)
    q = torch.round(Wg / scale).clamp(-7, 7)
    q_u4 = (q + 8).to(torch.uint8)
    hi = q_u4[..., 0::2]
    lo = q_u4[..., 1::2]
    packed = ((hi << 4) | lo).reshape(En, N, K // 2)
    scales = scale.squeeze(-1).float()
    return packed, scales


def dequant_u4(packed, scales, group_size):
    """packed uint8 [E,N,K//2] + scales [E,N,G] -> float [E,N,K]."""
    En, N, Khalf = packed.shape
    K = Khalf * 2
    hi = ((packed >> 4) & 0xF).float()
    lo = (packed & 0xF).float()
    q = torch.stack([hi, lo], dim=-1).reshape(En, N, K)  # 2j=hi, 2j+1=lo
    q = q - 8.0
    s = scales.repeat_interleave(group_size, dim=-1)  # [E,N,K]
    return q * s


def silu(x):
    return x * torch.sigmoid(x)


def ref_moe(hidden, w13f, w2f, topk_ids, topk_weights, inter):
    Tn, H = hidden.shape
    out = torch.zeros(Tn, H, dtype=torch.float32)
    for t in range(Tn):
        for i in range(topk_ids.shape[1]):
            e = int(topk_ids[t, i])
            w = float(topk_weights[t, i])
            x = hidden[t].float() @ w13f[e].t()          # [2*inter]
            gate, up = x[:inter], x[inter:]
            act = silu(gate) * up                          # [inter]
            y = act @ w2f[e].t()                           # [hidden]
            out[t] += w * y
    return out


def main():
    print(f"torch {torch.__version__}, vllm {vllm.__version__}")
    print(f"device0: {torch.xpu.get_device_name(0)}")

    # random weights
    w13_f = torch.randn(E, 2 * INTER, HIDDEN, dtype=torch.float32) * 0.5
    w2_f = torch.randn(E, HIDDEN, INTER, dtype=torch.float32) * 0.5

    w13_p, w13_s = quantize_u4(w13_f, GS)   # [E,2I,H/2] u8, [E,2I,H/GS] f32
    w2_p, w2_s = quantize_u4(w2_f, GS)       # [E,H,I/2] u8, [E,H,I/GS] f32

    # dequantized float weights for the reference (matches what the kernel computes)
    w13_dq = dequant_u4(w13_p, w13_s, GS)    # [E,2I,H]
    w2_dq = dequant_u4(w2_p, w2_s, GS)       # [E,H,I]

    hidden = (torch.randn(T, HIDDEN, dtype=torch.float32) * 0.5).to(DT).to(DEV)
    topk_ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device=DEV)
    topk_w = torch.softmax(torch.randn(T, TOPK, device=DEV, dtype=torch.float32), dim=-1)

    # --- kernel path ---
    moe = XpuFusedMoe(
        w13_p.to(DEV), w13_s.to(DEV).to(DT), None,
        w2_p.to(DEV), w2_s.to(DEV).to(DT), None,
        TOPK, "silu", E,
    )
    output = torch.empty(T, HIDDEN, dtype=DT, device=DEV)
    moe.apply(output, hidden, topk_w, topk_ids)
    torch.xpu.synchronize()
    out_k = output.cpu().float()

    # --- reference ---
    ref = ref_moe(hidden.cpu().float(), w13_dq.cpu(), w2_dq.cpu(),
                  topk_ids.cpu(), topk_w.cpu(), INTER)

    ad = (out_k - ref).abs()
    denom = ref.abs().clamp(min=1e-6)
    rel = (ad / denom)
    cos = torch.nn.functional.cosine_similarity(
        out_k.flatten(), ref.flatten(), dim=0).item()
    print(f"\nmax_abs={ad.max().item():.4f}  mean_abs={ad.mean().item():.5f}  "
          f"max_rel={rel.max().item():.4f}  cos={cos:.6f}")
    ok = ad.max().item() < 0.05 and cos > 0.999
    print("PASS" if ok else "FAIL")

    # --- bandwidth on expert-shaped grouped GEMM (decode: 1 token/expert) ---
    print("\n=== grouped GEMM throughput (bf16 A, int4 B) ===")
    for tokens in (1, 8, 32):
        h = (torch.randn(tokens, HIDDEN, dtype=torch.float32) * 0.5).to(DT).to(DEV)
        tid = torch.randint(0, E, (tokens, TOPK), dtype=torch.int32, device=DEV)
        tw = torch.softmax(torch.randn(tokens, TOPK, device=DEV, dtype=torch.float32), dim=-1)
        o = torch.empty(tokens, HIDDEN, dtype=DT, device=DEV)
        for _ in range(10):
            moe.apply(o, h, tw, tid)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        iters = 50
        for _ in range(iters):
            moe.apply(o, h, tw, tid)
        torch.xpu.synchronize()
        dt = (time.perf_counter() - t0) / iters
        # bytes ~ A(bf16) + W13(int4) + W2(int4) + D(bf16), per active expert set
        print(f"  tokens={tokens:3d}: {dt*1e3:9.3f} ms/step")

    print("\nDONE")


if __name__ == "__main__":
    main()