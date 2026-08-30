# Phase 0: synthetic W4A16 MoE grouped-GEMM test on Arc Pro B70
# Mirrors vllm-xpu-kernels tests/fused_moe/test_grouped_gemm.py::test_xe_grouped_gemm_int4
# exactly: random uint8 packed B, package dequantize_uint4 for the reference,
# package implement_zp to repack for the kernel, raw cutlass_grouped_gemm op.
import time

import torch

import vllm  # noqa: F401  (registers _xpu_C / _moe_C ops)
from vllm_xpu_kernels.fused_moe_interface import (
    cutlass_grouped_gemm_xe2,
    implement_zp,
)

torch.manual_seed(7)
DEV = "xpu"
ops = torch.ops._xpu_C


def dequantize_uint4(qweight, scales, group_size):
    # qweight: [n, k//2] uint8 ; scales: [n, group_num]
    k = qweight.shape[1] * 2
    n = qweight.shape[0]
    dev = qweight.device
    unpack_idx = torch.tensor([0, 1], device=dev)
    data = qweight[:, [i // 2 for i in range(k)]]
    shift = (unpack_idx[[i % 2 for i in range(k)]].to(torch.int32) * 4)
    dst_data = (data.to(torch.int32) >> shift) & 0xF
    expand_scales = scales[:, [i // group_size for i in range(k)]]
    weight_16 = (dst_data - 8) * expand_scales
    return weight_16.to(scales.dtype)


def init_rows_for_experts(tokens, topk, num_rows_per_expert):
    n_experts = num_rows_per_expert.numel()
    rand = torch.rand(tokens, n_experts, device=num_rows_per_expert.device)
    topk_idx = torch.topk(rand, topk, dim=1).indices
    flat_idx = topk_idx.flatten()
    num_rows_per_expert += torch.bincount(flat_idx, minlength=n_experts)


def run_case(m, n, k, e, topk, dtype, has_bias):
    group_size = 128
    group_num = k // group_size
    total_m = m * topk
    input_A = torch.randn((total_m, k), dtype=dtype, device=DEV).contiguous()
    ref_A = input_A
    input_B_uint4 = (
        torch.randint(0, 0xFF, [e, n, k // 2], device=DEV).to(torch.uint8)
    )
    random_exponents = torch.randint(-3, 4, (e, n, group_num), device=DEV)
    scale_B = torch.pow(2.0, random_exponents.float()).to(dtype)
    bias = (
        torch.randn((e, n), dtype=dtype, device=DEV) * 100 if has_bias else None
    )

    input_B_16 = torch.empty(e, n, k, dtype=dtype, device=DEV)
    input_B_int4 = torch.empty_like(input_B_uint4).to(torch.int8)
    for i in range(e):
        input_B_16[i] = dequantize_uint4(input_B_uint4[i], scale_B[i], group_size)
        input_B_int4[i] = implement_zp(input_B_uint4[i])

    num_rows_per_expert = torch.zeros(e, device=DEV, dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)

    output = torch.empty((total_m, n), dtype=dtype, device=DEV)
    cutlass_grouped_gemm_xe2(
        input_A, input_B_int4, scale_B, bias, output,
        num_rows_per_expert, n, k, e,
    )
    torch.xpu.synchronize()

    # reference (fp32 accumulate, like the kernel's mma)
    ref = []
    pre = 0
    for i in range(e):
        cur = int(num_rows_per_expert[i])
        if cur == 0:
            continue
        inp = ref_A[pre:pre + cur, :].to(torch.float32)
        w = input_B_16[i, :, :].to(torch.float32)
        out = inp @ w.T
        if has_bias:
            out = out + bias[i].to(torch.float32)
        ref.append(out.to(dtype))
        pre += cur
    ref = torch.cat(ref, dim=0) if ref else torch.empty(0, n, dtype=dtype, device=DEV)

    out_c = output.cpu().float()
    ref_c = ref.cpu().float()
    if ref_c.numel() == 0:
        return True, 0.0, 1.0
    ad = (out_c - ref_c).abs()
    cos = torch.nn.functional.cosine_similarity(
        out_c.flatten(), ref_c.flatten(), dim=0
    ).item()
    ok = ad.max().item() < 0.05 and cos > 0.999
    print(
        f"  m={m:3d} n={n:5d} k={k:5d} e={e:3d} topk={topk} "
        f"{str(dtype):15s} bias={int(has_bias)} "
        f"max_abs={ad.max().item():.4f} cos={cos:.6f} {'PASS' if ok else 'FAIL'}"
    )
    return ok, ad.max().item(), cos


def bench(m, n, k, e, topk, dtype):
    group_size = 128
    group_num = k // group_size
    total_m = m * topk
    input_A = torch.randn((total_m, k), dtype=dtype, device=DEV).contiguous()
    input_B_uint4 = (
        torch.randint(0, 0xFF, [e, n, k // 2], device=DEV).to(torch.uint8)
    )
    scale_B = torch.pow(
        2.0, torch.randint(-3, 4, (e, n, group_num), device=DEV).float()
    ).to(dtype)
    input_B_int4 = torch.empty_like(input_B_uint4).to(torch.int8)
    for i in range(e):
        input_B_int4[i] = implement_zp(input_B_uint4[i])
    num_rows_per_expert = torch.zeros(e, device=DEV, dtype=torch.int32)
    init_rows_for_experts(m, topk, num_rows_per_expert)
    output = torch.empty((total_m, n), dtype=dtype, device=DEV)
    for _ in range(10):
        cutlass_grouped_gemm_xe2(
            input_A, input_B_int4, scale_B, None, output,
            num_rows_per_expert, n, k, e,
        )
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    iters = 100
    for _ in range(iters):
        cutlass_grouped_gemm_xe2(
            input_A, input_B_int4, scale_B, None, output,
            num_rows_per_expert, n, k, e,
        )
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    print(f"  m={m:3d} n={n:5d} k={k:5d} e={e:3d} topk={topk}: {dt*1e3:9.3f} ms/step")
    return dt


if __name__ == "__main__":
    print(f"torch {torch.__version__}, vllm {vllm.__version__}")
    print(f"device0: {torch.xpu.get_device_name(0)}")
    print(f"is_xe2_arch={ops.is_xe2_arch()}  is_xe3_arch={ops.is_xe3_arch()}")

    print("\n=== correctness (upstream int4 grouped gemm shapes) ===")
    allok = True
    for dtype in (torch.bfloat16, torch.float16):
        for has_bias in (False, True):
            for (m, n, k) in ((1, 640, 2560), (8, 640, 2560), (32, 640, 2560)):
                allok &= run_case(m, n, k, 16, 1, dtype, has_bias)[0]

    print("\n=== throughput (bf16 A, int4 B, expert-shaped) ===")
    for m in (1, 8, 32, 256):
        bench(m, 640, 2560, 16, 1, torch.bfloat16)

    print("\nALL PASS" if allok else "\nSOME CHECKS FAILED")