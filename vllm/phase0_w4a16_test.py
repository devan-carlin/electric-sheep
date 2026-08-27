# Phase 0: synthetic W4A16 GEMM test on Arc Pro B70
# Mirrors vllm-xpu-kernels tests/test_int4_gemm_onednn.py conventions.
import time

import torch

import vllm  # noqa: F401  (triggers _xpu_C op registration)

torch.manual_seed(1234)
DEV = "xpu"
ops = torch.ops._xpu_C

print(f"torch {torch.__version__}, vllm {vllm.__version__}")
print(f"device0: {torch.xpu.get_device_name(0)}")
print(f"is_xe2_arch={ops.is_xe2_arch()}  is_xe3_arch={ops.is_xe3_arch()}")

try:
    from vllm_xpu_kernels.quantization._quantize_convert import dequantize

    HAVE_DEQ = True
except Exception as e:  # noqa: BLE001
    HAVE_DEQ = False
    print(f"WARNING: no dequantize helper ({e}); using local unpack")


def rand_int4(size, dtype=torch.int32, device=DEV):
    # same trick as upstream test: size//2 int8 bytes reinterpreted as int32
    rand = torch.randint(-128, 128, [size // 2], device=device).to(torch.int8)
    return rand.view(dtype=dtype)


def local_dequant(w_packed, scales, group_size):
    # w_packed: [K/8, N] int32, 8 int4 per int32 (lane i = bits 4i..4i+3)
    w = w_packed.to(torch.int64)
    K, N = w_packed.shape[0] * 8, w_packed.shape[1]
    lanes = []
    for i in range(8):
        v = (w >> (4 * i)) & 0xF
        v = torch.where(v >= 8, v - 16, v).to(torch.float32)
        lanes.append(v)
    # interleave lanes back into K order: element k = lane (k % 8), row k // 8
    wu = torch.stack(lanes, dim=1).reshape(K, N)  # [K/8, 8, N] -> [K, N]
    G = K // group_size
    s = scales.to(torch.float32).repeat_interleave(group_size, dim=0)  # [K, N]
    return (wu - 8.0) * s  # symmetric: zp=8


def make_case(M, N, K, group_size=128, dtype=torch.bfloat16):
    A = torch.randn([M, K], device=DEV, dtype=dtype)
    W = rand_int4(K * N).reshape(K // 8, N)
    G = K // group_size
    S = (torch.rand([G, N], device=DEV, dtype=dtype) * 0.05 + 0.01)
    ZP = torch.tensor([8], dtype=torch.int8, device=DEV)
    Wba = W.t().contiguous().t()  # BA layout, as in upstream test
    return A, W, S, ZP, Wba


def ref_out(A, W, S, group_size):
    if HAVE_DEQ:
        # symmetric mode: dequantize takes None for zero_points (zp=8 implicit)
        Wf = dequantize(W, S, None, group_size, None).cpu().float()
    else:
        Wf = local_dequant(W.cpu(), S.cpu(), group_size)
    return A.cpu().float() @ Wf


def check(M, N, K, dtype, group_size=128):
    A, W, S, ZP, Wba = make_case(M, N, K, group_size, dtype)
    out = ops.int4_gemm_w4a16(A, Wba, torch.Tensor(), S, ZP, group_size, None)
    ref = ref_out(A, W, S, group_size)
    out_c = out.cpu().float()
    ad = (out_c - ref).abs()
    cos = torch.nn.functional.cosine_similarity(
        out_c.flatten(), ref.flatten(), dim=0
    ).item()
    ok = ad.max().item() < 0.1 and cos > 0.999
    print(
        f"  M={M:4d} N={N:5d} K={K:5d} {str(dtype):22s} "
        f"max_abs={ad.max().item():.4f} cos={cos:.6f} {'PASS' if ok else 'FAIL'}"
    )
    return ok


def bench(M, N, K, dtype, iters=100):
    A, W, S, ZP, Wba = make_case(M, N, K, 128, dtype)
    for _ in range(10):
        out = ops.int4_gemm_w4a16(A, Wba, torch.Tensor(), S, ZP, 128, None)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = ops.int4_gemm_w4a16(A, Wba, torch.Tensor(), S, ZP, 128, None)
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    nbytes = N * K // 2 + M * K * 2 + M * N * 2  # int4 W + bf16 A + bf16 D
    gbps = nbytes / dt / 1e9
    tflops = 2 * M * N * K / dt / 1e12
    print(
        f"  M={M:4d} N={N:5d} K={K:5d}: {dt * 1e3:9.3f} ms  "
        f"{gbps:8.1f} GB/s  {tflops:7.3f} TFLOPS"
    )
    return dt


if __name__ == "__main__":
    print("\n=== correctness (expert-shaped: N=640, K=2560, g=128) ===")
    allok = True
    for dtype in (torch.bfloat16, torch.float16):
        for M in (1, 8, 32):
            allok &= check(M, 640, 2560, dtype)
    # upstream test shapes (fp16)
    print("\n=== correctness (upstream shapes, fp16) ===")
    for (m, n, k) in ((1, 4096, 11008), (32, 4096, 4096)):
        allok &= check(m, n, k, torch.float16)

    print("\n=== bandwidth / throughput (bf16, expert-shaped) ===")
    for M in (1, 8, 32, 256):
        bench(M, 640, 2560, torch.bfloat16)

    print("\n=== bandwidth (bf16, attention-shaped: N=K=2560) ===")
    for M in (1, 32, 256):
        bench(M, 2560, 2560, torch.bfloat16)

    print("\nALL PASS" if allok else "\nSOME CHECKS FAILED")