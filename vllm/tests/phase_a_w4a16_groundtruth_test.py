"""Phase A / A9 debug: FP8 ground-truth comparison for the W4A16 int4 path.

Disambiguates the token-soup root cause. The prior real-weight test compared
the XPU kernel output against MY `local_dequant` CPU reference and FAILED
(cos ~ -0.3), but that cannot tell us WHICH side is wrong:

  (a) kernel correct, my local_dequant reference wrong  -> soup in model def
  (b) kernel / int4 packing convention wrong            -> W4A16 path is the bug

Ground truth = the FP8 model's TRUE weights (already on disk):
  - full-attn q/k/v/o_proj : exact bf16 (in modules_to_not_convert, not fp8)
  - MoE experts            : fp8 `weight` * `weight_scale_inv` (128x128 block)

Two checks per tensor:
  1. weight-level : cos(local_dequant(W4A16), W_true)
       high -> my dequant convention recovers the true weights
  2. GEMM-level   : cos(int4_gemm_w4a16(...), A @ W_true.t())
       high -> the kernel output matches the true GEMM

Decision:
  both high  -> W4A16 path is correct -> soup is in the model definition
  either low -> int4 packing convention is the bug -> fix the W4A16 path

Single GPU, memory-light. Run:  python phase_a_w4a16_groundtruth_test.py
"""

import glob
import json
import os

import torch

import vllm  # noqa: F401  (registers torch.ops._xpu_C)
from safetensors import safe_open

W4A16 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
FP8 = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8"
DEV = "xpu"
ops = torch.ops._xpu_C

print(f"torch {torch.__version__}, vllm {vllm.__version__}")
print(f"device0: {torch.xpu.get_device_name(0)}")
print(f"is_xe3_arch={ops.is_xe3_arch()}")


def local_dequant(w_packed, scales, group_size):
    """Dequantize [out, in/8] int32 + [out, in/g] scales -> [out, in] float32.

    Nibble convention (matches phase 0 + the XPU kernel): within each int32,
    input element k (k=0..7) lives in nibble (k % 8) of byte (k // 8); on a
    little-endian host the int32->uint8 view exposes byte0 = [n1<<4 | n0] so
    the lower nibble = the lower input index. Symmetric, zero-point = 8.
    """
    w = w_packed.to(torch.int64)
    N, K = w_packed.shape[0], w_packed.shape[1] * 8  # N=out, K=in
    lanes = []
    for i in range(8):
        v = (w >> (4 * i)) & 0xF
        v = torch.where(v >= 8, v - 16, v).to(torch.float32)
        lanes.append(v)
    wu = torch.stack(lanes, dim=2).reshape(N, K)
    G = K // group_size
    s = scales.to(torch.float32).repeat_interleave(group_size, dim=1)  # [N, K]
    return (wu - 8.0) * s  # symmetric: zp=8


def _load(model_dir, key):
    """Load a single tensor `key` from `model_dir` via its weight_map."""
    idx = json.load(open(glob.glob(model_dir + "/*.index.json")[0]))
    wm = idx["weight_map"]
    f = os.path.join(model_dir, wm[key])
    with safe_open(f, framework="pt") as sf:
        return sf.get_tensor(key)


def fp8_block_dequant(weight, scale_inv, block=128):
    """Dequant a block-scaled fp8 weight -> float32 [out, in]."""
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return weight.to(torch.float32) * s


def check(name, kind, M=16):
    """`kind` is 'full' (bf16 ground truth) or 'moe' (fp8 block ground truth)."""
    # --- W4A16 side (what the kernel consumes) ---
    w_packed = _load(W4A16, name + ".weight_packed")  # [out, in/8] int32
    w_scale = _load(W4A16, name + ".weight_scale")    # [out, in/128] bf16
    w_shape = _load(W4A16, name + ".weight_shape")    # [out, in] int32
    out, inp = int(w_shape[0]), int(w_shape[1])
    assert w_packed.shape == (out, inp // 8), (w_packed.shape, out, inp)
    assert w_scale.shape == (out, inp // 128), (w_scale.shape, out, inp)

    # --- FP8 ground-truth side (the TRUE weights) ---
    if kind == "full":
        W_true = _load(FP8, name + ".weight").to(torch.float32)  # bf16 -> f32
    else:  # moe
        w_fp8 = _load(FP8, name + ".weight")
        s_inv = _load(FP8, name + ".weight_scale_inv")
        W_true = fp8_block_dequant(w_fp8, s_inv, 128)  # [out, in] f32
    assert W_true.shape == (out, inp), (W_true.shape, out, inp)

    tag = name.split("layers.")[-1]
    print(f"\n=== {tag}  [out={out}, in={inp}]  ({kind}) ===")

    # --- Check 1: weight-level (my dequant vs ground truth) ---
    W_local = local_dequant(w_packed.cpu(), w_scale.cpu(), 128)  # [out, in] f32
    cos_w = torch.nn.functional.cosine_similarity(
        W_local.flatten(), W_true.flatten(), dim=0
    ).item()

    # --- Check 2: GEMM-level (kernel output vs true GEMM) ---
    torch.manual_seed(0)
    A = torch.randn(M, inp, dtype=torch.bfloat16, device=DEV)
    w_q = w_packed.to(DEV)                       # [out, in/8] int32
    w_s = w_scale.to(DEV).t().contiguous()       # [in/128, out]
    w_zp = torch.tensor([8], dtype=torch.int8, device=DEV)
    out_xpu = ops.int4_gemm_w4a16(A, w_q.t(), None, w_s, w_zp, 128, None)

    A_cpu = A.cpu().float()
    ref_true = A_cpu @ W_true.t()                # [M, out]  (ground truth)
    ref_local = A_cpu @ W_local.t()              # [M, out]  (my dequant)
    out_c = out_xpu.cpu().float()

    cos_x_true = torch.nn.functional.cosine_similarity(
        out_c.flatten(), ref_true.flatten(), dim=0
    ).item()
    cos_x_local = torch.nn.functional.cosine_similarity(
        out_c.flatten(), ref_local.flatten(), dim=0
    ).item()

    print(f"  [1] cos(W_local, W_true)      = {cos_w:+.6f}")
    print(f"  [2] cos(out_xpu, ref_true)    = {cos_x_true:+.6f}")
    print(f"  [3] cos(out_xpu, ref_local)   = {cos_x_local:+.6f}")
    return cos_w, cos_x_true, cos_x_local


def main():
    results = {}
    # Full-attention W4A16 linears (layer 3 is a full_attention layer).
    base = "model.language_model.layers.3.self_attn"
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        results[f"{proj}"] = check(f"{base}.{proj}", "full")
    # A GDN-layer MoE expert is W4A16 too (layer 0 is linear_attention).
    results["exp0.gate"] = check(
        "model.language_model.layers.0.mlp.experts.0.gate_proj", "moe")
    results["exp0.down"] = check(
        "model.language_model.layers.0.mlp.experts.0.down_proj", "moe")

    print("\n" + "=" * 64)
    print(f"{'tensor':<14}{'cos(W,Wt)':>12}{'cos(x,xt)':>12}{'cos(x,xl)':>12}")
    for k, (cw, cx_t, cx_l) in results.items():
        print(f"{k:<14}{cw:>+12.4f}{cx_t:>+12.4f}{cx_l:>+12.4f}")

    # Decision: int4 quantization keeps cos below 1.0; > 0.95 is a strong match.
    # (my local_dequant is a *reference* I wrote, so it can be wrong even when
    #  the kernel is right -- that is exactly what this test disambiguates)
    thr = 0.95
    w_ok = all(r[0] > thr for r in results.values())
    x_ok = all(r[1] > thr for r in results.values())
    print("\n" + "=" * 64)
    print(f"my dequant convention correct (cos(W,Wt)>{thr}) : {w_ok}")
    print(f"kernel convention correct     (cos(x,xt)>{thr}) : {x_ok}")
    if w_ok and x_ok:
        print("VERDICT: W4A16 path is CORRECT -> soup is in the model definition")
        print("         (HC flow / GDN wiring / attention / mrope).")
    elif (not w_ok) and (not x_ok):
        print("VERDICT: BOTH my dequant and the kernel disagree with ground truth.")
        print("         -> the W4A16 checkpoint packing itself is the problem,")
        print("            or the FP8 ground truth is not comparable.")
    elif w_ok and not x_ok:
        print("VERDICT: my dequant is right, the KERNEL is wrong.")
        print("         -> fix XPUwNa16LinearKernel / _process_weights_xpu.")
    else:  # not w_ok and x_ok
        print("VERDICT: the KERNEL is right, my local_dequant reference is wrong.")
        print("         -> W4A16 vLLM path is fine; soup is in the model definition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())