"""Phase A / A9 debug: validate W4A16 dense linear with REAL checkpoint weights.

The token-soup bug is most likely an int4 nibble-packing convention mismatch
between the checkpoint's `pack-quantized` int32 layout and the XPU
`int4_gemm_w4a16` kernel. Phase 0 validated the kernel with SYNTHETIC weights
packed in the kernel's own convention; this test uses REAL checkpoint weights
to check the convention actually matches.

Loads one real W4A16 linear (layer-3 self_attn.q_proj) from the W4A16
checkpoint, dequantizes on CPU using the kernel's nibble convention, runs the
XPU kernel exactly as vLLM's XPUwNa16LinearKernel.apply_weights does, and
compares against the CPU dequant reference.

  PASS -> checkpoint packing matches the kernel; dense-linear path is correct.
  FAIL -> packing convention mismatch is the root cause of the soup.

Single GPU, memory-light. Run:  python phase_a_w4a16_realweight_test.py
"""

import glob
import json
import os

import torch

import vllm  # noqa: F401  (registers torch.ops._xpu_C)
from safetensors import safe_open

MODEL = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
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
    # element k = lane (k % 8), row k // 8  ->  [N, K/8, 8] -> [N, K]
    wu = torch.stack(lanes, dim=2).reshape(N, K)
    G = K // group_size
    s = scales.to(torch.float32).repeat_interleave(group_size, dim=1)  # [N, K]
    return (wu - 8.0) * s  # symmetric: zp=8


def check(name, M=16):
    idx = json.load(open(glob.glob(MODEL + "/*.index.json")[0]))
    wm = idx["weight_map"]
    f = os.path.join(MODEL, wm[name + ".weight_packed"])
    with safe_open(f, framework="pt") as sf:
        w_packed = sf.get_tensor(name + ".weight_packed")  # [out, in/8] int32
        w_scale = sf.get_tensor(name + ".weight_scale")  # [out, in/128] bf16
        w_shape = sf.get_tensor(name + ".weight_shape")  # [out, in] int32
    out, inp = int(w_shape[0]), int(w_shape[1])
    assert w_packed.shape == (out, inp // 8), (w_packed.shape, out, inp)
    assert w_scale.shape == (out, inp // 128), (w_scale.shape, out, inp)
    print(f"\n=== {name.split('.')[-3]}.{name.split('.')[-2]} "
          f"[out={out}, in={inp}] ===")

    # CPU dequant reference: W [out, in], ref = A @ W^T
    W_cpu = local_dequant(w_packed.cpu(), w_scale.cpu(), 128)  # [out, in] f32

    torch.manual_seed(0)
    A = torch.randn(M, inp, dtype=torch.bfloat16, device=DEV)

    # XPU kernel in vLLM's EXACT format (XPUwNa16LinearKernel.apply_weights):
    #   w_q = weight_packed [out, in/8] int32  (as loaded, no transpose)
    #   w_s = weight_scale.t().contiguous()    [in/128, out]
    #   w_zp = tensor([8]) int8, group_size=128, w_gidx=None
    #   out = int4_gemm_w4a16(x, w_q.t(), None, w_s, w_zp, 128, None)
    w_q = w_packed.to(DEV)  # [out, in/8] int32
    w_s = w_scale.to(DEV).t().contiguous()  # [in/128, out]
    w_zp = torch.tensor([8], dtype=torch.int8, device=DEV)
    out_xpu = ops.int4_gemm_w4a16(A, w_q.t(), None, w_s, w_zp, 128, None)

    ref = (A.cpu().float() @ W_cpu.t())  # [M, out]
    out_c = out_xpu.cpu().float()
    ad = (out_c - ref).abs()
    cos = torch.nn.functional.cosine_similarity(
        out_c.flatten(), ref.flatten(), dim=0
    ).item()
    ok = cos > 0.999
    print(f"  max_abs_err={ad.max().item():.4f}  cos={cos:.6f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    allok = True
    # Full-attention W4A16 linears (layer 3 is a full_attention layer).
    base = "model.language_model.layers.3.self_attn"
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        allok &= check(f"{base}.{proj}")
    # A GDN-layer MoE expert is W4A16 too (layer 0 is linear_attention).
    allok &= check("model.language_model.layers.0.mlp.experts.0.gate_proj")
    allok &= check("model.language_model.layers.0.mlp.experts.0.down_proj")

    print("\n" + "=" * 60)
    if allok:
        print("PASS: real-weight W4A16 matches CPU dequant reference")
        print("-> checkpoint int4 packing matches the XPU kernel convention.")
        print("-> dense-linear + expert GEMM paths are correct; the soup bug")
        print("   is in the model definition (HC / GDN / attention / mrope).")
    else:
        print("FAIL: real-weight W4A16 does NOT match the CPU reference.")
        print("-> int4 nibble-packing convention mismatch is the root cause.")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())