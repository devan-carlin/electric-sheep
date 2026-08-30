"""Phase A / A9 debug: FP8 ground-truth test for the MoE int4 path.

The dense W4A16 path (int4_gemm_w4a16) is verified correct vs FP8 ground
truth. The MoE path is a DIFFERENT kernel: XpuFusedMoe ->
cutlass_grouped_gemm_interface, with a compact signed int4 packing
(implement_zp). This test runs that kernel for a single expert and compares
against the true MoE output computed from the FP8 model's true weights.

  out_true = (silu(A @ Wgate.T) * (A @ Wup.T)) @ Wdown.T

  PASS (cos > 0.95) -> MoE int4 path is correct -> soup is in the model def.
  FAIL              -> the MoE int4 kernel / packing is the bug.

Single GPU, memory-light. Run:  python phase_a_moe_groundtruth_test.py
"""

import glob
import json
import os

import torch

import vllm  # noqa: F401  (registers torch.ops._xpu_C)
from safetensors import safe_open
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

W4A16 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
FP8 = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8"
DEV = "xpu"

print(f"torch {torch.__version__}, vllm {vllm.__version__}")
print(f"device0: {torch.xpu.get_device_name(0)}")


def _load(model_dir, key):
    idx = json.load(open(glob.glob(model_dir + "/*.index.json")[0]))
    wm = idx["weight_map"]
    f = os.path.join(model_dir, wm[key])
    with safe_open(f, framework="pt") as sf:
        return sf.get_tensor(key)


def fp8_block_dequant(weight, scale_inv, block=128):
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return weight.to(torch.float32) * s


def main():
    # Expert 0 of layer 0 (a GDN layer, but the MoE block is identical).
    base = "model.language_model.layers.0.mlp.experts.0"
    hidden = 2560      # K
    inter = 640        # N (moe_intermediate)
    group = 128

    # --- W4A16 side: build the MoE kernel's input format ---
    # gate/up: weight_packed [N, K/8] int32 -> uint8 [N, K/2]
    # down:    weight_packed [K, N/8] int32 -> uint8 [K, N/2]
    gate_p = _load(W4A16, base + ".gate_proj.weight_packed")   # [640, 320] i32
    up_p = _load(W4A16, base + ".up_proj.weight_packed")       # [640, 320] i32
    down_p = _load(W4A16, base + ".down_proj.weight_packed")   # [2560, 80] i32
    gate_s = _load(W4A16, base + ".gate_proj.weight_scale")    # [640, 20] bf16
    up_s = _load(W4A16, base + ".up_proj.weight_scale")        # [640, 20] bf16
    down_s = _load(W4A16, base + ".down_proj.weight_scale")    # [2560, 5] bf16

    assert gate_p.shape == (inter, hidden // 8), gate_p.shape
    assert down_p.shape == (hidden, inter // 8), down_p.shape

    # w13: [1, 2N, K/2] uint8  (gate rows then up rows)
    w13 = torch.cat(
        [gate_p.view(torch.uint8), up_p.view(torch.uint8)], dim=0
    ).unsqueeze(0).contiguous()  # [1, 1280, 1280]
    # w2:  [1, K, N/2] uint8
    w2 = down_p.view(torch.uint8).unsqueeze(0).contiguous()  # [1, 2560, 320]

    w13_s = torch.cat([gate_s, up_s], dim=0).unsqueeze(0).contiguous()  # [1,1280,20]
    w2_s = down_s.unsqueeze(0).contiguous()  # [1, 2560, 5]

    # --- FP8 ground truth: true weights [N, K] / [K, N] ---
    Wgate_true = fp8_block_dequant(
        _load(FP8, base + ".gate_proj.weight"),
        _load(FP8, base + ".gate_proj.weight_scale_inv"))
    Wup_true = fp8_block_dequant(
        _load(FP8, base + ".up_proj.weight"),
        _load(FP8, base + ".up_proj.weight_scale_inv"))
    Wdown_true = fp8_block_dequant(
        _load(FP8, base + ".down_proj.weight"),
        _load(FP8, base + ".down_proj.weight_scale_inv"))
    assert Wgate_true.shape == (inter, hidden), Wgate_true.shape
    assert Wdown_true.shape == (hidden, inter), Wdown_true.shape

    print(f"\n=== MoE expert 0 (layer 0)  [K={hidden}, N={inter}] ===")
    print(f"  w13 {tuple(w13.shape)} {w13.dtype}  w2 {tuple(w2.shape)} {w2.dtype}")

    # --- Run the XPU fused-MoE kernel (single token -> expert 0) ---
    torch.manual_seed(0)
    A = torch.randn(1, hidden, dtype=torch.bfloat16, device=DEV)

    moe = XpuFusedMoe(
        w13=w13.to(DEV),
        w13_scales=w13_s.to(DEV),
        w13_bias=None,
        w2=w2.to(DEV),
        w2_scales=w2_s.to(DEV),
        w2_bias=None,
        n_experts_per_token=1,
        activation="silu",
        num_experts=1,
        ep_rank=0,
        ep_size=1,
    )
    output = torch.empty(1, hidden, dtype=torch.bfloat16, device=DEV)
    topk_weights = torch.ones(1, 1, dtype=torch.float32, device=DEV)
    topk_ids = torch.zeros(1, 1, dtype=torch.int32, device=DEV)
    moe.apply(output=output, hidden_states=A,
              topk_weights=topk_weights, topk_ids=topk_ids)
    out_xpu = output.cpu().float()

    # --- True MoE output on CPU ---
    A_cpu = A.cpu().float()
    gate = A_cpu @ Wgate_true.t()
    up = A_cpu @ Wup_true.t()
    act = torch.nn.functional.silu(gate) * up
    out_true = act @ Wdown_true.t()  # [1, hidden]

    cos = torch.nn.functional.cosine_similarity(
        out_xpu.flatten(), out_true.flatten(), dim=0
    ).item()
    ad = (out_xpu - out_true).abs()
    print(f"  cos(out_xpu, out_true) = {cos:+.6f}")
    print(f"  max_abs_err = {ad.max().item():.4f}  "
          f"mean_abs_err = {ad.mean().item():.4f}")
    print(f"  out_xpu[:8]  = {out_xpu.flatten()[:8].tolist()}")
    print(f"  out_true[:8] = {out_true.flatten()[:8].tolist()}")

    ok = cos > 0.95
    print("\n" + "=" * 60)
    if ok:
        print("PASS: MoE int4 path matches ground truth.")
        print("-> dense + MoE GEMM paths are both correct.")
        print("-> the soup bug is in the model definition (HC / GDN / attn / mrope).")
    else:
        print("FAIL: MoE int4 path does NOT match ground truth.")
        print("-> the MoE fused kernel / int4 packing is the root cause.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())