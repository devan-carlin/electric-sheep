"""Phase A / A2: validate HyperConnection mix/combine math.

Compares the vLLM `HyperConnection` module against a direct reference
implementation of the ggml math from llama.cpp qwen4exp.cpp
(build_hc_mix / build_hc_combine). Runs on CPU in fp32.
"""

import sys

import torch

sys.path.insert(
    0, "/home/dc/electric-sheep/vllm/.venv/lib/python3.12/site-packages"
)

from vllm.model_executor.models.qwen4_exp import HyperConnection  # noqa: E402
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig  # noqa: E402


def ref_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    # x: [T, hc, n_embd]; reduce over last dim (one residual stream)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)


def ref_mix(
    x, w_norm, w_down, w_up, w_inject, hc, n_embd, eps, has_inject=True
):
    # x: [T, hc, n_embd]
    xn = ref_rms_norm(x, eps)
    xn_flat = xn.reshape(x.shape[0], hc * n_embd)
    xn_flat = xn_flat * w_norm
    lo = (w_down @ xn_flat.t()).t()  # [T, lowrank]
    lo = torch.nn.functional.silu(lo / hc)
    gate = torch.sigmoid(lo @ w_up.t())  # [T, hc_dim]
    gated = (xn_flat * gate)  # [T, hc_dim]
    gated = gated.reshape(x.shape[0], hc, n_embd)
    mixed = gated.mean(dim=1)  # [T, n_embd]
    inject = None
    if has_inject:
        inject = (w_inject @ xn_flat.t()).t()  # [T, hc]
    return mixed, inject


def ref_combine(residual, block_out, inject, hc):
    # residual: [T, hc, n_embd]; block_out: [T, n_embd]; inject: [T, hc]
    w = 2.0 * torch.sigmoid(inject / hc)  # [T, hc]
    w = w.unsqueeze(-1)  # [T, hc, 1]
    b = block_out.unsqueeze(1)  # [T, 1, n_embd]
    return residual + b * w


def main():
    torch.manual_seed(0)
    hc, n_embd, lowrank, T = 4, 2560, 320, 17
    eps = 1e-6

    cfg = Qwen4ExpTextConfig(
        hidden_size=n_embd, hc_count=hc, hc_lowrank=lowrank, rms_norm_eps=eps
    )
    mod = HyperConnection(cfg, has_inject=True)
    # fp32 weights for a clean numeric comparison
    with torch.no_grad():
        mod.hc_norm.weight = torch.nn.Parameter(
            torch.randn(hc * n_embd) * 0.1
        )
        mod.input_mix_weight_down.weight = torch.nn.Parameter(
            torch.randn(lowrank, hc * n_embd) * 0.05
        )
        mod.input_mix_weight_up.weight = torch.nn.Parameter(
            torch.randn(hc * n_embd, lowrank) * 0.05
        )
        mod.block_inject_weight.weight = torch.nn.Parameter(
            torch.randn(hc, hc * n_embd) * 0.05
        )

    x = torch.randn(T, hc, n_embd)
    block_out = torch.randn(T, n_embd)

    mixed, inject = mod.mix(x)
    ref_mixed, ref_inject = ref_mix(
        x,
        mod.hc_norm.weight,
        mod.input_mix_weight_down.weight,
        mod.input_mix_weight_up.weight,
        mod.block_inject_weight.weight,
        hc,
        n_embd,
        eps,
    )
    dm = (mixed - ref_mixed).abs().max().item()
    di = (inject - ref_inject).abs().max().item()
    print(f"mix  max|diff| = {dm:.3e}")
    print(f"inject max|diff| = {di:.3e}")
    assert dm < 1e-4, f"mix mismatch: {dm}"
    assert di < 1e-4, f"inject mismatch: {di}"

    # combine: residual + block_out * 2*sigmoid(inject/hc)
    residual = torch.randn(T, hc, n_embd)
    out = mod.combine(residual, block_out, inject)
    ref_out = ref_combine(residual, block_out, ref_inject, hc)
    dc = (out - ref_out).abs().max().item()
    print(f"combine max|diff| = {dc:.3e}")
    assert dc < 1e-4, f"combine mismatch: {dc}"

    # untrained-injection property: zero inject -> 2*sigmoid(0)=1 -> plain add
    zero_inject = torch.zeros(T, hc)
    out0 = mod.combine(residual, block_out, zero_inject)
    ref0 = residual + block_out.unsqueeze(1)
    d0 = (out0 - ref0).abs().max().item()
    print(f"zero-inject (plain add) max|diff| = {d0:.3e}")
    assert d0 < 1e-5, f"zero-inject mismatch: {d0}"

    # has_inject=False (top-level mixer): mix only, no inject
    head = HyperConnection(cfg, has_inject=False)
    with torch.no_grad():
        head.hc_norm.weight = torch.nn.Parameter(
            torch.randn(hc * n_embd) * 0.1
        )
        head.input_mix_weight_down.weight = torch.nn.Parameter(
            torch.randn(lowrank, hc * n_embd) * 0.05
        )
        head.input_mix_weight_up.weight = torch.nn.Parameter(
            torch.randn(hc * n_embd, lowrank) * 0.05
        )
    mixed_h, inject_h = head.mix(x)
    assert inject_h is None
    ref_h, _ = ref_mix(
        x,
        head.hc_norm.weight,
        head.input_mix_weight_down.weight,
        head.input_mix_weight_up.weight,
        None,
        hc,
        n_embd,
        eps,
        has_inject=False,
    )
    dh = (mixed_h - ref_h).abs().max().item()
    print(f"head-mix max|diff| = {dh:.3e}")
    assert dh < 1e-4, f"head-mix mismatch: {dh}"

    print("PASS: HyperConnection mix/combine matches ggml reference")


if __name__ == "__main__":
    main()