"""Pure-PyTorch reimplementation of ONE MoE layer (layer 2), compared to vLLM's
captured output.

Goal: isolate whether the MoE layer (routing + expert GEMM + dispatch + shared
expert) is correct. We recompute from GROUND-TRUTH checkpoint weights (FP8
dequantized) + the captured input, and compare to vLLM's captured output.

If they match  -> MoE layer is correct, rule it out.
If they diverge -> the bug is in the MoE layer (routing / GEMM / dispatch / shared).

Uses the FP8 checkpoint (experts are float8_e4m3fn, 128x128 block scales).
"""
import os
import torch

torch.manual_seed(0)

CKPT = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8"
CAPTURE = "/tmp/qwen4exp_moe_capture_rank0.pt"
LAYER = 2
TOPK = 10
N_EXPERT = 512
HID = 2560
INTER = 640
BLOCK = 128

dev = "cpu"
torch.set_num_threads(os.cpu_count() or 8)


def load_shard_tensor(key):
    import json
    from safetensors import safe_open
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    shard = idx[key]
    with safe_open(f"{CKPT}/{shard}", framework="pt") as st:
        return st.get_tensor(key)


def dequant_fp8(w_fp8, scale):
    """w_fp8: [R, C] float8_e4m3fn, scale: [R/BLOCK, C/BLOCK] -> [R, C] float32."""
    R, C = w_fp8.shape
    sr, sc = scale.shape
    assert R % sr == 0 and C % sc == 0, (R, C, sr, sc)
    br, bc = R // sr, C // sc
    s = scale.to(torch.float32)
    s = s.repeat_interleave(br, dim=0).repeat_interleave(bc, dim=1)  # [R, C]
    return w_fp8.to(torch.float32) * s


def silu(x):
    return x * torch.sigmoid(x)


def main():
    cap = torch.load(CAPTURE, map_location="cpu")
    x = cap["input"].to(torch.float32)          # [T, HID]
    out_vllm = cap["output"].to(torch.float32)  # [T, HID]
    T = x.shape[0]
    print(f"capture: input {tuple(x.shape)} output {tuple(out_vllm.shape)}")

    # ---- Load gate + shared expert gate ----
    gate = load_shard_tensor(f"model.language_model.layers.{LAYER}.mlp.gate.weight").to(torch.float32)  # [512, 2560]
    sh_gate = load_shard_tensor(f"model.language_model.layers.{LAYER}.mlp.shared_expert_gate.weight").to(torch.float32)  # [1, 2560]
    print(f"gate {tuple(gate.shape)} sh_gate {tuple(sh_gate.shape)}")

    # ---- Routing: softmax -> topk -> renorm (matches vLLM kernel + llama.cpp) ----
    logits = x @ gate.T  # [T, 512]
    probs = torch.softmax(logits, dim=-1)
    topk_w, topk_ids = torch.topk(probs, TOPK, dim=-1)  # [T, 10]
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)  # renormalize
    print(f"routing: topk_ids range [{topk_ids.min().item()}, {topk_ids.max().item()}]")

    # ---- Load + dequant all experts (gate/up/down) ----
    print("loading + dequantizing 512 experts ...")
    gate_w = torch.empty(N_EXPERT, INTER, HID, dtype=torch.float32)
    up_w = torch.empty(N_EXPERT, INTER, HID, dtype=torch.float32)
    down_w = torch.empty(N_EXPERT, HID, INTER, dtype=torch.float32)
    for e in range(N_EXPERT):
        base = f"model.language_model.layers.{LAYER}.mlp.experts.{e}"
        gate_w[e] = dequant_fp8(load_shard_tensor(f"{base}.gate_proj.weight"),
                                load_shard_tensor(f"{base}.gate_proj.weight_scale_inv"))
        up_w[e] = dequant_fp8(load_shard_tensor(f"{base}.up_proj.weight"),
                              load_shard_tensor(f"{base}.up_proj.weight_scale_inv"))
        down_w[e] = dequant_fp8(load_shard_tensor(f"{base}.down_proj.weight"),
                                load_shard_tensor(f"{base}.down_proj.weight_scale_inv"))
        if e % 64 == 0:
            print(f"  expert {e}/{N_EXPERT}")
    print("experts loaded")

    # ---- Routed expert computation (grouped, per-expert loop) ----
    # Flatten to (T*TOPK) rows, each tagged with its expert.
    x_rep = x.repeat_interleave(TOPK, dim=0)          # [T*10, HID]
    ids = topk_ids.reshape(-1)                        # [T*10]
    w_rep = topk_w.reshape(-1)                        # [T*10]
    out_routed = torch.zeros_like(x_rep)              # [T*10, HID]
    for e in range(N_EXPERT):
        m = ids == e
        if not m.any():
            continue
        h = x_rep[m]                                   # [n, HID]
        g = h @ gate_w[e].T                            # [n, INTER]
        u = h @ up_w[e].T                              # [n, INTER]
        act = silu(g) * u                              # [n, INTER]
        out_routed[m] = act @ down_w[e].T              # [n, HID]
    # Weighted sum over the 10 slots per token.
    out_routed = (out_routed * w_rep.unsqueeze(-1)).reshape(T, TOPK, HID).sum(dim=1)  # [T, HID]

    # ---- Shared expert (bf16) ----
    sh_base = f"model.language_model.layers.{LAYER}.mlp.shared_expert"
    sh_gate_up = torch.cat([
        load_shard_tensor(f"{sh_base}.gate_proj.weight").to(torch.float32),  # [INTER, HID]
        load_shard_tensor(f"{sh_base}.up_proj.weight").to(torch.float32),
    ], dim=0)  # [2*INTER, HID]
    sh_down = load_shard_tensor(f"{sh_base}.down_proj.weight").to(torch.float32)  # [HID, INTER]
    gu = x @ sh_gate_up.T            # [T, 2*INTER]
    g, u = gu[:, :INTER], gu[:, INTER:]
    sh_out = (silu(g) * u) @ sh_down.T  # [T, HID]
    sh_out = torch.sigmoid(x @ sh_gate.T) * sh_out  # [T, HID]

    out_ref = out_routed + sh_out  # [T, HID]

    # ---- Compare ----
    diff = (out_ref - out_vllm).abs()
    denom = out_vllm.abs().clamp_min(1e-6)
    rel = (diff / denom)
    cos = torch.nn.functional.cosine_similarity(out_ref.flatten(), out_vllm.flatten(), dim=0).item()
    print("\n=== COMPARISON (pure-PyTorch ref vs vLLM capture) ===")
    print(f"ref  absmax={out_ref.abs().max().item():.4f} mean={out_ref.mean().item():.5f}")
    print(f"vllm absmax={out_vllm.abs().max().item():.4f} mean={out_vllm.mean().item():.5f}")
    print(f"max abs diff = {diff.max().item():.5f}")
    print(f"mean abs diff = {diff.mean().item():.6f}")
    print(f"mean rel diff = {rel.mean().item():.5f}")
    print(f"cosine similarity = {cos:.6f}")
    # Per-token cos
    tok_cos = torch.nn.functional.cosine_similarity(out_ref, out_vllm, dim=-1)
    print(f"per-token cos: min={tok_cos.min().item():.4f} mean={tok_cos.mean().item():.4f} max={tok_cos.max().item():.4f}")

    # Also compare routed-only (in case shared expert is NOT in the capture)
    cos_routed = torch.nn.functional.cosine_similarity(out_routed.flatten(), out_vllm.flatten(), dim=0).item()
    print(f"[routed-only] cosine vs vllm = {cos_routed:.6f}")


if __name__ == "__main__":
    main()