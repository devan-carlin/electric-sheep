"""Phase C: CPU reference for the layer-2 MoE block (4-GPU EP test).

Loads the captured MoE I/O (input + output) from the 4-GPU W4A16 run and
recomputes the MoE output on CPU from the FP8 'true' weights:
  router = softmax(gate @ x.T) -> top-10 -> renormalize
  routed = sum_e topk_w * silu(x @ Wg_e.T) * (x @ Wu_e.T) @ Wd_e.T
  shared = sigmoid(shared_gate @ x.T) * silu(x @ Wgu.T) @ Wd_shared
  out = routed + shared
Compares to the captured output. If it matches -> MoE/EP is fine, look at HC.
If it mismatches -> the EP MoE dispatch is the bug.
"""
import os, glob, torch
from safetensors import safe_open

FP8 = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8"
LAYER = 2
TOPK = 10
N_EXPERTS = 512
H = 2560
I = 640
BLOCK = 128

def fp8_block_dequant(weight, scale_inv, block=128):
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return weight.to(torch.float32) * s

def load_tensor(name):
    for f in sorted(glob.glob(FP8 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def main():
    cap = torch.load("/tmp/qwen4exp_moe_capture_rank0.pt", map_location="cpu")
    x = cap["input"].to(torch.float32)          # (4096, 2560)
    out_cap = cap["output"].to(torch.float32)   # (4096, 2560)
    print(f"captured input {tuple(x.shape)} out {tuple(out_cap.shape)}")

    base = f"model.language_model.layers.{LAYER}.mlp"
    gate = load_tensor(f"{base}.gate.weight").to(torch.float32)          # (512, 2560)
    se_gate = load_tensor(f"{base}.shared_expert.gate_proj.weight").to(torch.float32)
    se_up = load_tensor(f"{base}.shared_expert.up_proj.weight").to(torch.float32)
    se_down = load_tensor(f"{base}.shared_expert.down_proj.weight").to(torch.float32)
    se_gate_scalar = load_tensor(f"{base}.shared_expert_gate.weight").to(torch.float32)  # (1,2560)

    # ---- routing ----
    logits = x @ gate.T                          # (4096, 512)
    probs = torch.softmax(logits, dim=-1)
    topk_w, topk_idx = torch.topk(probs, TOPK, dim=-1)   # (4096, 10)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)   # renormalize
    print(f"router: logits absmax {logits.abs().max().item():.3f} "
          f"topk_w sum {topk_w.sum().item():.3f} (expect {x.shape[0]*TOPK})")

    # ---- shared expert (bf16, exact) ----
    # silu_and_mul: split gate/up, silu(gate)*up, then down
    shared = torch.nn.functional.silu(x @ se_gate.T) * (x @ se_up.T)   # (4096, 640)
    shared = shared @ se_down.T                                        # (4096, 2560)
    shared = torch.sigmoid(x @ se_gate_scalar.T) * shared   # (4096,1) * (4096,2560)
    print(f"shared expert: absmax {shared.abs().max().item():.4f} mean {shared.mean().item():.5f}")

    # ---- routed experts (only those actually selected) ----
    used = torch.unique(topk_idx)
    print(f"routed experts used: {used.numel()} / {N_EXPERTS}")
    routed = torch.zeros_like(x)
    for e in used.tolist():
        eb = f"{base}.experts.{e}"
        wg = fp8_block_dequant(load_tensor(f"{eb}.gate_proj.weight"),
                               load_tensor(f"{eb}.gate_proj.weight_scale_inv"), BLOCK)
        wu = fp8_block_dequant(load_tensor(f"{eb}.up_proj.weight"),
                               load_tensor(f"{eb}.up_proj.weight_scale_inv"), BLOCK)
        wd = fp8_block_dequant(load_tensor(f"{eb}.down_proj.weight"),
                               load_tensor(f"{eb}.down_proj.weight_scale_inv"), BLOCK)
        h = torch.nn.functional.silu(x @ wg.T) * (x @ wu.T)   # (4096, 640)
        eout = h @ wd.T                                        # (4096, 2560)
        mask = (topk_idx == e)                                 # (4096, 10)
        w = (topk_w * mask).sum(dim=-1, keepdim=True)          # (4096, 1)
        routed += w * eout
    print(f"routed experts: absmax {routed.abs().max().item():.4f} mean {routed.mean().item():.5f}")

    ref = routed + shared
    print(f"reference out: absmax {ref.abs().max().item():.4f} mean {ref.mean().item():.5f}")

    # ---- compare ----
    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    def relerr(a, b):
        return ((a - b).norm() / (b.norm() + 1e-9)).item()
    print("\n=== COMPARISON (captured vs CPU reference) ===")
    print(f"cosine(captured, ref)        = {cos(out_cap, ref):.6f}")
    print(f"cosine(captured, shared)     = {cos(out_cap, shared):.6f}")
    print(f"cosine(captured, routed)     = {cos(out_cap, routed):.6f}")
    print(f"relerr(captured, ref)        = {relerr(out_cap, ref):.4f}")
    print(f"relerr(captured, shared)     = {relerr(out_cap, shared):.4f}")
    print(f"max|captured - ref|          = {(out_cap-ref).abs().max().item():.4f}")
    print(f"max|captured - shared|       = {(out_cap-shared).abs().max().item():.4f}")
    print(f"absmax: captured {out_cap.abs().max().item():.4f}  ref {ref.abs().max().item():.4f}  "
          f"shared {shared.abs().max().item():.4f}  routed {routed.abs().max().item():.4f}")

if __name__ == "__main__":
    main()