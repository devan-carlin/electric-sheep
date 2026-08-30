import torch, os
caps = [torch.load(f"/tmp/qwen4exp_moe_capture_rank{r}.pt", map_location="cpu") for r in range(4)]
c0 = caps[0]
print("keys:", list(c0.keys()))
print("layer_idx:", c0["layer_idx"], "ep_rank:", c0["ep_rank"], "ep_size:", c0["ep_size"])
x = c0["input"]; o = c0["output"]
print("input shape", tuple(x.shape), x.dtype, "absmax", x.abs().max().item(), "mean", x.mean().item())
print("output shape", tuple(o.shape), o.dtype, "absmax", o.abs().max().item(), "mean", o.mean().item())
# Are inputs identical across ranks? (EP should replicate input)
for r in range(1,4):
    same_in = torch.equal(caps[r]["input"], x)
    same_out = torch.equal(caps[r]["output"], o)
    in_diff = (caps[r]["input"]-x).abs().max().item()
    out_diff = (caps[r]["output"]-o).abs().max().item()
    print(f"rank{r}: input_equal={same_in} (maxdiff {in_diff:.4g})  output_equal={same_out} (maxdiff {out_diff:.4g})")
# output/input ratio
print("out_absmax / in_absmax =", o.abs().max().item()/x.abs().max().item())