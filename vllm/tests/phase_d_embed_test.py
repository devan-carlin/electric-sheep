#!/usr/bin/env python3
"""
Non-circular EMBEDDING test.

The HC capture is at layer 0, where resid_in = embedding broadcast to 4
identical streams (layer 0 has residual=None -> broadcast). So:
  resid_in[0, s, :] should be identical across s, and equal to
  embed_tokens[input_ids[0]] from the checkpoint.

Compare vLLM's captured resid_in (layer 0) to checkpoint embed_tokens[input_ids].
input_ids from the PLE capture (same 13:23 run): [760, 6511, 314, 9338, 369].
"""
import torch
import torch.nn.functional as F
from safetensors import safe_open

EMB = "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-FP8/model-00130-of-00131.safetensors"
INPUT_IDS = [760, 6511, 314, 9338, 369]


def main():
    # vLLM's captured layer-0 resid_in (embedding broadcast to 4 streams)
    hc = torch.load("/tmp/qwen4exp_hc_capture_rank0.pt", map_location="cpu")
    resid_in = hc["resid_in"].float()  # (5, 4, 2560)
    T = resid_in.shape[0]
    print(f"resid_in: {tuple(resid_in.shape)}")

    # Check: are the 4 streams identical (clean broadcast)?
    for t in range(T):
        streams = [resid_in[t, s] for s in range(4)]
        max_diff = max((streams[0] - s).abs().max().item() for s in streams[1:])
        print(f"  token {t}: max stream diff = {max_diff:.6f} (should be ~0 if clean broadcast)")

    # Load checkpoint embedding
    with safe_open(EMB, framework="pt") as f:
        emb_w = f.get_tensor("model.language_model.embed_tokens.weight").float()  # (248320, 2560)
    print(f"\nembed_tokens.weight: {tuple(emb_w.shape)}")

    # Compute embedding from checkpoint
    ids = torch.tensor(INPUT_IDS)
    emb_ref = emb_w[ids]  # (5, 2560)

    # Compare to vLLM's resid_in (stream 0)
    print(f"\n=== EMBEDDING: vLLM resid_in[0,0,:] vs checkpoint embed_tokens[input_ids] ===")
    for t in range(T):
        vllm_emb = resid_in[t, 0]  # stream 0
        ref_emb = emb_ref[t]
        cos = F.cosine_similarity(vllm_emb, ref_emb, dim=0).item()
        diff = (vllm_emb - ref_emb).abs()
        print(f"  token {t} (id={INPUT_IDS[t]}): cos={cos:.6f} max_diff={diff.max().item():.6f} "
              f"vllm_absmax={vllm_emb.abs().max().item():.4f} ref_absmax={ref_emb.abs().max().item():.4f}")

    # Also try: maybe vLLM's resid_in is NOT the raw embedding (could be scaled/processed).
    # Check the ratio of magnitudes.
    print(f"\n=== Magnitude ratio (vllm / ref) per token ===")
    for t in range(T):
        vllm_emb = resid_in[t, 0]
        ref_emb = emb_ref[t]
        ratio = (vllm_emb.norm() / ref_emb.norm()).item()
        print(f"  token {t}: norm ratio = {ratio:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()