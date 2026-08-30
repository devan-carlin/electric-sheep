"""Phase C: verify the lm_head -> logits step.

final_hidden is captured (5, 2560). lm_head is a separate BF16 weight
(248320, 2560), tie_word_embeddings=False. Compute logits = final_hidden @
lm_head.T on CPU and compare the top-5 to the actual logprobs in the run log.
If they match -> lm_head is correct, bug is upstream (final_hidden wrong).
If they mismatch -> lm_head / logits is the bug.
"""
import glob, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"

def load(name):
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def main():
    d = torch.load("/tmp/qwen4exp_final_capture_R0.pt", map_location="cpu")
    final_hidden = d["final_hidden"].float()  # (T, 2560)
    T = final_hidden.shape[0]
    print(f"final_hidden {tuple(final_hidden.shape)} norm {final_hidden.norm().item():.4f}")

    lm_head = load("lm_head.weight").float()  # (248320, 2560)
    print(f"lm_head {tuple(lm_head.shape)} absmax {lm_head.abs().max().item():.4f}")

    # logits for the LAST token (position T-1) -> generates the first output token
    h = final_hidden[-1]  # (2560,)
    logits = h @ lm_head.T  # (248320,)
    logprobs = F.log_softmax(logits, dim=-1)
    topv, topi = torch.topk(logprobs, 5)
    print("\n=== CPU logits (last token) top-5 ===")
    for i in range(5):
        print(f"  {topv[i].item():.4f}  id={topi[i].item()}")

    # Also check: is the final_hidden degenerate? (flat logprobs symptom)
    print(f"\nlogits absmax {logits.abs().max().item():.4f} std {logits.std().item():.4f} "
          f"min {logits.min().item():.4f} max {logits.max().item():.4f}")
    # entropy of the distribution
    p = torch.softmax(logits, dim=-1)
    entropy = -(p * logprobs).sum().item()
    import math
    print(f"entropy {entropy:.4f} nats (uniform = {math.log(248320):.4f})")

if __name__ == "__main__":
    main()