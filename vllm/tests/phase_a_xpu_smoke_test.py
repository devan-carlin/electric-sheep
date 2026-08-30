"""Phase A / A9: real XPU generation smoke test for Qwen4-Exp.

Uses the full vLLM engine (vllm.LLM) on 4x Arc Pro B70 XPU with TP=4 + EP.
This exercises the real worker (xccl init), the real DefaultModelLoader,
KV-cache allocation, and a short generation. PLE is stubbed to zero in
Phase A, so output is the dense-attention / GDN / MoE / HC core only.

Run via:  bash phase_a_xpu_smoke.sh
"""

import os
import sys

os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_USE_MODELSCOPE", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29513")

MODEL_PATH = os.environ.get(
    "SMOKE_MODEL_PATH", "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
)

PROMPTS = [
    "Explain what a hyper-connection is in one sentence.",
    "The capital of France is",
    "Write a haiku about GPUs.",
]


def main():
    import torch

    print("torch:", torch.__version__, "xpu available:", torch.xpu.is_available(), flush=True)
    print("visible xpus:", torch.xpu.device_count(), flush=True)

    from vllm import LLM, SamplingParams

    print("constructing LLM (TP=4, EP, fp8 KV, max_model_len=4096)...", flush=True)
    ep = os.environ.get("SMOKE_EP", "1") == "1"
    print(f"enable_expert_parallel={ep}", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        enable_expert_parallel=ep,
        dtype="bfloat16",
        max_model_len=int(os.environ.get("SMOKE_MAX_LEN", "4096")),
        max_num_seqs=1,
        gpu_memory_utilization=float(os.environ.get("SMOKE_GPU_UTIL", "0.85")),
        kv_cache_dtype="fp8",
        trust_remote_code=False,
        enforce_eager=True,  # skip graph capture for the smoke test
    )

    print("LLM constructed. Generating...", flush=True)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        top_p=1.0,
    )
    outs = llm.generate(PROMPTS, sampling)

    print("\n" + "=" * 70, flush=True)
    print("GENERATION RESULTS", flush=True)
    print("=" * 70, flush=True)
    for prompt, out in zip(PROMPTS, outs):
        print(f"\n--- PROMPT: {prompt!r}")
        print(f"--- OUTPUT: {out.outputs[0].text!r}")
        print(f"    finish_reason={out.outputs[0].finish_reason} "
              f"num_tokens={len(out.outputs[0].token_ids)}")

    # Coherence gate: every completion must be non-empty and not all-NaN.
    ok = True
    for out in outs:
        t = out.outputs[0].text
        if not t.strip():
            ok = False
            print("FAIL: empty completion", flush=True)
    if ok:
        print("\nPASS: A9 smoke test produced non-empty completions", flush=True)
    else:
        print("\nFAIL: A9 smoke test", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()