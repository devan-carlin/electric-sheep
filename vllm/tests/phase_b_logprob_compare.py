"""Compare vLLM first-token logprobs vs llama.cpp baseline (ground truth).

Baseline (port 8090, greedy) for 'The capital of France is':
  top-5: Paris(-0.739), \n\n(-2.817), \n(-3.416), a(-3.420), located(...)
If vLLM's top-1 differs, the prefill is wrong.
"""
import os
import torch

os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton"))
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29521")

from vllm import LLM, SamplingParams

MODEL = os.environ.get(
    "QWEN4EXP_TEST_MODEL", "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
)


def main():
    mem_util = float(os.environ.get("QWEN4EXP_MEM_UTIL", "0.85"))
    cpu_offload_gb = float(os.environ.get("QWEN4EXP_CPU_OFFLOAD_GB", "0"))
    llm_kwargs = dict(
        model=MODEL,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=mem_util,
        kv_cache_dtype="fp8",
        trust_remote_code=False,
        enforce_eager=True,
    )
    if cpu_offload_gb > 0:
        llm_kwargs["cpu_offload_gb"] = cpu_offload_gb
    llm = LLM(**llm_kwargs)

    prompts = [
        "The capital of France is",
        "Write a haiku about GPUs.",
        "Explain what a hyper-connection is in one sentence.",
    ]

    sp = SamplingParams(temperature=0.0, max_tokens=8, logprobs=5)
    outs = llm.generate(prompts, sp)
    for p, o in zip(prompts, outs):
        print(f"### PROMPT: {p!r}")
        lp = o.outputs[0].logprobs
        if lp:
            first = lp[0]
            items = sorted(first.items(), key=lambda kv: kv[1].logprob)[:5]
            for tok, info in items:
                print(f"   {info.logprob:8.3f}  id={info.rank}  {tok!r}")
        print()


if __name__ == "__main__":
    main()