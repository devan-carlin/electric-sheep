"""Decode throughput benchmark for INT8 vs INT4 (same TP, same settings).

Applies the chat template (Qwen3.5 is a chat model; raw prompts can EOS
immediately). Measures steady-state decode tok/s after a warmup pass so
first-step Triton JIT doesn't skew the number.
"""
import time
import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1]
TP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
WARMUP_TOKENS = 64
BENCH_TOKENS = 512
USER_PROMPT = (
    "Write a detailed explanation of how a modern CPU out-of-order execution "
    "engine works, covering the fetch, decode, rename, dispatch, execution, "
    "and retirement stages, and how branch prediction and the reorder buffer "
    "interact. Be thorough."
)


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,  # custom int8 op has no FakeTensor impl; compile path fails
    )
    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": USER_PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(prompt, str):
        prompt = prompt[0] if isinstance(prompt, list) else str(prompt)

    # Warmup (JIT + any capture)
    sp_w = SamplingParams(temperature=0.0, max_tokens=WARMUP_TOKENS)
    llm.generate([prompt], sp_w)

    # Timed run
    sp = SamplingParams(temperature=0.0, max_tokens=BENCH_TOKENS)
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    o = out[0].outputs[0]
    n = len(o.token_ids)
    prompt_len = len(out[0].prompt_token_ids)
    print("=== BENCHMARK ===")
    print(f"model: {MODEL}")
    print(f"tp: {TP}")
    print(f"finish_reason: {o.finish_reason}")
    print(f"prompt_tokens: {prompt_len}")
    print(f"completion_tokens: {n}")
    print(f"wall_time_s: {elapsed:.3f}")
    print(f"total_tok_per_s (incl prefill): {n / elapsed:.2f}")
    print("=== END ===")


if __name__ == "__main__":
    main()
