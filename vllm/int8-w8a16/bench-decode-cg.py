"""Decode throughput benchmark with optional cudagraph (enforce_eager flag).

Usage: python bench-decode-cg.py <model_path> [tp] [eager|cg]
  eager -> enforce_eager=True
  cg    -> enforce_eager=False (cudagraph capture)
"""
import time
import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1]
TP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MODE = sys.argv[3] if len(sys.argv) > 3 else "cg"
ENFORCE_EAGER = MODE == "eager"

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
        enforce_eager=ENFORCE_EAGER,
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
    print(f"mode: {MODE} (enforce_eager={ENFORCE_EAGER})")
    print(f"finish_reason: {o.finish_reason}")
    print(f"prompt_tokens: {prompt_len}")
    print(f"completion_tokens: {n}")
    print(f"wall_time_s: {elapsed:.3f}")
    print(f"total_tok_per_s (incl prefill): {n / elapsed:.2f}")
    print(f"decode_tok_per_s (approx): {(n - 1) / elapsed:.2f}")
    print("=== END ===")


if __name__ == "__main__":
    main()
