"""Control test: run the INT4 AutoRound model on the same prompt.
If INT4 is sane and INT8 loops, the problem is INT8-specific (quantization or
XPU int8 load/fusion), not the model or vLLM path.
"""
import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/models/devan-carlin-Qwen3.8-27B-int4-AutoRound"
PROMPT = "What is 17 * 23? Answer with just the number."


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    out = llm.generate([PROMPT], sp)
    toks = out[0].outputs[0].token_ids
    text = out[0].outputs[0].text
    print("=== MODEL:", MODEL, "===")
    print("=== NUM TOKENS:", len(toks), "===")
    print("=== RAW IDS:", list(toks), "===")
    print("=== TEXT repr:", repr(text), "===")
    dec = llm.get_tokenizer()
    from collections import Counter
    c = Counter(toks)
    print("=== TOP REPEATED ===")
    for t, n in c.most_common(5):
        print(f"  id={t} count={n} -> {dec.decode([t])!r}")
    print("=== END ===")


if __name__ == "__main__":
    main()
