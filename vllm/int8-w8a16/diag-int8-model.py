"""Sanity-check an INT8 W8A16 model: raw token IDs + per-token decode + repetition.
Usage: python diag-int8-model.py <model_path> [tp]
"""
import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/models/devan-carlin-Qwen3.8-27B--ara-int8-w8a16"
TP = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    out = llm.generate(["What is 17 * 23? Answer with just the number."], sp)
    toks = out[0].outputs[0].token_ids
    text = out[0].outputs[0].text
    print("=== NUM TOKENS:", len(toks), "===")
    print("=== RAW IDS:", list(toks), "===")
    print("=== TEXT repr:", repr(text), "===")
    dec = llm.get_tokenizer()
    print("=== PER-TOKEN DECODE ===")
    for i, t in enumerate(toks):
        try:
            s = dec.decode([t])
        except Exception as e:
            s = f"<err {e}>"
        print(f"  [{i:2d}] id={t:6d} -> {s!r}")
    from collections import Counter

    c = Counter(toks)
    print("=== TOP REPEATED ===")
    for t, n in c.most_common(5):
        print(f"  id={t} count={n} -> {dec.decode([t])!r}")
    print("=== END ===")


if __name__ == "__main__":
    main()
