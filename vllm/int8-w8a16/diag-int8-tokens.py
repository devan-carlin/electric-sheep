"""Diagnose the empty-output issue: print raw token IDs and per-token decodes
to see whether the model is looping on a special token or producing whitespace.
"""

from vllm import LLM, SamplingParams

MODEL = "/mnt/data/models/lued-Qwen3.8-27B-INT8-W8A16-MTP"


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
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
    # per-token decode
    dec = llm.get_tokenizer()
    print("=== PER-TOKEN DECODE ===")
    for i, t in enumerate(toks):
        try:
            s = dec.decode([t])
        except Exception as e:
            s = f"<err {e}>"
        print(f"  [{i:2d}] id={t:6d} -> {s!r}")
    # repetition check
    from collections import Counter
    c = Counter(toks)
    print("=== TOP REPEATED ===")
    for t, n in c.most_common(5):
        print(f"  id={t} count={n} -> {dec.decode([t])!r}")
    print("=== END ===")


if __name__ == "__main__":
    main()
