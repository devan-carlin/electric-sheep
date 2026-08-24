"""Compare against the base (non-, bf16) Qwen3.8-27B model with the
same prompt used for the INT8 diag. If the base model produces sane text, the
loop is quantization/model-specific. If it also loops, it's a prompt/config issue.
"""

from vllm import LLM, SamplingParams

MODEL = "/mnt/data/models/Qwen3.8-27B"
PROMPT = "What is 17 * 23? Answer with just the number."


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    out = llm.generate([PROMPT], sp)
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
