"""Smoke test: load the lued INT8 W8A16 model on the B70 XPU stack and run a
short generation. Must be a real file (not a heredoc) because vllm's spawn
multiprocessing re-imports __main__ by path.
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
        enforce_eager=True,  # skip graph capture for the smoke test
    )
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    out = llm.generate(["What is 17 * 23? Answer with just the number."], sp)
    print("=== OUTPUT ===")
    print(out[0].outputs[0].text)
    print("=== END ===")


if __name__ == "__main__":
    main()
