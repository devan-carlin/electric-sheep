"""Control test: Qwen3.5-4B (GDN arch) on 4 GPUs (TP=4).

The earlier control test ran TP=1 (single GPU) and produced correct output.
qwen4exp runs TP=4 + expert parallel. This tests whether the 4-GPU GDN path
(TP-sharded in_proj/out_proj + kernel tp_size=4) is the bug.
If Qwen3.5-4B ALSO produces garbage on TP=4 -> 4-GPU GDN path is the bug.
If it still produces "Paris" -> 4-GPU GDN path is fine, bug is elsewhere.
"""
import os
from vllm import LLM, SamplingParams

MODEL = os.environ.get("CONTROL_MODEL", "/home/dc/electric-sheep/models/Qwen3.5-4B")
TP = int(os.environ.get("CONTROL_TP", "4"))


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=40)
    outs = llm.generate(["The capital of France is"], sp)
    print("=== CONTROL TEST: Qwen3.5-4B (GDN arch) TP=%d ===" % TP)
    print(repr(outs[0].outputs[0].text))


if __name__ == "__main__":
    main()