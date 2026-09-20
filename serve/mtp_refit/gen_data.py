#!/usr/bin/env python3
"""Offline MTP re-fit data generator.

Loads the Flash-Next W4A16 target via the venv vLLM offline API (TP4, spec OFF,
eager, no prefix caching, max_num_seqs=1) and generates diverse completions.
The B70_MTP_DATAGEN_HOOK tracer in Qwen4ExpModel captures, per token, the wide
hyper-connection residual state the MTP drafter consumes, appending to
B70_MTP_DATAGEN_TRACE (set in the environment before launch).

Sampling matches production serving (temp 0.7, top_p 0.8, top_k 20,
presence_penalty 1.5) so the captured target distribution is what the draft
must learn to predict.

Run via run_datagen.sh (sets oneAPI + venv + device env), e.g.:
  bash run_datagen.sh            # uses defaults below
Env overrides: DATAGEN_PROMPTS, DATAGEN_TRACE, DATAGEN_MAX_TOKENS,
               DATAGEN_COUNT (subset of prompts), DATAGEN_OUT
"""
import json
import os
import sys
import time

MODEL = "/mnt/data/models/devancarlin-Qwen3.8-Flash-Next-heretic-2-W4A16-BF16src"
PROMPTS = os.environ.get("DATAGEN_PROMPTS", "prompts.jsonl")
TRACE = os.environ.get("DATAGEN_TRACE", "trace.bin")
OUT = os.environ.get("DATAGEN_OUT", "gen_meta.json")
MAX_TOKENS = int(os.environ.get("DATAGEN_MAX_TOKENS", "2048"))
COUNT = int(os.environ.get("DATAGEN_COUNT", "0"))  # 0 = all prompts


def main():
    assert os.environ.get("B70_MTP_DATAGEN_TRACE"), \
        "B70_MTP_DATAGEN_TRACE must be set (run via run_datagen.sh)"
    print(f"[datagen] trace -> {os.environ['B70_MTP_DATAGEN_TRACE']}", flush=True)
    print(f"[datagen] prompts <- {PROMPTS}", flush=True)

    with open(PROMPTS) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if COUNT > 0:
        items = items[:COUNT]
    print(f"[datagen] {len(items)} prompts, max_tokens={MAX_TOKENS}", flush=True)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        enable_expert_parallel=True,  # required: WNA16 static group scales need EP under TP4
        dtype="bfloat16",
        kv_cache_dtype="fp8",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,          # one sequence at a time -> clean trace parsing
        enable_prefix_caching=False,  # recompute every token so all are captured
        enforce_eager=True,       # no XPU/cuda graphs -> hook's .cpu()/file IO is safe
        trust_remote_code=False,
    )
    print(f"[datagen] model loaded in {time.time()-t0:.0f}s", flush=True)

    sp = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=MAX_TOKENS,
        seed=1234,
    )

    # Use chat() so the checkpoint's embedded chat template is applied,
    # matching production traffic. The hook captures wide states regardless.
    conversations = [[{"role": "user", "content": it["text"]}] for it in items]

    t1 = time.time()
    results = llm.chat(conversations, sampling_params=sp, use_tqdm=True)
    dt = time.time() - t1

    n_tok = sum(len(o.token_ids) for r in results for o in r.outputs)
    meta = {
        "n_prompts": len(items),
        "n_generated_tokens": n_tok,
        "wall_seconds": round(dt, 1),
        "tok_per_s": round(n_tok / dt, 2) if dt > 0 else 0,
        "max_tokens": MAX_TOKENS,
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                     "presence_penalty": 1.5},
        "trace": os.environ["B70_MTP_DATAGEN_TRACE"],
    }
    with open(OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[datagen] done: {n_tok:,} tokens in {dt/60:.1f} min "
          f"({meta['tok_per_s']} tok/s)", flush=True)
    print(f"[datagen] meta -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
