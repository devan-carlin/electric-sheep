#!/usr/bin/env python3
"""Long-context benchmark client for a running vLLM OpenAI server.

Builds a deterministic prompt of EXACTLY --prompt-tokens tokens (seed-rotated
so different runs use different prompts), streams a completion, and measures:
  - ttft_s        : request start -> first generated token (prefill latency)
  - prefill_tok_s : prompt_tokens / ttft
  - decode_tok_s  : (gen_tokens-1) / (last_token_t - first_token_t)
  - wall_s        : total request time
"""
import argparse
import json
import time

from openai import OpenAI
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir (for tokenizer)")
    ap.add_argument("--served", default="qwen-128k")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--prompt-tokens", type=int, default=130816)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    base = ("The quick brown fox jumps over the lazy dog near the riverbank "
            "while the sun sets over the hills and the wind changes direction. ")
    base_ids = tok.encode(base)
    # deterministic per-seed rotation -> different prompt text per run
    rot = (a.seed * 7) % len(base_ids)
    unit = base_ids[rot:] + base_ids[:rot]
    reps = a.prompt_tokens // len(unit) + 1
    ids = (unit * reps)[: a.prompt_tokens]

    client = OpenAI(base_url=a.base_url + "/v1", api_key="none")
    t0 = time.perf_counter()
    first_tok_t = None
    last_tok_t = None
    n_content = 0
    usage = None
    stream = client.completions.create(
        model=a.served,
        prompt=ids,
        max_tokens=a.gen_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.choices:
            ch = chunk.choices[0]
            # openai 2.x legacy /v1/completions exposes .text; the chat API
            # exposes .delta.content. Handle both.
            text = getattr(ch, "text", None)
            if text is None and getattr(ch, "delta", None) is not None:
                text = ch.delta.content
            if text:
                now = time.perf_counter()
                if first_tok_t is None:
                    first_tok_t = now
                last_tok_t = now
                n_content += 1
    wall = time.perf_counter() - t0

    ttft = (first_tok_t - t0) if first_tok_t else None
    gen_tokens = usage.completion_tokens if usage else n_content
    decode = 0.0
    if gen_tokens > 1 and first_tok_t and last_tok_t and last_tok_t > first_tok_t:
        decode = (gen_tokens - 1) / (last_tok_t - first_tok_t)

    res = {
        "prompt_tokens": len(ids),
        "gen_tokens": gen_tokens,
        "ttft_s": round(ttft, 3) if ttft else None,
        "prefill_tok_s": round(len(ids) / ttft, 1) if ttft else None,
        "decode_tok_s": round(decode, 2),
        "wall_s": round(wall, 2),
    }
    print(json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
