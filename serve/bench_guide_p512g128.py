#!/usr/bin/env python3
"""Guide-exact p512/g128 benchmark for the docker A/B test.

Same methodology as the MikeCaldera/SergiioB replication guide:
- exact 512-token prompt built via /tokenize
- one same-shape warmup
- 5 measured runs, 128 gen tokens, concurrency 1, temp 0, ignore_eos
- rate = (completion_tokens - 1) / (request_end - first_generated_token)
- reports median/mean/min/max decode + median TTFT

Usage: python3 bench_guide_p512g128.py [BASE_URL] [MODEL]
Defaults: http://127.0.0.1:11436  qwen38
"""
import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11436"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen38"

seed = (
    "A retrieval augmented generation system retrieves relevant source passages, "
    "constructs a grounded context, sends the evidence to a language model, and "
    "generates a concise answer based on the supplied information. "
)

text = seed
while True:
    req = urllib.request.Request(
        BASE + "/tokenize",
        data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        tok = json.load(r)
    if len(tok["tokens"]) >= 512:
        break
    text += seed

ids = tok["tokens"][:512]
req = urllib.request.Request(
    BASE + "/detokenize",
    data=json.dumps({"model": MODEL, "tokens": ids}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    text = json.load(r)["prompt"]


def run():
    payload = {
        "model": MODEL,
        "prompt": text,
        "max_tokens": 128,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    end = None
    usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                end = time.perf_counter()
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices", [])
            if choices and choices[0].get("text") and first is None:
                first = time.perf_counter()
    if end is None:
        end = time.perf_counter()
    ct = usage["completion_tokens"]
    pt = usage["prompt_tokens"]
    return {
        "prompt": pt,
        "completion": ct,
        "ttft": first - start,
        "post": end - first,
        "rate": (ct - 1) / (end - first),
    }


print("Warmup...")
w = run()
print(f"warmup: {w['rate']:.2f} tok/s  TTFT={w['ttft']:.4f}s")

rates = []
ttfts = []
for i in range(1, 6):
    r = run()
    rates.append(r["rate"])
    ttfts.append(r["ttft"])
    print(
        f"run {i}: {r['rate']:.2f} tok/s  "
        f"TTFT={r['ttft']:.4f}s  "
        f"prompt={r['prompt']} completion={r['completion']}"
    )

print()
print(f"MEDIAN DECODE: {statistics.median(rates):.2f} tok/s")
print(f"MEAN DECODE:   {statistics.mean(rates):.2f} tok/s")
print(f"MIN/MAX:       {min(rates):.2f} / {max(rates):.2f} tok/s")
print(f"MEDIAN TTFT:   {statistics.median(ttfts):.4f} s")
