#!/usr/bin/env python3
"""
quick-ab-throughput.py — Rough same-prompt throughput A/B between the two
start-all vLLM instances (qwen-256k @ 8088, gemma-31b @ 8089).

Sends ONE identical prompt to both models (thinking disabled on Qwen), a few
repetitions each, and reports per-request time + output tok/s. The two models
run in parallel (separate GPUs), so total wall time ~= the slower model's time.

This is a rough single-stream decode-throughput probe, not a saturated-serving
benchmark. No concurrency, no 4K output — finishes in a couple of minutes.
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# One prompt that reliably yields a few hundred tokens of prose.
PROMPT = (
    "Write a detailed, vivid 400-word description of a bustling night market "
    "in a rain-soaked cyberpunk megacity. Include sensory detail: sights, "
    "sounds, smells, and the mood of the crowd."
)
MAX_TOKENS = 1024
REPS = 3

MODELS = [
    # (name, port, extra_body)
    ("qwen-256k", 8088, {"chat_template_kwargs": {"enable_thinking": False}}),
    ("gemma-31b", 8089, None),
]


def one_request(port, model, extra_body):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    if extra_body:
        body.update(extra_body)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    dt = time.time() - t0
    usage = out.get("usage", {})
    n_out = usage.get("completion_tokens", 0)
    content = out["choices"][0]["message"].get("content") or ""
    return {
        "model": model,
        "time_s": round(dt, 2),
        "out_tokens": n_out,
        "tok_s": round(n_out / dt, 1) if dt > 0 else 0.0,
        "content_head": content[:80].replace("\n", " "),
    }


def bench_model(name, port, extra_body):
    rows = []
    for r in range(REPS):
        rows.append(one_request(port, name, extra_body))
    times = [x["time_s"] for x in rows]
    toks = [x["tok_s"] for x in rows]
    return {
        "model": name,
        "reps": rows,
        "mean_time_s": round(sum(times) / len(times), 2),
        "mean_tok_s": round(sum(toks) / len(toks), 1),
    }


def main():
    wall0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        futs = {
            ex.submit(bench_model, name, port, extra): name
            for name, port, extra in MODELS
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                results[name] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[name] = {"model": name, "error": repr(e)}

    print("=" * 62)
    print(f"Same-prompt A/B  |  prompt ~{len(PROMPT.split())} words  |  "
          f"max_tokens={MAX_TOKENS}  |  reps={REPS}  |  thinking OFF (qwen)")
    print("=" * 62)
    for name, _port, _extra in MODELS:
        res = results.get(name, {})
        if "error" in res:
            print(f"\n{name}: ERROR {res['error']}")
            continue
        print(f"\n{name}  (mean {res['mean_tok_s']} tok/s, "
              f"{res['mean_time_s']}s/req)")
        for i, r in enumerate(res["reps"], 1):
            print(f"  rep{i}: {r['time_s']:>7.2f}s  {r['out_tokens']:>5} tok  "
                  f"{r['tok_s']:>6} tok/s   {r['content_head']!r}")
    print("\n" + "=" * 62)
    print(f"total wall time: {time.time() - wall0:.1f}s")
    print("=" * 62)


if __name__ == "__main__":
    main()