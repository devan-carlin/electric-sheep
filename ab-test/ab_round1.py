#!/usr/bin/env python3
"""Blind A/B prose test: 31B QAT MTP (8089) vs 26B-A4B (8090).

Writes a blinded markdown file (round1_blind.md) for the user to judge,
and a separate answer key (round1_key.txt) that must NOT be opened first.
"""
import json
import os
import random
import urllib.request
import concurrent.futures

OUT_DIR = "/home/dc/electric-sheep/ab-test"
BLIND = os.path.join(OUT_DIR, "round1_blind.md")
KEY = os.path.join(OUT_DIR, "round1_key.txt")

ENDPOINTS = {
    "31b": ("http://127.0.0.1:8089/v1/chat/completions", "gemma-31b-qat-mtp"),
    "26b": ("http://127.0.0.1:8090/v1/chat/completions", "gemma-26b-a4b"),
}

PROMPTS = [
    ("Fiction scene",
     "Write a short scene of 300-400 words. A lighthouse keeper discovers the "
     "light has been off for three nights, and no one in the village seems to "
     "notice. End on an unsettling detail."),
    ("Explainer",
     "Explain in clear prose for a smart non-specialist why the sky is blue but "
     "sunsets are red. 250-350 words. No jargon without explanation."),
    ("Argument",
     "Write a 300-word argument for why cities should ban single-use plastic "
     "bags. Anticipate and rebut the strongest counterargument."),
    ("Dialogue",
     "Write a 250-word exchange between a retired chess grandmaster and his "
     "12-year-old student, the morning after the student lost a match they "
     "should have won. Let their personalities show in the dialogue."),
    ("Essay",
     "Write a 500-word essay on the experience of waiting - airports, queues, "
     "loading screens - and what it reveals about modern life. Keep a "
     "consistent thread from first sentence to last."),
]


def call(url, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    content = (m.get("content") or "").strip()
    reasoning_len = len((m.get("reasoning_content") or "").strip())
    return content, reasoning_len


def call_with_retry(url, model, prompt):
    # Gemma4 thinks first; if the budget is eaten by reasoning, retry bigger.
    content, rlen = call(url, model, prompt, 3000)
    if len(content) < 80:
        content, rlen = call(url, model, prompt, 8000)
    return content, rlen


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for i, (title, prompt) in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] {title} ...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f31 = ex.submit(call_with_retry, *ENDPOINTS["31b"], prompt)
            f26 = ex.submit(call_with_retry, *ENDPOINTS["26b"], prompt)
            c31, r31 = f31.result()
            c26, r26 = f26.result()
        a_is_31 = random.random() < 0.5
        a, b = (c31, c26) if a_is_31 else (c26, c31)
        results.append((i, title, prompt, a, b,
                        "31b" if a_is_31 else "26b", r31, r26))
        print(f"    31b: {len(c31)} ch content / {r31} ch reasoning | "
              f"26b: {len(c26)} ch content / {r26} ch reasoning", flush=True)

    lines = [
        "# Blind A/B - prose quality (round 1)",
        "",
        "Two models answered each prompt. Pick the better response (A or B).",
        "Judge on prose quality: clarity, voice, coherence, style, detail. "
        "Not length.",
        "Reply with your picks, e.g. `1A 2B 3A 4B 5A` (or 'tie' for any).",
        "",
    ]
    for (i, title, prompt, a, b, *_rest) in results:
        lines += [f"## Prompt {i} - {title}", "",
                  f"> {prompt}", "",
                  "### Response A", "", a, "",
                  "### Response B", "", b, ""]
    with open(BLIND, "w") as f:
        f.write("\n".join(lines))

    with open(KEY, "w") as f:
        f.write("ANSWER KEY - do not read until picks are in\n\n")
        for (i, title, _p, _a, _b, a_model, _r31, _r26) in results:
            b_model = "26b" if a_model == "31b" else "31b"
            f.write(f"Prompt {i} ({title}): A = {a_model}, B = {b_model}\n")

    print(f"\nWrote {BLIND}")
    print(f"Key at {KEY} (do not open until picks are in)")


if __name__ == "__main__":
    main()