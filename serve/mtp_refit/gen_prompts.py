#!/usr/bin/env python3
"""Generate a diverse, seeded calibration prompt suite for MTP re-fit data gen.

Categories mirror real serving traffic: coding, math/reasoning, chat/instruction,
and long-form. Content is synthesized from topic pools + templates with a fixed
seed so runs are reproducible. Output: JSONL of {"id": int, "text": str}.

Usage:
  python3 gen_prompts.py --count 300 --seed 1234 --out prompts.jsonl
"""
import argparse
import json
import random

TOPICS = [
    "distributed systems", "memory management", "database indexing",
    "network protocols", "compiler design", "operating system scheduling",
    "cryptography", "machine learning pipelines", "kubernetes orchestration",
    "real-time rendering", "signal processing", "graph algorithms",
    "concurrent programming", "file systems", "load balancing",
    "cache coherence", "virtualization", "stream processing",
    "version control internals", "container networking",
]

CODE_TASKS = [
    "parse a CSV file with quoted fields and validate each row",
    "implement a thread-safe LRU cache with a maximum size",
    "merge k sorted linked lists into one sorted list",
    "detect cycles in a directed graph and return the cycle",
    "implement a rate limiter using the token bucket algorithm",
    "write a retry wrapper with exponential backoff and jitter",
    "parse a nested JSON config and apply defaults",
    "implement a sliding-window aggregator over a stream of values",
    "validate and normalize an email address",
    "implement a simple expression evaluator for arithmetic",
    "build an in-memory event bus with pub/sub and wildcards",
    "implement a consistent hash ring for sharding keys",
    "write a diff algorithm that produces minimal edits",
    "implement a priority queue with decrease-key support",
    "parse a date range string and expand it into days",
    "implement a circuit breaker with half-open probing",
    "write a memoized recursive Fibonacci with an explicit stack",
    "implement a topological sort that breaks ties deterministically",
    "build a key-value store with TTL expiration",
    "implement a lock-free single-producer single-consumer ring buffer",
]

REWRITE_TONES = ["formal", "casual", "concise", "empathetic", "technical",
                 "persuasive", "apologetic", "confident"]

SUMMARY_SEEDS = [
    "a postmortem of a production outage caused by a cascading retry storm",
    "the tradeoffs between eventual consistency and strong consistency",
    "why tail latency is harder to optimize than mean latency",
    "the design of a distributed consensus protocol",
    "how garbage collection pauses affect real-time systems",
    "the economics of edge computing versus centralized cloud",
    "the history and evolution of relational database query optimizers",
    "why microservices can increase operational complexity",
    "the role of observability in debugging distributed systems",
    "the tension between developer velocity and system reliability",
]


def _pick(rng, pool, n=1):
    return rng.sample(pool, n) if len(pool) >= n else pool


def gen_coding(rng):
    task = rng.choice(CODE_TASKS)
    style = rng.choice(["Python", "TypeScript", "Go", "Rust"])
    kind = rng.choice([
        f"Write a {style} function that {task}. Include type hints and docstrings.",
        f"Implement {task} in {style}. Handle edge cases and add unit tests.",
        f"Here is a buggy {style} snippet for '{task}'. Find and fix the bug:\n"
        f"def solve(x):\n    return x\nExplain the fix.",
        f"Refactor the following {style} code for clarity and performance. Task: {task}.",
    ])
    return kind


def gen_math(rng):
    a, b = rng.randint(2, 97), rng.randint(2, 97)
    c = rng.randint(1, 50)
    n = rng.randint(2, 9)
    kind = rng.choice([
        f"Solve step by step: a warehouse ships {a} boxes per hour for {b} hours, "
        f"then {c} boxes are returned. How many boxes remain? Show your work.",
        f"A pipe fills a tank in {a} minutes and another drains it in {b} minutes. "
        f"If both are open, how long to fill? Solve carefully.",
        f"Prove or disprove: for all integers k>=1, {n}^2 - 1 is divisible by 8. Give a rigorous argument.",
        f"There are {a} red and {b} blue balls in an urn. You draw {c} without replacement. "
        f"What is the probability of at least one red? Compute exactly.",
    ])
    return kind


def gen_chat(rng):
    tone = rng.choice(REWRITE_TONES)
    kind = rng.choice([
        f"Rewrite this in a {tone} tone: 'We missed the deadline because of a dependency issue.'",
        f"List {rng.randint(5, 10)} practical ways to reduce technical debt in a fast-moving team, with brief rationale for each.",
        f"Compare {_pick(rng, TOPICS, 2)[0]} and {_pick(rng, TOPICS, 2)[1]} in a balanced table, then give a recommendation for a small team.",
        f"Explain {_pick(rng, TOPICS, 1)[0]} to a smart non-specialist in under 200 words.",
        f"Draft a short status update to stakeholders about a delayed feature, in a {tone} tone.",
    ])
    return kind


def gen_longform(rng):
    topic = rng.choice(TOPICS)
    kind = rng.choice([
        f"Write a detailed, well-structured explanation of {topic}. Cover the core ideas, "
        f"common pitfalls, and at least one concrete example. Aim for depth.",
        f"Draft a technical design document outline for a system centered on {topic}. "
        f"Include goals, non-goals, architecture, data flow, failure modes, and rollout plan.",
        f"Summarize the key lessons from: {_pick(rng, SUMMARY_SEEDS, 1)[0]}. "
        f"Then propose three actionable takeaways for an engineering org.",
        f"Write a thorough tutorial introducing {topic}, from first principles to an advanced nuance. "
        f"Use headings and short code or math where helpful.",
    ])
    return kind


def generate(count, seed):
    rng = random.Random(seed)
    cats = [gen_coding, gen_math, gen_chat, gen_longform]
    out = []
    for i in range(count):
        fn = cats[i % len(cats)] if rng.random() < 0.8 else rng.choice(cats)
        out.append({"id": i, "text": fn(rng)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="prompts.jsonl")
    a = ap.parse_args()
    items = generate(a.count, a.seed)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} prompts -> {a.out}")
    # show a couple samples
    for it in items[:3]:
        print(f"  [{it['id']}] {it['text'][:80]}...")


if __name__ == "__main__":
    main()
