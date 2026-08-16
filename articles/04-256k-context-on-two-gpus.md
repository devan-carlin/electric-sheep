# Fitting a 256K Context Window on Two 32 GB GPUs

*The KV cache is the whole game. Here's the exact math for a 27B hybrid model
at its full native context, tensor-parallel across two Intel Arc GPUs.*

---

Qwen 3.8 27B has a **256K native context window**. That's 262,144 tokens —
enough to hold an entire codebase, a long technical document, or a multi-hour
conversation in a single prompt. The question is whether you can actually
*serve* it, and on how many GPUs.

We run it tensor-parallel across **two** of our four Intel Arc Pro B70 GPUs
(32 GB each), leaving the other two free for a second model. The model is
INT4-quantized (W4A16, AutoRound). It fits. But "it fits" hides a tradeoff
worth understanding: **the context window and your concurrency are the same
resource.**

## Where the memory goes

On each GPU, the 32 GB is split into three buckets:

1. **Model weights.** INT4 at 27B is ~18 GB total, so ~9 GB per GPU at TP=2.
2. **KV cache.** Everything left over. This is where your context lives.
3. **Activations + overhead.** A few GB, mostly constant.

The KV cache is the variable. It's sized from *free* VRAM, not from the
context length you request. vLLM computes how many tokens it can hold and
reports it:

```
Available KV cache memory: 13.36 GiB
GPU KV cache size: 843,486 tokens
Maximum concurrency for 262,144 tokens per request: 3.22x
```

That last line is the whole story.

## The tradeoff: context length vs. concurrency

The KV cache holds a fixed number of *total* tokens — here, ~843K. Every
request consumes tokens from that pool proportional to its sequence length.
So:

| Max context per request | Concurrent requests |
|------------------------|--------------------|
| 32,768 (32K) | ~25 |
| 65,536 (64K) | ~12 |
| 131,072 (128K) | ~6 |
| 262,144 (256K) | **~3** |

The pool doesn't grow when you ask for a longer window. You're just taking
*bigger slices* of the same pie. At 256K, you can serve about **three**
full-length requests at once. At 32K, you could serve ~25.

This is not a bug or a limitation of the hardware. It's arithmetic. A 256K
sequence's KV cache is 8× a 32K sequence's KV cache. Same memory, fewer
sequences.

## Why we chose 256K anyway

For single-user, long-document work — the actual use case — concurrency is
irrelevant. You're not serving 25 parallel clients. You're feeding one long
document and asking questions about it. In that regime, the 256K window is
strictly better: the whole document fits, nothing gets truncated, and the
model can reason across the entire thing.

The 3.22x concurrency number only matters if you expect parallel load. If you
do, you have three levers:

- **More GPUs.** TP=4 doubles the KV pool (and halves per-GPU weights),
  roughly doubling concurrency at the same context length.
- **Shorter max context.** If your documents are actually 32K, set
  `--max-model-len 32768` and reclaim the concurrency.
- **fp8 KV cache.** We use `--kv-cache-dtype fp8`, which halves the KV memory
  vs. fp16. It's already in the numbers above; dropping it would halve
  concurrency again.

## The config

```bash
--tensor-parallel-size 2
--max-model-len 262144
--max-num-seqs 8
--kv-cache-dtype fp8
--enable-prefix-caching
--gpu-memory-utilization 0.85
```

`--max-num-seqs 8` is the *scheduler* ceiling; the KV pool is the *real*
ceiling. The engine will never actually run 8 concurrent 256K sequences — the
KV cache only supports ~3. The flag just caps how many the scheduler will
*queue*.

## Takeaway

- **Context length and concurrency draw from the same KV cache pool.** A
  longer window doesn't cost more memory per se — it costs you *other
  requests*.
- **Read the "Maximum concurrency" line in vLLM's startup log.** It tells you
  exactly the tradeoff you're making, in one number.
- **Match the window to the workload.** Single-user long documents → max out
  the context. Multi-client serving → keep it short and add GPUs.
- **fp8 KV cache is a free 2× on context capacity.** If your backend supports
  it, turn it on.
