# We Fixed the Crash. Then Found the "Speedup" Made It 33% Slower.

*Speculative decoding is supposed to be free lunch. On Intel Arc, it wasn't.*

---

There's a feature in vLLM called **multi-token prediction** (MTP), a form of
speculative decoding. The pitch is simple: the model has an extra lightweight
layer that *drafts* the next token, the main model *verifies* it in the same
forward pass, and if it's right you get two tokens for the price of one. More
tokens per step. Faster generation. It's one of those features that sounds too
good to be true, which is exactly why we turned it on.

We couldn't even turn it on at first — see the companion piece on the int64
pointer overflow. But once that was fixed and MTP was actually running, we
measured it. And the result was the opposite of the pitch.

## The numbers

Same model, same hardware (4× Intel Arc Pro B70, tensor-parallel 4), same
INT4 quantization. The only difference: MTP on vs. off.

| Config | Throughput |
|--------|-----------|
| No MTP (baseline) | **52.2 tok/s** |
| MTP, 1 draft token | **34.9 tok/s** |

MTP was **33% slower**. Not a rounding error. Not a warmup artifact. We ran
three 1,500-token prompts each way to be sure.

## But the drafts were *good*

Here's the part that made us stop and think. We instrumented the acceptance
rate — how often the drafted token was actually accepted by the main model.

- Mean acceptance length: **~1.85 tokens per step**
- Per-position acceptance: **80–90%**

The draft layer was *working*. It was guessing the next token correctly most of
the time. By the textbook argument, that should be a win: you're getting nearly
two tokens per step instead of one.

So why was it slower?

## The cost model the pitch leaves out

Speculative decoding isn't free. Every step now costs:

1. The **draft layer's forward pass** — an extra transformer layer, run every
   step, to produce the guess.
2. The **main model processing two tokens** instead of one — the verify step
   has to score the drafted token *and* the real one.

On a GPU where a single forward pass is already the bottleneck, you've just
made every step do *more work*. The acceptance rate tells you how often the
extra work *pays off*. The throughput tells you whether it pays off *enough*.

On this hardware, it didn't. The draft layer is a full transformer layer — it's
not "lightweight" in the way the pitch implies. And the XPU backend's per-step
overhead is high enough that the extra work per step swallows the benefit of
occasionally getting two tokens.

The math: you need the acceptance rate to be high *and* the draft to be cheap
relative to the main model. We had the first. We did not have the second.

## The verdict

For this model, on this hardware, **the fastest stable configuration is no
MTP.** ~52 tok/s, no speculative decoding, no extra layer, no crash to debug.

That's a negative result, and it's worth stating plainly: **a feature that is
a net win on one backend can be a net loss on another.** The same MTP layer
that helps on a fast CUDA GPU can hurt on an XPU where the per-step cost is
different. "Turn on speculative decoding" is not a universal optimization. It
is a bet on the ratio of draft cost to acceptance rate, and that ratio is
hardware-specific.

## Takeaway

- **Measure speculative decoding on your actual hardware.** The acceptance
  rate is necessary but not sufficient. What matters is acceptance rate
  *divided by* the extra cost per step, and that denominator changes with the
  backend.
- **A high acceptance rate does not guarantee a speedup.** If the draft is
  expensive, you can be accepted 90% of the time and still lose.
- **Negative results are results.** "MTP is slower here" is a useful,
  publishable finding. It saves the next person the experiment.
