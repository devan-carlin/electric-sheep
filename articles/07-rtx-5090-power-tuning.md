# What a 70% Power Cap Actually Does to an RTX 5090

*LLM inference is memory-bandwidth-bound, not compute-bound. That one fact
changes the whole power-tuning calculus.*

> **Status: draft.** Real measurements; needs a final pass (a photo of the
> Afterburner slider) before posting.

---

## The hook

An RTX 5090 draws 450–500W at stock. For a home AI box that's a lot of heat,
a lot of fan noise, and a lot of electricity, for a workload that, it turns
out, barely needs the power.

The reason is a single, counterintuitive fact: **LLM inference is
memory-bandwidth-bound, not compute-bound.** The bottleneck is streaming
weights from VRAM to the compute units, not raw FLOPS. So most of that 500W
is being spent on compute headroom the workload never uses.

## The tradeoff curve

We measured a 27B Q4 model across power caps:

| Setting | Power | Boost clock | 27B Q4 throughput |
|---------|-------|-------------|-------------------|
| Stock (no cap) | ~450–500W | ~2000+ MHz | ~15–20 tok/s |
| **70% (~315W)** | ~315W | ~1800 MHz | **~14–18 tok/s (~10% slower)** |
| 60% (~270W) | ~270W | ~1650 MHz | ~13–16 tok/s (~15% slower) |
| 50% (~225W) | ~225W | ~1500 MHz | ~12–15 tok/s (~20% slower) |

**The 70% cap is the sweet spot.** You save ~135W (and the heat and noise that
comes with it) for a ~10% throughput hit that's barely noticeable in
interactive use. Below 70%, the throughput starts to fall faster than the
power, because you're finally touching the memory-clock side of the equation.

## Why the curve is shaped that way

- **Above ~70%:** you're paying for compute the workload doesn't use. Cutting
  it is nearly free.
- **Below ~70%:** the GPU starts to downclock memory along with the core, and
  *that* is the actual bottleneck. Now every watt you save costs real
  throughput.

So the power cap has a "knee," and for a memory-bound workload on a 5090 it's
around 70%.

## How to set it

MSI Afterburner → **Power Limit** slider → 70%. Optionally:

- **Temp Limit** to 80–85°C (a safety ceiling if the power cap isn't enough).
- **Core Clock** offset of −100 to −200 MHz (caps boost independently).
- **Memory Clock** offset: leave at stock unless you've measured a benefit;
  this is the side that hurts.

Use the **NVIDIA Studio driver**, not Game Ready, for stability over long
inference runs.

## Verification

```powershell
# Poll power, clocks, and temperature every 2 seconds
nvidia-smi --query-gpu=power.draw,power.limit,temperature.gpu,clocks.gr,clocks.mem --format=csv -l 2
```

You should see `power.draw` holding under the cap, `power.limit` at your
target wattage, and `temperature.gpu` a few degrees below stock. Run the same
inference workload before and after, and compare tok/s.

## Links

- Power-tuning guide (full Afterburner walkthrough, voltage curve, `nvidia-smi`
  reference):
  [rtx-5090-power-tuning.md](https://github.com/devan-carlin/electric-sheep/blob/main/docs/guides/rtx-5090-power-tuning.md)
- The Intel Arc side of the same story (sysfs caps, no GUI):
  [10-arc-b70-power-tuning.md](https://github.com/devan-carlin/electric-sheep/blob/main/articles/10-arc-b70-power-tuning.md)

## Takeaway

- **Profile the workload before you tune.** If it's memory-bound (most LLM
  inference is), the compute headroom is wasted power.
- **There's a knee.** For a 5090 running a 27B Q4 model, ~70% power is where
  "free savings" turn into "real cost."
- **Measure, don't assume.** The exact knee depends on the model, the
  quantization, and the batch size. The shape of the curve is general; the
  number is yours to find.
