# Undervolting Intel Arc B70 GPUs for LLM Inference

*Four Intel Arc Pro B70 GPUs at stock draw ~800W combined. Capping each at
160W saves ~160W for a 5-10% throughput hit, and the GPUs run quieter, cooler,
and more efficient.*

> **Status: draft.** Restructured as a problem/solution deep dive. Needs a
> final pass (a `--status` screenshot, a real stock-wattage read) before
> posting.

---

## The problem

I run four Intel Arc Pro B70 GPUs (Battlemage, 32 GB each) on an Ubuntu
26.04 server for LLM inference. Each card is rated at 230W TDP. Under
sustained decode they draw about 200W, below the ceiling but still a lot.
That is ~800W of GPU power, all at the same time, for a workload that barely
needs it.

The reason is a counterintuitive fact about LLM inference: **decode is
memory-bandwidth-bound, not compute-bound.** The bottleneck is streaming
weights from VRAM to the compute units, not raw FLOPS. Most of that 200W per
card pays for compute headroom the workload never uses.

The cost of leaving it at stock:

- ~800W of GPU power draw, plus the CPU and the rest of the box
- Sustained heat and fan noise from four cards at full tilt
- Electricity you pay for every hour the server runs
- Diminishing returns: more watts, but almost no more tokens per second

## The solution

On Windows you would open MSI Afterburner and drag a power slider. On Linux
there is no Afterburner. The `xe` driver (standard on Ubuntu 26.04 with
oneAPI) exposes the controls as plain sysfs files, and a systemd unit makes
them survive reboots.

Two controls do the job:

| sysfs path | What it sets | Units |
|------------|--------------|-------|
| `/sys/class/hwmon/hwmon*/power1_cap` | Power limit | microwatts (160W = 160000000) |
| `/sys/class/drm/card*/device/gt_max_freq_mhz` | Max GPU clock | MHz |

Set the power cap on every Intel GPU:

```bash
# 160W power cap on every Intel GPU (match by driver name, not index)
for d in /sys/class/hwmon/hwmon*; do
  [ "$(cat $d/name 2>/dev/null)" = "xe" ] && echo 160000000 > $d/power1_cap
done
```

Set the GPU clock ceiling:

```bash
# Cap the GPU clock at 1100 MHz
for c in /sys/class/drm/card*/device; do
  echo 1100 > $c/gt_max_freq_mhz
done
```

One gotcha matters here: `hwmon0`, `hwmon1`, and so on are assigned at probe
time and **change across reboots**. A script that hardcodes `hwmon2` will cap
the wrong GPU, or no GPU, after a restart. Always loop over all `hwmon*`
directories and match on `name` (`xe` or `i915`).

To make it persist, a oneshot systemd service re-applies the caps, the
frequency limits, and the CPU governor at boot:

```ini
[Unit]
Description=Intel Arc B70 Power & Frequency Limits
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '<re-apply caps + freqs + governor>'

[Install]
WantedBy=multi-user.target
```

The companion script (`setup-gpu-power-limits.sh`) does all of this: power
caps, `gt_max_freq_mhz` / `gt_boost_freq_mhz`, and the CPU governor set to
`performance` (so the CPU does not downclock during prompt processing).
`--status` shows current caps and draw. `--reset` restores stock.

One clarification on the name: this is a **power cap plus a frequency
limit**, not a true voltage offset. The `xe` driver does not expose a voltage
control, so "undervolting" on Arc means capping power and clock.

## The results

I measured a 27B Q4 model across power caps, per GPU:

| Setting | Power/GPU | Boost clock | 27B Q4 throughput |
|---------|-----------|-------------|-------------------|
| Stock (no cap) | ~200W | ~1450 MHz | ~12-15 tok/s |
| **160W cap** | ~160W | ~1050 MHz | **~11-14 tok/s (~5-10% slower)** |
| 140W cap | ~140W | ~950 MHz | ~10-13 tok/s (~15% slower) |
| 120W cap | ~120W | ~850 MHz | ~9-12 tok/s (~20% slower) |

The curve has a knee. Above it, you are paying for compute the workload does
not use, so cutting it is nearly free. Below it, the GPU starts downclocking
memory along with the core, and that is the actual bottleneck. Every watt
saved past the knee costs real throughput.

For a memory-bound workload on the B70, the knee sits around the 160W cap.

## The payoff

In spite of a minor hit to throughput, the capped GPUs are a better deal:

- **Quieter.** Lower power means lower fan speed. Four cards at 160W are
  noticeably quieter than four at 200W.
- **Cooler.** Less power in means less heat to move. The box stays cooler
  under sustained load.
- **More energy efficient.** ~40W saved per GPU, ~160W across the four cards,
  for a 5-10% throughput reduction that is barely noticeable in interactive
  use.
- **Most of the throughput stays.** You keep the majority of stock tokens per
  second while shedding the wasted compute headroom.

Verify it on your own box:

```bash
# Current caps + draw
sudo ./setup-gpu-power-limits.sh --status

# Continuous GPU monitor
sudo intel_gpu_top
```

Run the same inference workload before and after, and compare tok/s.

## Takeaway

- **Profile the workload before you tune.** If it is memory-bound (most LLM
  inference is), the compute headroom is wasted power.
- **There is a knee.** For a B70 running a 27B Q4 model, ~160W is where free
  savings turn into real cost.
- **Measure, don't assume.** The shape of the curve is general. The number is
  yours to find.

## Links

- Power-tuning guide (full sysfs reference, both methods):
  [arc-b70-power-tuning.md](https://github.com/devan-carlin/electric-sheep/blob/main/docs/guides/arc-b70-power-tuning.md)
- Setup script (caps + freqs + governor + systemd):
  [setup-gpu-power-limits.sh](https://github.com/devan-carlin/electric-sheep/blob/main/build/common/setup-gpu-power-limits.sh)