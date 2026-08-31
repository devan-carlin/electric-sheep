# Undervolting Intel Arc B70 GPUs for LLM Inference

*LLM inference is memory-bandwidth-bound. On Intel Arc you can cap power and
clock through plain sysfs files, no GUI tool, and save ~160W across four GPUs
for a 5-10% throughput hit.*

> **Status: draft.** Real measurements from the power-tuning guide; needs a
> final pass (tighten the table, add a `--status` screenshot) before posting.

---

## The hook

Four Intel Arc Pro B70 GPUs at stock draw near their 200W TDP each. That is
~800W of GPU power for a workload that, it turns out, barely needs it.

The reason is the same counterintuitive fact as on NVIDIA cards: **LLM
inference is memory-bandwidth-bound, not compute-bound.** The bottleneck is
streaming weights from VRAM to the compute units, not raw FLOPS. Most of that
200W pays for compute headroom the workload never uses.

The difference from an RTX 5090: on Linux there is no Afterburner. The `xe`
driver exposes power and frequency controls as plain sysfs files, and a
systemd unit makes it survive reboots. No GUI, no driver hacks.

## The tradeoff curve

Measured a 27B Q4 model across power caps (per GPU):

| Setting | Power/GPU | Boost clock | 27B Q4 throughput |
|---------|-----------|-------------|-------------------|
| Stock (no cap) | ~200W | ~1450 MHz | ~12-15 tok/s |
| **160W cap** | ~160W | ~1050 MHz | **~11-14 tok/s (~5-10% slower)** |
| 140W cap | ~140W | ~950 MHz | ~10-13 tok/s (~15% slower) |
| 120W cap | ~120W | ~850 MHz | ~9-12 tok/s (~20% slower) |

**The 160W cap is the sweet spot.** ~40W saved per GPU, ~160W total across the
four cards, for a 5-10% throughput reduction that is barely noticeable in
interactive use.

## Why the curve has a knee

- **Above the knee:** you are paying for compute the workload does not use.
  Cutting it is nearly free.
- **Below the knee:** the GPU starts downclocking memory along with the core,
  and that is the actual bottleneck. Every watt saved now costs real
  throughput.

For a memory-bound workload on the B70, the knee sits around the 160W cap.

## How it works on Linux

The `xe` driver (standard on Ubuntu 26.04 with oneAPI) exposes:

| sysfs path | What | Units |
|------------|------|-------|
| `/sys/class/hwmon/hwmon*/power1_cap` | Power limit | microwatts (160W = 160000000) |
| `/sys/class/hwmon/hwmon*/power1_average` | Current draw | microwatts |
| `/sys/class/drm/card*/device/gt_max_freq_mhz` | Max GPU clock | MHz |
| `/sys/class/drm/card*/device/gt_boost_freq_mhz` | Boost clock limit | MHz |

Two controls, a few lines of shell:

```bash
# 160W power cap on every Intel GPU (match by driver name, not index)
for d in /sys/class/hwmon/hwmon*; do
  [ "$(cat $d/name 2>/dev/null)" = "xe" ] && echo 160000000 > $d/power1_cap
done

# GPU clock ceiling
for c in /sys/class/drm/card*/device; do
  echo 1100 > $c/gt_max_freq_mhz
done
```

### The gotcha: hwmon indices are not stable

`hwmon0`, `hwmon1`, and so on are assigned at probe time and **change across
reboots**. A script that hardcodes `hwmon2` will cap the wrong GPU, or no
GPU, after a restart. Always loop over all `hwmon*` directories and match on
`name` (`xe` or `i915`).

### Persistence

A oneshot systemd service re-applies caps, frequency limits, and the CPU
governor at boot:

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
caps, `gt_max_freq_mhz` / `gt_boost_freq_mhz`, CPU governor set to
`performance` (so the CPU does not downclock during prompt processing), and
the systemd unit. `--status` shows current caps and draw. `--reset` restores
stock.

## Caveats

- This is a **power cap + frequency limit**, not a true voltage offset. The
  `xe` driver does not expose a voltage control, so "undervolting" on Arc
  means capping power and clock.
- The throughput numbers are per-GPU ranges from a 27B Q4 workload. Your knee
  depends on the model, quantization, and batch size.
- Compute-bound workloads (training, large-batch prefill) will cost more than
  10% under the cap.

## Verification

```bash
# Current caps + draw
sudo ./setup-gpu-power-limits.sh --status

# Continuous GPU monitor
sudo intel_gpu_top
```

Run the same inference workload before and after, and compare tok/s.

## Summary

- LLM inference is memory-bound, so most of a GPU's stock power is wasted
  compute headroom.
- On Intel Arc, power and frequency caps are plain sysfs files. No GUI tool.
- 160W per B70 is the sweet spot: ~160W saved across four GPUs for 5-10%
  throughput.
- Parse hwmon by driver name, not index. Persist with a oneshot systemd unit.

## Links

- Power-tuning guide (full sysfs reference, both methods):
  [arc-b70-power-tuning.md](https://github.com/devan-carlin/electric-sheep/blob/main/docs/guides/arc-b70-power-tuning.md)
- Setup script (caps + freqs + governor + systemd):
  [setup-gpu-power-limits.sh](https://github.com/devan-carlin/electric-sheep/blob/main/build/common/setup-gpu-power-limits.sh)

## Takeaway

- **Profile the workload before you tune.** If it is memory-bound (most LLM
  inference is), the compute headroom is wasted power.
- **There is a knee.** For a B70 running a 27B Q4 model, ~160W is where free
  savings turn into real cost.
- **Measure, don't assume.** The shape of the curve is general. The number is
  yours to find.