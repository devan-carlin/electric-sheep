# Intel Arc Pro B70 Power & Frequency Tuning (Ubuntu)

## Overview

This guide covers reducing power consumption and heat output on 4× Intel Arc
Pro B70 GPUs (32 GB each, `xe` driver) using sysfs power caps and frequency
limits, with persistence via systemd. The companion script is
`scripts/common/setup-gpu-power-limits.sh`.

Unlike the RTX 5090 (Windows, MSI Afterburner), the B70 exposes its power and
frequency controls through Linux sysfs — no GUI tool needed.

## Why Underclock for LLM Inference?

LLM inference is **memory-bandwidth bound**, not compute-bound. The bottleneck
is streaming weights from VRAM to the compute units, not raw FLOPS. This means:

- Lower GPU clock → less power/heat, similar token/sec
- Lower power cap → thermal headroom, stable sustained performance
- Combined → quieter system, lower electricity cost, less thermal throttling risk

## Typical B70 Trade-offs

| Setting | Power/GPU | Boost Clock | 27B Q4 Throughput |
|---------|-----------|-------------|-------------------|
| Stock (no cap) | ~175W | ~1450 MHz | ~12–15 tok/s |
| **160W cap** | ~160W | ~1050 MHz | **~11–14 tok/s (~5–10% slower)** |
| 140W cap | ~140W | ~950 MHz | ~10–13 tok/s (~15% slower) |
| 120W cap | ~120W | ~850 MHz | ~9–12 tok/s (~20% slower) |

**The 160W cap is the sweet spot:** ~36W saved per GPU (144W total across 4
GPUs) with a 5–10% throughput reduction that's barely noticeable in
interactive use.

## Prerequisites

1. **Intel `xe` driver** loaded (standard on Ubuntu 26.04 with oneAPI 2026.1)
2. **hwmon enabled** — the `xe` driver exposes `/sys/class/hwmon/hwmon*/power1_cap`
3. **sudo access** — writing to sysfs requires root
4. (Optional) `intel-gpu-tools` for `intel_gpu_top` monitoring:
   ```bash
   sudo apt install intel-gpu-tools
   ```

## Method 1: The Script (Recommended)

The repo ships a full-featured script that handles power caps, GPU frequency
limits, CPU governor, and systemd persistence in one shot:

```bash
# Apply defaults (160W cap, 1100 MHz max, 1200 MHz boost, persistent)
sudo bash ~/electric-sheep/scripts/common/setup-gpu-power-limits.sh

# Custom power cap
sudo bash ~/electric-sheep/scripts/common/setup-gpu-power-limits.sh --watts 140

# Custom frequency limits
sudo bash ~/electric-sheep/scripts/common/setup-gpu-power-limits.sh --watts 160 --gt-max 1000 --gt-boost 1100

# Apply without persistence (current session only)
sudo bash ~/electric-sheep/scripts/common/setup-gpu-power-limits.sh --no-persist

# Check current state
sudo bash ~/electric-sheep/scripts/common/setup-gpu-power-limits.sh --status

# Remove all limits, restore stock
sudo bash ~/electric-sheep/scripts/common/setup-gpu-power-limits.sh --reset
```

### What it does

1. **Power caps** — writes microwatts to `/sys/class/hwmon/hwmon*/power1_cap`
   for every hwmon interface named `i915` or `xe` (one per GPU).
2. **GPU frequency limits** — writes to `/sys/class/drm/card*/device/`:
   - `gt_max_freq_mhz` — hard ceiling on GPU clock
   - `gt_boost_freq_mhz` — transient boost limit
3. **CPU governor** — sets all cores to `performance` (prevents the CPU from
   downclocking during prompt processing / tokenization).
4. **Persistence** — creates `/etc/systemd/system/arc-b70-power-limits.service`
   (a oneshot that re-applies all limits at boot).

### The systemd service

```ini
[Unit]
Description=Intel Arc B70 Power & Frequency Limits
After=multi-user.target
Wants=systemd-modules-load.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '<inline script that re-applies caps + freqs + governor>'

[Install]
WantedBy=multi-user.target
```

Enabled with `systemctl enable arc-b70-power-limits.service`. Removed by
`--reset`.

## Method 2: Minimal rc.local (Power Cap Only)

If you only want the power cap (no frequency limits, no CPU governor), a
minimal `/etc/rc.local` approach works. This is what the root-level
`setup-power-clamps.sh` deploys:

```bash
#!/bin/bash
# /etc/rc.local — minimal B70 power clamp

for hwmon_dir in /sys/class/hwmon/hwmon*; do
    if [ -f "$hwmon_dir/name" ]; then
        hwmon_name=$(cat "$hwmon_dir/name")
        if [[ "$hwmon_name" == "i915" || "$hwmon_name" == "xe" ]]; then
            if [ -f "$hwmon_dir/power1_cap" ]; then
                echo 160000000 > "$hwmon_dir/power1_cap"
            fi
        fi
    fi
done

# CPU governor (optional)
if [ -d /sys/devices/system/cpu/cpufreq ]; then
    echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
fi

exit 0
```

Plus the standard `rc-local.service` systemd unit to run it at boot. This is
simpler but gives you less control (no frequency caps, no per-GPU tuning).

**Use Method 1 unless you have a specific reason not to.**

## The sysfs Interfaces

| Path | What | Units |
|------|------|-------|
| `/sys/class/hwmon/hwmon*/name` | Driver name (`xe` or `i915`) | string |
| `/sys/class/hwmon/hwmon*/power1_cap` | Power limit | microwatts (160W = 160000000) |
| `/sys/class/hwmon/hwmon*/power1_average` | Current power draw | microwatts |
| `/sys/class/drm/card*/device/gt_max_freq_mhz` | Max GPU clock | MHz |
| `/sys/class/drm/card*/device/gt_boost_freq_mhz` | Boost clock limit | MHz |
| `/sys/class/drm/card*/device/gt_min_freq_mhz` | Min GPU clock | MHz |
| `/sys/class/drm/card*/device/gt_cur_freq_mhz` | Current GPU clock | MHz |
| `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` | CPU governor | string |

**Note:** hwmon indices (`hwmon0`, `hwmon1`, …) are **not stable across
reboots**. Always parse by `name` (looking for `xe` or `i915`), never by index.
This is why both scripts loop over all `hwmon*` dirs and check the name.

## Verification

### Check power caps and current draw

```bash
for hwmon_dir in /sys/class/hwmon/hwmon*; do
    if [ -f "$hwmon_dir/name" ]; then
        name=$(cat "$hwmon_dir/name")
        if [[ "$name" == "i915" || "$name" == "xe" ]]; then
            cap=$(cat "$hwmon_dir/power1_cap" 2>/dev/null || echo "n/a")
            avg=$(cat "$hwmon_dir/power1_average" 2>/dev/null || echo "n/a")
            echo "$hwmon_dir ($name): cap=${cap}μW avg=${avg}μW"
        fi
    fi
done
```

### Check GPU frequencies

```bash
for card in /sys/class/drm/card*/device; do
    [ -f "$card/gt_cur_freq_mhz" ] && \
        echo "$(basename $(dirname $card)): cur=$(cat $card/gt_cur_freq_mhz) MHz max=$(cat $card/gt_max_freq_mhz) MHz"
done
```

### Check CPU governor

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

### Continuous monitoring (intel_gpu_top)

```bash
sudo intel_gpu_top -d 1 -1   # 1-second sample, then exit
sudo intel_gpu_top            # continuous (Ctrl+C to stop)
```

## Performance Testing

Run the same inference workload before and after:

```bash
# Quick throughput check (27B INT4, 2048-token prompt)
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-ara-int4","messages":[{"role":"user","content":"Explain quantum entanglement in detail."}],"max_tokens":512}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['completion_tokens'])"
```

Record:
- **Token generation speed** (tokens/sec)
- **Power draw** (from `power1_average`)
- **Temperature** (from `power1_average` context or `intel_gpu_top`)
- **Fan noise** (subjective)

### Expected Results at 160W Cap

| Metric | Stock | 160W Cap | Change |
|--------|-------|----------|--------|
| Power/GPU | ~175W | ~160W | −15W (−9%) |
| Power (4× total) | ~700W | ~640W | −60W |
| Boost clock | ~1450 MHz | ~1050 MHz | −28% |
| Token/sec (27B Q4) | ~13 tok/s | ~12 tok/s | −8% |
| Temperature | ~70°C | ~60°C | −10°C |
| Fan noise | Medium | Low | Noticeably quieter |

## Troubleshooting

### `power1_cap` not found

- The `xe` driver must be loaded: `lsmod | grep xe`
- hwmon must be enabled: check `ls /sys/class/hwmon/` for entries
- On some kernel/driver versions, the file is `power1_average` only (read-only).
  If `power1_cap` doesn't exist, your driver version may not support power
  limiting via sysfs. Check `dmesg | grep -i xe` for clues.

### hwmon indices shift after reboot

This is expected. **Never hardcode `hwmon3`** — always parse by `name`. Both
scripts in this repo do this correctly.

### Cap doesn't seem to apply

- Verify you wrote microwatts, not watts: 160W = `160000000` (160 × 10^6)
- Check `power1_average` after a few seconds of load — it should stay below
  the cap
- Some workloads (idle) won't hit the cap; the limit only engages under load

### Frequency writes fail

- `gt_max_freq_mhz` and `gt_boost_freq_mhz` require the `xe` driver (not
  `i915`). On older kernels with `i915`, these files may not exist.
- The values must be within the hardware's supported range. Writing above the
  hardware max is silently clamped; writing below `gt_min_freq_mhz` may fail.

### Service doesn't run at boot

```bash
sudo systemctl status arc-b70-power-limits.service
journalctl -u arc-b70-power-limits.service --since today
```

Common cause: the `xe` driver isn't loaded yet when the service runs. The
`After=multi-user.target` + `Wants=systemd-modules-load.service` ordering
should handle this, but if not, add `After=systemd-modules-load.service`
explicitly.

## Comparison: B70 (Ubuntu) vs 5090 (Windows)

| Aspect | Intel Arc B70 (4×, Ubuntu) | RTX 5090 (1×, Windows) |
|--------|---------------------------|----------------------|
| Tool | sysfs + systemd | MSI Afterburner |
| Power cap method | `power1_cap` sysfs (μW) | Power Limit slider (%) |
| Frequency control | `gt_max_freq_mhz` / `gt_boost_freq_mhz` | Core Clock offset (MHz) |
| Persistence | systemd oneshot service | Auto-start with Windows |
| Target cap | 160W per GPU (640W total) | 315W (70% of 450W) |
| Savings | ~60W system-wide | ~135W |
| Throughput hit | ~5–10% | ~10% |
| Complexity | Low (one script) | Medium (GUI + profiles) |

The B70 is actually *easier* to tune than the 5090: no GUI, no driver
restarts, no profile management. Just write a number to a file. The tradeoff
is less granularity — you can't undervolt the B70 the way you can the 5090.
