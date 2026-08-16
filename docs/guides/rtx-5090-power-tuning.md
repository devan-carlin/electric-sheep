# RTX 5090 Power & Frequency Tuning with MSI Afterburner

## Overview

This guide covers reducing power consumption and heat output on the RTX 5090 (32 GB VRAM) using MSI Afterburner, with minimal impact on LLM inference throughput.

## Why Underclock for LLM Inference?

LLM inference is **memory-bandwidth bound**, not compute-bound. The bottleneck is streaming weights from VRAM to the compute units, not raw FLOPS. This means:

- Lower GPU clock → less power/heat, similar token/sec
- Lower power cap → thermal headroom, stable sustained performance
- Combined → quieter system, lower electricity cost, less thermal throttling risk

### Typical RTX 5090 Trade-offs

| Setting | Power Draw | Boost Clock | Impact on 27B Q4 Inference |
|---|---|---|---|
| Stock (no cap) | ~450-500W | ~2000+ MHz | Baseline (~15-20 tok/s) |
| 70% power (~315W) | ~315W | ~1800 MHz | ~14-18 tok/s (~10% slower) |
| 60% power (~270W) | ~270W | ~1650 MHz | ~13-16 tok/s (~15% slower) |
| 50% power (~225W) | ~225W | ~1500 MHz | ~12-15 tok/s (~20% slower) |

**The 70% power cap (~315W) is the sweet spot:** ~135W saved with ~10% throughput reduction that's barely noticeable in interactive use.

## Prerequisites

1. **MSI Afterburner** (latest version): [Download](https://www.msi.com/Lab/Afterburner)
2. **RTDY 2026.1.0** (oneAPI runtime) — already installed on Ubuntu; Windows equivalent is the NVIDIA driver
3. **NVIDIA Studio Driver** (recommended over Game Ready for stability during long inference runs)

## Step-by-Step Configuration

### 1. Install MSI Afterburner

- Run the installer, accept defaults
- Check "Allow access to voltage control" when prompted (optional, not required for power limits)
- Launch MSI Afterburner

### 2. Set Power Limit

1. Click the **Power Limit** slider (⚡ icon or "Power Limit [mW]")
2. Drag to **70%** (or your target percentage)
3. The display will show the target wattage (e.g., ~315W for 70% of 450W)

### 3. Set Temperature Limit (Optional)

1. Click the **Temp Limit** slider
2. Set to **80-85°C** (stock is usually 90°C)
3. This provides a safety ceiling if power limiting isn't enough

### 4. Set Core Clock Offset (Optional Fine-Tuning)

1. Click the **Core Clock (MHz)** slider
2. Set to **-100 to -200 MHz** (negative offset)
3. This caps the boost clock independently of the power limit

### 5. Set Memory Clock Offset (Optional)

1. Click the **Memory Clock (MHz)** slider
2. Leave at **0** or set to **-100 MHz** if you want to reduce VRAM power
3. **Warning:** Reducing memory clock *will* impact inference speed more than core clock, since inference is memory-bound

### 6. Apply Settings

1. Click the **Checkmark (✓)** button to apply
2. Verify the changes took effect (sliders should show the new values)

### 7. Save Profile & Auto-Start

1. Click the **Profile** button (1-5 slots)
2. Select **Profile 1** and click **Save**
3. Open **Settings** (gear icon)
4. Enable:
   - "Apply current settings on startup"
   - "Apply after DirectX/OpenGL initialization"
   - "Start with Windows" (minimized to tray)

## Verification

### Check Current Power Draw

Open PowerShell and run:
```powershell
nvidia-smi --query-gpu=power.draw,power.limit,temperature.gpu,clocks.gr,clocks.mem,utilization.gpu --format=csv -l 2
```

This polls every 2 seconds. You should see:
- `power.draw` staying below your cap
- `power.limit` showing your target wattage
- `temperature.gpu` lower than stock (~65-75°C vs ~80-85°C)

### Check Clock Speeds

```powershell
nvidia-smi --query-gpu=clocks.gr,clocks.mem,clocks.max.gr --format=csv
```

Verify the core clock is below stock boost (~1800 MHz vs ~2000+ MHz at 70%).

## Performance Testing

### Before/After Comparison

Run the same inference workload and compare:

```powershell
# Test prompt (27B model, 2048 context)
$testPrompt = "Explain the concept of quantum entanglement in detail, including its implications for quantum computing and cryptography."

# Time the response
Measure-Command {
    # Your inference command here (e.g., llama.cpp, vLLM API call)
}
```

Record:
- **Token generation speed** (tokens/sec)
- **Power draw** (from nvidia-smi)
- **Temperature** (from nvidia-smi)
- **Fan speed** (from nvidia-smi or MSI Afterburner)

### Expected Results at 70% Power

| Metric | Stock | 70% Cap | Change |
|---|---|---|---|
| Power Draw | ~450W | ~315W | -135W (-30%) |
| Temperature | ~80°C | ~65-70°C | -10-15°C |
| Token/sec (27B Q4) | ~17 tok/s | ~15 tok/s | -12% |
| Fan Noise | Medium | Low | Noticeably quieter |

## Advanced: Undervolting (Optional)

The RTX 5090 supports voltage-frequency curve editing in MSI Afterburner. This can reduce power further without sacrificing clock speeds.

### Steps

1. Press **Ctrl+F** to unlock the voltage slider (if available on your card)
2. Click the **Voltage (mV)** slider
3. Click the graph icon to open the voltage-frequency curve editor
4. For each frequency point above your target boost clock, drag the voltage point **down** by 50-100mV
5. Apply and test stability

### Warning

- Undervolting can cause instability or crashes under heavy load
- Test thoroughly with a stress test (FurMark, 3DMark) before relying on it for inference
- If you experience crashes, increase the voltage by 25mV increments until stable

## Alternative: NVIDIA Control Panel Power Management

If you don't want MSI Afterburner running, you can set power management via NVIDIA Control Panel:

1. Open **NVIDIA Control Panel**
2. Go to **Manage 3D Settings** → **Global Settings**
3. Set **Power Management Mode** to **Prefer Maximum Performance** (paradoxically, this prevents dynamic clock scaling that causes spikes)
4. For power limiting, you'll still need MSI Afterburner or a third-party tool

## Alternative: nvapi / Command-Line Power Limit

For headless or automated setups, you can use NVIDIA's API or third-party tools:

```powershell
# Using nvidia-smi to set power limit (requires root/admin)
nvidia-smi -pl 315  # Set power limit to 315W
```

**Note:** The `-pl` flag may not persist across reboots. MSI Afterburner's auto-start is more reliable for desktop use.

## Troubleshooting

### Settings Not Applying

- Ensure MSI Afterburner is running as Administrator
- Check that the NVIDIA driver is up to date
- Try restarting the NVIDIA Display Driver (Win+Ctrl+Shift+B)

### Crashes or Instability

- Reduce the power limit further (e.g., 60% instead of 70%)
- Reset core clock offset to 0
- Disable undervolting if enabled
- Update MSI Afterburner to the latest version

### Power Limit Greyed Out

- Some OEM cards (laptops, pre-builts) lock the power limit
- Check if your HP Omen 45L allows power limit adjustments
- If locked, you can still adjust core/memory clock offsets

## Comparison: Ubuntu B70 vs Windows 5090

See `arc-b70-power-tuning.md` for the full B70 guide (sysfs caps, frequency
limits, systemd persistence, and the `setup-gpu-power-limits.sh` script).

| Aspect | Ubuntu B70 (4×) | Windows 5090 (1×) |
|---|---|---|
| Tool | sysfs / systemd | MSI Afterburner |
| Power Cap Method | `power1_cap` sysfs file | Power Limit slider |
| Frequency Control | `gt_max_freq_mhz` sysfs | Core Clock offset |
| Persistence | systemd service | Auto-start with Windows |
| Target Cap | 160W per GPU (640W total) | 315W (70% of 450W) |
| Savings | ~60W system-wide | ~135W |
| Throughput Impact | ~5-10% | ~10% |

## References

- [MSI Afterburner Official Site](https://www.msi.com/Lab/Afterburner)
- [NVIDIA Power Management Documentation](https://docs.nvidia.com/deploy/nvidia-smi/)
- [RTX 5090 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
