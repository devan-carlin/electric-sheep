# Ubuntu B70 Utilities

Utility scripts and documentation for the 4× Intel Arc Pro B70 server.

## Scripts

### `01-build-all.sh`

Orchestrates builds for vLLM and/or llama.cpp with interactive prompts or CLI flags.

```bash
./01-build-all.sh                    # Interactive mode (choose what to build)
./01-build-all.sh --all              # Build everything (prereqs → vLLM → llama.cpp)
./01-build-all.sh --vllm             # Build vLLM only
./01-build-all.sh --llama            # Build llama.cpp only
./01-build-all.sh --prereqs          # Install prerequisites only
./01-build-all.sh --status           # Show current build status
```

### `03-convert-int4-autoround.sh`

Downloads a full-precision model and quantizes to INT4 AutoRound format.

```bash
./03-convert-int4-autoround.sh --repo meta-llama/Llama-3.1-8B --output-dir ~/electric-sheep/models/Llama-3.1-8B-int4-AutoRound
./03-convert-int4-autoround.sh --repo meta-llama/Llama-3.1-8B --bits 4 --group-size 128 --sym
```

### `download-model.sh`

Downloads models in vLLM's native HuggingFace safetensors format.

```bash
./download-model.sh qwen3.6-27b                          # Download by alias
./download-model.sh Intel/Qwen3.6-27B-int4-AutoRound     # Download by repo
./download-model.sh --list                               # List known models
./download-model.sh --status                             # Check existing models
```

**About GGUF files:** vLLM serves GGUF directly — no conversion needed. If you have a `.gguf` file, just copy it to `~/electric-sheep/models/` and serve it. GGUF cannot be "converted" to native safetensors because it's already quantized (lossy). The only way to get native format is downloading the original HuggingFace repo.

### `compare-venvs.sh`

Compares two vLLM virtual environments side-by-side.

```bash
./compare-venvs.sh ~/electric-sheep/vllm/.venv ~/other-venv
```

Compares:
- Python version
- Core packages (vllm, torch, triton-xpu, setuptools)
- GPU detection (`torch.xpu.is_available()`)
- `_C` import status
- Environment variables
- Dependency conflicts

### `setup-gpu-power-limits.sh`

Reduces power consumption and heat on 4× Arc B70 GPUs with minimal impact on inference throughput.

```bash
sudo ./setup-gpu-power-limits.sh              # Apply defaults (160W per GPU)
sudo ./setup-gpu-power-limits.sh --watts 140  # Custom power cap
sudo ./setup-gpu-power-limits.sh --status     # Show current state
sudo ./setup-gpu-power-limits.sh --reset      # Remove all limits
sudo ./setup-gpu-power-limits.sh --no-persist # Session-only (no reboot persistence)
```

#### Why Underclocking Works for LLM Inference

LLM inference is **memory-bandwidth bound**, not compute-bound. The bottleneck is streaming weights from VRAM to compute units, not raw FLOPS. Lowering the GPU clock reduces power/heat significantly while token generation speed only drops ~5-10%.

#### Power Cap Trade-offs (per GPU)

| Cap | Power Saved | Throughput Impact | Use Case |
|---|---|---|---|
| 160W (default) | ~9% | ~5-10% slower | Sweet spot for daily use |
| 140W | ~20% | ~15-20% slower | Quiet operation, hot rooms |
| 120W | ~31% | ~25-30% slower | Maximum efficiency, batch jobs |

#### Persistence

By default, the script creates a systemd service (`arc-b70-power-limits.service`) that survives reboots. Use `--no-persist` for temporary testing.

#### How It Works

The script writes to sysfs interfaces exposed by the `xe` driver:

- **Power cap:** `/sys/class/hwmon/hwmon*/power1_cap` (microwatts)
- **Max frequency:** `/sys/class/drm/card*/device/gt_max_freq_mhz`
- **Boost frequency:** `/sys/class/drm/card*/device/gt_boost_freq_mhz`
- **CPU governor:** `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`

## Quick Reference

### Check GPU Status
```bash
# Power and frequency status
sudo ./setup-gpu-power-limits.sh --status

# GPU utilization (if intel_gpu_top installed)
intel_gpu_top

# VRAM usage
intel_gpu_top -d 1 -1
```

### Compare Environments
```bash
# Compare current venv with a backup
./compare-venvs.sh ~/electric-sheep/vllm/.venv ~/electric-sheep/vllm/.venv-backup
```

### Reset Everything
```bash
# Remove GPU power limits
sudo ./setup-gpu-power-limits.sh --reset
```
