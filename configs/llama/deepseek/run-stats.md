# DeepSeek V4-Flash — Run Statistics

**Model:** `unsloth/DeepSeek-V4-Flash-0731-GGUF`  
**Quantization:** UD-IQ3_XXS (~3-bit, very aggressive)  
**Shards:** 4 GGUF files (98 GB total)  
**Backend:** llama.cpp SYCL, layer split across 4× Intel Arc Pro B70  
**Date:** 2026-08-08

---

## Configuration

| Parameter | Value |
|---|---|
| Context window | 65,536 tokens (64K) with DSpark / 98,304 (96K) without |
| Batch size | 4,096 |
| CPU threads | 6 |
| CPU batch threads | 6 |
| GPU split mode | layer |
| Flash attention | auto |
| KV cache dtype | f16 (default; q8_0 unsupported on SYCL) |
| Load mode | mmap |
| DSpark | Enabled (draft-dspark, n_max=3) |
| Port | 8080 |

---

## Performance Benchmarks

### Prompt Processing (DSpark, 35K prompt)

| Tokens Processed | Time | Throughput |
|---|---|---|
| 4,096 | 42.0 s | **97.5 t/s** |
| 8,192 | 87.5 s | **93.6 t/s** |
| 16,384 | 192.5 s | **85.1 t/s** |
| 24,576 | 312.9 s | **78.6 t/s** |
| 32,768 | 451.1 s | **72.6 t/s** |
| 35,226 (total) | 498.5 s | **70.7 t/s** |

Prompt throughput degrades as context fills (more KV cache lookups per token).

### Token Generation (DSpark)

| Metric | Value |
|---|---|
| **Avg generation speed** | **15.75 tokens/s** |
| **3-second rolling avg** | **~16.3 tokens/s** |
| Steady-state | ~16.3 t/s (very stable) |

### Token Generation (No DSpark, for comparison)

| Metric | Value |
|---|---|
| **Avg generation speed** | **12.80 tokens/s** |
| **3-second rolling avg** | **13.04 tokens/s** |
| Steady-state | ~13.2 tokens/s |

### DSpark Speedup

| Metric | No DSpark | With DSpark | Speedup |
|---|---|---|---|
| Generation speed | 12.80 t/s | 15.75 t/s | **1.23x** |
| 3s rolling avg | 13.04 t/s | 16.3 t/s | **1.25x** |

### Model Load Time

| Metric | Value |
|---|---|
| Cold load (with DSpark) | ~2 min 22 sec |
| Model size | 98 GB (4 shards) + 11 GB drafter |

---

## VRAM Usage (64K Context + DSpark)

| GPU | Used | Total | % Used | Headroom |
|---|---|---|---|---|
| GPU 0 | 29.3 GiB | 31.9 GiB | 92% | 2.6 GiB |
| **GPU 1** | **31.6 GiB** | **31.9 GiB** | **99%** | **0.3 GiB** |
| GPU 2 | 27.7 GiB | 31.9 GiB | 87% | 4.2 GiB |
| GPU 3 | 25.3 GiB | 31.9 GiB | 79% | 6.6 GiB |

**GPU 1 is the bottleneck** (99% utilization). DSpark drafter model landed on GPU 1 after main model layers filled it. Very tight — monitor for OOM on long conversations.

### VRAM Usage (96K Context, No DSpark — for comparison)

| GPU | Used | Total | % Used | Headroom |
|---|---|---|---|---|
| GPU 0 | 26.2 GiB | 31.9 GiB | 82% | 5.7 GiB |
| GPU 1 | 24.9 GiB | 31.9 GiB | 78% | 7.0 GiB |
| **GPU 2** | **28.2 GiB** | **31.9 GiB** | **88%** | **3.7 GiB** |
| GPU 3 | 24.7 GiB | 31.9 GiB | 78% | 7.2 GiB |

**GPU 2 is the bottleneck** (88% utilization). Uneven distribution is expected with layer split — some layers are larger than others.

### Power & Thermals (under load)

| GPU | Frequency | Power | Temp |
|---|---|---|---|
| GPU 0 | 2800 MHz | 84 W | 50°C |
| GPU 1 | 2800 MHz | 87 W | 50°C |
| GPU 2 | 2800 MHz | 84 W | 51°C |
| GPU 3 | 2800 MHz | 81 W | 50°C |

**Total power draw:** ~336 W across all 4 GPUs

---

## Context Window Testing

| Context | Worst GPU VRAM | Status | Notes |
|---|---|---|---|
| 32K | 27.3 GiB (86%) | ✅ Stable | Baseline |
| 96K | 28.2 GiB GPU 2 (88%) | ✅ Stable | No DSpark, comfortable headroom |
| 64K + DSpark | 31.6 GiB GPU 1 (99%) | ⚠️ Tight | GPU 1 is bottleneck, 0.3 GiB headroom |
| 128K | — | ⚠️ Not tested | Would likely hit 93-95% |

**Recommendation:** 96K without DSpark is safest. With DSpark, 64K works but GPU 1 is at 99% — monitor for OOM on long conversations.

## DSpark (Speculative Decoding)

DSpark is DeepSeek's native speculative decoding algorithm (superior to MTP). It requires a drafter model file (~11 GB Q8_0).

| Feature | Standard | With DSpark |
|---|---|---|
| **Decoding speed** | 12.80 t/s | 15.75 t/s (1.23x) |
| **3s rolling avg** | 13.04 t/s | 16.3 t/s (1.25x) |
| **VRAM overhead** | — | +10 GB |
| **Context (safe)** | 96K | 64K |
| **Worst GPU** | GPU 2 (88%) | GPU 1 (99%) |
| **Drafter file** | — | `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` |

### DSpark Flags

```bash
-md dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
--spec-type draft-dspark \
--spec-draft-n-max 3
```

**Note:** Do not use `-ngld 99` — it causes DSpark init failure (`dflash requires ctx_other`). Let llama.cpp auto-assign drafter layers.

### Download DSpark Drafter

```bash
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
    --include '*dspark-DeepSeek-V4-Flash-0731-Q8_0*' \
    --local-dir ~/electric-sheep/models/unsloth-DeepSeek-V4-Flash-0731-GGUF/
```

### Recommended Inference Params (from Unsloth)

| Parameter | Value | Notes |
|---|---|---|
| `temperature` | 1.0 | Default |
| `top-p` | 1.0 | General tasks |
| `top-p` | 0.95 | Agentic/tool-calling tasks |
| `min-p` | 0.01 | Token filtering |
| `reasoning` | on | High effort (default) |

---

## Known Limitations

- **q8_0 KV cache unsupported** — SYCL backend lacks `concat` kernel for q8_0 type. Use f16 (default) instead.
- **Layer split unevenness** — GPU 2 consistently uses ~2-3 GiB more than others due to larger layers landing on it.
- **Flash attention** — works with `--flash-attn auto` but MKL FA not available (SYCL-only path).

---

## Commands

### Start server
```bash
cd ~/electric-sheep/configs/llama/deepseek
./start-deepseek-v4-flash.sh
```

### Check VRAM (nvtop)
```bash
nvtop
```

### Quick VRAM snapshot (PyTorch XPU)
```bash
source ~/electric-sheep/vllm/.venv/bin/activate
python3 -c "
import torch
for i in range(torch.xpu.device_count()):
    alloc = torch.xpu.memory_allocated(i) / 1e9
    reserved = torch.xpu.memory_reserved(i) / 1e9
    props = torch.xpu.get_device_properties(i)
    total = props.total_memory / 1e9
    print(f'GPU {i}: {alloc:.1f}GB alloc / {reserved:.1f}GB reserved / {total:.1f}GB total ({reserved/total*100:.0f}%)')
"
```

### Test generation
```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 100
    }'
```

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-08 | Initial run: 96K context, 12.8 t/s generation, 58 t/s prompt processing |
