# XPU Ecosystem Notes

**Purpose**: Track findings, limitations, workarounds, and useful tidbits from exploring the Intel XPU / oneAPI ecosystem.

---

## FP8 Support on XPU

### Status (2026-08-10)

| Component | Status | Notes |
|-----------|--------|-------|
| `torch.float8_e4m3fn` dtype | ✅ Defined | Exists in PyTorch XPU |
| `Float8_e4m3fnStorage` | ❌ Missing | Cannot deserialize FP8 safetensors |
| IPEX FP8 runtime | ❌ Dead | IPEX archived Mar 2026, no further development |
| FP8 safetensor loading | ❌ Not supported | Requires storage backend in core PyTorch |
| FP8 compute kernels | ❌ Missing | No GEMM/attention in FP8 for Battlemage |
| Native PyTorch XPU | ✅ Active | Intel upstreamed all XPU support to PyTorch core |

### Key Findings

- **PyTorch XPU defines FP8 dtypes but lacks storage backends** — The dtype exists (`torch.float8_e4m3fn`) but `Float8_e4m3fnStorage` doesn't, so safetensors can't deserialize FP8 weights
- **FP8 tensor creation works** — You can create empty FP8 tensors and convert FP32→FP8 in memory (`torch.empty(dtype=torch.float8_e4m3fn, device="xpu")`)
- **FP8 safetensor deserialization fails** — The storage backend needed to read FP8 weights from disk doesn't exist; this is the actual blocker
- **Workaround**: Load FP16 with `low_gpu_mem_usage=True` (sequential layer processing), then quantize

### IPEX Status (Archived March 30, 2026)

- **IPEX is dead** — Repository archived, read-only since Mar 30, 2026
- **Final release**: 2.8.0 (CPU), 2.8.10+xpu (XPU) — Aug 2025
- **Intel's new strategy**: All XPU/CPU optimization development upstreamed to native PyTorch core
- **Migration path**: Remove `import intel_extension_for_pytorch`, use standard PyTorch directly
- **IPEX FP8 was prototype-only** — Runtime conversion (`fp8_autocast`), no storage backend, only FP8 Linear op supported
- **IPEX never had FP8 storage for XPU** — Zero FP8 storage mentions in any XPU release (2.3.110 through 2.8.10)
- **FP8 was CPU-only in IPEX** — 2.6.0 had "FP8 KV cache" for Xeon 6 AMX, never for XPU

### Adding FP8 Storage Yourself — Difficulty Assessment

**Scope**: Patch PyTorch core C++ to add `Float8_e4m3fnStorage` backend for XPU device type.

**What needs to change**:
1. `torch/csrc/StorageMethods.cpp` — Add FP8 dtype to XPU device case in storage allocation
2. `torch-xpu-ops` — Register FP8 storage allocation (oneAPI `malloc` for raw bytes)
3. Build PyTorch from source with XPU support (~2-4 hour build)

**The good**: FP8 storage is just raw bytes — no special memory format or alignment. Allocation is trivial.

**The bad**:
- Requires PyTorch C++ internals knowledge
- Need to build PyTorch from source with XPU
- Even with storage working, FP8 compute kernels (GEMM, attention) don't exist for Battlemage
- So you'd load FP8 to XPU memory, then cast to FP16 for compute

**Estimated effort**:

| Scope | Time | Feasibility |
|-------|------|-------------|
| Storage allocation only (load FP8 from disk) | 1-2 weeks C++ work | Doable for experienced PyTorch dev |
| Storage + FP8→FP16 cast on XPU | 2-3 weeks | Doable |
| Full FP8 compute (GEMM, attention) | 6+ months | Not feasible solo |

**Better path**: File issue on `pytorch/pytorch` or `intel/torch-xpu-ops` — Intel's upstream team now owns XPU support, and FP8 storage is a small patch for them.

### FP8 Format Reference

- **E4M3** (`float8_e4m3fn`): 4-bit exponent, 3-bit mantissa, finite-only — for activations/weights
- **E5M2** (`float8_e5m2fn`): 5-bit exponent, 2-bit mantissa — for gradients
- **E4M3 FNUZ** (`float8_e4m3fnuz`): No NaN, dedicated +inf bit, dedicated zero — hardware-specific (NVIDIA Hopper+)
- FP8 range: ~4496 max value (E4M3), very limited precision — requires scaling to avoid overflow

### References

- IPEX FP8 docs (archived): https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.40+xpu/tutorials/features/float8.html
- IPEX future announcement: https://github.com/intel/intel-extension-for-pytorch/issues/867
- IPEX releases (archived): https://github.com/intel/intel-extension-for-pytorch/releases
- PyTorch XPU get started: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
- torch-xpu-ops: https://github.com/intel/torch-xpu-ops
- Runebook FP8 storage article: https://runebook.dev/en/docs/pytorch/storage/torch.UntypedStorage.float8_e4m3fnuz

### Search Terms for Future Research

```
pytorch xpu float8_e4m3fnstorage support
pytorch xpu float8 deserialization safetensors
site:github.com/pytorch/pytorch xpu float8 storage
site:github.com/intel/torch-xpu-ops float8
intel pytorch xpu fp8 support roadmap
```

---

## SGLang XPU Support

### Status (2026-08-10)

| Feature | Status | Notes |
|---------|--------|-------|
| XPU device support | ✅ Working | `--device xpu` |
| Intel attention backend | ✅ Working | `--attention-backend intel_xpu` |
| XPU graph capture | ⚠️ Experimental | Opt-in, multiple backends |
| GGUF/quantized models | ❌ Not supported | Critical gap for large models |
| Speculative decoding | ❌ Not implemented | No DSpark support |
| MoE models | ❌ Not tested | Only dense models (3B-8B) verified |
| Prefill-decode disaggregation | ⚠️ Experimental | Via NIXL |

### Verdict

Not worth switching from llama.cpp until:
1. GGUF/quantized model support lands
2. Speculative decoding implemented
3. MoE models tested on XPU

### Reference

- SGLang XPU docs: https://docs.sglang.io/docs/hardware-platforms/xpu
- Deployment guide: `docs/guides/sglang-xpu-deployment.md`

---

## INT4 AutoRound on XPU

### Status (2026-08-10)

| Aspect | Status | Notes |
|--------|--------|-------|
| auto-round library | ✅ Works | Processes layers sequentially with `low_gpu_mem_usage=True` |
| FP8→INT4 conversion | ⚠️ Works with workaround | Must load FP16 first (no FP8 deserialization) |
| Parallel layer processing | ⚠️ Limited | `low_gpu_mem_usage=True` forces sequential |
| CPU threading | ✅ Works | All 24 threads utilized |
| XPU acceleration | ⚠️ Partial | Some ops on XPU, others fall back to CPU |

### Benchmark Settings (Speed Optimized)

| Parameter | Benchmark | Full Quality |
|-----------|-----------|-------------|
| Iterations | 100 | 1000 |
| Samples | 32 | 128 |
| Seq length | 512 | 2048 |
| `low_gpu_mem_usage` | **true** (forced) | true |
| damp_percent | 0.05 | 0.01 |
| search_method | rms | ammp |

### Memory Requirements

| Model | FP8 Source | FP16 Load | INT4 Output |
|-------|-----------|-----------|-------------|
| DeepSeek V4-Flash (284B) | ~400GB | ~568GB | ~155-165GB |

System: 247GB RAM + 128GB VRAM = 375GB total → **FP16 load OOMs without `low_gpu_mem_usage=True`**

---

## Balanced Split Mode (llama.cpp)

### Status (2026-08-10)

| Aspect | Status | Notes |
|--------|--------|-------|
| Patch applied | ✅ Working | `--split-mode balanced` |
| Quantization-aware | ✅ Working | Reads tensor sizes from GGUF metadata |
| MoE-aware | ✅ Working | Expert layers balanced across GPUs |
| DSpark drafter split | ⚠️ Sequential | Drafter doesn't inherit balanced mode |
| VRAM distribution | ✅ Improved | 35% spread → 13% spread |

### VRAM Comparison

| Mode | SYCL0 | SYCL1 | SYCL2 | SYCL3 | Spread |
|------|-------|-------|-------|-------|--------|
| Sequential | 22.3GB | 22.3GB | 22.3GB | **30.2GB** | 35% |
| Balanced | 25.7GB | 22.8GB | 25.0GB | 23.5GB | 13% |

### References

- Patch script: `scripts/common/apply-balanced-split-mode.sh`
- User guide: `docs/guides/balanced-split-mode.md`
- PR docs: `docs/guides/pr-balanced-split-mode.md`

---

## SYCL Limitations

### Known Issues

| Issue | Status | Impact |
|-------|--------|--------|
| `concat` no quantized types | ❌ Unsupported | `ggml_sycl_op_concat: unsupported types: dst: q8_0` |
| Quantized KV cache multi-GPU | ❌ Crashes | Only works with `--split-mode none` (single GPU) |
| 196K context OOM | ❌ Crashes | `GGML_ASSERT(false)` in concat.cpp |
| 128K context f16 KV | ✅ Works | Max practical context window |

### Workarounds

- Use f16 KV cache (default) for multi-GPU setups
- Keep context ≤ 128K for 4× Arc B70
- ubatch-size ≤ 128 for sequential mode (avoids OOM on warmup)

---

## Hardware Notes

### 4× Intel Arc Pro B70

| Spec | Value |
|------|-------|
| VRAM per GPU | 32 GB (31.89 GiB) |
| Total VRAM | 128 GB |
| Architecture | Battlemage G31 |
| PCIe | Gen 4 ×16 |
| Power | ~45-47W idle, varies under load |
| Driver | Level Zero (oneAPI 2026.1.1) |

### System

| Spec | Value |
|------|-------|
| CPU | AMD Threadripper PRO 3945WX (12C/24T) |
| RAM | 247 GB |
| OS | Ubuntu 26.04 LTS |
| Kernel | 7.0.0-29 |
| Compiler | icpx (oneAPI 2026.1.1) |

---

## Useful Commands

```bash
# Check XPU FP8 support
python3 -c "import torch; print('float8_e4m3fn:', hasattr(torch, 'float8_e4m3fn'))"

# Check SYCL devices
sycl-ls | grep level_zero

# Check GPU memory
clinfo | grep -A5 "Global Mem"

# Check oneAPI version
icpx --version

# Check PyTorch XPU status
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'XPU: {torch.xpu.is_available()}')"

# Check available IPEX versions (archived, last: 2.8.0)
pip3 index versions intel-extension-for-pytorch
```

---

## TODO / Future Revisit

- [ ] Check if PyTorch XPU adds `Float8_e4m3fnStorage` support
- [ ] Re-evaluate SGLang when GGUF + speculative decoding land
- [ ] Test INT4 AutoRound quality vs IQ2_XXS on DeepSeek V4-Flash
- [ ] Investigate if IPEX FP8 runtime can help with memory during quantization
- [ ] Check if vLLM XPU support improves (currently limited)
- [ ] Monitor llama.cpp upstream for balanced split mode PR acceptance
