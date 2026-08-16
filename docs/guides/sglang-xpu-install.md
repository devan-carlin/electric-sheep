# SGLang XPU Install Guide

**Created**: 2026-08-16
**Hardware**: ai-server (Ubuntu 26.04, 4x Intel Arc Pro B70, Level Zero)
**Upstream reference**: [SGLang XPU Documentation](https://docs.sglang.io/docs/hardware-platforms/xpu)
**Related**: [sglang-xpu-deployment.md](sglang-xpu-deployment.md) (launch, benchmarking, limitations, comparison vs llama.cpp)

---

## Overview

SGLang XPU is optimized for Intel Arc Pro B-Series and Arc B-Series (Battlemage) GPUs. Two install paths:

1. **From source (only supported path)** — no pip wheel exists for XPU.
2. **Docker** — official `xpu.Dockerfile` provided in the repo.

SGLang XPU currently supports BF16 models only (no GGUF/quantized weights). Verified models are small (Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-1.5B, tested on Arc B580).

---

## Path 1: Install From Source

### Prerequisites

- Conda
- PyTorch XPU (see [PyTorch XPU getting started](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) for the underlying XPU dependency setup)
- Intel GPU driver (`xe` on this box)

### Steps

```bash
# 1. Create and activate a conda environment
conda create -n sgl-xpu python=3.12 -y
conda activate sgl-xpu

# 2. Install PyTorch XPU.
# Set the XPU index as primary channel to avoid pulling the larger
# CUDA-enabled build and prevent runtime issues.
pip3 install torch==2.12.0+xpu torchvision==0.27.0+xpu torchaudio==2.11.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu

# 3. xgrammar pulls in CUDA-enabled triton which conflicts with XPU,
# so install it without deps, then add the one real dependency.
pip3 install xgrammar --no-deps
pip3 install apache-tvm-ffi

# 4. Clone SGLang and pin a version
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# 5. Use the XPU-specific build config
cd python
cp pyproject_xpu.toml pyproject.toml

# 6. Install dependencies and build the main package
pip install --upgrade pip setuptools
pip install -v . --extra-index-url https://download.pytorch.org/whl/xpu
```

### Verify

```bash
python3 -c "import sglang; print(sglang.__version__)"
python3 -c "import torch; print(torch.xpu.device_count())"  # expect 4 on this box
```

---

## Path 2: Install Using Docker

The official Dockerfile is at [docker/xpu.Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/xpu.Dockerfile). Replace `<secret>` with a HuggingFace access token.

```bash
# Clone the repo
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the image
docker build -t sglang-xpu:latest -f xpu.Dockerfile .

# Run the container
docker run \
    -it \
    --privileged \
    --ipc=host \
    --network=host \
    --user root \
    --group-add $(getent group video | cut -d: -f3) \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v /dev/shm:/dev/shm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    -e "HF_TOKEN=<secret>" \
    sglang-xpu:latest /bin/bash
```

---

## Smoke Test: Launch the Serving Engine

Minimal launch (single GPU):

```bash
sglang serve \
    --model-path <MODEL_ID_OR_PATH> \
    --trust-remote-code \
    --disable-overlap-schedule \
    --device xpu \
    --host 0.0.0.0
```

Multi-GPU with the Intel-optimized attention backend:

```bash
sglang serve \
    --model-path <MODEL_ID_OR_PATH> \
    --trust-remote-code \
    --disable-overlap-schedule \
    --device xpu \
    --host 0.0.0.0 \
    --tp 2 \
    --attention-backend intel_xpu \
    --page-size 64
```

Notes:
- `--attention-backend intel_xpu` supports `--page-size` of 32, 64, or 128 only.
- `--disable-overlap-schedule` is in the official example; overlap scheduling is not supported on XPU.
- For 4x B70, use `--tp 4`.

Full launch options, XPU graph capture, and P/D disaggregation are covered in [sglang-xpu-deployment.md](sglang-xpu-deployment.md).

---

## Arc B70 (Battlemage) Notes

- The official optimized-model list was verified on Arc B580; Arc Pro B70 is in the same Battlemage family and is the target platform per the doc header, but expect to validate per-model.
- GPU selection for multi-process setups uses `ZE_AFFINITY_MASK` (Level Zero), e.g. `ZE_AFFINITY_MASK=0` for GPU 0, `ZE_AFFINITY_MASK=1,2,3` for the rest.
- `UCX_POSIX_USE_PROC_LINK=n` is required when using the NIXL KV-transfer backend (P/D disaggregation) to avoid UCX shared-memory transport issues.
- BF16 only: a 27B-class model in BF16 needs ~54GB, which fits on 2x B70 with `--tp 2` but not 1x. Quantized (GGUF) models are not supported by SGLang XPU.

---

## Known Limitations (from upstream doc)

| Feature | Status |
|---------|--------|
| Quantized / GGUF models | Not supported |
| Speculative decoding | Not implemented |
| Memory saver (`--enable-memory-saver`) | Not supported |
| Two-batch overlap (`--enable-two-batch-overlap`) | Not supported |
| XPU graph capture | Experimental, opt-in (decode `full`, prefill `tc_piecewise` / `breakable`) |
| P/D disaggregation | Experimental, NIXL backend, tested on Qwen3-0.6B and Qwen2.5-7B-Instruct |
