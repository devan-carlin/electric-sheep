You are a senior GPU systems engineer. Your task is to build a high-performance,
from-scratch inference engine for the **Intel Arc B770 (Battlemage, 32 GB)** on this
**Ubuntu** server, using **SYCL 2020 / oneAPI (Level Zero backend)**. The target is to
replicate the architecture of **NInfer** — a proven, high-performance single-GPU
inference engine — on Intel hardware.

## 1. Get the reference material

Clone both repos (they are private; you have access):

```bash
git clone https://github.com/devan-carlin/ninfer.git
git clone https://github.com/devan-carlin/ninfer-porting-kit.git
```

- `ninfer/` — the reference engine (C++/CUDA, RTX 5090). This is your **architecture
  reference only** — you are NOT porting the CUDA kernels, you are re-implementing the
  same design in SYCL.
- `ninfer-porting-kit/` — the porting docs + the hardware-agnostic `bench/` harness.

**Read this first, in full:** `ninfer-porting-kit/docs/intel-arc-port-handoff.md`.
It contains the hardware reality check, stack selection, feature-by-feature mapping,
the adaptive graph-allowance design, the risk-ordered challenge list, the build order,
and the definition of done. Also skim `ninfer/README.md` and `ninfer/docs/performance.md`
for the reference architecture and the 5090 performance numbers (calibration only).

## 2. Set your expectations (do not skip this)

The B770 has ~512 GB/s memory bandwidth vs. the 5090's ~1,792 GB/s, and its XMX matrix
units are **INT8-only** (no 4-bit, no FP8). Decode is bandwidth-bound. **Target ~80–110
tok/s MTP3 decode at C=1, not ~200.** The goal is to build the same architecture and
extract every last drop of 512 GB/s — not to match the 5090. 256k context fits (INT8
weights ~14 GB + INT8 KV ~9.5 GB ≈ 24 GB of 32 GB).

## 3. CRITICAL: do the Week-0 microbenchmarks BEFORE writing any engine code

The entire architecture hinges on three numbers. **Do not write a single line of engine
code until you have measured these and reported them back to me.** Write small standalone
SYCL/oneDNN programs for each:

1. **oneDNN `u8s8` GEMM on XMX** — at the actual prefill shapes (e.g. the Qwen3.8-27B
   MLP/attention projection dims). Measure achieved TOPS. This tells us whether XMX is
   actually being used or if we're falling back to ALU math (a 5–10× difference).
2. **Level Zero graph capture round-trip** — capture a small command list into a
   `zext_graph`, replay it, measure the launch overhead vs. a non-graph command list.
   This tells us whether the CUDA-graphs analog (the single biggest decode win) is even
   available on this driver.
3. **Raw memory bandwidth** — a simple read/write kernel. Should hit ~450+ GB/s.

**Report these three numbers to me with your oneAPI/SYCL/oneDNN/Level Zero versions
(`sycl-ls`, `icpx --version`, oneDNN version) before proceeding.** If oneDNN XMX GEMM is
weak or graph capture is unsupported, stop and report — the architecture needs
rethinking before we invest in engine code.

## 4. Build order (after the microbenchmarks are green)

- **Phase 1 — MTP0 baseline:** weight loading, INT8 GEMM prefill, custom attention,
  paged KV (INT8 group-64), greedy decode. **Target: correct output at ~30 tok/s.**
  Validate token-by-token against **IPEX (PyTorch XPU)** running the same Qwen3.8-27B
  checkpoint on a fixed seed corpus — use IPEX purely as a numerical reference, then
  discard it.
- **Phase 2 — graphs + MTP:** Level Zero graph capture on the decode path, MTP
  draft/verify round, acceptance tracking. **Target: ~90 tok/s.**
- **Phase 3 — the NInfer polish:** prefix reuse, adaptive graph allowance (re-fit the
  constants on the B770 — see §5 of the handoff doc; the 5090 constants do NOT transfer),
  and wire up the `bench/` harness from the porting-kit repo (adjust its `EXE`/`MODEL`
  constants to your build layout).

## 5. Key design constraints (from the reference architecture)

- **From-scratch, closed-set, no PyTorch in the hot path.** Hand-tuned SYCL kernels for
  the exact Qwen3.8-27B topology (GDN linear attention, GQA, SwiGLU, MTP draft layers).
- **oneDNN for GEMM** (the only reliable XMX path); custom SYCL kernels for attention,
  KV quant/dequant, sampling, and MTP round logic.
- **Deterministic memory layout at startup** — compute the full persistent + workspace +
  graph-allowance reservation in one pass before serving.
- **Paged KV with INT8 group-64** — note the B770's small L2; 4K-token pages may hurt
  more than on the 5090, so be prepared to tune page size / gather strategy.
- **GPU-agnostic serving layer** — port `ninfer/src/serve/` (OpenAI-compatible HTTP)
  largely unchanged.

## 6. How to work and report

- Work in small, verifiable increments. After each phase, report: what you built, the
  measured throughput, and how it compares to the target.
- When you hit a wall (driver limitation, oneDNN gap, kernel that won't hit bandwidth),
  **stop and report the specific blocker with evidence** rather than guessing.
- Keep a running `PORTING_NOTES.md` in your working directory: decisions made, measured
  numbers, dead ends, and open questions. I will read it.
- Do not commit model artifacts (multi-GB) or build outputs to git.

## 7. Definition of done

1. A `serve` binary on this Ubuntu + Arc B770 box serving the OpenAI-compatible HTTP API.
2. Correct output, validated token-by-token against IPEX on a fixed seed corpus.
3. **~80–110 tok/s MTP3 decode** at C=1.
4. **256k context** with paged INT8 KV fitting in 32 GB.
5. Level Zero graph capture active on the decode path (verify via a
   `graphs=X MiB/Y MiB` startup log line, analogous to NInfer's).
6. The `bench/` harness running against it and producing a dashboard.

**Start now with §3 — the three microbenchmarks. Report the numbers before writing
engine code.**