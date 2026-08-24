# NInfer → Intel Arc (Battlemage) Port — Notes

Running log: decisions, measured numbers, dead ends, open questions.

## Environment (measured 2026-08-23)

| Component | Version |
|---|---|
| OS | Ubuntu (server) |
| GPU | 4x Intel Arc Pro B70 (Battlemage, 32 GB each) — prompt says B770; same Xe2 |
| CPU | AMD Ryzen Threadripper PRO 3945WX (24 cores) |
| RAM | 247 GB |
| `icpx` | 2026.1.1 (2026.1.1.20260724) |
| oneDNN | 3.11.4 (`/opt/intel/oneapi/dnnl/2026.0`) |
| Level Zero driver | 2.20.2.0 (26.22.38646.7) |
| Level Zero loader | `/usr/lib/x86_64-linux-gnu/libze_loader.so.1.28.6` |
| SYCL runtime | `libsycl.so.9` (oneAPI 2026.1) |

oneDNN GPU backend is built into `libdnnl.so` (links `libsycl.so.9`); driven via
the SYCL interop API (`dnnl_sycl.hpp`).

## Target

- Artifact: `models/ninfer/qwen3_8_27b/qwen3_8_27b.ninfer` (SHA-verified vs published manifest).
- GPU 0.
- Validation: reference server `192.168.68.54:8080` (live NInfer) + local IPEX.

## Weight format decision (locked)

- Artifact is mixed groupwise-int: MLP gate/up `Q4G64` (5.71 GiB), MLP down +
  GDN/attention proj `Q5G64` (~7.0 GiB), embedding/head/MTP `W8G32` (~2.7 GiB),
  norms/conv `BF16` (0.06 GiB). Packed total **16.95 GiB**.
- XMX is INT8-only. **Keep int4 storage, dequant to int8 in-kernel.**
- True int8 = 28.18 GiB → does NOT fit with 256k KV (8.25 GiB) in 32 GB.
- Packed int4 + KV@256k = 25.2 GiB → fits, 6.8 GiB headroom.
- Decode ceiling @512 GB/s: ~30 tok/s MTP0 (packed) vs ~18 tok/s (true int8).

## Model shapes (from artifact, [N,K] = out rows x in cols)

- hidden 5120, 64 layers (16 full-attn at 3,7,...,63; 48 GDN)
- MLP `gate_up [34816,5120]`, `down [5120,17408]`
- full-attn `query_key [7168,5120]`, `gate_value [7168,5120]`, `output [5120,6144]`
- GDN `query_key [4096,5120]`, `value_z [12288,5120]`, `output [5120,6144]`
- MTP `input_projection [5120,10240]`, `query_key_gate_value [14336,5120]`
- vocab 248320, embedding/output_head `[248320,5120]`

## Week-0 microbenchmarks (measured 2026-08-23)

All three gates PASS.

### 1. oneDNN GEMM TOPS (SYCL interop, `bench1_gemm`)

oneDNN matmul convention: `dst[M,N] = src[M,K] @ weights[K,N]` (weights are
[K,N], not [N,K]).

Integer (s8 x s8 -> s32) — the XMX path:

| shape | M=8192 (prefill) | M=1 (decode) |
|---|---|---|
| mlp/gate_up [34816,5120] | 10.15 ms, **287.7 TOPS** | 0.333 ms, 1.07 TOPS |
| mlp/down [5120,17408] | 7.00 ms, **208.6 TOPS** | 0.152 ms, 1.18 TOPS |
| attn/query_key [7168,5120] | 1.95 ms, **307.7 TOPS** | 0.049 ms, 1.49 TOPS |
| attn/output [5120,6144] | 1.68 ms, **307.5 TOPS** | 0.030 ms, 2.10 TOPS |
| mtp/input_proj [5120,10240] | 3.06 ms, **280.8 TOPS** | 0.085 ms, 1.24 TOPS |

bf16 reference (ALU path): 116.7 / 136.2 / 135.3 / 135.8 / 134.4 TOPS.

- **XMX confirmed**: int8 prefill is ~2.2x bf16 (208-308 vs 116-136 TOPS).
- Decode (M=1) is latency-bound (~0.03-0.33 ms/layer-proj), not TOPS-bound —
  consistent with a bandwidth-bound decode.

### 2. Level Zero graph capture (`probe_graph`)

- Old clone API `zeCommandListCreateCloneExp`: **NOT supported** (0x78000003).
- Driver advertises `ZE_experimental_record_replay_graph` — a newer record/replay
  graph API. Local headers (1.28.6) predate it; real signatures from
  `intel/compute-runtime` `level_zero/include/level_zero/driver_experimental/zex_graph.h`.
  Functions use the `Exp` suffix, resolved via `zeDriverGetExtensionFunctionAddress`.
- Full pipeline verified end-to-end (all SUCCESS, copies actually executed):
  `zeGraphCreateExp` -> `zeCommandListBeginCaptureIntoGraphExp` (immediate CL)
  -> append N copies -> `zeCommandListEndGraphCaptureExp`
  -> `zeCommandListInstantiateGraphExp` -> `zeCommandListAppendGraphExp`.
- **Gotcha**: the executing command list must match the capture list's mode
  (immediate). A non-immediate append "succeeds" silently but the graph never runs.
- Host overhead (64 ops/step, K=2000): fresh non-imm CL + 64 appends ~60 us/step
  vs fresh immediate CL + 1 graph append ~37 us/step. The 37 us includes
  per-step immediate-CL create/destroy; a persistent-CL pattern should be <10 us.
  **Graphs are a clear win for decode.**
- Known issue: probe segfaults in teardown (after `zeMemFree`, before queue
  destroy) — a driver cleanup quirk, after all measurements complete.

### 3. Raw memory bandwidth (`bench3_bandwidth`, 1 GiB)

| op | time | bandwidth |
|---|---|---|
| read-only | 0.678 ms | **1582.9 GB/s** |
| write-only | 1.837 ms | **584.5 GB/s** |
| copy (r+w) | 3.956 ms | **542.8 GB/s** |

Well above the 450+ GB/s target.

## Dead ends / notes

- `ocloc` is the legacy OpenCL tool — no SPIR-V output. Graph bench uses
  memory-copy commands as the work proxy (measures submission overhead, which
  is what graphs eliminate).
- First oneDNN probe (plain `engine::gpu`) rejected even int8×int8 — that path
  needs a SYCL context. The real bench uses the SYCL interop path.
- oneDNN matmul weights are [K,N]; passing [N,K] makes every shape fail with
  "dimension dst:1 is inconsistent with weights:1".
- SYCL `h.fill` on large buffers (>=~100 MB) throws `UR_RESULT_ERROR_DEVICE_LOST`
  on this driver. Skip fills in the GEMM bench (data content is irrelevant for
  TOPS; the warmup loop handles JIT).
- `zeMemAllocShared` requires a valid `ze_host_mem_alloc_desc_t` (not nullptr)
  or it segfaults.
- `zeDriverGetExtensionProperties` must be called once with the full array
  (pass `&n` + `ext.data()`), not per-index with a stale count (buffer overflow).

## Open questions

- ~~Does oneDNN take int4 weights directly?~~ **No.** All 4-bit matmul variants
  (s8xs4, s8xu4, bf16xs4, bf16xu4, bf16xf4_e2m1) return NOT SUPPORTED on the GPU
  engine. XMX is INT8-only. We dequant int4 -> int8 in-kernel and feed oneDNN
  int8 (or write our own XMX GEMM).
- Page size for paged KV given the small L2 (4K-token pages may hurt).