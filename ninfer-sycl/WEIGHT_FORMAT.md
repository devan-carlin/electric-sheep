# Weight format (verified against artifact)

Source of truth: `ninfer/src/artifact/storage_layouts.cpp`, `reader.cpp`, and the
dequant atoms in `ninfer/src/ops/linear/{q4,q5,q6,w8}/*_rowsplit_*.cuh`.
All geometry verified byte-exact against all 9 quantized tensor shapes in
`models/ninfer/qwen3_8_27b/qwen3_8_27b.ninfer`.

## Container (.ninfer v2)

- 16-byte prefix: magic `4e 49 4e 46 45 52 00 02` ("NINFER\0\2") + u64 LE `json_bytes`.
- UTF-8 JSON object directory, padded to 4096-byte boundary.
- `payload_offset = align_up(16 + json_bytes, 4096)`.
- Object `offset` fields are relative to `payload_offset`.
- Reader mmaps the file (POSIX `open`+`mmap`), parses JSON, exposes `payload(name)`.
- `read_direct` requires 4096-byte alignment (offset, size, dest).

## Tensor formats present in the artifact

| format        | tensors | group | base B/grp | high B/grp | scale |
|---------------|---------|-------|-----------|-----------|-------|
| Q4G64_F16S    | 183     | 64    | 32        | 0         | f16   |
| Q5G64_F16S    | 246     | 64    | 32        | 8         | f16   |
| Q6G64_F16S    | 1       | 64    | 32        | 16        | f16   |
| W8G32_F16S    | 9       | 32    | 32        | 0         | f16   |
| BF16          | 582     | -     | -         | -         | -     |
| FP32          | 96      | -     | -         | -         | -     |
| I32           | 1       | -     | -         | -         | -     |

## row-split-k128-v1 geometry (3-plane, NOT interleaved)

For a tensor of shape `[rows, cols]`:

```
padded_columns   = align_up(cols, 128)
groups_per_row   = padded_columns / group_size
groups           = rows * groups_per_row
low_plane_bytes  = groups * base_bytes_per_group        # 32 B for all
high_plane_bytes = groups * high_bytes_per_group        # 0/8/16
high_plane_offset= align_up(low_plane_bytes, 256)
scale_plane_offset = high_plane_offset + align_up(high_plane_bytes, 256)
scale_plane_bytes  = groups * 2                          # f16 scales
encoded_bytes      = scale_plane_offset + scale_plane_bytes
```

Layout in memory: `[low codes plane][high bits plane][f16 scale plane]`,
each plane 256-byte aligned.

- Low plane: 4-bit codes, 2 per byte, 64 values = 32 bytes/group.
- High plane: extra bits (Q5: 1 bit/value = 8 B/grp; Q6: 2 bits/value = 16 B/grp).
- Scale plane: one f16 per group.

## Dequant math (per format)

All produce `dequant = signed_code * scale` where `scale` is the f16 group scale.

### Q4 (bias 8, range -8..7)
```
q0 = (code & 0x0f) ^ 8) - 8
q1 = ((code >> 4) ^ 8) - 8
```

### Q5 (bias 16, range -16..15)
```
u  = lo4 | (hi1 << 4)          # 5-bit unsigned
s  = (u ^ 16) - 16
```
hi1 is the 5th bit, stored in the high plane (1 bit per value, 8 B/group).

### Q6 (bias 32, range -32..31)
```
u  = lo4 | (hi2 << 4)          # 6-bit unsigned
s  = (u ^ 32) - 32
```
hi2 is the top 2 bits, stored in the high plane (2 bits per value, 16 B/group).
High-plane bit order (from `q6_rowsplit_storage.cuh`): for value index `i` in a
group, `high_byte = high[i >> 1]`, `shift = (i & 1) * 4`,
`hi2 = (high_byte >> shift) & 3`.

### W8 (group 32)
```
s = (int8)byte                  # direct signed
```

## XMX int8 GEMM strategy (design decision)

Intel XMX is int8-only (no 4/5/6-bit, no FP8). Decode is bandwidth-bound.

Two viable paths:
1. **Prefill (compute-bound):** dequant codes to int8, quantize activations to
   int8, run int8 x int8 -> int32 GEMM on XMX, then apply per-group f16 weight
   scale x activation scale to the int32 accumulator -> bf16. Per-group scale
   varies along K, so accumulate per-group partial int32 sums then rescale.
2. **Decode (bandwidth-bound):** custom SYCL kernel that reads packed weights
   directly, dequantizes in-register, does the dot product (GEMV). Mirrors the
   reference SIMT GEMV. Minimizes weight traffic (the bottleneck).

Phase 1 (MTP0 baseline) targets correct output at ~30 tok/s. Start with the
decode GEMV path for all linear layers; add the int8 XMX GEMM path for prefill
once decode is correct.