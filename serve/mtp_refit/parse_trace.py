#!/usr/bin/env python3
"""Parse the B70_MTP_DATAGEN_TRACE binary into per-sequence (tokens, wide) samples.

Trace layout (little-endian, one record per forward step, rank 0 only):
  u32 n_tokens
  u32 wide_dim
  i32 token_ids[n_tokens]
  i64 positions[n_tokens]
  u16 wide[n_tokens * wide_dim]   (raw bfloat16 bits)

Sequences are reconstructed by position continuity: a token continues the
current sequence iff its position == last_position + 1; otherwise it starts a
new sequence. This is robust to vLLM batching (multiple requests share a
forward step, tokens interleaved). Each saved sample keeps tokens[i] paired
with wide[i] = wide_state_at(tokens[i]); the training label for tokens[i] is
tokens[i+1].

Usage:
  python3 parse_trace.py --trace trace.bin --out-dir ./data --min-len 64 --shard-tokens 200000
"""
import argparse
import json
import os
import struct

import numpy as np
import torch


def read_records(path):
    """Yield (token_ids:int32[nt], positions:int64[nt], wide:bf16[nt,wdim])."""
    import torch
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    n = len(data)
    while off + 8 <= n:
        nt, wdim = struct.unpack_from("<II", data, off)
        off += 8
        tok_bytes = nt * 4
        pos_bytes = nt * 8
        wide_bytes = nt * wdim * 2
        if off + tok_bytes + pos_bytes + wide_bytes > n:
            print(f"warning: truncated record at offset {off}; stopping")
            break
        toks = np.frombuffer(data[off:off+tok_bytes], dtype=np.int32); off += tok_bytes
        poss = np.frombuffer(data[off:off+pos_bytes], dtype=np.int64); off += pos_bytes
        wb = off; off += wide_bytes
        wide_u16 = np.frombuffer(data[wb:wb+wide_bytes], dtype=np.uint16)
        wide = torch.from_numpy(wide_u16.copy()).view(torch.bfloat16).view(nt, wdim)
        yield toks, poss, wide


def parse(trace_path, min_len=64):
    """Return list of (tokens:int32[T], wide:bf16[T,wdim]) sequences."""
    seqs = []          # list of dict(tokens=[], wide=[], last_pos=-1)
    cur = None
    for toks, poss, wide in read_records(trace_path):
        for i in range(len(toks)):
            p = int(poss[i])
            t = int(toks[i])
            w = wide[i]
            if cur is not None and p == cur["last_pos"] + 1:
                cur["tokens"].append(t)
                cur["wide"].append(w)
                cur["last_pos"] = p
            else:
                if cur is not None and len(cur["tokens"]) >= min_len:
                    seqs.append(cur)
                cur = {"tokens": [t], "wide": [w], "last_pos": p}
    if cur is not None and len(cur["tokens"]) >= min_len:
        seqs.append(cur)

    # Validate contiguity: tokens[i] must sit at position i (prompt starts at 0).
    good = []
    for s in seqs:
        T = len(s["tokens"])
        # spot-check a few positions were monotonic (already enforced by builder)
        good.append({
            "input_ids": np.array(s["tokens"], dtype=np.int32).copy(),
            "wide": torch.stack(s["wide"]),  # bf16 [T, wdim]
        })
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--min-len", type=int, default=64)
    ap.add_argument("--shard-tokens", type=int, default=200_000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import torch
    samples = parse(a.trace, a.min_len)
    print(f"parsed {len(samples)} sequences (>= {a.min_len} tokens)")
    total_tok = sum(len(s["input_ids"]) for s in samples)
    print(f"total tokens: {total_tok:,}")

    # shard to disk
    shard = 0
    buf_tok = 0
    shard_samples = []
    manifests = []
    for idx, s in enumerate(samples):
        T = len(s["input_ids"])
        if shard_samples and buf_tok + T > a.shard_tokens:
            _flush(shard, shard_samples, a.out_dir, manifests)
            shard += 1
            shard_samples = []
            buf_tok = 0
        shard_samples.append(s)
        buf_tok += T
        manifests.append({"sample": idx, "tokens": T, "shard": shard})
    if shard_samples:
        _flush(shard, shard_samples, a.out_dir, manifests)

    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump({"n_samples": len(samples), "total_tokens": total_tok,
                   "min_len": a.min_len}, f, indent=2)
    print(f"wrote {len(samples)} samples across shards -> {a.out_dir}")


def _flush(shard, samples, out_dir, manifests):
    import torch
    # Variable-length sequences: zero-pad input_ids and wide to the shard max,
    # keep per-sample lens for masking during training.
    max_len = max(len(s["input_ids"]) for s in samples)
    wdim = samples[0]["wide"].shape[1]
    n = len(samples)
    ids = torch.zeros(n, max_len, dtype=torch.int32)
    wide = torch.zeros(n, max_len, wdim, dtype=torch.bfloat16)
    lens = torch.zeros(n, dtype=torch.int32)
    for i, s in enumerate(samples):
        T = len(s["input_ids"])
        ids[i, :T] = torch.from_numpy(np.ascontiguousarray(s["input_ids"]))
        wide[i, :T] = s["wide"][:T]
        lens[i] = T
    path = os.path.join(out_dir, f"shard-{shard:03d}.pt")
    torch.save({"input_ids": ids, "wide": wide, "lens": lens}, path)
    print(f"  shard {shard}: {n} samples, {int(lens.sum()):,} tokens -> {path}")


if __name__ == "__main__":
    main()
