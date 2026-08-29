"""Phase B: one-time prep of the PLE n-gram table.

The W4A16 checkpoint stores the PLE table as 128 shards:
    model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight
each [2500012, 160] BF16. shard_i covers global rows [i*2500012, (i+1)*2500012).

This script concatenates them (in shard order) into a single
[320001536, 160] BF16 tensor and saves it to a file the model memory-maps
(torch.load(mmap=True)). All TP ranks mmap the same file, so the OS page
cache is shared and physical host RAM ~= 102GB (not 4x).

Usage:
    python phase_b_ple_table_prep.py [out_path]

Env:
    PLE_CKPT   path to the W4A16 checkpoint dir (default below)
Default out_path: /mnt/data/ple_cache/ple_table_qwen4exp.pt
"""

import os
import sys
import time

import torch
from safetensors import safe_open

# Override with PLE_CKPT to point at your local copy of the W4A16 checkpoint.
CKPT = os.environ.get("PLE_CKPT", "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16")
BASE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{}.weight"
N_SHARDS = 128
SHARD_ROWS = 2500012
HEAD_DIM = 160
TOTAL_ROWS = N_SHARDS * SHARD_ROWS  # 320001536


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/ple_cache/ple_table_qwen4exp.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        print(f"already exists: {out_path} ({os.path.getsize(out_path)/1e9:.1f} GB). Skipping.")
        return

    import json
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    print(f"allocating [{TOTAL_ROWS}, {HEAD_DIM}] bf16 "
          f"({TOTAL_ROWS*HEAD_DIM*2/1e9:.1f} GB) ...", flush=True)
    t0 = time.time()
    table = torch.empty(TOTAL_ROWS, HEAD_DIM, dtype=torch.bfloat16)

    for i in range(N_SHARDS):
        key = BASE.format(i)
        fname = wm[key]
        with safe_open(os.path.join(CKPT, fname), framework="pt") as sf:
            shard = sf.get_tensor(key)
        assert shard.shape == (SHARD_ROWS, HEAD_DIM), f"shard {i} shape {shard.shape}"
        assert shard.dtype == torch.bfloat16, f"shard {i} dtype {shard.dtype}"
        table[i*SHARD_ROWS:(i+1)*SHARD_ROWS].copy_(shard)
        if i % 16 == 0 or i == N_SHARDS - 1:
            print(f"  shard {i:3d}/{N_SHARDS-1} "
                  f"({time.time()-t0:6.1f}s)", flush=True)
        del shard

    # Spot-check: shard 0 row 0 and last shard last row must match the source.
    with safe_open(os.path.join(CKPT, wm[BASE.format(0)]), framework="pt") as sf:
        s0 = sf.get_tensor(BASE.format(0))
    assert torch.equal(table[0], s0[0]), "shard0 row0 mismatch"
    with safe_open(os.path.join(CKPT, wm[BASE.format(N_SHARDS-1)]), framework="pt") as sf:
        sl = sf.get_tensor(BASE.format(N_SHARDS-1))
    assert torch.equal(table[TOTAL_ROWS-1], sl[-1]), "last shard last row mismatch"
    print("spot-check OK (shard0 row0, last shard last row).", flush=True)

    print(f"saving to {out_path} ...", flush=True)
    t1 = time.time()
    torch.save({"table": table}, out_path)
    print(f"saved {os.path.getsize(out_path)/1e9:.1f} GB in {time.time()-t1:.1f}s. "
          f"total {time.time()-t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()