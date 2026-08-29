#!/usr/bin/env python3
"""Convert Qwen3.8-Flash-Next -> our own W4A16 (compressed-tensors
pack-quantized int4, group-128 symmetric), replicating the VnimanieAI port's
recipe. Source may be FP8 (dequant F8->BF16 first) or BF16 (direct).

Usage:
  fp8_to_w4a16.py [SRC] [DST]
  SRC default: /mnt/data/models/unsloth-Qwen3.8-Flash-Next-BF16
  DST default: /mnt/data/models/devancarlin-Qwen3.8-Flash-Next-W4A16-BF16src

Recipe (matches the port):
  - int4 g128 symmetric (scale = group_maxabs / 7):
      * main-LM routed experts:  model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight
      * main-LM full-attn:       model.language_model.layers.N.self_attn.{q,k,v,o}_proj.weight
  - BF16 copy-through (dequant F8->BF16 first if the source tensor is F8):
      * everything else (mtp.*, visual.*, linear_attn, shared_expert, mlp.gate,
        hyper_connection, embed_tokens, lm_head, norms, biases)
  - EXCLUDED (vLLM reads the PLE table from the mmap'd BF16 file, never the ckpt):
      * any key containing ".ngram_embedding."  (the 102GB table)
      * any key containing ".indexer."
  - DROPPED: every "weight_scale_inv" (F8 block scale; int4 path uses weight_scale)

Output layout per quantized weight (matches the port + vLLM XPUwNa16 kernel):
  weight_packed : I32  [N, K//8]   (8 int4/word, nibble j = element k%8, unsigned q+8)
  weight_scale  : BF16 [N, K//128]
  weight_shape  : I32  [N, K]
"""
import glob, json, os, re, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# argv-overridable; defaults to the clean BF16 source (single quant, no double-quant)
SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-BF16"
DST = sys.argv[2] if len(sys.argv) > 2 else "/mnt/data/models/devancarlin-Qwen3.8-Flash-Next-W4A16-BF16src"
SHARD_BYTES = 4 * 1024**3  # ~4GB per output shard

# --- classification helpers ---
EXPERT_RE = re.compile(
    r'^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$'
)
SELFATTN_RE = re.compile(
    r'^model\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj\.weight$'
)
def is_quantize(key):
    return bool(EXPERT_RE.match(key) or SELFATTN_RE.match(key))
def is_skip(key):
    return (".ngram_embedding." in key) or (".indexer." in key) or key.endswith("weight_scale_inv")

# --- int4 g128 symmetric quantize + pack ---
def quantize_pack(w_bf16: torch.Tensor):
    """w_bf16: [N, K] BF16 -> (packed I32 [N,K//8], scale BF16 [N,K//128], shape I32 [N,K])"""
    N, K = w_bf16.shape
    assert K % 128 == 0, f"K={K} not multiple of 128"
    w32 = w_bf16.to(torch.float32)
    ng = K // 128
    wg = w32.reshape(N, ng, 128)
    maxabs = wg.abs().amax(dim=2)                       # [N, ng]
    scale = (maxabs / 7.0).clamp_min(1e-9)              # [N, ng]
    q = torch.round(wg / scale.unsqueeze(2))            # [N, ng, 128]
    q = q.clamp(-7, 7).reshape(N, K).to(torch.int32)    # signed [-7,7]
    qu = q + 8                                          # unsigned [1,15]
    packed = torch.zeros(N, K // 8, dtype=torch.int32)
    for j in range(8):
        packed |= (qu[:, j::8] << (4 * j))
    return packed, scale.to(torch.bfloat16), torch.tensor([N, K], dtype=torch.int32)

# --- F8 -> BF16 dequant using 128x128 block scales ---
def dequant_f8(w_f8: torch.Tensor, s_inv: torch.Tensor) -> torch.Tensor:
    N, K = w_f8.shape
    wf = w_f8.to(torch.float32)
    sf = s_inv.to(torch.float32)
    # s_inv is [N//128, K//128] (or transposed); broadcast to [N, K]
    if sf.shape == (N // 128, K // 128):
        sf_rep = sf.repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    elif sf.shape == (K // 128, N // 128):
        sf_rep = sf.t().repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    else:
        # fallback: nearest reshape
        sf_rep = sf.reshape(N // 128, K // 128).repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    return (wf * sf_rep).to(torch.bfloat16)

def main():
    os.makedirs(DST, exist_ok=True)
    src_shards = sorted(glob.glob(os.path.join(SRC, "*.safetensors")))
    print(f"source shards: {len(src_shards)}", flush=True)

    # We need weight + its weight_scale_inv together for F8 dequant. Source shards
    # may split them, so first build a key -> (shard_idx) map and a per-shard key list.
    print("pass 1: index source keys...", flush=True)
    shard_keys = []          # list of list[key]
    key_shard = {}           # key -> shard_idx
    for si, fp in enumerate(src_shards):
        with safe_open(fp, framework="pt") as sf:
            keys = list(sf.keys())
        shard_keys.append(keys)
        for k in keys:
            key_shard[k] = si

    # For each F8 weight we need its scale_inv; find which shard holds it.
    # Build a fetch cache: we open shards on demand and keep tensors small.
    # Strategy: iterate over ALL source keys in a stable order; for quantize targets
    # we need the paired scale_inv (possibly in another shard) -> fetch it.
    all_keys = []
    for keys in shard_keys:
        all_keys.extend(keys)
    # de-dup preserve order
    seen = set(); ordered = []
    for k in all_keys:
        if k not in seen:
            seen.add(k); ordered.append(k)
    print(f"total unique source keys: {len(ordered)}", flush=True)

    # on-demand tensor fetch with a small LRU of open shards
    open_sf = {}
    def get_tensor(key):
        si = key_shard[key]
        if si not in open_sf:
            # evict oldest if too many open (safe_open has no close(); drop ref)
            if len(open_sf) > 8:
                old = next(iter(open_sf)); del open_sf[old]
            open_sf[si] = safe_open(src_shards[si], framework="pt")
        return open_sf[si].get_tensor(key)

    # output sharding
    cur = {}          # key -> tensor
    cur_bytes = 0
    total_size = 0
    out_shards = []   # list of (filename, {key:tensor})
    index = {}        # key -> filename
    def flush():
        nonlocal cur, cur_bytes, total_size
        if not cur:
            return
        fn = f"model-{len(out_shards)+1:05d}.safetensors"
        path = os.path.join(DST, fn)
        save_file(cur, path)
        for k in cur:
            index[k] = fn
        total_size += cur_bytes
        out_shards.append(fn)
        print(f"  wrote {fn} ({len(cur)} tensors, {cur_bytes/1e9:.2f} GB)", flush=True)
        cur = {}; cur_bytes = 0

    t0 = time.time()
    n_quant = n_copy = n_skip = 0
    for i, key in enumerate(ordered):
        if is_skip(key):
            n_skip += 1
            continue
        w = get_tensor(key)
        if is_quantize(key):
            # experts are F8 in source (need dequant); self_attn is BF16 in source
            if w.dtype == torch.float8_e4m3fn:
                s_inv = get_tensor(key.replace(".weight", ".weight_scale_inv"))
                w = dequant_f8(w, s_inv)
            packed, scale, shape = quantize_pack(w)
            cur[key.replace(".weight", ".weight_packed")] = packed
            cur[key.replace(".weight", ".weight_scale")] = scale
            cur[key.replace(".weight", ".weight_shape")] = shape
            cur_bytes += packed.numel()*4 + scale.numel()*2 + shape.numel()*4
            n_quant += 1
        else:
            # copy-through; dequant F8 (MTP experts) to BF16
            if w.dtype == torch.float8_e4m3fn:
                s_inv = get_tensor(key.replace(".weight", ".weight_scale_inv"))
                w = dequant_f8(w, s_inv)
            cur[key] = w
            cur_bytes += w.numel() * w.element_size()
            n_copy += 1
        if cur_bytes >= SHARD_BYTES:
            flush()
        if (i+1) % 2000 == 0:
            el = time.time()-t0
            print(f"  [{i+1}/{len(ordered)}] {el:.0f}s  quant={n_quant} copy={n_copy} skip={n_skip}", flush=True)

    flush()
    open_sf.clear()

    # write index (total_size tracked incrementally)
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": index}, f, indent=2)

    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)
    print(f"  quantized(int4): {n_quant}  copied(BF16): {n_copy}  skipped: {n_skip}", flush=True)
    print(f"  output shards: {len(out_shards)}  total_size: {total_size/1e9:.1f} GB", flush=True)
    print(f"  output dir: {DST}", flush=True)

if __name__ == "__main__":
    main()