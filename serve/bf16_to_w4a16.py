#!/usr/bin/env python3
"""Convert the official BF16 Qwen3.8-Flash-Next checkpoint -> our own W4A16
(compressed-tensors pack-quantized int4, group-128 symmetric), replicating the
VnimanieAI port's recipe but sourced from the clean BF16 weights (single quant,
no FP8 double-quant).

Usage:
  bf16_to_w4a16.py [SRC] [DST]
  SRC default: /mnt/data/models/unsloth-Qwen3.8-Flash-Next-BF16
  DST default: /mnt/data/models/devancarlin-Qwen3.8-Flash-Next-W4A16-BF16src

Key layout difference vs the FP8 source:
  The BF16 source stores routed experts FUSED and without a per-expert index:
      model.language_model.layers.L.mlp.experts.gate_up_proj  [E, 2*I, H]
      model.language_model.layers.L.mlp.experts.down_proj     [E, H, I]
  (E=512 experts, I=640 intermediate, H=2560 hidden). We split these into the
  per-expert layout vLLM + the working checkpoint expect:
      experts.E.gate_proj  = gate_up_proj[E, :I, :]   (gate = first half)
      experts.E.up_proj    = gate_up_proj[E, I:, :]   (up   = second half)
      experts.E.down_proj  = down_proj[E, :, :]
  Split order (gate=first, up=second) is confirmed by vLLM's
  packed_modules_mapping {"gate_up_proj": ["gate_proj","up_proj"]} and cross-
  checked against the FP8 source (cos 0.9996, cross ~0.008).

Recipe (matches the port):
  - int4 g128 symmetric (scale = group_maxabs / 7):
      * main-LM routed experts (split from fused tensors above)
      * main-LM full-attn:  model.language_model.layers.N.self_attn.{q,k,v,o}_proj.weight
  - BF16 copy-through:
      * everything else (mtp.*, visual.*, linear_attn, shared_expert, mlp.gate,
        hyper_connection, embed_tokens, lm_head, norms, biases, PLE small tensors)
  - EXCLUDED (vLLM reads the PLE n-gram table from the mmap'd file, never the ckpt):
      * any key containing ".ngram_embedding."  (the 102GB table, 128 shards)
      * any key containing ".indexer."
  - DROPPED: any "weight_scale_inv" (none present in the BF16 source; kept for safety)

Output layout per quantized weight (matches the port + vLLM XPUwNa16 kernel):
  weight_packed : I32  [N, K//8]   (8 int4/word, nibble j = element k%8, unsigned q+8)
  weight_scale  : BF16 [N, K//128]
  weight_shape  : I32  [N, K]
"""
import glob, json, os, re, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/models/unsloth-Qwen3.8-Flash-Next-BF16"
DST = sys.argv[2] if len(sys.argv) > 2 else "/mnt/data/models/devancarlin-Qwen3.8-Flash-Next-W4A16-BF16src"
SHARD_BYTES = 4 * 1024**3  # ~4GB per output shard

# --- classification helpers ---
GATEUP_RE = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$')
DOWN_RE   = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.down_proj$')
SELFATTN_RE = re.compile(
    r'^model\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj\.weight$'
)
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

def main():
    os.makedirs(DST, exist_ok=True)
    src_shards = sorted(glob.glob(os.path.join(SRC, "*.safetensors")))
    print(f"source: {SRC}", flush=True)
    print(f"dest:   {DST}", flush=True)
    print(f"source shards: {len(src_shards)}", flush=True)

    # pass 1: index source keys
    print("pass 1: index source keys...", flush=True)
    shard_keys = []
    key_shard = {}
    for si, fp in enumerate(src_shards):
        with safe_open(fp, framework="pt") as sf:
            keys = list(sf.keys())
        shard_keys.append(keys)
        for k in keys:
            key_shard[k] = si

    all_keys = []
    for keys in shard_keys:
        all_keys.extend(keys)
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
            if len(open_sf) > 8:
                old = next(iter(open_sf)); del open_sf[old]
            open_sf[si] = safe_open(src_shards[si], framework="pt")
        return open_sf[si].get_tensor(key)

    # output sharding
    cur = {}
    cur_bytes = 0
    total_size = 0
    out_shards = []
    index = {}
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

    def emit_quant(base_key, w):
        """quantize [N,K] and add packed/scale/shape under base_key (which ends in .weight)."""
        nonlocal cur_bytes
        packed, scale, shape = quantize_pack(w)
        cur[base_key.replace(".weight", ".weight_packed")] = packed
        cur[base_key.replace(".weight", ".weight_scale")] = scale
        cur[base_key.replace(".weight", ".weight_shape")] = shape
        cur_bytes += packed.numel()*4 + scale.numel()*2 + shape.numel()*4

    t0 = time.time()
    n_expert = n_selfattn = n_copy = n_skip = 0
    for i, key in enumerate(ordered):
        if is_skip(key):
            n_skip += 1
            continue

        m = GATEUP_RE.match(key)
        if m:
            # fused [E, 2*I, H] -> per-expert gate (first I) + up (second I)
            t = get_tensor(key)
            E, twoI, H = t.shape
            I = twoI // 2
            L = m.group(1)
            for e in range(E):
                gate = t[e, :I, :].contiguous()
                up   = t[e, I:, :].contiguous()
                emit_quant(f"model.language_model.layers.{L}.mlp.experts.{e}.gate_proj.weight", gate)
                emit_quant(f"model.language_model.layers.{L}.mlp.experts.{e}.up_proj.weight", up)
            del t
            n_expert += 2 * E
            if cur_bytes >= SHARD_BYTES:
                flush()
            continue

        m = DOWN_RE.match(key)
        if m:
            # fused [E, H, I] -> per-expert down [H, I]
            t = get_tensor(key)
            E, H, I = t.shape
            L = m.group(1)
            for e in range(E):
                down = t[e, :, :].contiguous()
                emit_quant(f"model.language_model.layers.{L}.mlp.experts.{e}.down_proj.weight", down)
            del t
            n_expert += E
            if cur_bytes >= SHARD_BYTES:
                flush()
            continue

        if SELFATTN_RE.match(key):
            w = get_tensor(key)
            if w.dtype == torch.float8_e4m3fn:  # no-op for BF16 source
                s_inv = get_tensor(key.replace(".weight", ".weight_scale_inv"))
                w = w.to(torch.bfloat16)
            emit_quant(key, w)
            n_selfattn += 1
            if cur_bytes >= SHARD_BYTES:
                flush()
            continue

        # copy-through
        w = get_tensor(key)
        if w.dtype == torch.float8_e4m3fn:  # no-op for BF16 source
            s_inv = get_tensor(key.replace(".weight", ".weight_scale_inv"))
            w = w.to(torch.bfloat16)
        cur[key] = w
        cur_bytes += w.numel() * w.element_size()
        n_copy += 1
        if cur_bytes >= SHARD_BYTES:
            flush()
        if (i+1) % 200 == 0:
            el = time.time()-t0
            print(f"  [{i+1}/{len(ordered)}] {el:.0f}s  expert={n_expert} selfattn={n_selfattn} copy={n_copy} skip={n_skip}", flush=True)

    flush()
    open_sf.clear()

    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": index}, f, indent=2)

    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)
    print(f"  expert proj quantized(int4): {n_expert}  self_attn quantized: {n_selfattn}", flush=True)
    print(f"  copied(BF16): {n_copy}  skipped: {n_skip}", flush=True)
    print(f"  output shards: {len(out_shards)}  total_size: {total_size/1e9:.1f} GB", flush=True)
    print(f"  output dir: {DST}", flush=True)

if __name__ == "__main__":
    main()