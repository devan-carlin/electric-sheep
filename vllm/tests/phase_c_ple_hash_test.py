"""Verify vLLM's PLE n-gram hash matches llama.cpp's exactly.

llama.cpp (qwen4exp.cpp llm_graph_input_ple::set_input):
  ctx[0]=tok; ctx[s]=prev(s) with EOS reset (if ctx[s-1]==eos then ctx[s]=eos)
  mixed2 = ctx[0]*m[0] ^ ctx[1]*m[1]          -> heads 0..per_gram-1
  mixed3 = mixed2 ^ ctx[2]*m[2]               -> heads per_gram..n_heads-1
  row[h] = mixed % vocab[h] + offset[h]
vLLM (_compute_rows) should produce identical rows.
Compare against the captured vLLM rows from /tmp/ple_debug_rank0.pt.
"""
import glob, torch
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
MASK64 = (1 << 64) - 1

def load(name):
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name)
    raise KeyError(name)

def main():
    m = load("model.language_model.layers.1.ple.ple_embedding.layer_multipliers").tolist()
    offs = load("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets").tolist()
    vocs = load("model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes").tolist()
    print(f"multipliers={m}")
    print(f"offsets={offs}")
    print(f"vocab_sizes={vocs}")
    per_gram = 8
    n_heads = 16
    eos = 248044  # from GGUF metadata (ple_eos_token_id)

    d = torch.load("/tmp/ple_debug_rank0.pt", map_location="cpu")
    ids = d["input_ids"].tolist()
    rows_vllm = d["rows"]  # (5, 16)

    # llama.cpp reference: fresh sequence, predecessors = EOS
    prev1 = eos
    prev2 = eos
    rows_ref = []
    for tok in ids:
        c0 = tok
        c1 = prev1
        c2 = eos if prev1 == eos else prev2
        mixed2 = (c0 * m[0] ^ c1 * m[1]) & MASK64
        mixed3 = (mixed2 ^ (c2 * m[2])) & MASK64
        row = []
        for h in range(per_gram):
            row.append((mixed2 % vocs[h]) + offs[h])
        for h in range(per_gram, n_heads):
            row.append((mixed3 % vocs[h]) + offs[h])
        rows_ref.append(row)
        prev2 = prev1
        prev1 = tok

    print("\n=== COMPARISON (vLLM captured vs llama.cpp reference) ===")
    all_match = True
    for i in range(len(ids)):
        match = rows_vllm[i].tolist() == rows_ref[i]
        all_match = all_match and match
        print(f"t{i} id={ids[i]}: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print(f"   vllm: {rows_vllm[i].tolist()}")
            print(f"   ref : {rows_ref[i]}")
    print(f"\nALL MATCH: {all_match}")

if __name__ == "__main__":
    main()