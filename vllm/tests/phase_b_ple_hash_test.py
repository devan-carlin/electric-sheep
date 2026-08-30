"""Phase B: validate the PLE n-gram row-index hash against llama.cpp's logic.

llama.cpp (qwen4exp.cpp set_input) computes, per token with context
ctx = [t0, t1, t2] (t0 = current, t1 = prev1, t2 = prev2):

    for n in 2..n_gram:
        mixed = ctx[0]*m[0] ^ ctx[1]*m[1] ^ ... ^ ctx[n-1]*m[n-1]   (uint64)
        base  = (n-2) * per_gram
        for g in 0..per_gram-1:
            h = base + g
            row[h] = mixed % vocab[h] + offset[h]

So heads 0..(per_gram-1) use the 2-gram hash, heads per_gram.. use the 3-gram.
An EOS anywhere in the window resets everything at/after it to EOS.

This script re-implements that in Python and checks it against a hand-computed
example using the REAL checkpoint multipliers/offsets/vocabs. No table needed.
"""

import json

MODEL = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"

# Real values (from GGUF metadata, confirmed == checkpoint).
MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
OFFSETS = [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228,
           140000297, 160000374, 180000455, 200000548, 220000655, 240000802,
           260000955, 280001114, 300001275]
VOCABS = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069,
          20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159,
          20000161, 20000171]
NGRAM = 3
PER_GRAM = 8
N_HEADS = (NGRAM - 1) * PER_GRAM  # 16
EOS = 248044
MASK64 = (1 << 64) - 1


def compute_rows_for_token(ctx):
    """ctx = [t0, t1, t2] (current, prev1, prev2). Returns [N_HEADS] row ids."""
    rows = [0] * N_HEADS
    for n in range(2, NGRAM + 1):
        mixed = (ctx[0] * MULTIPLIERS[0]) & MASK64
        for j in range(1, n):
            mixed = (mixed ^ ((ctx[j] * MULTIPLIERS[j]) & MASK64)) & MASK64
        base = (n - 2) * PER_GRAM
        for g in range(PER_GRAM):
            h = base + g
            rows[h] = (mixed % VOCABS[h]) + OFFSETS[h]
    return rows


def main():
    # Hand example: t0=5, t1=7, t2=3 (no EOS).
    ctx = [5, 7, 3]
    m2 = (5 * MULTIPLIERS[0] ^ 7 * MULTIPLIERS[1]) & MASK64
    m3 = (m2 ^ (3 * MULTIPLIERS[2])) & MASK64
    print(f"2-gram mixed = {m2}")
    print(f"3-gram mixed = {m3}")
    print(f"head0 (2g) row = {m2 % VOCABS[0] + OFFSETS[0]}")
    print(f"head8 (3g) row = {m3 % VOCABS[8] + OFFSETS[8]}")

    rows = compute_rows_for_token(ctx)
    print(f"\nall 16 rows = {rows}")

    # Verify head0 and head8 match the hand computation.
    assert rows[0] == m2 % VOCABS[0] + OFFSETS[0], "head0 mismatch"
    assert rows[8] == m3 % VOCABS[8] + OFFSETS[8], "head8 mismatch"
    # heads 0-7 all use the 2-gram mixed; heads 8-15 use the 3-gram mixed.
    for h in range(8):
        assert rows[h] == m2 % VOCABS[h] + OFFSETS[h], f"head{h} (2g) mismatch"
    for h in range(8, 16):
        assert rows[h] == m3 % VOCABS[h] + OFFSETS[h], f"head{h} (3g) mismatch"
    print("\nPASS: hash matches hand computation for all 16 heads.")

    # EOS reset: if prev1 is EOS, the 3-gram window is [t0, EOS, EOS].
    ctx_eos = [5, EOS, EOS]
    rows_eos = compute_rows_for_token(ctx_eos)
    m2e = (5 * MULTIPLIERS[0] ^ EOS * MULTIPLIERS[1]) & MASK64
    m3e = (m2e ^ (EOS * MULTIPLIERS[2])) & MASK64
    assert rows_eos[0] == m2e % VOCABS[0] + OFFSETS[0]
    assert rows_eos[8] == m3e % VOCABS[8] + OFFSETS[8]
    print("PASS: EOS-window hash consistent.")

    # All rows must be within the table (max offset+vocab < 320,001,536).
    max_row = max(max(rows), max(rows_eos))
    table_rows = 128 * 2500012
    print(f"max row = {max_row}  (< table rows {table_rows}: {max_row < table_rows})")
    assert max_row < table_rows


if __name__ == "__main__":
    main()