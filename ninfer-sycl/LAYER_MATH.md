# Layer math (Qwen3.8-27B, verified from reference)

Source: `ninfer/src/targets/qwen3_6/impl/runtime/text_context_impl.h`
(`attn_mix`, `gdn_mix`), `ninfer/src/ops/kernel/gdn_gating.cuh`,
`ninfer/src/ops/linear_attention/gated_delta_net/recurrent.cuh`.

## Topology
- 64 layers. Full-attention at layers 3,7,...,63 (16 layers, `(layer+1)%4==0`).
- GDN (gated delta net) at the other 48 layers.
- hidden 5120, intermediate 17408, vocab 248320.
- rms_epsilon 1e-6, rope_theta 1e7, rotary_dim 64.
- kAttentionScale 0.0625, kGdnScale 0.08838834764831845.

## Per-layer residual pattern
Both layer types: `x = x + block(rmsnorm(x, input_norm))`, then
`x = x + mlp(rmsnorm(x, post_attention_norm))`.

## Full-attention layer (16)
1. `h = rmsnorm(x, input_norm)`
2. `q, gate, k, v = attention_proj(h)`
   - `query_key [7168, 5120]` = q (6144) + k (1024)
   - `gate_value [7168, 5120]` = gate (6144) + v (1024)
   - q/gate: 24 heads x 256; k/v: 4 heads x 256
3. `qn = rmsnorm(q, query_norm)` (per head, dim 256)
   `kn = rmsnorm(k, key_norm)` (per head, dim 256)
4. `rope(qn, kn)` on first 64 dims of each head, theta 1e7
5. `a = gqa_attention(qn, kn, v, kv_cache)` — 24 q / 4 kv heads, scale 0.0625
6. `a = sigmoid(gate) * a`
7. `x = x + output_proj(a)`  (`attention/output [5120, 6144]`)

## GDN layer (48)
1. `h = rmsnorm(x, input_norm)`
2. `q, k = query_key_proj(h)`  (`gdn/query_key [4096, 5120]` = q 2048 + k 2048)
   `v, z = value_z_proj(h)`    (`gdn/value_z [12288, 5120]` = v 6144 + z 6144)
   - q/k: 16 heads x 128; v/z: 48 heads x 128
3. Gating (per v-head, 48):
   - `a = a_proj(h)` [48, 5120], `b = b_proj(h)` [48, 5120]
   - `g = -exp(A_log[h]) * softplus(a + dt_bias[h])`
   - `beta = sigmoid(b)`
4. `qkv_c = causal_conv1d_silu([q,k,v], convolution [4, 10240], conv_state)`
   - conv_dim = 2*key_dim + value_dim = 2*2048 + 6144 = 10240, kernel 4
5. `o = gated_delta_net(q, k, v, g, beta, scale=kGdnScale, normalize_qk=true, state)`
   - state: [48 v-heads, 128 dv, 128 dk] fp32
   - head map: qk_head = h_v / 3 (48 v-heads -> 16 qk-heads, group 3)
6. `on = gated_rmsnorm(o, gdn_norm [128], z)` (per v-head, dim 128)
7. `x = x + output_proj(on)`  (`gdn/output [5120, 6144]`)

### Gated DeltaNet recurrence (per v-head, per token)
```
alpha = exp(g)
for each dv row r:
    partial = dot(state[r, :], k)          # state [dv, dk]
    delta   = beta * (v[r] - alpha * partial)
    state[r, :] = alpha * state[r, :] + delta * k
out[r] = dot(state[r, :], q) * scale
```
q and k are L2-normalized per head (eps 1e-6) before the recurrence.

## MLP (all 64 layers)
1. `h = rmsnorm(x, post_attention_norm)`
2. `gate, up = gate_up_proj(h)`  (`mlp/gate_up [34816, 5120]` = 2 x 17408)
3. `h = silu(gate) * up`
4. `x = x + down_proj(h)`  (`mlp/down [5120, 17408]`)

## Embedding / output
- `x = token_embedding[token]`  (`text/token_embedding [248320, 5120]` W8)
- `h = rmsnorm(x, final_norm)`
- `logits = output_head(h)`  (`text/output_head [248320, 5120]` W8)
- `token = argmax(logits)`

## Tensor name map (per layer)
GDN layer `text/layers/{i}`:
- `input_norm` [5120] BF16
- `gdn/a_log` [48] FP32, `gdn/dt_bias` [48] FP32
- `gdn/convolution` [4, 10240] BF16
- `gdn/a_projection` [48, 5120] BF16, `gdn/b_projection` [48, 5120] BF16
- `gdn/query_key` [4096, 5120] Q4, `gdn/value_z` [12288, 5120] Q5
- `gdn/norm` [128] BF16
- `gdn/output` [5120, 6144] Q5
- `post_attention_norm` [5120] BF16
- `mlp/gate_up` [34816, 5120] Q4, `mlp/down` [5120, 17408] Q5

Full-attn layer `text/layers/{i}`:
- `input_norm` [5120] BF16
- `attention/query_key` [7168, 5120] Q4
- `attention/gate_value` [7168, 5120] Q5
- `attention/query_norm` [256] BF16, `attention/key_norm` [256] BF16
- `attention/output` [5120, 6144] Q5
- `post_attention_norm` [5120] BF16
- `mlp/gate_up` [34816, 5120] Q4, `mlp/down` [5120, 17408] Q5

Global:
- `text/token_embedding` [248320, 5120] W8
- `text/final_norm` [5120] BF16
- `text/output_head` [248320, 5120] W8