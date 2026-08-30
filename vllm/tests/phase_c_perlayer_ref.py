"""Phase C: per-layer CPU reference to find the FIRST diverging layer.

Runs the full 48-layer forward on CPU from the CORRECT embedding (absolute
anchor) using the checkpoint weights, and compares the wide residual state
after each layer to the captured (buggy-run) resid_out. The first layer where
cosine drops = the bug.

Every block type's math was verified in isolation (GDN L0, PLE L1, MoE L2,
FA L3, HC, final mixer, lm_head). This test threads them together to localize
the divergence.
"""
import glob, math, os, torch
import torch.nn.functional as F
from safetensors import safe_open

W4 = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
TABLE = "/mnt/data/ple_cache/ple_table_qwen4exp.pt"
HC, N_EMBD = 4, 2560
HC_DIM = HC * N_EMBD
LOWRANK = 320
EPS = 1e-6
# GDN
NUM_K, NUM_V, HK, HV = 16, 48, 128, 128
# FA
NUM_Q, NUM_KV, HEAD_DIM, ROT_DIM, BASE = 24, 2, 256, 64, 1e7
# MoE
N_EXPERTS, TOPK, MOE_I = 512, 10, 640
GROUP = 128

LAYER_TYPES = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention',
 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
PLE_LAYERS = {1}  # 0-based

# ---- weight cache ----
_KEY2FILE = {}
def _build_index():
    for f in sorted(glob.glob(W4 + "/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                _KEY2FILE[k] = f
_build_index()
_WCACHE = {}
def load(name):
    if name in _WCACHE:
        return _WCACHE[name]
    f = _KEY2FILE.get(name)
    if f is None:
        raise KeyError(name)
    with safe_open(f, framework="pt") as sf:
        t = sf.get_tensor(name)
    _WCACHE[name] = t
    return t

def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()

def dequant_w4a16(packed, scale, group=GROUP):
    out, in8 = packed.shape
    inp = in8 * 8
    p = packed.to(torch.int64)
    shifts = torch.arange(0, 32, 4)
    q = (p.unsqueeze(-1) >> shifts).to(torch.int32) & 0xF
    q = q.reshape(out, inp).to(torch.float32)
    s = scale.to(torch.float32).repeat_interleave(group, dim=1)
    return (q - 8.0) * s

def rmsnorm_gemma(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + EPS) * (1.0 + w)

def grouped_rmsnorm(x):
    # per-stream standard RMSNorm over n_embd (no affine)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + EPS)

def hc_mix(resid, hc_norm, w_down, w_up, w_inject):
    T = resid.shape[0]
    xn = grouped_rmsnorm(resid)
    xn_flat = xn.reshape(T, HC_DIM) * hc_norm
    lo = xn_flat @ w_down.T
    lo = F.silu(lo / HC)
    gate = torch.sigmoid(lo @ w_up.T)
    gated = (xn_flat * gate).reshape(T, HC, N_EMBD)
    mix_out = gated.mean(dim=1)
    inject = xn_flat @ w_inject.T
    return mix_out, inject

def hc_combine(resid, block_out, inject):
    w = 2.0 * torch.sigmoid(inject / HC)
    return resid + block_out.unsqueeze(1) * w.unsqueeze(-1)

def pure_python_gdn(x, base):
    in_qkv = load(f"{base}.in_proj_qkv.weight").float()
    in_z = load(f"{base}.in_proj_z.weight").float()
    in_b = load(f"{base}.in_proj_b.weight").float()
    in_a = load(f"{base}.in_proj_a.weight").float()
    A_log = load(f"{base}.A_log").float()
    dt_bias = load(f"{base}.dt_bias").float()
    conv_w = load(f"{base}.conv1d.weight").float()
    norm_w = load(f"{base}.norm.weight").float()
    out_w = load(f"{base}.out_proj.weight").float()
    T = x.shape[0]
    qkvz = x @ torch.cat([in_qkv, in_z], dim=0).T
    ba = x @ torch.cat([in_b, in_a], dim=0).T
    q_dim = NUM_K * HK
    v_dim = NUM_V * HV
    z = qkvz[:, q_dim * 2 + v_dim:].reshape(T, NUM_V, HV)
    qkv = qkvz[:, :q_dim * 2 + v_dim]
    w2d = conv_w.view(conv_w.shape[0], conv_w.shape[2])
    xx = qkv.transpose(0, 1).unsqueeze(0)
    xp = F.pad(xx, (3, 0))
    conv = F.conv1d(xp, w2d.unsqueeze(1), groups=w2d.shape[0])
    conv = F.silu(conv).transpose(1, 2).squeeze(0)
    q = conv[:, :q_dim].view(T, NUM_K, HK)
    k = conv[:, q_dim:q_dim * 2].view(T, NUM_K, HK)
    v = conv[:, q_dim * 2:].view(T, NUM_V, HV)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    # FLA kernel scales q by 1/sqrt(K) after l2norm (K = head_k_dim).
    q = q * (1.0 / (HK ** 0.5))
    b = ba[:, :NUM_V]
    a = ba[:, NUM_V:]
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)
    beta = torch.sigmoid(b)
    rep = NUM_V // NUM_K
    # llama.cpp uses ggml_repeat_4d = TILE (v-head h -> k-head h % NUM_K).
    # vLLM/FLA may use INTERLEAVE (v-head h -> k-head h // rep). Switchable.
    if os.environ.get("GDN_REPEAT", "interleave") == "tile":
        q = q.repeat(1, rep, 1)
        k = k.repeat(1, rep, 1)
    else:
        q = q.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
    state = torch.zeros(NUM_V, HV, HK, dtype=torch.float32)
    outs = []
    for t in range(T):
        qt, kt, vt = q[t], k[t], v[t]
        gt, bt = g[t], beta[t].unsqueeze(-1)
        decay = torch.exp(gt).unsqueeze(-1).unsqueeze(-1)
        state = state * decay
        pred = torch.bmm(state, kt.unsqueeze(-1)).squeeze(-1)
        delta = bt * (vt - pred)
        state = state + torch.bmm(delta.unsqueeze(-1), kt.unsqueeze(1))
        out_t = torch.bmm(state, qt.unsqueeze(-1)).squeeze(-1)
        outs.append(out_t)
    core = torch.stack(outs, dim=0)
    # post-kernel: RMSNormGated (norm before gate, sigmoid) + out_proj
    core2 = core.reshape(-1, HV)
    z2 = z.reshape(-1, HV)
    var = core2.pow(2).mean(dim=-1, keepdim=True)
    xn = core2 * torch.rsqrt(var + EPS)
    zgated = xn * norm_w * torch.sigmoid(z2)
    zgated = zgated.reshape(T, NUM_V * HV)
    return zgated @ out_w.T

def manual_rope(x, positions, cos_cache, sin_cache):
    c = cos_cache[positions].unsqueeze(1)
    s = sin_cache[positions].unsqueeze(1)
    x1 = x[..., :ROT_DIM // 2]
    x2 = x[..., ROT_DIM // 2:ROT_DIM]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out = x.clone()
    out[..., :ROT_DIM // 2] = o1
    out[..., ROT_DIM // 2:ROT_DIM] = o2
    return out

def full_attention(x, base, positions, cos_cache, sin_cache):
    q_w = dequant_w4a16(load(f"{base}.q_proj.weight_packed"), load(f"{base}.q_proj.weight_scale"))
    k_w = dequant_w4a16(load(f"{base}.k_proj.weight_packed"), load(f"{base}.k_proj.weight_scale"))
    v_w = dequant_w4a16(load(f"{base}.v_proj.weight_packed"), load(f"{base}.v_proj.weight_scale"))
    o_w = dequant_w4a16(load(f"{base}.o_proj.weight_packed"), load(f"{base}.o_proj.weight_scale"))
    qn_w = load(f"{base}.q_norm.weight").float()
    kn_w = load(f"{base}.k_norm.weight").float()
    T = x.shape[0]
    w_full = torch.cat([q_w, k_w, v_w], dim=0)  # [13312, 2560]
    qkv = x @ w_full.T
    q_gate = qkv[:, :12288].view(T, NUM_Q, 512)
    k = qkv[:, 12288:12800].view(T, NUM_KV, HEAD_DIM)
    v = qkv[:, 12800:13312].view(T, NUM_KV, HEAD_DIM)
    q_pre, gate = torch.chunk(q_gate, 2, dim=-1)  # (T,24,256)
    q_n = rmsnorm_gemma(q_pre, qn_w)
    k_n = rmsnorm_gemma(k, kn_w)
    q_rope = manual_rope(q_n, positions, cos_cache, sin_cache)
    k_rope = manual_rope(k_n, positions, cos_cache, sin_cache)
    # GQA attention: q head h -> kv head h // 12
    scale = HEAD_DIM ** 0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    attn_heads = []
    for h in range(NUM_Q):
        kvh = h // (NUM_Q // NUM_KV)
        scores = torch.einsum("td,id->ti", q_rope[:, h], k_rope[:, kvh]) / scale
        scores = scores.masked_fill(mask, float("-inf"))
        a = torch.softmax(scores, dim=-1)
        out_h = a @ v[:, kvh]  # (T, 256)
        attn_heads.append(out_h)
    attn = torch.stack(attn_heads, dim=1).reshape(T, NUM_Q * HEAD_DIM)  # (T, 6144)
    gate = gate.reshape(T, NUM_Q * HEAD_DIM)
    attn_gated = attn * torch.sigmoid(gate)
    return attn_gated @ o_w.T

_MOE_CACHE = {}
def moe(x, base):
    T = x.shape[0]
    gate_w = load(f"{base}.gate.weight").float()
    se_gate = load(f"{base}.shared_expert.gate_proj.weight").float()
    se_up = load(f"{base}.shared_expert.up_proj.weight").float()
    se_down = load(f"{base}.shared_expert.down_proj.weight").float()
    se_gate_scalar = load(f"{base}.shared_expert_gate.weight").float()
    logits = x @ gate_w.T
    probs = torch.softmax(logits, dim=-1)
    topk_w, topk_idx = torch.topk(probs, TOPK, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    # shared expert
    shared = F.silu(x @ se_gate.T) * (x @ se_up.T)
    shared = shared @ se_down.T
    shared = torch.sigmoid(x @ se_gate_scalar.T) * shared
    # routed experts
    used = torch.unique(topk_idx)
    routed = torch.zeros_like(x)
    for e in used.tolist():
        key = (base, e)
        if key not in _MOE_CACHE:
            eb = f"{base}.experts.{e}"
            wg = dequant_w4a16(load(f"{eb}.gate_proj.weight_packed"), load(f"{eb}.gate_proj.weight_scale"))
            wu = dequant_w4a16(load(f"{eb}.up_proj.weight_packed"), load(f"{eb}.up_proj.weight_scale"))
            wd = dequant_w4a16(load(f"{eb}.down_proj.weight_packed"), load(f"{eb}.down_proj.weight_scale"))
            _MOE_CACHE[key] = (wg, wu, wd)
        wg, wu, wd = _MOE_CACHE[key]
        h = F.silu(x @ wg.T) * (x @ wu.T)
        eout = h @ wd.T
        mask = (topk_idx == e)
        w = (topk_w * mask).sum(dim=-1, keepdim=True)
        routed = routed + w * eout
    return routed + shared

def ple_forward(resid, rows, base):
    key_proj = load(f"{base}.key_proj.weight").float()
    value_proj = load(f"{base}.value_proj.weight").float()
    norm_key = load(f"{base}.norm_key.weight").float()
    norm_query = load(f"{base}.norm_query.weight").float()
    norm_conv = load(f"{base}.norm_conv.weight").float()
    conv_w = load(f"{base}.conv1d.weight").float()
    T = resid.shape[0]
    table = torch.load(TABLE, mmap=True, weights_only=True)["table"]
    emb = table[rows.reshape(-1)].reshape(T, -1).float()
    def gnorm(x, w):
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        xn = x * torch.rsqrt(var + EPS)
        return (xn.reshape(T, HC_DIM) * w).reshape(T, HC, N_EMBD)
    key = gnorm(emb @ key_proj.T, norm_key)
    value = emb @ value_proj.T
    query = gnorm(resid, norm_query)
    s = (key * query).sum(dim=-1) / math.sqrt(N_EMBD)
    gate = torch.sigmoid(torch.sign(s) * torch.sqrt(torch.clamp(s.abs(), min=1e-6)))
    gated = value.unsqueeze(1) * gate.unsqueeze(-1)
    normalized = gnorm(gated, norm_conv)
    KERN, DIL = 4, 3
    HIST = (KERN - 1) * DIL
    x = normalized.reshape(T, HC_DIM)
    w = conv_w.squeeze(1)
    state = torch.zeros(HIST, HC_DIM)
    padded = torch.cat([state, x], dim=0)
    acc = torch.zeros(T, HC_DIM)
    for k in range(KERN):
        offset = HIST - (KERN - 1 - k) * DIL
        acc = acc + padded[offset:offset + T].float() * w[:, k].float()
    conv_out = F.silu(acc)
    return resid + gated + conv_out.reshape(T, HC, N_EMBD)

def main():
    emb = torch.load("/tmp/qwen4exp_embed_capture.pt", map_location="cpu")
    embed = emb["embed"].float()
    T = embed.shape[0]
    positions = torch.arange(T, dtype=torch.long)
    inv_freq = 1.0 / (BASE ** (torch.arange(0, ROT_DIM, 2, dtype=torch.float) / ROT_DIM))
    t = torch.arange(262144, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_cache, sin_cache = freqs.cos(), freqs.sin()

    # PLE rows (deterministic from input_ids; reuse captured rows)
    ple_cap = torch.load("/tmp/qwen4exp_ple_capture_rank0.pt", map_location="cpu")
    ple_rows = ple_cap["rows"]

    # wide state
    resid = embed.unsqueeze(1).expand(-1, HC, -1).contiguous()  # (T, hc, n_embd)

    print(f"T={T} starting per-layer CPU reference\n")
    print(f"{'L':>2} {'type':<6} {'cos(ref,cap)':>12} {'ref_norm':>10} {'cap_norm':>10}")
    first_div = None
    for L in range(48):
        ltype = LAYER_TYPES[L]
        base = f"model.language_model.layers.{L}"
        # PLE
        if L in PLE_LAYERS:
            resid = ple_forward(resid, ple_rows, f"{base}.ple")
        # attn sub-block
        a_base = f"{base}.attn_hyper_connection"
        cur, inject = hc_mix(resid, load(f"{a_base}.hc_norm.weight").float(),
                             load(f"{a_base}.input_mix_weight_down.weight").float(),
                             load(f"{a_base}.input_mix_weight_up.weight").float(),
                             load(f"{a_base}.block_inject_weight.weight").float())
        if ltype == "linear_attention":
            attn_out = pure_python_gdn(cur, f"{base}.linear_attn")
        else:
            attn_out = full_attention(cur, f"{base}.self_attn", positions, cos_cache, sin_cache)
        resid = hc_combine(resid, attn_out, inject)
        # mlp sub-block
        m_base = f"{base}.mlp_hyper_connection"
        cur2, inject2 = hc_mix(resid, load(f"{m_base}.hc_norm.weight").float(),
                               load(f"{m_base}.input_mix_weight_down.weight").float(),
                               load(f"{m_base}.input_mix_weight_up.weight").float(),
                               load(f"{m_base}.block_inject_weight.weight").float())
        mlp_out = moe(cur2, f"{base}.mlp")
        resid = hc_combine(resid, mlp_out, inject2)
        # compare
        cap = torch.load(f"/tmp/qwen4exp_layer{L:02d}_resid.pt", map_location="cpu")["resid_out"].float()
        c = cos(resid, cap)
        flag = ""
        if first_div is None and c < 0.99:
            first_div = L
            flag = "  <-- FIRST DIVERGENCE"
        print(f"{L:>2} {ltype[:4]:<6} {c:>12.6f} {resid.norm().item():>10.3f} {cap.norm().item():>10.3f}{flag}")
        if L % 8 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"\nFirst diverging layer: {first_div}")

if __name__ == "__main__":
    main()