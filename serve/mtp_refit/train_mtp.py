#!/usr/bin/env python3
"""Standalone MTP dense-core re-fit trainer for Qwen4-Exp (Flash-Next).

Reimplements the ``Qwen4ExpMultiTokenPredictor`` draft head in plain PyTorch
(vLLM is inference-only, no autograd path) and trains ONLY the dense core on
one GPU. Frozen: the 512 routed experts, ``embed_tokens``, and ``lm_head``
(all loaded from the checkpoint). Trainable (~89M): fc stage, the single
full-attention decoder layer (HC mixers + attention + shared expert + router),
and the final hyper-connection mixer.

Data: parsed datagen shards (see parse_trace.py) -- per-sequence
``input_ids`` (int32), ``wide`` (bf16 [T, hc*n_embd]), ``lens``. For each
position t the draft sees (embed(token_t), wide_state_t) and predicts
token_{t+1}. Loss = CE(lm_head(draft_out_t), token_{t+1}).

Conventions mirrored EXACTLY from the vLLM fork (~/vllm-fresh-venv):
  * every RMSNorm uses DEVIATION gamma: true_gamma = (1 + w)
    (HyperConnection.hc_norm, pre_fc_norm_*, and GemmaRMSNorm q/k_norm).
  * RoPE: neox-style, theta=1e7, partial_rotary_factor=0.25 -> rotary_dim=64
    (first 64 of head_dim=256 rotate; remainder passthrough). Cache indexed
    by ABSOLUTE position.
  * Attention: GQA 24q / 2kv, head_dim 256, sigmoid output gate (q_proj is
    doubled to 12288 = 24*256*2), o_proj [2560, 6144].
  * MoE: 512 experts, top-10, norm_topk_prob (renormalize), + gated shared
    expert (sigmoid(shared_expert_gate(x)) * silu(gate)*up -> down).
  * HyperConnection mix/combine: grouped-RMSNorm -> gamma(1+w) -> low-rank
    down -> silu(x/hc) -> up -> sigmoid gate -> x*gate -> mean-collapse;
    combine = residual + block_out * 2*sigmoid(inject/hc).

Usage:
  # fidelity check (shipped weights, no training) -- must reproduce ~63% pos-1
  python train_mtp.py --data-dir ./data --ckpt <CKPT> --device xpu:0 \
      --mode fidelity --eval-n 8
  # train the dense core
  python train_mtp.py --data-dir ./data --ckpt <CKPT> --device xpu:0 \
      --epochs 3 --lr 1e-4 --seq-batch 1 --out-dir ./checkpoints
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_text_config(ckpt: str):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    return cfg.get("text_config", cfg)


class Cfg:
    def __init__(self, tc):
        self.hidden_size = tc["hidden_size"]                 # 2560
        self.hc = tc["hc_count"]                             # 4
        self.lowrank = tc["hc_lowrank"]                      # 320
        self.eps = tc["rms_norm_eps"]                        # 1e-6
        self.vocab_size = tc["vocab_size"]                   # 248320
        self.n_heads = tc["num_attention_heads"]             # 24
        self.n_kv = tc["num_key_value_heads"]                # 2
        self.head_dim = tc["head_dim"]                       # 256
        self.moe_int = tc["moe_intermediate_size"]           # 640
        self.n_experts = tc["num_experts"]                   # 512
        self.top_k = tc["num_experts_per_tok"]               # 10
        self.shared_int = tc["shared_expert_intermediate_size"]  # 640
        rp = tc.get("rope_parameters", {})
        self.rope_theta = rp.get("rope_theta", 10000000)
        self.partial_rot = rp.get("partial_rotary_factor",
                                  tc.get("partial_rotary_factor", 1.0))
        self.max_pos = tc.get("max_position_embeddings", 262144)
        self.rotary_dim = int(self.head_dim * self.partial_rot)   # 64


# --------------------------------------------------------------------------- #
# Norms (deviation gamma: true = 1 + w)
# --------------------------------------------------------------------------- #
def rms_deviation(x, w, eps):
    """x: [..., D]; w: [D] (deviation). Returns x * rsqrt(mean(x^2)+eps) * (1+w)."""
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps).to(x.dtype)
    gamma = (1.0 + w.to(torch.float32)).to(xn.dtype)
    return xn * gamma


def _param(shape, scale=0.02):
    """Small-random-initialized Parameter. Safe for real training: load_weights
    overwrites these with checkpoint values; for any unloaded key this is a
    sane default, and it keeps the CPU self-test numerically stable."""
    p = nn.Parameter(torch.empty(*shape))
    nn.init.normal_(p.data, mean=0.0, std=scale)
    return p


def grouped_rms_collapse(x, w, hc, n_embd, eps):
    """x: [T, hc, n_embd]; w: [hc*n_embd] (deviation). Per-stream RMSNorm +
    gamma, then mean-collapse over streams -> [T, n_embd]."""
    T = x.shape[0]
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps).to(x.dtype)
    gamma = (1.0 + w.to(torch.float32)).view(hc, n_embd).to(xn.dtype)
    xn = xn * gamma
    return xn.mean(dim=1)


# --------------------------------------------------------------------------- #
# HyperConnection (mirrors qwen4_exp.HyperConnection)
# --------------------------------------------------------------------------- #
class HyperConnection(nn.Module):
    def __init__(self, c: Cfg, has_inject: bool):
        super().__init__()
        self.c = c
        self.has_inject = has_inject
        hd = c.hc * c.hidden_size
        self.hc_norm = _param([hd])
        self.down = _param([c.lowrank, hd])
        self.up = _param([hd, c.lowrank])
        if has_inject:
            self.inject = _param([c.hc, hd])

    def mix(self, x):
        # x: [T, hc, n_embd]
        c = self.c
        xn = grouped_rms_collapse_no_mean(x, self.hc_norm, c.hc, c.hidden_size, c.eps)
        flat = xn.reshape(x.shape[0], -1)
        lo = F.linear(flat, self.down)
        lo = F.silu(lo / c.hc)
        gate = torch.sigmoid(F.linear(lo, self.up))
        gated = (flat * gate).reshape(x.shape[0], c.hc, c.hidden_size)
        mixed = gated.mean(dim=1)
        inject = F.linear(flat, self.inject) if self.has_inject else None
        return mixed, inject

    def combine(self, residual, block_out, inject):
        w = 2.0 * torch.sigmoid(inject / self.c.hc)          # [T, hc]
        w = w.unsqueeze(-1)                                   # [T, hc, 1]
        b = block_out.unsqueeze(1)                            # [T, 1, n_embd]
        return residual + b * w


def grouped_rms_collapse_no_mean(x, w, hc, n_embd, eps):
    """Per-stream RMSNorm + gamma(1+w), NO mean-collapse (returns [T,hc,n_embd])."""
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps).to(x.dtype)
    gamma = (1.0 + w.to(torch.float32)).view(hc, n_embd).to(xn.dtype)
    return xn * gamma


# --------------------------------------------------------------------------- #
# Full-attention block (mirrors Qwen3NextAttention, eager, TP1)
# --------------------------------------------------------------------------- #
class FullAttention(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.hidden_size, c.n_heads * c.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.n_kv * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.n_kv * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.n_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm = _param([c.head_dim])
        self.k_norm = _param([c.head_dim])
        # RoPE cache (fp32 source, cast to bf16 at use). Indexed by abs pos.
        rd = c.rotary_dim
        inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        t = torch.arange(min(c.max_pos, 32768), dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def _rope(self, x, positions):
        # x: [T, H, D]; neox-style rotate over the first rotary_dim dims.
        # cos/sin caches hold rotary_dim/2 entries; pair dim j with dim j+rd/2.
        c = self.c
        rd = c.rotary_dim
        cos = self.cos_cache[positions].to(x.dtype).unsqueeze(1)   # [T,1,rd/2]
        sin = self.sin_cache[positions].to(x.dtype).unsqueeze(1)   # [T,1,rd/2]
        if rd < c.head_dim:
            rot = x[..., :rd]
            passthrough = x[..., rd:]
        else:
            rot = x
            passthrough = None
        x1 = rot[..., :rd // 2]
        x2 = rot[..., rd // 2:]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        rot_out = torch.cat([o1, o2], dim=-1)                       # [T,H,rd]
        if passthrough is not None:
            return torch.cat([rot_out, passthrough], dim=-1)
        return rot_out

    def forward(self, h, positions):
        # h: [T, n_embd]; positions: [T] int64 (absolute)
        c = self.c
        T = h.shape[0]
        qg = self.q_proj(h).view(T, c.n_heads, c.head_dim * 2)
        q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:]
        k = self.k_proj(h).view(T, c.n_kv, c.head_dim)
        v = self.v_proj(h).view(T, c.n_kv, c.head_dim)
        q = rms_deviation(q, self.q_norm, c.eps)
        k = rms_deviation(k, self.k_norm, c.eps)
        q = self._rope(q, positions)
        k = self._rope(k, positions)
        # GQA: expand kv heads to match q heads
        k = k.repeat_interleave(c.n_heads // c.n_kv, dim=1)
        v = v.repeat_interleave(c.n_heads // c.n_kv, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn * torch.sigmoid(gate)
        out = self.o_proj(attn.reshape(T, -1))
        return out


# --------------------------------------------------------------------------- #
# MoE block (mirrors Qwen3NextSparseMoeBlock, TP1, frozen experts)
# --------------------------------------------------------------------------- #
class MoEBlock(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.gate = nn.Linear(c.hidden_size, c.n_experts, bias=False)
        self.shared_gate = nn.Linear(c.hidden_size, 1, bias=False)
        # shared expert (trainable)
        self.shared_gate_proj = nn.Linear(c.hidden_size, c.shared_int, bias=False)
        self.shared_up_proj = nn.Linear(c.hidden_size, c.shared_int, bias=False)
        self.shared_down_proj = nn.Linear(c.shared_int, c.hidden_size, bias=False)
        # routed experts (FROZEN)
        self.expert_gate_up = _param([c.n_experts, 2 * c.moe_int, c.hidden_size])
        self.expert_down = _param([c.n_experts, c.hidden_size, c.moe_int])
        self.expert_gate_up.requires_grad_(False)
        self.expert_down.requires_grad_(False)

    def _shared(self, x):
        gu = torch.cat([F.silu(self.shared_gate_proj(x)) * self.shared_up_proj(x)], dim=-1)
        out = self.shared_down_proj(gu)
        out = torch.sigmoid(self.shared_gate(x)) * out
        return out

    def _routed(self, x):
        # x: [T, n_embd]; per-token expert selection via bmm.
        c = self.c
        T = x.shape[0]
        logits = self.gate(x)                                   # [T, E]
        topv, topi = torch.topk(logits, c.top_k, dim=-1)        # [T, k]
        w = F.softmax(topv.float(), dim=-1)                     # renormalize
        w = w.to(x.dtype)
        x3 = x.unsqueeze(-1)                                    # [T, n_embd, 1]
        acc = torch.zeros_like(x)
        for j in range(c.top_k):
            idx = topi[:, j]                                    # [T]
            wi = w[:, j:j + 1]                                  # [T, 1]
            # A=[T,out,in] @ B=[T,in,1] -> [T,out,1]; gate_up is [E,2I,n_embd]
            gu = torch.bmm(self.expert_gate_up[idx], x3).squeeze(-1)      # [T, 2I]
            g, u = gu.chunk(2, dim=-1)
            act = F.silu(g) * u                                 # [T, I]
            # down is [E,n_embd,I]: A=[T,n_embd,I] @ B=[T,I,1] -> [T,n_embd,1]
            d = torch.bmm(self.expert_down[idx], act.unsqueeze(-1)).squeeze(-1)  # [T, n_embd]
            acc = acc + wi * d
        return acc

    def forward(self, x):
        return self._shared(x) + self._routed(x)


# --------------------------------------------------------------------------- #
# Decoder layer (mirrors Qwen4ExpDecoderLayer, full_attention, no PLE)
# --------------------------------------------------------------------------- #
class DraftDecoderLayer(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.attn_hc = HyperConnection(c, has_inject=True)
        self.self_attn = FullAttention(c)
        self.mlp_hc = HyperConnection(c, has_inject=True)
        self.mlp = MoEBlock(c)

    def forward(self, x, residual, positions):
        # x: [T, n_embd]; residual: [T, hc, n_embd] | None
        c = self.c
        if residual is None:
            residual = x.unsqueeze(1).expand(-1, c.hc, -1).contiguous()
        cur, inject = self.attn_hc.mix(residual)
        cur = self.self_attn(cur, positions)
        residual = self.attn_hc.combine(residual, cur, inject)
        cur, inject = self.mlp_hc.mix(residual)
        cur = self.mlp(cur)
        residual = self.mlp_hc.combine(residual, cur, inject)
        return residual.mean(dim=1), residual


# --------------------------------------------------------------------------- #
# Draft head (mirrors Qwen4ExpMultiTokenPredictor)
# --------------------------------------------------------------------------- #
class MTPDraftHead(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.pre_fc_norm_embedding_w = _param([c.hidden_size])
        self.pre_fc_norm_hidden_w = _param([c.hc * c.hidden_size])
        self.fc_embedding = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.fc_hidden = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.layer = DraftDecoderLayer(c)
        self.mixer = HyperConnection(c, has_inject=False)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def _fc_stage(self, emb, wide):
        e = rms_deviation(emb, self.pre_fc_norm_embedding_w, self.c.eps)
        h = grouped_rms_collapse(wide, self.pre_fc_norm_hidden_w,
                                 self.c.hc, self.c.hidden_size, self.c.eps)
        return self.fc_embedding(e) + self.fc_hidden(h)

    def draft_forward(self, input_ids, wide_flat, positions):
        # input_ids: [T] int64 ; wide_flat: [T, hc*n_embd] ; positions: [T] int64
        c = self.c
        emb = self.embed_tokens(input_ids)
        wide = wide_flat.view(-1, c.hc, c.hidden_size)
        x = self._fc_stage(emb, wide)
        _collapsed, wide_out = self.layer(x, None, positions)
        out, _ = self.mixer.mix(wide_out)
        return out

    def logits(self, input_ids, wide_flat, positions):
        out = self.draft_forward(input_ids, wide_flat, positions)
        return self.lm_head(out)


# --------------------------------------------------------------------------- #
# Weight loading (checkpoint -> module)
# --------------------------------------------------------------------------- #
CHECKPOINT_MAP = {
    # name in checkpoint (mtp.* prefix stripped handled below) -> (module_attr_path, transform)
}


def load_weights(model: MTPDraftHead, ckpt: str, device: str, verbose=True):
    """Load mtp.*, embed_tokens, lm_head from safetensors into the module."""
    from safetensors import safe_open
    idx_path = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
    else:
        wm = {}
        for f in sorted(os.listdir(ckpt)):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(ckpt, f), framework="pt") as sf:
                    for k in sf.keys():
                        wm[k] = f

    c = model.c
    want = {}
    for k in wm:
        if k.startswith("mtp.") or k in ("lm_head.weight",) or "embed_tokens" in k:
            want[k] = wm[k]

    # group by file
    by_file = {}
    for k, f in want.items():
        by_file.setdefault(f, []).append(k)

    def put(param: nn.Parameter, t: torch.Tensor):
        param.data.copy_(t.to(device=device, dtype=param.dtype))

    loaded = set()
    for f, keys in by_file.items():
        path = os.path.join(ckpt, f)
        with safe_open(path, framework="pt") as sf:
            for k in keys:
                t = sf.get_tensor(k)
                nk = k
                if nk.startswith("mtp."):
                    nk = nk[len("mtp."):]
                if nk == "lm_head.weight":
                    put(model.lm_head.weight, t); loaded.add(k)
                elif "embed_tokens" in nk:
                    put(model.embed_tokens.weight, t); loaded.add(k)
                elif nk == "pre_fc_norm_embedding.weight":
                    put(model.pre_fc_norm_embedding_w, t); loaded.add(k)
                elif nk == "pre_fc_norm_hidden.weight":
                    put(model.pre_fc_norm_hidden_w, t); loaded.add(k)
                elif nk == "fc_embedding.weight":
                    put(model.fc_embedding.weight, t); loaded.add(k)
                elif nk == "fc_hidden.weight":
                    put(model.fc_hidden.weight, t); loaded.add(k)
                elif nk == "hyper_connection_mixer.hc_norm.weight":
                    put(model.mixer.hc_norm, t); loaded.add(k)
                elif nk == "hyper_connection_mixer.input_mix_weight_down.weight":
                    put(model.mixer.down, t); loaded.add(k)
                elif nk == "hyper_connection_mixer.input_mix_weight_up.weight":
                    put(model.mixer.up, t); loaded.add(k)
                elif nk.startswith("layers.0."):
                    sub = nk[len("layers.0."):]
                    L = model.layer
                    if sub == "attn_hyper_connection.hc_norm.weight":
                        put(L.attn_hc.hc_norm, t); loaded.add(k)
                    elif sub == "attn_hyper_connection.input_mix_weight_down.weight":
                        put(L.attn_hc.down, t); loaded.add(k)
                    elif sub == "attn_hyper_connection.input_mix_weight_up.weight":
                        put(L.attn_hc.up, t); loaded.add(k)
                    elif sub == "attn_hyper_connection.block_inject_weight.weight":
                        put(L.attn_hc.inject, t); loaded.add(k)
                    elif sub == "mlp_hyper_connection.hc_norm.weight":
                        put(L.mlp_hc.hc_norm, t); loaded.add(k)
                    elif sub == "mlp_hyper_connection.input_mix_weight_down.weight":
                        put(L.mlp_hc.down, t); loaded.add(k)
                    elif sub == "mlp_hyper_connection.input_mix_weight_up.weight":
                        put(L.mlp_hc.up, t); loaded.add(k)
                    elif sub == "mlp_hyper_connection.block_inject_weight.weight":
                        put(L.mlp_hc.inject, t); loaded.add(k)
                    elif sub == "self_attn.q_proj.weight":
                        put(L.self_attn.q_proj.weight, t); loaded.add(k)
                    elif sub == "self_attn.k_proj.weight":
                        put(L.self_attn.k_proj.weight, t); loaded.add(k)
                    elif sub == "self_attn.v_proj.weight":
                        put(L.self_attn.v_proj.weight, t); loaded.add(k)
                    elif sub == "self_attn.o_proj.weight":
                        put(L.self_attn.o_proj.weight, t); loaded.add(k)
                    elif sub == "self_attn.q_norm.weight":
                        put(L.self_attn.q_norm, t); loaded.add(k)
                    elif sub == "self_attn.k_norm.weight":
                        put(L.self_attn.k_norm, t); loaded.add(k)
                    elif sub == "mlp.gate.weight":
                        put(L.mlp.gate.weight, t); loaded.add(k)
                    elif sub == "mlp.shared_expert_gate.weight":
                        put(L.mlp.shared_gate.weight, t); loaded.add(k)
                    elif sub == "mlp.shared_expert.gate_proj.weight":
                        put(L.mlp.shared_gate_proj.weight, t); loaded.add(k)
                    elif sub == "mlp.shared_expert.up_proj.weight":
                        put(L.mlp.shared_up_proj.weight, t); loaded.add(k)
                    elif sub == "mlp.shared_expert.down_proj.weight":
                        put(L.mlp.shared_down_proj.weight, t); loaded.add(k)
                    elif sub == "mlp.experts.gate_up_proj":
                        put(L.mlp.expert_gate_up, t); loaded.add(k)
                    elif sub == "mlp.experts.down_proj":
                        put(L.mlp.expert_down, t); loaded.add(k)
                    else:
                        if verbose:
                            print(f"  [warn] unmapped mtp key: {k}")
                else:
                    if verbose:
                        print(f"  [warn] unmapped key: {k}")
    if verbose:
        print(f"loaded {len(loaded)} tensors from {len(by_file)} files")
    return loaded


def freeze_frozen(model: MTPDraftHead):
    """Freeze experts + embed + lm_head; keep dense core trainable."""
    for p in model.parameters():
        p.requires_grad_(False)
    # trainable dense core
    for name in ["pre_fc_norm_embedding_w", "pre_fc_norm_hidden_w"]:
        getattr(model, name).requires_grad_(True)
    model.fc_embedding.weight.requires_grad_(True)
    model.fc_hidden.weight.requires_grad_(True)
    L = model.layer
    for mod in [L.attn_hc, L.mlp_hc]:
        for n in ["hc_norm", "down", "up", "inject"]:
            p = getattr(mod, n, None)
            if isinstance(p, nn.Parameter):
                p.requires_grad_(True)
    for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        getattr(L.self_attn, n).weight.requires_grad_(True)
    L.self_attn.q_norm.requires_grad_(True)
    L.self_attn.k_norm.requires_grad_(True)
    for n in ["gate", "shared_gate", "shared_gate_proj", "shared_up_proj", "shared_down_proj"]:
        getattr(L.mlp, n).weight.requires_grad_(True)
    for n in ["hc_norm", "down", "up"]:
        p = getattr(model.mixer, n)
        if isinstance(p, nn.Parameter):
            p.requires_grad_(True)


def count_params(model):
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return train, frozen


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_shards(data_dir):
    import glob
    paths = sorted(glob.glob(os.path.join(data_dir, "shard-*.pt")))
    shards = [torch.load(p, map_location="cpu") for p in paths]
    print(f"loaded {len(shards)} shards from {data_dir}")
    return shards


def iter_sequences(shards, min_len=16, seed=0):
    """Yield (input_ids:int64[T], wide:bf16[T,wdim]) for each valid sequence."""
    for sh in shards:
        ids = sh["input_ids"].long()
        wide = sh["wide"]
        lens = sh["lens"].tolist()
        for i, T in enumerate(lens):
            if T < min_len:
                continue
            yield ids[i, :T].contiguous(), wide[i, :T].contiguous()


# --------------------------------------------------------------------------- #
# Eval: greedy pos-1 match vs target next-token
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_match(model, seqs, device, max_seq=None):
    model.eval()
    correct = total = 0
    losses = []
    n_used = 0
    for input_ids, wide in seqs:
        if max_seq and n_used >= max_seq:
            break
        n_used += 1
        T = input_ids.shape[0]
        if T < 2:
            continue
        src = input_ids[:T - 1].to(device)
        tgt = input_ids[1:T].to(device)
        wf = wide[:T - 1].to(device)
        pos = torch.arange(T - 1, device=device, dtype=torch.int64)
        with torch.autocast(device_type=xpu_dtype(device), dtype=torch.bfloat16):
            lg = model.logits(src, wf, pos).float()
        loss = F.cross_entropy(lg, tgt)
        losses.append(loss.item())
        pred = lg.argmax(dim=-1)
        correct += (pred == tgt).sum().item()
        total += tgt.numel()
    model.train()
    return {"match": correct / max(total, 1), "loss": sum(losses) / max(len(losses), 1),
            "n_tokens": total}


def xpu_dtype(device):
    return "xpu" if device.startswith("xpu") else "cuda"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--mode", choices=["fidelity", "train"], default="train")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-len", type=int, default=16)
    ap.add_argument("--eval-n", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default="./checkpoints")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    dt = xpu_dtype(device)

    tc = load_text_config(args.ckpt)
    c = Cfg(tc)
    print(f"cfg: hidden={c.hidden_size} hc={c.hc} heads={c.n_heads}/{c.n_kv} "
          f"hd={c.head_dim} rotary_dim={c.rotary_dim} experts={c.n_experts} "
          f"topk={c.top_k} vocab={c.vocab_size}")

    model = MTPDraftHead(c).to(device=device, dtype=torch.bfloat16)
    load_weights(model, args.ckpt, device)
    freeze_frozen(model)
    tr, fr = count_params(model)
    print(f"params: trainable={tr/1e6:.1f}M frozen={fr/1e6:.1f}M total={(tr+fr)/1e6:.1f}M")

    shards = load_shards(args.data_dir)

    if args.mode == "fidelity":
        seqs = list(iter_sequences(shards, args.min_len))[:args.eval_n]
        r = eval_match(model, seqs, device)
        print(f"FIDELITY (shipped weights): pos-1 match={r['match']:.4f} "
              f"loss={r['loss']:.4f} n_tokens={r['n_tokens']}")
        print("expect ~0.63 match if the reimplementation is faithful.")
        return

    # ---- train ---- #
    os.makedirs(args.out_dir, exist_ok=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # reserve a held-out eval set (last N sequences)
    all_seqs = list(iter_sequences(shards, args.min_len))
    eval_seqs = all_seqs[-args.eval_n:] if len(all_seqs) > args.eval_n else []
    train_seqs = all_seqs[:-args.eval_n] if eval_seqs else all_seqs
    print(f"sequences: train={len(train_seqs)} eval={len(eval_seqs)}")

    global_step = 0
    best_match = -1.0
    t0 = time.time()
    for epoch in range(args.epochs):
        torch.random.manual_seed(args.seed + epoch)
        perm = torch.randperm(len(train_seqs)).tolist()
        epoch_loss, epoch_tok = 0.0, 0
        for si in perm:
            input_ids, wide = train_seqs[si]
            T = input_ids.shape[0]
            if T < 2:
                continue
            src = input_ids[:T - 1].to(device)
            tgt = input_ids[1:T].to(device)
            wf = wide[:T - 1].to(device)
            pos = torch.arange(T - 1, device=device, dtype=torch.int64)
            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dt, dtype=torch.bfloat16):
                lg = model.logits(src, wf, pos).float()
                loss = F.cross_entropy(lg, tgt)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            epoch_loss += loss.item() * (T - 1)
            epoch_tok += (T - 1)
            global_step += 1
            if global_step % 20 == 0:
                el = epoch_loss / max(epoch_tok, 1)
                print(f"ep{epoch} step{global_step} loss={el:.4f} "
                      f"tok/s={epoch_tok/(time.time()-t0+1e-9):.0f}", flush=True)
            if args.eval_every and global_step % args.eval_every == 0 and eval_seqs:
                r = eval_match(model, eval_seqs, device)
                print(f"  EVAL step{global_step}: match={r['match']:.4f} loss={r['loss']:.4f}")
                if r["match"] > best_match:
                    best_match = r["match"]
                    ck = os.path.join(args.out_dir, "best.pt")
                    torch.save({
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "cfg": vars(c), "match": best_match, "step": global_step,
                        "epoch": epoch,
                    }, ck)
                    print(f"  saved best ({best_match:.4f}) -> {ck}")
        print(f"epoch {epoch} done: avg_loss={epoch_loss/max(epoch_tok,1):.4f} "
              f"tokens={epoch_tok:,} wall={time.time()-t0:.0f}s")

    if eval_seqs:
        r = eval_match(model, eval_seqs, device)
        print(f"FINAL: pos-1 match={r['match']:.4f} loss={r['loss']:.4f}")
    print(f"done in {time.time()-t0:.0f}s; best_match={best_match:.4f}; out={args.out_dir}")


if __name__ == "__main__":
    main()
