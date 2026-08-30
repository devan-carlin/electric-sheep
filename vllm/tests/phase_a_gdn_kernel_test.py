"""Phase A / A9 debug: XPU GDN kernel isolation test.

M-RoPE is ruled out (standard RoPE gives the identical soup). The prime
suspect is the XPU GDN kernel `torch.ops._xpu_C.gdn_attention`, specifically
its `reorder_input=True` (non-interleaved / Qwen3.5) path. Layer 0 is GDN,
so a broken GDN corrupts the residual stream from the first token -> soup.

The kernel takes `projected_states_qkvz` [T, 16384] = [q;k;v;z] and
`projected_states_ba` [T, 96] = [b;a], and outputs `core_attn_out` and `z`.
In the non-interleaved layout, `z` = the last 6144 dims of qkvz, reshaped to
[T, 48, 128]. The kernel extracts z internally; if `reorder_input` is buggy,
z comes out scrambled.

This test:
  1. loads a 1-layer model on XPU (real GDN weights + kv_cache)
  2. runs in_proj on random tokens -> projected_qkvz, projected_ba
  3. calls the RAW XPU kernel op directly (no forward context needed)
  4. compares the kernel's `z` output vs the expected z (pure layout check)
  5. compares `core_attn_out` vs a pure-Python delta-rule reference

  z matches + core matches -> GDN kernel is fine -> soup is elsewhere
  z mismatched             -> reorder_input layout bug is the root cause

Single GPU. Run:  python phase_a_gdn_kernel_test.py
"""

import glob
import json
import os

import torch
import torch.nn.functional as F

import vllm  # noqa: F401  (registers torch.ops._xpu_C)
from safetensors import safe_open

MODEL = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
DEV = "xpu"
T = 8  # tokens in the single prefill sequence


def build_vllm_config():
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model_config = ModelConfig(
        model=MODEL, dtype="bfloat16", runner="auto",
        trust_remote_code=False,
    )
    for cfg in (model_config.hf_config, model_config.hf_text_config):
        if cfg is not None and hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = 1
    if isinstance(model_config.hf_config.text_config, object):
        model_config.hf_config.text_config.num_hidden_layers = 1

    parallel_config = ParallelConfig(
        tensor_parallel_size=1, pipeline_parallel_size=1,
        enable_expert_parallel=False,
    )
    return VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=CacheConfig(),
        scheduler_config=SchedulerConfig.default_factory(),
        device_config=DeviceConfig(device=DEV),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )


def _load_ckpt(key):
    idx = json.load(open(glob.glob(MODEL + "/*.index.json")[0]))
    wm = idx["weight_map"]
    f = os.path.join(MODEL, wm[key])
    with safe_open(f, framework="pt") as sf:
        return sf.get_tensor(key).cpu().float()


def _cos(a, b):
    a = a.detach().cpu().float().flatten()
    b = b.detach().cpu().float().flatten()
    n = min(a.numel(), b.numel())
    return torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0).item()


def pure_python_gdn(layer, projected_qkvz, projected_ba, T):
    """Pure-Python GDN reference on CPU. Returns (core_out, z) in float32.

    Mirrors the non-interleaved (Qwen3.5) layout:
      qkvz = [q(2048); k(2048); v(6144); z(6144)]
      ba   = [b(48); a(48)]
    conv1d (silu) on the qkv part only, l2norm q/k, gating, delta rule.
    Returns the RAW recurrent output (the z-gate is applied by forward_xpu
    after the kernel, so it must not be applied here).
    """
    dev = "cpu"
    qkvz = projected_qkvz.to(dev).float()   # [T, 16384]
    ba = projected_ba.to(dev).float()       # [T, 96]

    num_k = layer.num_k_heads      # 16
    num_v = layer.num_v_heads      # 48
    hk = layer.head_k_dim          # 128
    hv = layer.head_v_dim          # 128
    q_dim = num_k * hk             # 2048
    v_dim = num_v * hv             # 6144

    qkv = qkvz[:, : q_dim * 2 + v_dim]   # [T, 10240]  (q;k;v)
    z = qkvz[:, q_dim * 2 + v_dim:]      # [T, 6144]   (z)

    # conv1d (causal, kernel 4, silu) on qkv, per channel
    w = layer.conv1d.weight.data.to(dev).float()  # [10240, 1, 4]
    w2d = w.view(w.shape[0], w.shape[2])          # [10240, 4]
    x = qkv.transpose(0, 1).unsqueeze(0)          # [1, 10240, T]
    xp = F.pad(x, (3, 0))                         # [1, 10240, T+3]
    conv = F.conv1d(xp, w2d.unsqueeze(1), groups=w2d.shape[0])  # [1,10240,T]
    conv = F.silu(conv).transpose(1, 2).squeeze(0)              # [T, 10240]

    # split q/k/v
    q = conv[:, :q_dim].view(T, num_k, hk)
    k = conv[:, q_dim:q_dim * 2].view(T, num_k, hk)
    v = conv[:, q_dim * 2:].view(T, num_v, hv)

    # l2norm q, k (per head)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    # gating: g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)
    b = ba[:, :num_v]
    a = ba[:, num_v:]
    A_log = layer.A_log.data.to(dev).float()      # [48]
    dt_bias = layer.dt_bias.data.to(dev).float()  # [48]
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)   # [T, 48]
    beta = torch.sigmoid(b)                        # [T, 48]

    # GQA: each k-head serves (num_v // num_k) v-heads. Broadcast q,k to v.
    rep = num_v // num_k
    q = q.repeat_interleave(rep, dim=1)           # [T, 48, 128]
    k = k.repeat_interleave(rep, dim=1)           # [T, 48, 128]

    # gated delta rule (recurrent), per v-head, float32.
    # State S = h^T, shape [head, v_dim, k_dim] (h is the k -> v map).
    #   pred  = S @ k            (v-space prediction)
    #   delta = beta * (v - pred)
    #   S     = S * decay + delta (x) k
    #   out   = S @ q
    state = torch.zeros(num_v, hv, hk, device=dev, dtype=torch.float32)
    outs = []
    for t in range(T):
        qt = q[t]                                  # [48, 128]
        kt = k[t]                                  # [48, 128]
        vt = v[t]                                  # [48, 128]
        gt = g[t]                                  # [48]
        bt = beta[t].unsqueeze(-1)                 # [48, 1]
        decay = torch.exp(gt).unsqueeze(-1).unsqueeze(-1)  # [48,1,1]
        pred = torch.bmm(state, kt.unsqueeze(-1)).squeeze(-1)  # [48,128]
        delta = bt * (vt - pred)                   # [48, 128]
        state = state * decay + torch.bmm(
            delta.unsqueeze(-1), kt.unsqueeze(1))  # [48, v, k]
        out_t = torch.bmm(state, qt.unsqueeze(-1)).squeeze(-1)  # [48,128]
        outs.append(out_t)
    core = torch.stack(outs, dim=0)                # [T, 48, 128]
    return core, z


def selfcheck_delta_rule():
    """Hand-verifiable T=1 mechanics check of the recurrent loop.

    Zero initial state, decay=1:  S = delta (x) k,  out = S @ q
    =>  out1 = beta1 * v1 * (k1 . q1)
    """
    torch.manual_seed(1)
    H, HK, HV = 2, 3, 4
    q = F.normalize(torch.randn(1, H, HK), dim=-1)
    k = F.normalize(torch.randn(1, H, HK), dim=-1)
    v = torch.randn(1, H, HV)
    beta = torch.rand(1, H)
    state = torch.zeros(H, HV, HK)
    pred = torch.bmm(state, k[0].unsqueeze(-1)).squeeze(-1)
    delta = beta[0].unsqueeze(-1) * (v[0] - pred)
    state = state + torch.bmm(delta.unsqueeze(-1), k[0].unsqueeze(1))
    out = torch.bmm(state, q[0].unsqueeze(-1)).squeeze(-1)
    expected = beta[0].unsqueeze(-1) * v[0] * (k[0] * q[0]).sum(-1, keepdim=True)
    err = (out - expected).abs().max().item()
    ok = err < 1e-5
    print(f"selfcheck T=1 max|err| = {err:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as ps

    if not selfcheck_delta_rule():
        raise SystemExit("delta-rule selfcheck failed; reference is broken")

    vllm_config = build_vllm_config()
    mc = vllm_config.model_config
    print(f"num_hidden_layers={mc.hf_text_config.num_hidden_layers}")

    with set_current_vllm_config(vllm_config):
        ps.init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="xccl")
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            backend="xccl")
        torch.set_default_dtype(mc.dtype)

        from vllm.model_executor.models.qwen4_exp import Qwen4ExpModel

        model = Qwen4ExpModel(vllm_config=vllm_config)
        la = model.layers[0].linear_attn
        print(f"GDN: k_heads={la.num_k_heads} v_heads={la.num_v_heads} "
              f"hk={la.head_k_dim} hv={la.head_v_dim} "
              f"interleaved={la.gqa_interleaved_layout}")

        # Load layer-0 weights (GDN + HC + embed).
        from vllm.model_executor.model_loader.weight_utils import (
            filter_duplicate_safetensors_files,
            filter_files_not_needed_for_inference,
            safetensors_weights_iterator,
        )

        files = sorted(glob.glob(os.path.join(MODEL, "*.safetensors")))
        idx_files = glob.glob(os.path.join(MODEL, "*.index.json"))
        if idx_files:
            files = filter_duplicate_safetensors_files(files, MODEL, idx_files[0])
        files = filter_files_not_needed_for_inference(files)

        def filtered(weights):
            for name, tensor in weights:
                if name.startswith("model.language_model."):
                    name = name[len("model.language_model."):]
                if (name.startswith("embed_tokens.")
                        or name.startswith("layers.0.")):
                    yield name, tensor

        weights = filtered(
            safetensors_weights_iterator(files, use_tqdm_on_load=True))
        model.load_weights(weights)

        # The standalone loader does not run process_weights_after_loading,
        # so some params (e.g. the unquantized in_proj) may still be on CPU.
        # Move everything to the XPU so the forward runs on-device.
        for p in model.parameters():
            if p.device.type != "xpu":
                p.data = p.data.to(DEV)
        for buf in model.buffers():
            if buf is not None and buf.device.type != "xpu":
                buf.data = buf.data.to(DEV)

        # Allocate the GDN kv_cache: [conv_state, ssm_state].
        # conv_state: SD layout [num_slots, 3, conv_dim=10240]
        # ssm_state:  [num_slots, 48, 128, 128]
        num_slots = 4
        conv_dim = la.key_dim * 2 + la.value_dim  # 10240
        conv_state = torch.zeros(
            num_slots, la.conv_kernel_size - 1, conv_dim,
            dtype=torch.bfloat16, device=DEV)
        ssm_state = torch.zeros(
            num_slots, la.num_v_heads, la.head_v_dim, la.head_k_dim,
            dtype=torch.float32, device=DEV)
        la.kv_cache = (conv_state, ssm_state)

        # Random input tokens -> in_proj projections.
        torch.manual_seed(0)
        hidden = torch.randn(T, mc.hf_text_config.hidden_size,
                             dtype=torch.bfloat16, device=DEV)
        projected_qkvz, _ = la.in_proj_qkvz(hidden)   # [T, 16384]
        projected_ba, _ = la.in_proj_ba(hidden)       # [T, 96]
        print(f"projected_qkvz {tuple(projected_qkvz.shape)}  "
              f"projected_ba {tuple(projected_ba.shape)}")

        # Expected z = last 6144 dims of qkvz, reshaped [T, 48, 128].
        q_dim = la.key_dim
        v_dim = la.value_dim
        z_start = q_dim * 2 + v_dim
        expected_z = projected_qkvz[:, z_start:].reshape(
            T, la.num_v_heads, la.head_v_dim)

        # Call the RAW XPU kernel op directly (single prefill sequence).
        core_out = torch.zeros(
            T, la.num_v_heads, la.head_v_dim,
            dtype=torch.bfloat16, device=DEV)
        z_out = torch.empty_like(core_out)
        conv_weights = la.conv1d.weight.view(
            la.conv1d.weight.size(0), la.conv1d.weight.size(2))

        torch.ops._xpu_C.gdn_attention(
            core_out,
            z_out,
            projected_qkvz,
            projected_ba,
            la.num_k_heads,
            la.num_v_heads,
            la.head_k_dim,
            la.head_v_dim,
            conv_state=conv_state,
            ssm_state=ssm_state,
            conv_weights=conv_weights,
            conv_bias=la.conv1d.bias,
            activation=la.activation,
            A_log=la.A_log,
            dt_bias=la.dt_bias,
            num_prefills=1,
            num_decodes=0,
            num_spec_decodes=0,
            has_initial_state=torch.zeros(1, dtype=torch.bool, device=DEV),
            non_spec_query_start_loc=torch.tensor(
                [0, T], dtype=torch.int32, device=DEV),
            non_spec_token_indx=torch.arange(
                T, dtype=torch.int32, device=DEV),
            non_spec_state_indices_tensor=torch.zeros(
                1, dtype=torch.int32, device=DEV),
            spec_query_start_loc=None,
            spec_token_indx=None,
            spec_state_indices_tensor=None,
            num_accepted_tokens=None,
            num_actual_tokens=T,
            tp_size=1,
            reorder_input=not la.gqa_interleaved_layout,
        )
        torch.xpu.synchronize()

        print("\n=== CHECK 1: z extraction (pure layout, no delta rule) ===")
        cz = _cos(z_out, expected_z)
        print(f"  cos(z_kernel, z_expected) = {cz:+.6f}  "
              f"{'OK' if cz > 0.999 else 'MISMATCH'}")
        print(f"  z_kernel[:3]  = {z_out.flatten()[:3].tolist()}")
        print(f"  z_expected[:3]= {expected_z.flatten()[:3].tolist()}")

        print("\n=== CHECK 2: core_attn_out vs pure-Python delta rule ===")
        ref_core, _ = pure_python_gdn(la, projected_qkvz, projected_ba, T)
        cc = _cos(core_out, ref_core)
        print(f"  cos(core_kernel, core_ref) = {cc:+.6f}  "
              f"{'OK' if cc > 0.99 else 'MISMATCH'}")
        print(f"  core_kernel[:6] = {core_out.flatten()[:6].tolist()}")
        print(f"  core_ref[:6]    = {ref_core.flatten()[:6].tolist()}")

        print("\n" + "=" * 60)
        z_ok = cz > 0.999
        c_ok = cc > 0.99
        if z_ok and c_ok:
            print("PASS: XPU GDN kernel matches reference (z + core).")
            print("-> GDN is fine; soup is elsewhere (MoE routing? HC flow?).")
        elif not z_ok:
            print("FAIL: z extraction is WRONG -> reorder_input layout bug.")
            print("-> This is the root cause of the token soup.")
        else:
            print("z is correct but core_attn_out mismatches the reference.")
            print("-> delta-rule / conv / gating bug in the XPU kernel.")

        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()