"""Phase A / A9 debug: lm_head + MoE-gate placement verification.

GDN, dense W4A16, MoE GEMM, HC, M-RoPE, and weight placement (layer 0) are
all cleared. The two remaining high-likelihood suspects for COMPLETE soup:

  1. lm_head  -- untied [248320, 2560] BF16, lives on the ForCausalLM wrapper.
     A wrong/missing lm_head = uniform garbage logits = exactly this soup.
  2. MoE gate -- ReplicatedLinear(2560 -> 512) router. Determines the top-10
     expert selection in all 48 layers. A misplaced gate weight -> wrong
     experts picked -> garbage.

This test loads the full Qwen4ExpForCausalLM (1 layer) on a single XPU and:
  - compares lm_head.weight vs checkpoint (cos)
  - compares mlp.gate.weight + shared_expert_gate.weight vs checkpoint (cos)
  - FUNCTIONAL routing check: run the loaded gate on random hidden states,
    take top-10, and compare the selected expert SETS against a CPU reference
    computed from the checkpoint gate weight. If placement is right, the
    selected sets must be identical (top-k is deterministic).

Single GPU. Run:  python phase_a_lmhead_moe_gate_test.py
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
TOPK = 10
N_TOKENS = 16


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
    return F.cosine_similarity(a[:n], b[:n], dim=0).item()


def main():
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as ps

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

        from vllm.model_executor.models.qwen4_exp import Qwen4ExpForCausalLM

        model = Qwen4ExpForCausalLM(vllm_config=vllm_config)
        print(f"lm_head type = {type(model.lm_head).__name__}")
        print(f"lm_head is embed? {model.lm_head is model.model.embed_tokens}")

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

        # Keep only the tensors this 1-layer wrapper actually has. Yield the
        # ORIGINAL checkpoint names; the ForCausalLM mapper strips
        # `model.language_model.` -> `model.` for routing.
        def filtered(weights):
            for name, tensor in weights:
                if (name.startswith("lm_head.")
                        or name.startswith("model.language_model.embed_tokens.")
                        or name.startswith("model.language_model.layers.0.")
                        or name.startswith("model.language_model.hyper_connection_mixer.")):
                    yield name, tensor

        weights = filtered(
            safetensors_weights_iterator(files, use_tqdm_on_load=True))
        loaded = model.load_weights(weights)
        print(f"loaded {len(loaded)} tensors")

        # Move any CPU params to XPU (standalone loader skips
        # process_weights_after_loading).
        for p in model.parameters():
            if p.device.type != "xpu":
                p.data = p.data.to(DEV)

        # ------------------------------------------------------------------
        # CHECK 1: lm_head placement
        # ------------------------------------------------------------------
        print("\n=== CHECK 1: lm_head placement ===")
        lm = model.lm_head
        # ParallelLMHead stores weight as [vocab, hidden] (TP=1 -> full).
        lm_w = lm.weight.data
        print(f"  lm_head.weight shape = {tuple(lm_w.shape)}")
        ckpt_lm = _load_ckpt("lm_head.weight")
        print(f"  ckpt lm_head shape   = {tuple(ckpt_lm.shape)}")
        c = _cos(lm_w, ckpt_lm)
        print(f"  cos(lm_head, ckpt) = {c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}")

        # ------------------------------------------------------------------
        # CHECK 2: MoE gate + shared_expert_gate placement
        # ------------------------------------------------------------------
        print("\n=== CHECK 2: MoE gate placement ===")
        mlp = model.model.layers[0].mlp
        gate_w = mlp.gate.weight.data
        ckpt_gate = _load_ckpt(
            "model.language_model.layers.0.mlp.gate.weight")
        c = _cos(gate_w, ckpt_gate)
        print(f"  gate: cos={c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}  "
              f"shape={tuple(gate_w.shape)}")

        se_gate_w = mlp.shared_expert_gate.weight.data
        ckpt_se = _load_ckpt(
            "model.language_model.layers.0.mlp.shared_expert_gate.weight")
        c = _cos(se_gate_w, ckpt_se)
        print(f"  shared_expert_gate: cos={c:+.6f}  "
              f"{'OK' if c > 0.999 else 'MISMATCH'}  shape={tuple(se_gate_w.shape)}")

        # ------------------------------------------------------------------
        # CHECK 3: FUNCTIONAL routing (top-10 expert selection)
        # ------------------------------------------------------------------
        print("\n=== CHECK 3: functional MoE routing (top-10) ===")
        torch.manual_seed(0)
        hidden = torch.randn(
            N_TOKENS, mc.hf_text_config.hidden_size,
            dtype=torch.bfloat16, device=DEV)

        # vLLM gate output (router logits) -> top-10
        router_logits, _ = mlp.gate(hidden)          # [T, 512]
        vllm_topk = torch.topk(router_logits, TOPK, dim=-1).indices.cpu()

        # CPU reference from the checkpoint gate weight (true weights).
        hidden_cpu = hidden.cpu().float()
        ref_logits = hidden_cpu @ ckpt_gate.t()      # [T, 512]
        ref_topk = torch.topk(ref_logits, TOPK, dim=-1).indices

        # Compare the selected expert SETS (order-independent).
        match = 0
        for t in range(N_TOKENS):
            if set(vllm_topk[t].tolist()) == set(ref_topk[t].tolist()):
                match += 1
        print(f"  top-10 expert-set agreement = {match}/{N_TOKENS}")
        # Also report how many individual picks agree (order-sensitive).
        exact = int((vllm_topk == ref_topk).all(dim=-1).sum().item())
        print(f"  top-10 exact-order agreement = {exact}/{N_TOKENS}")
        print(f"  vllm top-10[0] = {vllm_topk[0].tolist()}")
        print(f"  ref  top-10[0] = {ref_topk[0].tolist()}")

        print("\n" + "=" * 60)
        lm_ok = _cos(lm_w, ckpt_lm) > 0.999
        gate_ok = _cos(gate_w, ckpt_gate) > 0.999
        route_ok = match == N_TOKENS
        if lm_ok and gate_ok and route_ok:
            print("PASS: lm_head + MoE gate placement + routing all correct.")
            print("-> soup is in the full 48-layer stack (HC flow / EP / attn).")
        else:
            print("FAIL: placement/routing bug found.")
            print(f"  lm_head ok={lm_ok}  gate ok={gate_ok}  routing ok={route_ok}")

        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()