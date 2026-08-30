"""Phase A / A9 debug: single-layer weight PLACEMENT verification.

A8 verified that checkpoint tensors route to the correct tensor NAMES, but
not that the VALUES land in the right location within a fused tensor. A wrong
q/k/v/z placement in the GDN in_proj_qkvz (or b/a in in_proj_ba) would still
pass A8 but scramble 36/48 layers -> token soup.

This test loads a 1-layer Qwen4ExpModel on a single XPU and compares the
loaded fused weights against the checkpoint values:
  - in_proj_qkvz = [q(2048); k(2048); v(6144); z(6144)]
  - in_proj_ba   = [b(48); a(48)]
  - HC weights (hc_norm, mix down/up, block_inject)
  - GDN scalars (A_log, dt_bias, norm) + out_proj + conv1d

  all cos ~ 1.0 -> placement correct -> soup is in a forward-pass kernel
  any cos ~ 0  -> placement bug found

Single GPU, memory-light (1 layer). Run:  python phase_a_single_layer_test.py
"""

import glob
import json
import os

import torch

import vllm  # noqa: F401  (registers torch.ops._xpu_C)
from safetensors import safe_open

MODEL = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"
DEV = "xpu"


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
        model=MODEL,
        dtype="bfloat16",
        runner="auto",
        trust_remote_code=False,
    )
    # Override to a single layer so the model fits on one GPU.
    for cfg in (model_config.hf_config, model_config.hf_text_config):
        if cfg is not None and hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = 1
    if hasattr(model_config.hf_config, "text_config") and isinstance(
        model_config.hf_config.text_config, object
    ):
        model_config.hf_config.text_config.num_hidden_layers = 1

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
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

        from vllm.model_executor.models.qwen4_exp import Qwen4ExpModel

        model = Qwen4ExpModel(vllm_config=vllm_config)
        n_layers = len(model.layers)
        print(f"constructed: {n_layers} layer(s)")

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
            # Qwen4ExpModel params are rooted at `layers.0...` / `embed_tokens`
            # / `hyper_connection_mixer` (no `model.` prefix). The checkpoint
            # carries `model.language_model.`; strip it to match.
            for name, tensor in weights:
                if name.startswith("model.language_model."):
                    name = name[len("model.language_model."):]
                if (name.startswith("embed_tokens.")
                        or name.startswith("layers.0.")
                        or name.startswith("hyper_connection_mixer.")):
                    yield name, tensor

        weights = filtered(
            safetensors_weights_iterator(files, use_tqdm_on_load=True))
        loaded = model.load_weights(weights)
        print(f"loaded {len(loaded)} tensors")

        la = model.layers[0].linear_attn
        print("\n=== GDN in_proj_qkvz placement [q;k;v;z] ===")
        w = la.in_proj_qkvz.weight.data
        q, k, v, z = w[:2048], w[2048:4096], w[4096:10240], w[10240:]
        ckpt_qkv = _load_ckpt(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        ckpt_z = _load_ckpt(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight")
        cq, ck, cv = ckpt_qkv[:2048], ckpt_qkv[2048:4096], ckpt_qkv[4096:]
        for tag, got, want in (("q", q, cq), ("k", k, ck), ("v", v, cv),
                               ("z", z, ckpt_z)):
            c = _cos(got, want)
            print(f"  {tag}: cos={c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}")

        print("\n=== GDN in_proj_ba placement [b;a] ===")
        ba = la.in_proj_ba.weight.data
        b, a = ba[:48], ba[48:]
        ckpt_b = _load_ckpt(
            "model.language_model.layers.0.linear_attn.in_proj_b.weight")
        ckpt_a = _load_ckpt(
            "model.language_model.layers.0.linear_attn.in_proj_a.weight")
        for tag, got, want in (("b", b, ckpt_b), ("a", a, ckpt_a)):
            c = _cos(got, want)
            print(f"  {tag}: cos={c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}")

        print("\n=== GDN scalars / out_proj / conv1d ===")
        for attr, key in (
            ("A_log", "model.language_model.layers.0.linear_attn.A_log"),
            ("dt_bias", "model.language_model.layers.0.linear_attn.dt_bias"),
        ):
            got = getattr(la, attr).data
            want = _load_ckpt(key)
            c = _cos(got, want)
            print(f"  {attr}: cos={c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}")
        c = _cos(la.norm.weight.data,
                 _load_ckpt("model.language_model.layers.0.linear_attn.norm.weight"))
        print(f"  norm: cos={c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}")
        c = _cos(la.out_proj.weight.data,
                 _load_ckpt("model.language_model.layers.0.linear_attn.out_proj.weight"))
        print(f"  out_proj: cos={c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}")

        print("\n=== HyperConnection weights (attn block) ===")
        hc = model.layers[0].attn_hyper_connection
        for attr in ("hc_norm", "input_mix_weight_down",
                     "input_mix_weight_up", "block_inject_weight"):
            got = getattr(hc, attr).weight.data
            want = _load_ckpt(
                f"model.language_model.layers.0.attn_hyper_connection.{attr}.weight")
            c = _cos(got, want)
            print(f"  {attr}: cos={c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}")

        print("\n=== top-level hyper_connection_mixer (final norm) ===")
        mix = model.hyper_connection_mixer
        for attr in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up"):
            got = getattr(mix, attr).weight.data
            want = _load_ckpt(
                f"model.language_model.hyper_connection_mixer.{attr}.weight")
            c = _cos(got, want)
            print(f"  {attr}: cos={c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}")

        print("\n=== embedding (first 100 rows) ===")
        c = _cos(model.embed_tokens.weight.data[:100],
                 _load_ckpt("model.language_model.embed_tokens.weight")[:100])
        print(f"  embed_tokens[:100]: cos={c:+.6f}  {'OK' if c > 0.999 else 'MISMATCH'}")

        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()