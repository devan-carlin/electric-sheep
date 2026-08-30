"""Phase A / A8: CPU weight-routing dry-run for Qwen4-Exp.

Constructs the full model on CPU (TP=1) and runs ``load_weights`` over the
real W4A16 safetensors checkpoint. Validates that every checkpoint tensor
routes to the correct parameter via the weight mappers + module structure.
This does NOT test XPU GEMM (that is Phase C); it tests weight routing only.

Run with:  VLLM_TARGET_DEVICE=cpu .venv/bin/python phase_a_load_test.py
"""

import os
import sys

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("VLLM_USE_MODELSCOPE", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")

MODEL_PATH = "/mnt/data/models/VnimanieAI-Qwen3.8-Flash-Next-W4A16"


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
        model=MODEL_PATH,
        dtype="bfloat16",
        runner="auto",
        trust_remote_code=False,
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=CacheConfig(),
        scheduler_config=SchedulerConfig.default_factory(),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )
    return vllm_config


def init_distributed():
    from vllm.distributed import parallel_state as ps

    ps.init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        backend="gloo",
    )
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )


def main():
    from vllm.config import set_current_vllm_config

    # Force the CPU platform so the CPU WNA16 kernel is selected (the XPU
    # platform auto-activates on this box and its kernels reject the CPU
    # activation dtype). This is a weight-routing dry-run only.
    import vllm.platforms as platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms.current_platform = CpuPlatform()

    vllm_config = build_vllm_config()
    mc = vllm_config.model_config
    print("hf_config:", type(mc.hf_config).__name__, "model_type:", mc.hf_config.model_type)
    print("quant_config:", type(vllm_config.quant_config).__name__ if vllm_config.quant_config else None)

    with set_current_vllm_config(vllm_config):
        init_distributed()

        from vllm.model_executor.models.registry import ModelRegistry

        arch = mc.hf_config.architectures[0]
        print("architecture:", arch)
        model_cls, resolved = ModelRegistry.resolve_model_cls(arch, mc)
        print("model class:", model_cls.__module__ + "." + model_cls.__name__)

        print("constructing model (allocates ~62GB W4A16 weights on CPU)...")
        model = model_cls(vllm_config=vllm_config)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"total parameters: {n_params:,}")

        from vllm.model_executor.model_loader.weight_utils import (
            safetensors_weights_iterator,
        )

        print("loading weights from checkpoint...")
        weights = safetensors_weights_iterator(MODEL_PATH)
        loaded = model.load_weights(weights)
        print(f"loaded {len(loaded)} weight tensors")

        import glob
        import json

        idx_files = glob.glob(os.path.join(MODEL_PATH, "*.index.json"))
        all_names = set()
        if idx_files:
            wm = json.load(open(idx_files[0]))["weight_map"]
            all_names = set(wm.keys())
        else:
            for f in glob.glob(os.path.join(MODEL_PATH, "*.safetensors")):
                from safetensors import safe_open

                with safe_open(f, framework="pt") as st:
                    all_names.update(st.keys())

        def to_vllm(name):
            if name.startswith("model.language_model."):
                return "model." + name[len("model.language_model."):]
            return name

        should_load = {
            n for n in all_names
            if not n.startswith("mtp.")
            and not n.startswith("model.visual.")
            and ".ple." not in n
        }
        expected_vllm = {to_vllm(n) for n in should_load}
        loaded_norm = set(loaded)
        missing = expected_vllm - loaded_norm
        extra = loaded_norm - expected_vllm
        print(f"checkpoint tensors (excl mtp/visual): {len(should_load)}")
        print(f"loaded: {len(loaded_norm)}")
        print(f"MISSING (expected but not loaded): {len(missing)}")
        for m in sorted(missing)[:40]:
            print("   MISSING:", m)
        print(f"EXTRA (loaded but not expected): {len(extra)}")
        for e in sorted(extra)[:40]:
            print("   EXTRA:", e)

        if not missing:
            print("PASS: all non-mtp/visual checkpoint tensors routed to a parameter")
        else:
            print("FAIL: some tensors did not route")
            sys.exit(1)


if __name__ == "__main__":
    main()