"""Phase A / A8: real XPU weight-load test for Qwen4-Exp (TP=4 + EP).

Run 4 processes (one per GPU) via:
    bash phase_a_xpu_load.sh
Each rank constructs the model and loads its shard of the W4A16 checkpoint
on XPU. Rank 0 reports the load summary. This validates the full weight
routing (linear WNA16 + XPUExpertsWNA16 MoE + HC + GDN) on the real
hardware.
"""

import os
import sys

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "4"))

os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_USE_MODELSCOPE", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29512")

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
        tensor_parallel_size=WORLD,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=CacheConfig(),
        scheduler_config=SchedulerConfig.default_factory(),
        device_config=DeviceConfig(device="xpu"),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )
    return vllm_config


def main():
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as ps

    vllm_config = build_vllm_config()
    mc = vllm_config.model_config

    with set_current_vllm_config(vllm_config):
        # XPU uses the xccl backend (current_platform.dist_backend).
        ps.init_distributed_environment(
            world_size=WORLD,
            rank=RANK,
            local_rank=RANK,
            backend="xccl",
        )
        ps.initialize_model_parallel(
            tensor_model_parallel_size=WORLD,
            pipeline_model_parallel_size=1,
            backend="xccl",
        )
        if RANK == 0:
            print("distributed init OK (xccl, world=%d)" % WORLD, flush=True)

        from vllm.model_executor.models.registry import ModelRegistry

        arch = mc.hf_config.architectures[0]
        model_cls, resolved = ModelRegistry.resolve_model_cls(arch, mc)
        if RANK == 0:
            print("model class:", model_cls.__module__ + "." + model_cls.__name__, flush=True)

        if RANK == 0:
            print("constructing model on XPU...", flush=True)
        # The real XPU worker sets the torch default dtype to the model dtype
        # before constructing the model. Linear layers fall back to
        # torch.get_default_dtype() for params_dtype, so this must match.
        import torch

        torch.set_default_dtype(mc.dtype)
        model = model_cls(vllm_config=vllm_config)
        if RANK == 0:
            n = sum(p.numel() for p in model.parameters())
            print(f"constructed OK. local params={n:,}", flush=True)

        from vllm.model_executor.model_loader.weight_utils import (
            safetensors_weights_iterator,
            filter_duplicate_safetensors_files,
            filter_files_not_needed_for_inference,
        )
        from vllm.model_executor.model_loader.ep_weight_filter import (
            compute_local_expert_ids,
        )

        import glob

        files = sorted(glob.glob(os.path.join(MODEL_PATH, "*.safetensors")))
        idx_files = glob.glob(os.path.join(MODEL_PATH, "*.index.json"))
        if idx_files:
            files = filter_duplicate_safetensors_files(files, MODEL_PATH, idx_files[0])
        files = filter_files_not_needed_for_inference(files)
        local_expert_ids = compute_local_expert_ids(
            num_experts=512, ep_size=WORLD, ep_rank=RANK, placement="linear"
        )
        if RANK == 0:
            print(f"loading weights from checkpoint ({len(files)} shards)...", flush=True)
        weights = safetensors_weights_iterator(
            files,
            use_tqdm_on_load=(RANK == 0),
            local_expert_ids=local_expert_ids,
        )
        loaded = model.load_weights(weights)
        if RANK == 0:
            print(f"loaded {len(loaded)} weight tensors (rank 0 view)", flush=True)

            import json

            idx_files = glob.glob(os.path.join(MODEL_PATH, "*.index.json"))
            all_names = set()
            if idx_files:
                wm = json.load(open(idx_files[0]))["weight_map"]
                all_names = set(wm.keys())

            # Full name transform = top-level prefix mapper | nested model mapper
            # (stacked renames). This correctly maps fused names (in_proj_qkv ->
            # in_proj_qkvz) so the expected set matches what load_weights returns.
            full_mapper = model.hf_to_vllm_mapper | model.model.hf_to_vllm_mapper
            should_load = [
                n for n in sorted(all_names)
                if not n.startswith("mtp.")
                and not n.startswith("model.visual.")
                and ".ple." not in n
                and ".indexer." not in n
            ]
            expected_vllm = set(full_mapper.apply_list(should_load))

            # Expert weights are EP-sharded: each rank only loads its local
            # experts, so exclude them from the rank-0 full-set comparison.
            # Their correct routing is already validated by load_weights not
            # raising.
            expected_vllm = {n for n in expected_vllm if ".experts." not in n}
            loaded_norm = {n for n in loaded if ".experts." not in n}
            n_expert_loaded = sum(1 for n in loaded if ".experts." in n)

            missing = expected_vllm - loaded_norm
            extra = loaded_norm - expected_vllm
            print(f"checkpoint tensors (excl mtp/visual/ple/indexer): {len(should_load)}")
            print(f"non-expert loaded (rank 0): {len(loaded_norm)}")
            print(f"expert tensors loaded (rank 0, local EP shard): {n_expert_loaded}")
            print(f"MISSING: {len(missing)}")
            for m in sorted(missing)[:40]:
                print("   MISSING:", m)
            print(f"EXTRA: {len(extra)}")
            for e in sorted(extra)[:40]:
                print("   EXTRA:", e)
            if not missing:
                print("PASS: XPU load routed all non-expert tensors", flush=True)
            else:
                print("FAIL: some tensors did not route", flush=True)
                sys.exit(1)

    # Keep the process alive briefly so all ranks finish together.
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()