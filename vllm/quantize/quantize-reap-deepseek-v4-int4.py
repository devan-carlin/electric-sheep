#!/usr/bin/env python3
"""INT4 quantization of REAP-pruned DeepSeek-V4-Flash model.

Takes the REAP output (192 experts, BF16, ~147GB) and quantizes to INT4.
Expected output: ~75GB (fits in 128GB VRAM across 4x Arc B70).

Uses auto_round with all XPU patches applied.
"""

import os
import re
import signal
import sys
import threading
import time
import torch

print("=" * 60)
print("INT4 Quantization — REAP-pruned DeepSeek-V4-Flash")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"XPU count: {torch.xpu.device_count()}")
    for i in range(torch.xpu.device_count()):
        props = torch.xpu.get_device_properties(i)
        print(f"  XPU {i}: {props.name}, {props.total_memory / 1e9:.0f} GB")
print()

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "/mnt/data/models/DeepSeek-V4-Flash-0731--REAP-192experts"
OUTPUT_DIR = "/mnt/data/models/DeepSeek-V4-Flash-0731--REAP-192experts-INT4"

print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

# ============================================================
# Hang Watchdog
# ============================================================
class HangWatchdog:
    def __init__(self, timeout_minutes=30, output_dir=None):
        self.timeout = timeout_minutes * 60
        self.output_dir = output_dir
        self.last_activity = time.time()
        self.running = True
        self.pid = os.getpid()

    def start(self):
        t = threading.Thread(target=self._watch, daemon=True, name="hang-watchdog")
        t.start()
        print(f"[WATCHDOG] Started (timeout={self.timeout//60}min, PID={self.pid})")

    def _newest_mtime(self, directory):
        newest = 0
        if not directory or not os.path.exists(directory):
            return newest
        for root, dirs, files in os.walk(directory):
            for f in files:
                try:
                    mtime = os.path.getmtime(os.path.join(root, f))
                    newest = max(newest, mtime)
                except OSError:
                    pass
        return newest

    def _watch(self):
        while self.running:
            time.sleep(120)
            now = time.time()
            activity = False

            if self.output_dir:
                newest = self._newest_mtime(self.output_dir)
                if newest > self.last_activity:
                    activity = True

            try:
                total_alloc = sum(torch.xpu.memory_allocated(i)
                                  for i in range(torch.xpu.device_count()))
                if total_alloc == 0:
                    elapsed = now - self.last_activity
                    if elapsed > self.timeout:
                        print(f"\n⚠️  WATCHDOG: All XPU memory released for {elapsed//60:.0f}min!")
                        os.kill(self.pid, signal.SIGKILL)
                        return
            except Exception:
                pass

            if activity:
                self.last_activity = now
            elif now - self.last_activity > self.timeout:
                print(f"\n⚠️  WATCHDOG: No progress for {(now - self.last_activity)//60:.0f}min!")
                os.kill(self.pid, signal.SIGKILL)
                return

    def stop(self):
        self.running = False


watchdog = HangWatchdog(timeout_minutes=30, output_dir=OUTPUT_DIR)
watchdog.start()

# ============================================================
# Apply patches BEFORE importing AutoRound
# ============================================================

# PATCH 1: Disable MoE fused module replacement
print("[PATCH 1] Disabling MoE fused module replacement...")
import auto_round.modeling.fused_moe.replace_modules as replace_mod
_orig_apply = replace_mod.apply_replacements
def patched_apply(model, *args, **kwargs):
    print(">>> Skipping MoE fused replacement (INT4 tensors can't be nn.Parameter)")
    return model
replace_mod.apply_replacements = patched_apply

# PATCH 2: Fix DeepseekV4Attention forward signature
print("[PATCH 2] Fixing DeepseekV4Attention forward signature...")
import transformers.models.deepseek_v4.modeling_deepseek_v4 as dv4_module
_orig_attn_forward = dv4_module.DeepseekV4Attention.forward
def patched_attn_forward(self, hidden_states, **kwargs):
    pe = kwargs.get("position_embeddings")
    if isinstance(pe, tuple):
        kwargs["position_embeddings"] = {self.rope_layer_type: pe}
    elif pe is None or (isinstance(pe, dict) and len(pe) == 0):
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        head_dim = self.config.head_dim if hasattr(self.config, 'head_dim') else 512
        partial_rotary_factor = getattr(self.config, 'partial_rotary_factor', 64/512)
        qk_rope_head_dim = int(head_dim * partial_rotary_factor)
        rope_half_dim = qk_rope_head_dim // 2
        cos = torch.ones(batch_size, seq_len, rope_half_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        sin = torch.zeros(batch_size, seq_len, rope_half_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        kwargs["position_embeddings"] = {self.rope_layer_type: (cos, sin)}
    return _orig_attn_forward(self, hidden_states, **kwargs)
dv4_module.DeepseekV4Attention.forward = patched_attn_forward

# PATCH 3: Fix MoE gate forward signatures
print("[PATCH 3] Fixing MoE gate forward signatures...")
_orig_topk_forward = dv4_module.DeepseekV4TopKRouter.forward
def patched_topk_forward(self, hidden_states, input_ids=None):
    return _orig_topk_forward(self, hidden_states)
dv4_module.DeepseekV4TopKRouter.forward = patched_topk_forward

_orig_hash_forward = dv4_module.DeepseekV4HashRouter.forward
def patched_hash_forward(self, hidden_states, input_ids=None):
    if input_ids is None:
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
    return _orig_hash_forward(self, hidden_states, input_ids)
dv4_module.DeepseekV4HashRouter.forward = patched_hash_forward

# PATCH 4: Force eager MoE + strip FP8 quantization config
print("[PATCH 4] Forcing eager MoE + stripping FP8 quantization config...")
import json
config_path = os.path.join(MODEL_PATH, "config.json")
with open(config_path) as f:
    config = json.load(f)
config["_experts_implementation"] = "eager"
# CRITICAL: Remove FP8 quantization config so auto_round doesn't try FP8 dequantization
# on BF16 weights (causes inf/nan/overflow in loss)
if "quantization_config" in config:
    print(f"  Removing FP8 quantization_config: {config['quantization_config'].get('quant_method', '?')}")
    del config["quantization_config"]
if "torch_dtype" not in config or config["torch_dtype"] is None:
    config["torch_dtype"] = "bfloat16"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print("  config.json updated: eager MoE, FP8 config removed, torch_dtype=bfloat16")

# PATCH 5: Disable deepgemm
print("[PATCH 5] Disabling deepgemm FP8 kernel dispatch...")
import transformers.integrations.finegrained_fp8 as fg_fp8_module
def patched_disable_deepgemm(model):
    pass
fg_fp8_module._disable_deepgemm_on_multi_device = patched_disable_deepgemm
if hasattr(fg_fp8_module, 'FP8Experts'):
    fg_fp8_module.FP8Experts._deepgemm_disabled = True
if hasattr(fg_fp8_module, 'FP8Linear'):
    fg_fp8_module.FP8Linear._deepgemm_disabled = True

# PATCH 6: Fix multi-device tensor mismatch in quant_tensor_sym
print("[PATCH 6] Fixing multi-device tensor mismatch in quant_tensor_sym...")
import auto_round.data_type.int as int_dtype_module
_orig_quant_tensor_sym = int_dtype_module.quant_tensor_sym

def patched_quant_tensor_sym(
    tensor, bits=4, group_size=128, v=0,
    min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
    tensor_min=None, tensor_max=None, q_scale_thresh=1e-5,
    init_scale=None, **kwargs
):
    if isinstance(min_scale, torch.Tensor) and min_scale.device != tensor.device:
        min_scale = min_scale.to(tensor.device)
    if isinstance(max_scale, torch.Tensor) and max_scale.device != tensor.device:
        max_scale = max_scale.to(tensor.device)
    if isinstance(init_scale, torch.Tensor) and init_scale.device != tensor.device:
        init_scale = init_scale.to(tensor.device)
    if isinstance(tensor_min, torch.Tensor) and tensor_min.device != tensor.device:
        tensor_min = tensor_min.to(tensor.device)
    if isinstance(tensor_max, torch.Tensor) and tensor_max.device != tensor.device:
        tensor_max = tensor_max.to(tensor.device)
    return _orig_quant_tensor_sym(
        tensor, bits=bits, group_size=group_size, v=v,
        min_scale=min_scale, max_scale=max_scale,
        scale_dtype=scale_dtype, tensor_min=tensor_min, tensor_max=tensor_max,
        q_scale_thresh=q_scale_thresh, init_scale=init_scale, **kwargs
    )

int_dtype_module.quant_tensor_sym = patched_quant_tensor_sym

# PATCH 7: Fix wrapper forward for multi-device scales
print("[PATCH 7] Fixing wrapper forward for multi-device scales...")
import auto_round.wrapper as wrapper_module
_orig_wa_forward = wrapper_module.WrapperWALayer.forward

def patched_wa_forward(self, x):
    # Run stolen pre_hooks first
    for hook in self._stolen_pre_hooks:
        result = hook(self.orig_layer, (x,))
        if result is not None:
            x = result[0] if isinstance(result, tuple) else result

    import auto_round.envs as envs
    import math
    act_scale = envs.AR_ACT_SCALE
    act_max = self.orig_layer.act_max if hasattr(self.orig_layer, "act_max") else None
    max_scale = self.orig_layer.act_max_scale if math.isclose(act_scale, 1.0, rel_tol=1e-6) else act_scale
    min_scale = self.orig_layer.act_min_scale if math.isclose(act_scale, 1.0, rel_tol=1e-6) else act_scale

    # Move scale tensors to the same device as input x
    if isinstance(min_scale, torch.Tensor) and min_scale.device != x.device:
        min_scale = min_scale.to(x.device)
    if isinstance(max_scale, torch.Tensor) and max_scale.device != x.device:
        max_scale = max_scale.to(x.device)
    if isinstance(act_max, torch.Tensor) and act_max.device != x.device:
        act_max = act_max.to(x.device)

    if act_max is None:
        x, _, _ = self.orig_layer.act_quant_func(
            x,
            bits=self.orig_layer.act_bits,
            group_size=self.orig_layer.act_group_size,
            scale_dtype=self.orig_layer.scale_dtype,
            q_scale_thresh=self.orig_layer.q_scale_thresh,
            data_type=self.orig_layer.act_data_type,
            min_scale=min_scale,
            max_scale=max_scale,
        )
    else:
        x, _, _ = self.orig_layer.act_quant_func(
            x,
            bits=self.orig_layer.act_bits,
            group_size=self.orig_layer.act_group_size,
            scale_dtype=self.orig_layer.scale_dtype,
            q_scale_thresh=self.orig_layer.q_scale_thresh,
            data_type=self.orig_layer.act_data_type,
            act_max=act_max,
        )
    return self.orig_layer.forward(x)

wrapper_module.WrapperWALayer.forward = patched_wa_forward

# PATCH 8: No save_pretrained patch needed (AutoRound handles saving internally in v0.14.2)
print("[PATCH 8] Skipping save_pretrained patch (not needed in auto_round 0.14.2)")

print("\nAll patches applied.\n")

# ============================================================
# Additional patches from reference (patches 9-10)
# ============================================================

# PATCH 9: Fix regex error in revert_checkpoint_conversion_mapping
print("[PATCH 9] Fixing regex error in revert_checkpoint_conversion_mapping...")
import auto_round.utils.common as common_utils
import auto_round.compressors.shard_writer as shard_writer_module

def patched_revert_checkpoint_conversion_mapping(name, key_mapping):
    if "," in name:
        return ",".join(patched_revert_checkpoint_conversion_mapping(part, key_mapping) for part in name.split(","))
    for source_pattern, target_patterns in key_mapping.items():
        if isinstance(target_patterns, str):
            target_patterns = [target_patterns]
        for target_pattern in target_patterns:
            sp = source_pattern.lstrip("^")
            try:
                name, n_replace = re.subn(sp, target_pattern, name)
            except re.error:
                sp_stripped = re.sub(r"\(.*?\)", "", sp)
                if sp_stripped in name:
                    target_stripped = re.sub(r"\\(\d+)", "", target_pattern)
                    name = name.replace(sp_stripped, target_stripped)
                    return name
                continue
            if n_replace > 0:
                return name
    return name

common_utils.revert_checkpoint_conversion_mapping = patched_revert_checkpoint_conversion_mapping
shard_writer_module.revert_checkpoint_conversion_mapping = patched_revert_checkpoint_conversion_mapping

# PATCH 10: Fix OOM in mv_module_from_gpu
print("[PATCH 10] Fixing OOM in mv_module_from_gpu...")
import auto_round.utils.model as model_utils

def patched_mv_module_from_gpu(module):
    """Move module to CPU by moving parameters individually to avoid OOM."""
    if hasattr(module, "device"):
        dev = module.device
        if isinstance(dev, torch.device) and dev.type in ("cpu", "meta"):
            return module
        if isinstance(dev, str) and dev.split(":")[0] in ("cpu", "meta"):
            return module
    has_meta = any(p.device.type == "meta" for p in module.parameters())
    if not has_meta:
        has_meta = any(b.device.type == "meta" for b in module.buffers())
    if has_meta:
        for _, child in module.named_children():
            patched_mv_module_from_gpu(child)
        for attr_name in list(module._parameters.keys()):
            p = module._parameters[attr_name]
            if p is not None and p.device.type not in ("meta", "cpu"):
                module._parameters[attr_name] = torch.nn.Parameter(
                    p.to("cpu", non_blocking=True), requires_grad=p.requires_grad)
        for attr_name in list(module._buffers.keys()):
            b = module._buffers[attr_name]
            if b is not None and b.device.type not in ("meta", "cpu"):
                module._buffers[attr_name] = b.to("cpu", non_blocking=True)
        return module
    for _, child in module.named_children():
        patched_mv_module_from_gpu(child)
    for attr_name in list(module._parameters.keys()):
        p = module._parameters[attr_name]
        if p is not None and p.device.type not in ("meta", "cpu"):
            module._parameters[attr_name] = torch.nn.Parameter(
                p.to("cpu", non_blocking=True), requires_grad=p.requires_grad)
    for attr_name in list(module._buffers.keys()):
        b = module._buffers[attr_name]
        if b is not None and b.device.type not in ("meta", "cpu"):
            module._buffers[attr_name] = b.to("cpu", non_blocking=True)
    return module

model_utils.mv_module_from_gpu = patched_mv_module_from_gpu
import auto_round.compressors.data_driven as data_driven_module
data_driven_module.mv_module_from_gpu = patched_mv_module_from_gpu

# PATCH 11: Eager experts forward (BF16 version — no FP8 dequantization needed)
print("[PATCH 11] Setting up eager experts forward (BF16, no FP8 dequant)...")
import torch.nn.functional as F
import transformers.integrations.moe as moe_module

def _eager_experts_forward_bf16(self, hidden_states, top_k_index, top_k_weights):
    """Direct eager MoE forward for BF16 weights (no FP8 dequantization)."""
    with torch.no_grad():
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat = hidden_states.view(-1, hidden_dim)
        else:
            flat = hidden_states
            batch_size, seq_len = None, None
            hidden_dim = flat.shape[1]
        final = torch.zeros(flat.shape[0], flat.shape[1], dtype=torch.float32, device=flat.device)
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
        mask = mask.permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)
        for expert_idx in hit:
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = flat[token_idx]
            # BF16 weights — no dequantization needed
            if hasattr(self, 'has_gate') and self.has_gate:
                w_gu = self.gate_up_proj[expert_idx]
            else:
                w_gu = self.up_proj[expert_idx]
            if w_gu.shape[1] != current.shape[1]:
                w_gu = w_gu.T
            proj_out = F.linear(current, w_gu)
            if hasattr(self, 'has_gate') and self.has_gate:
                proj_out = self._apply_gate(proj_out)
            else:
                proj_out = self.act_fn(proj_out)
            w_d = self.down_proj[expert_idx]
            if w_d.shape[1] != proj_out.shape[1]:
                w_d = w_d.T
            proj_out = F.linear(proj_out, w_d)
            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            weighted = proj_out * routing_weights.to(proj_out.dtype)
            final.index_add_(0, token_idx, weighted.to(final.dtype))
        if batch_size is not None and seq_len is not None:
            return final.view(batch_size, seq_len, hidden_dim).to(hidden_states.dtype)
        return final.to(hidden_states.dtype)

dv4_module.DeepseekV4Experts.forward = _eager_experts_forward_bf16
if hasattr(fg_fp8_module, 'FP8Experts'):
    fg_fp8_module.FP8Experts.forward = _eager_experts_forward_bf16

def _patched_get_interface(self, experts_implementation, default):
    return _eager_experts_forward_bf16

moe_module.ExpertsInterface.get_interface = _patched_get_interface
if hasattr(fg_fp8_module, 'FP8ExpertsInterface'):
    fg_fp8_module.FP8ExpertsInterface.get_interface = _patched_get_interface

print("\nAll patches applied.\n")

# ============================================================
# Run quantization (matching reference script API)
# ============================================================
from auto_round import AutoRound

print("[1/3] Loading REAP-pruned model on xpu:0...")
t0 = time.time()
model = AutoRound(
    MODEL_PATH,
    scheme="INT4",
    group_size=128,
    sym=True,          # Symmetric quantization (simpler, less prone to overflow)
    iters=0,           # No iterative optimization (RTN only — avoids loss explosion)
    enable_quanted_input=False,  # No activation quantization (avoids FP8→INT4 chain)
    batch_size=4,
    amp=True,
    device_map="xpu:0",
    low_cpu_mem_usage=True,
    seed=42,
)
print(f"  Model loaded in {time.time() - t0:.0f}s")

print()
print("[2/3] Running calibration and quantization...")
print("  This will take a while (30-60 min expected)")
t1 = time.time()
model.quantize_and_save(OUTPUT_DIR)
print(f"  Quantization complete in {time.time() - t1:.0f}s")

watchdog.stop()
print("[WATCHDOG] Stopped — quantization completed successfully")

print()
print("[3/3] Checking output...")
import subprocess
result = subprocess.run(["du", "-sh", OUTPUT_DIR], capture_output=True, text=True)
print(f"  Output: {OUTPUT_DIR}")
print(f"  Output size: {result.stdout.strip()}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"INT4 QUANTIZATION COMPLETE")
print(f"{'=' * 60}")
print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print("\nDone!")