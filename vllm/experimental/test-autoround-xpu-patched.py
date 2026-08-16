#!/usr/bin/env python3
"""Test auto_round INT4 quantization on XPU with all patches applied.

Includes hang watchdog that kills the process if no progress is detected
for 30+ minutes (catches XPU driver context resets / spin loops).
"""

import os
import re
import signal
import sys
import threading
import time
import torch

print(f"PyTorch: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
print()

# ============================================================
# Hang Watchdog — kills process if no progress for 30+ minutes
# ============================================================
class HangWatchdog:
    """Background thread that monitors quantization progress.
    
    Detects silent hangs by checking:
    1. Output directory file modification times (shard writes)
    2. XPU memory allocation (all zero = GPU context lost)
    3. Stderr/stdout flush timestamps
    
    Kills the process with SIGKILL if no activity for timeout_minutes.
    """
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
        """Get the most recent file modification time in a directory tree."""
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
            time.sleep(120)  # Check every 2 minutes
            now = time.time()
            activity = False

            # Check 1: Output directory has new/modified files
            if self.output_dir:
                newest = self._newest_mtime(self.output_dir)
                if newest > self.last_activity:
                    activity = True

            # Check 2: XPU memory still allocated (context alive)
            try:
                total_alloc = sum(torch.xpu.memory_allocated(i) 
                                  for i in range(torch.xpu.device_count()))
                if total_alloc == 0:
                    # All GPUs idle — likely hung if we should be quantizing
                    elapsed = now - self.last_activity
                    if elapsed > self.timeout:
                        print(f"\n⚠️  WATCHDOG: All XPU memory released for {elapsed//60:.0f}min!")
                        print(f"   GPU context likely lost. Killing PID {self.pid}")
                        os.kill(self.pid, signal.SIGKILL)
                        return
            except Exception:
                pass  # XPU query failed, skip this check

            if activity:
                self.last_activity = now
            elif now - self.last_activity > self.timeout:
                print(f"\n⚠️  WATCHDOG: No progress for {(now - self.last_activity)//60:.0f}min!")
                print(f"   Output dir: {self.output_dir}")
                print(f"   Killing PID {self.pid}")
                os.kill(self.pid, signal.SIGKILL)
                return

    def stop(self):
        self.running = False


MODEL_PATH = "/home/dc/electric-sheep/models/DeepSeek-V4-Flash-0731-Abliterated-FP8"
OUTPUT_DIR = "/mnt/data/models/DeepSeek-V4-Flash-0731-Abliterated-INT4-xpu"

print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

# Start watchdog early so it catches hangs during model loading too
watchdog = HangWatchdog(timeout_minutes=30, output_dir=OUTPUT_DIR)
watchdog.start()

# ============================================================
# Apply patches BEFORE importing AutoRound
# ============================================================

# PATCH 1: Disable MoE fused module replacement (INT4 tensors can't be nn.Parameter)
print("[PATCH 1] Disabling MoE fused module replacement...")
import auto_round.modeling.fused_moe.replace_modules as replace_mod
_orig_apply = replace_mod.apply_replacements
def patched_apply(model, *args, **kwargs):
    print(">>> Skipping MoE fused replacement (INT4 tensors can't be nn.Parameter)")
    return model
replace_mod.apply_replacements = patched_apply

# PATCH 2: Fix DeepseekV4Attention forward signature (position_embeddings dict format)
print("[PATCH 2] Fixing DeepseekV4Attention forward signature...")
import transformers.models.deepseek_v4.modeling_deepseek_v4 as dv4_module
_orig_attn_forward = dv4_module.DeepseekV4Attention.forward
def patched_attn_forward(self, hidden_states, **kwargs):
    pe = kwargs.get("position_embeddings")
    # Handle tuple format (cos, sin) -> wrap in dict with the layer's rope type
    if isinstance(pe, tuple):
        kwargs["position_embeddings"] = {self.rope_layer_type: pe}
    elif pe is None or (isinstance(pe, dict) and len(pe) == 0):
        # Missing or empty dict -> construct dummy position embeddings (identity RoPE)
        # Real DeepseekV4RotaryEmbedding produces cos/sin with shape [B, seq_len, rope_half_dim]
        # We need to match this 3D shape for correct broadcasting in apply_rotary_pos_emb
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        head_dim = self.config.head_dim if hasattr(self.config, 'head_dim') else 512
        partial_rotary_factor = getattr(self.config, 'partial_rotary_factor', 64/512)
        qk_rope_head_dim = int(head_dim * partial_rotary_factor)
        rope_half_dim = qk_rope_head_dim // 2  # 32 for this model
        # cos: [B, seq_len, rope_half_dim] with ones (identity rotation), sin: zeros
        cos = torch.ones(batch_size, seq_len, rope_half_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        sin = torch.zeros(batch_size, seq_len, rope_half_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        kwargs["position_embeddings"] = {self.rope_layer_type: (cos, sin)}
    return _orig_attn_forward(self, hidden_states, **kwargs)
dv4_module.DeepseekV4Attention.forward = patched_attn_forward

# PATCH 3: Fix MoE gate forward signatures (TopKRouter/HashRouter ignore input_ids)
print("[PATCH 3] Fixing MoE gate forward signatures...")
_orig_topk_forward = dv4_module.DeepseekV4TopKRouter.forward
def patched_topk_forward(self, hidden_states, input_ids=None):
    return _orig_topk_forward(self, hidden_states)
dv4_module.DeepseekV4TopKRouter.forward = patched_topk_forward

_orig_hash_forward = dv4_module.DeepseekV4HashRouter.forward
def patched_hash_forward(self, hidden_states, input_ids=None):
    # HashRouter needs input_ids for tid2eid lookup
    if input_ids is None:
        # Create dummy input_ids (all zeros) - shape [B, S]
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
    return _orig_hash_forward(self, hidden_states, input_ids)
dv4_module.DeepseekV4HashRouter.forward = patched_hash_forward

# PATCH 4: Force eager MoE implementation
print("[PATCH 4] Forcing eager MoE implementation...")
import json
config_path = os.path.join(MODEL_PATH, "config.json")
with open(config_path) as f:
    config = json.load(f)
config["_experts_implementation"] = "eager"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print("  config.json updated: _experts_implementation = 'eager'")

# PATCH 5: Disable deepgemm FP8 kernel dispatch (force Triton fallback)
print("[PATCH 5] Disabling deepgemm FP8 kernel dispatch...")
import transformers.integrations.finegrained_fp8 as fg_fp8_module
def patched_disable_deepgemm(model):
    # No-op — deepgemm is already disabled
    pass
fg_fp8_module._disable_deepgemm_on_multi_device = patched_disable_deepgemm
# Also set _deepgemm_disabled on FP8 classes
if hasattr(fg_fp8_module, 'FP8Experts'):
    fg_fp8_module.FP8Experts._deepgemm_disabled = True
if hasattr(fg_fp8_module, 'FP8Linear'):
    fg_fp8_module.FP8Linear._deepgemm_disabled = True

# PATCH 7: Patch FP8 grouped matmul to prevent autograd errors
print("[PATCH 7] Patching FP8 grouped matmul to prevent autograd errors...")

# The FP8 grouped matmul on XPU is registered as a custom op without a backward formula.
# We need to patch the kernel source file to wrap the call in no_grad.
_fp8_kernel_file = '/home/dc/.cache/huggingface/hub/kernels--kernels-community--finegrained-fp8/snapshots/7cdb05d472d6c954c7d03182ed836ebfd4610df0/build/torch-xpu/grouped.py'

if os.path.exists(_fp8_kernel_file):
    # Read the kernel file
    with open(_fp8_kernel_file, 'r') as f:
        _fp8_kernel_content = f.read()
    
    # Check if already patched
    if '# PATCHED: Wrapped in no_grad to prevent autograd errors' not in _fp8_kernel_content:
        # Patch the w8a8_block_dynamic_fp8_matmul_grouped function
        # The function calls ops.w8a8_block_dynamic_fp8_matmul_grouped(...)
        # We wrap this call in torch.no_grad()
        _old_call = """    return ops.w8a8_block_dynamic_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size, output_dtype
    )"""
        _new_call = """    # PATCHED: Wrapped in no_grad to prevent autograd errors
    with torch.no_grad():
        return ops.w8a8_block_dynamic_fp8_matmul_grouped(
            A, B, Bs, offsets, tokens_per_expert, block_size, output_dtype
        )"""
        
        if _old_call in _fp8_kernel_content:
            _fp8_kernel_content = _fp8_kernel_content.replace(_old_call, _new_call)
            with open(_fp8_kernel_file, 'w') as f:
                f.write(_fp8_kernel_content)
            print(f"  Patched {_fp8_kernel_file}")
        else:
            print(f"  WARNING: Could not find target code in {_fp8_kernel_file}")
    else:
        print(f"  Kernel already patched")
else:
    print(f"  WARNING: Kernel file not found: {_fp8_kernel_file}")

# PATCH 8: Fix multi-device tensor mismatch in quant_tensor_sym
print("[PATCH 8] Fixing multi-device tensor mismatch in quant_tensor_sym...")
import auto_round.data_type.int as int_dtype_module
_orig_quant_tensor_sym = int_dtype_module.quant_tensor_sym

def patched_quant_tensor_sym(
    tensor,
    bits=4,
    group_size=128,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    init_scale=None,
    **kwargs
):
    """Patched version that ensures scale tensors are on the same device as the input tensor."""
    # Ensure all tensor arguments are on the same device as the input tensor
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
print("  Patched quant_tensor_sym to handle multi-device tensors")

# Also patch the wrapper forward to move scales to input device
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
print("  Patched WrapperWALayer.forward to handle multi-device scales")

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
            sp = source_pattern.lstrip("^")  # strip off un-needed chars and patterns
            # Don't strip capture groups - they're needed for backreferences in target
            try:
                name, n_replace = re.subn(sp, target_pattern, name)
            except re.error:
                # If regex fails, try simple string replacement
                sp_stripped = re.sub(r"\(.*?\)", "", sp)
                if sp_stripped in name:
                    target_stripped = re.sub(r"\\(\d+)", "", target_pattern)
                    name = name.replace(sp_stripped, target_stripped)
                    return name
                continue
            if n_replace > 0:
                return name
    return name

# Patch in both modules since shard_writer imports it directly
common_utils.revert_checkpoint_conversion_mapping = patched_revert_checkpoint_conversion_mapping
shard_writer_module.revert_checkpoint_conversion_mapping = patched_revert_checkpoint_conversion_mapping
print("  Patched revert_checkpoint_conversion_mapping in common_utils and shard_writer")

# PATCH 10: Fix OOM in mv_module_from_gpu by moving parameters individually
print("[PATCH 10] Fixing OOM in mv_module_from_gpu...")
import auto_round.utils.model as model_utils
_orig_mv_module_from_gpu = model_utils.mv_module_from_gpu

def _get_device_type(device_or_str):
    """Get device type string from torch.device or string like 'xpu:0'."""
    if isinstance(device_or_str, torch.device):
        return device_or_str.type
    if isinstance(device_or_str, str):
        return device_or_str.split(":")[0]
    return None

def patched_mv_module_from_gpu(module):
    """Move module to CPU by moving parameters individually to avoid OOM."""
    if hasattr(module, "device"):
        dev_type = _get_device_type(module.device)
        if dev_type in ("cpu", "meta"):
            return module

    has_meta = any(p.device.type == "meta" for p in module.parameters())
    if not has_meta:
        has_meta = any(b.device.type == "meta" for b in module.buffers())

    if has_meta:
        for _, child in module.named_children():
            patched_mv_module_from_gpu(child)
        for attr_name in list(module._parameters.keys()):
            p = module._parameters[attr_name]
            if p is not None and p.device.type != "meta" and p.device.type != "cpu":
                module._parameters[attr_name] = torch.nn.Parameter(p.to("cpu", non_blocking=True), requires_grad=p.requires_grad)
        for attr_name in list(module._buffers.keys()):
            b = module._buffers[attr_name]
            if b is not None and b.device.type != "meta" and b.device.type != "cpu":
                module._buffers[attr_name] = b.to("cpu", non_blocking=True)
        return module

    # Move parameters individually instead of module.to("cpu") to avoid OOM
    for _, child in module.named_children():
        patched_mv_module_from_gpu(child)
    for attr_name in list(module._parameters.keys()):
        p = module._parameters[attr_name]
        if p is not None and p.device.type != "meta" and p.device.type != "cpu":
            module._parameters[attr_name] = torch.nn.Parameter(p.to("cpu", non_blocking=True), requires_grad=p.requires_grad)
    for attr_name in list(module._buffers.keys()):
        b = module._buffers[attr_name]
        if b is not None and b.device.type != "meta" and b.device.type != "cpu":
            module._buffers[attr_name] = b.to("cpu", non_blocking=True)
    return module

model_utils.mv_module_from_gpu = patched_mv_module_from_gpu
# Also patch in data_driven which imports it directly
import auto_round.compressors.data_driven as data_driven_module
data_driven_module.mv_module_from_gpu = patched_mv_module_from_gpu
print("  Patched mv_module_from_gpu in model_utils and data_driven")

# PATCH 6: Define FP8 dequantization helper and patch experts forward
print("[PATCH 6] Forcing eager experts forward + FP8 dequantization...")
import transformers.integrations.moe as moe_module
import torch.nn.functional as F

def _dequantize_fp8_weight(weight, weight_scale_inv, block_size):
    """
    Dequantize FP8 weight to BF16 using block-wise scales.
    weight: [N, K] FP8 tensor (single expert slice)
    weight_scale_inv: various shapes depending on block granularity
    block_size: (block_n, block_k) e.g. (128, 128)
    """
    if weight is None or weight.element_size() > 1:
        return weight
    # Detach from computation graph to prevent backward through FP8 ops
    w = weight.detach().to(torch.bfloat16)
    s = weight_scale_inv.detach().to(torch.bfloat16) if weight_scale_inv is not None else None
    # Handle 3D+ scales by squeezing trailing singleton dimensions
    while s is not None and s.dim() > 2 and s.shape[-1] == 1:
        s = s.squeeze(-1)
    n, k = w.shape
    if s is None:
        return w
    if s.shape == w.shape:
        return w * s
    # Format 1: [N, K/block_k] - per-row, block-wise along K
    if s.dim() == 2 and s.shape[0] == n and k % s.shape[1] == 0:
        block_k = k // s.shape[1]
        s_expanded = s.unsqueeze(-1).expand(-1, -1, block_k).reshape(n, k)
        return w * s_expanded
    # Format 2: [N/block_n, K] - block-wise along N, per-column
    if s.dim() == 2 and s.shape[1] == k and n % s.shape[0] == 0:
        block_n = n // s.shape[0]
        s_expanded = s.unsqueeze(1).expand(-1, block_n, -1).reshape(n, k)
        return w * s_expanded
    # Format 3: [N/block_n, K/block_k] - full block-wise
    if s.dim() == 2:
        sn, sk = s.shape
        if n % sn == 0 and k % sk == 0:
            block_n, block_k = n // sn, k // sk
            s_expanded = s.unsqueeze(-1).unsqueeze(-1).expand(sn, block_n, sk, block_k).reshape(n, k)
            return w * s_expanded
    # Format 4: Per-row scale [N] or [N, 1]
    if s.shape[0] == n:
        return w * s.view(n, 1)
    # Format 5: Per-column scale [K] or [1, K]
    if s.numel() == k:
        return w * s.view(1, k)
    # Fallback
    return w

# CRITICAL: The @register_experts decorator at moe.py:576 creates a wrapper that calls
# get_interface() -> experts_forward(self, ...). The wrapper captures original_forward
# in its closure. When get_interface returns default (original_forward), it's the
# ORIGINAL forward from before @register_experts was applied, NOT our patched one.
#
# The recursion happens because @use_experts_implementation wraps @register_experts,
# creating nested wrappers. Each wrapper calls get_interface which returns the next
# wrapper in the chain, causing 42 recursive calls.
#
# FIX: Replace the wrapper function on the class entirely, AND patch get_interface
# to return our eager forward directly.

# Eager experts forward that handles BOTH 2D [T, H] and 3D [B, S, H] input
def _eager_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Direct eager MoE forward, handles 2D or 3D input. Wrapped in no_grad to avoid FP8 autograd."""
    # CRITICAL: Wrap entire forward in no_grad to prevent FP8 ops from entering autograd graph
    # The FP8 matmul ops don't have registered backward formulas on XPU
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
            current = flat[token_idx]  # [num_tokens_in_expert, H]
            # Get gate_up weight
            if hasattr(self, 'has_gate') and self.has_gate:
                raw_w = self.gate_up_proj[expert_idx]
                raw_s = self.gate_up_proj_scale_inv[expert_idx]
            else:
                raw_w = self.up_proj[expert_idx]
                raw_s = self.up_proj_scale_inv[expert_idx]
            w_gu = _dequantize_fp8_weight(raw_w, raw_s, getattr(self, 'block_size', (128, 128))) if raw_w.element_size() <= 1 else raw_w
            # FP8 weights stored as [in_features, out_features] but F.linear expects [out_features, in_features]
            if w_gu.shape[1] != current.shape[1]:
                w_gu = w_gu.T
            proj_out = F.linear(current, w_gu)
            if hasattr(self, 'has_gate') and self.has_gate:
                proj_out = self._apply_gate(proj_out)
            else:
                proj_out = self.act_fn(proj_out)
            # Get down weight
            raw_wd = self.down_proj[expert_idx]
            raw_sd = self.down_proj_scale_inv[expert_idx]
            w_d = _dequantize_fp8_weight(raw_wd, raw_sd, getattr(self, 'block_size', (128, 128))) if raw_wd.element_size() <= 1 else raw_wd
            if w_d.shape[1] != proj_out.shape[1]:
                w_d = w_d.T
            proj_out = F.linear(proj_out, w_d)
            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            weighted = proj_out * routing_weights.to(proj_out.dtype)
            final.index_add_(0, token_idx, weighted.to(final.dtype))
        if batch_size is not None and seq_len is not None:
            return final.view(batch_size, seq_len, hidden_dim).to(hidden_states.dtype)
        return final.to(hidden_states.dtype)

# Replace forward on both classes
dv4_module.DeepseekV4Experts.forward = _eager_experts_forward
print("  Replaced DeepseekV4Experts.forward with eager implementation")

if hasattr(fg_fp8_module, 'FP8Experts'):
    fg_fp8_module.FP8Experts.forward = _eager_experts_forward
    print("  Replaced FP8Experts.forward with eager implementation")

# CRITICAL: Patch get_interface to return our eager forward, NOT the default.
# This breaks the recursion chain because the wrapper at moe.py:576 calls
# experts_forward = get_interface(impl, original_forward), then experts_forward(self, ...).
# If get_interface returns our _eager_experts_forward, it bypasses the original chain.
# We need to patch it as a bound method, not classmethod.

def _patched_get_interface(self, experts_implementation, default):
    """Return our eager forward instead of the default, breaking the recursion chain."""
    return _eager_experts_forward

# Replace as a regular method (not classmethod)
moe_module.ExpertsInterface.get_interface = _patched_get_interface
if hasattr(fg_fp8_module, 'FP8ExpertsInterface'):
    fg_fp8_module.FP8ExpertsInterface.get_interface = _patched_get_interface
print("  Patched ExpertsInterface.get_interface to return eager forward")

print()
print("All patches applied. Loading model...")
print()

# Now import and run AutoRound
from auto_round import AutoRound

print("[1/3] Loading model on xpu:0 (single GPU to avoid multi-device context issues)...")
t0 = time.time()
model = AutoRound(
    MODEL_PATH,
    scheme="INT4",
    group_size=128,
    sym=False,
    iters=10,
    enable_quanted_input=True,
    batch_size=4,
    amp=True,
    device_map="xpu:0",  # Pin to single GPU — avoids cross-GPU tensor transfers and driver context issues
    low_cpu_mem_usage=True,
    seed=42,
)
print(f"  Model loaded in {time.time() - t0:.0f}s")

print()
print("[2/3] Running calibration and quantization...")
t1 = time.time()
model.quantize_and_save(OUTPUT_DIR)
print(f"  Quantization complete in {time.time() - t1:.0f}s")

# Stop watchdog — we succeeded
watchdog.stop()
print("[WATCHDOG] Stopped — quantization completed successfully")

print()
print("[3/3] Done! Output saved to:")
print(f"  {OUTPUT_DIR}")
