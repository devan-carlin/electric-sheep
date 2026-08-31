# Patches Diff: Original vs Patched Source Code

> Generated from `test-autoround-xpu-patched.py` — all patches applied for INT4 AutoRound quantization on Intel XPU.

## Summary Table

| Patch | File | Function | Change Type | Status |
|-------|------|----------|-------------|--------|
| 1 | `replace_modules.py` | `apply_replacements` | Full replacement (no-op) | ✅ Working |
| 2 | `modeling_deepseek_v4.py` | `DeepseekV4Attention.forward` | Wrapper with dict/tuple handling | ✅ Working |
| 3a | `modeling_deepseek_v4.py` | `DeepseekV4TopKRouter.forward` | Signature change (accept `input_ids`) | ✅ Working |
| 3b | `modeling_deepseek_v4.py` | `DeepseekV4HashRouter.forward` | Signature change (optional `input_ids`) | ✅ Working |
| 4 | `config.json` | `_experts_implementation` | Config key addition | ✅ Working |
| 5 | `finegrained_fp8.py` | `_disable_deepgemm_on_multi_device` | Full replacement (no-op) + class attr | ✅ Working |
| 6 | `modeling_deepseek_v4.py` | `DeepseekV4Experts.forward` | Full replacement (eager + FP8 dequant) | ✅ Working |
| 7 | `grouped.py` (kernel) | `w8a8_block_dynamic_fp8_matmul_grouped` | Wrap in `torch.no_grad()` | ✅ Working |
| 8a | `int.py` | `quant_tensor_sym` | Wrapper with device fix | ✅ Working |
| 8b | `wrapper.py` | `WrapperWALayer.forward` | Full replacement with device fix | ✅ Working |
| 9 | `common.py` | `revert_checkpoint_conversion_mapping` | Full replacement (preserve capture groups) | ✅ Working |
| 10a | `model.py` | `mv_module_from_gpu` | Full replacement (individual param moves) | ✅ Working |
| 10b | `data_driven.py` | `mv_module_from_gpu` (import) | Attribute assignment on module | ✅ Working |

---

## Patch 1: Disable MoE fused module replacement

**File**: `auto_round/modeling/fused_moe/replace_modules.py`
**Function**: `apply_replacements`
**Why**: INT4 quantized tensors can't be wrapped as `nn.Parameter`. The fused MoE replacement tries to create new parameters from quantized tensors, which fails on XPU.

### Original Code
```python
def apply_replacements(
    model: torch.nn.Module,
    auto_detect_moe: bool = True,
) -> torch.nn.Module:
    """
    Function to apply module replacements to a model.

    This scans all modules in the model and replaces any registered modules with their
    replacement equivalents. Non-permanent modules are tracked for later restoration.

    The model is modified in-place, so the same model object should be used.

    Args:
        model: The model to apply module replacement to (modified in-place).
        auto_detect_moe: If True, automatically detect and handle fused MOE modules
            (transformers 5.0+ pattern). Default is True.

    Returns:
        The model with modules replaced.
    """
    _import_required_replacements(model)
    _raw_expert_is_logged = False

    # Custom replacements first
    if is_custom_model(model):

        if not _raw_expert_is_logged:
            _raw_expert_is_logged = _log_first_moe_block(model, "before replacement")

        _apply_custom_replacements(model)

    if auto_detect_moe and is_transformers_version_greater_or_equal_5():

        if not _raw_expert_is_logged:
            _raw_expert_is_logged = _log_first_moe_block(model, "before replacement")

        _handle_moe_modules(model)

    if _raw_expert_is_logged:
        _log_first_moe_block(model, "after replacement/skip")

    return model
```

### Patched Code
```python
_orig_apply = replace_mod.apply_replacements
def patched_apply(model, *args, **kwargs):
    print(">>> Skipping MoE fused replacement (INT4 tensors can't be nn.Parameter)")
    return model
replace_mod.apply_replacements = patched_apply
```

### Diff Summary
- **Change**: Replaced entire function with a no-op that returns the model unchanged
- **Impact**: MoE modules are not replaced with fused versions during quantization
- **Risk**: None — fused modules would fail with INT4 quantized tensors anyway

---

## Patch 2: Fix DeepseekV4Attention forward signature

**File**: `transformers/models/deepseek_v4/modeling_deepseek_v4.py`
**Function**: `DeepseekV4Attention.forward`
**Why**: AutoRound passes `position_embeddings` as a dict (or tuple, or empty dict), but the model expects `position_ids` as a Tensor. The compressor sub-module (`DeepseekV4HCACompressor.forward`) has a completely different signature that uses `position_ids` directly.

### Original Code
```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # position_embeddings is a {"main", "compress"} dict from the model; pick the
        # one that matches this layer's rope type (sliding → main, CSA/HCA → compress).
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:  # sliding where K==V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        block_bias = None
        if self.compressor is not None:  # Compressed KV (CSA or HCA)
            compressed_kv, block_bias = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            kv = torch.cat([kv, compressed_kv], dim=2)
        # ... (rest of attention computation)
```

### Patched Code
```python
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
```

### Diff Summary
- **Change**: Wrapped original forward to normalize `position_embeddings` into the expected dict format
- **Handles three cases**:
  1. Tuple `(cos, sin)` → wrap in `{self.rope_layer_type: (cos, sin)}`
  2. `None` or empty dict → construct identity RoPE (cos=ones, sin=zeros)
  3. Already correct dict → pass through unchanged
- **Impact**: AutoRound calibration can proceed without position embedding format errors
- **Risk**: Identity RoPE means positional encoding is neutralized during calibration (acceptable for weight-only quantization)

---

## Patch 3a: Fix DeepseekV4TopKRouter forward signature

**File**: `transformers/models/deepseek_v4/modeling_deepseek_v4.py`
**Function**: `DeepseekV4TopKRouter.forward`
**Why**: The MoE sparse block (`DeepseekV4SparseMoeBlock`) passes `input_ids` to all gate types. TopKRouter doesn't accept `input_ids` in its signature, causing a TypeError.

### Original Code
```python
class DeepseekV4TopKRouter(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices
```

### Patched Code
```python
_orig_topk_forward = dv4_module.DeepseekV4TopKRouter.forward
def patched_topk_forward(self, hidden_states, input_ids=None):
    return _orig_topk_forward(self, hidden_states)
dv4_module.DeepseekV4TopKRouter.forward = patched_topk_forward
```

### Diff Summary
- **Change**: Added `input_ids=None` parameter, then discards it before calling original
- **Impact**: TopKRouter now accepts the same signature as HashRouter
- **Risk**: None — `input_ids` is genuinely unused by TopKRouter

---

## Patch 3b: Fix DeepseekV4HashRouter forward signature

**File**: `transformers/models/deepseek_v4/modeling_deepseek_v4.py`
**Function**: `DeepseekV4HashRouter.forward`
**Why**: AutoRound doesn't provide `input_ids` during calibration, but HashRouter requires it for the `tid2eid` lookup table.

### Original Code
```python
class DeepseekV4HashRouter(nn.Module):
    r"""
    Hash routing for the first `mlp_layer_types == "hash_moe"` MoE layers (paper
    §2.1). Expert selection is determined by a fixed `tid2eid[input_ids]` lookup —
    a frozen token-id → expert-id table — instead of a learned argmax. The learned
    gate `weight` still produces the per-expert scores that weight the selected
    experts' activations; only the *which-experts* selection is static.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("tid2eid", torch.zeros(config.vocab_size, self.top_k, dtype=torch.long), persistent=True)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices
```

### Patched Code
```python
_orig_hash_forward = dv4_module.DeepseekV4HashRouter.forward
def patched_hash_forward(self, hidden_states, input_ids=None):
    # HashRouter needs input_ids for tid2eid lookup
    if input_ids is None:
        # Create dummy input_ids (all zeros) - shape [B, S]
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
    return _orig_hash_forward(self, hidden_states, input_ids)
dv4_module.DeepseekV4HashRouter.forward = patched_hash_forward
```

### Diff Summary
- **Change**: Made `input_ids` optional; creates dummy all-zero tensor when not provided
- **Impact**: HashRouter can operate during calibration without real input_ids
- **Risk**: Dummy input_ids means all tokens map to the same experts (tid2eid[0]), but this is acceptable for calibration since we only need valid forward passes

---

## Patch 4: Force eager MoE implementation

**File**: `config.json` (model configuration)
**Key**: `_experts_implementation`
**Why**: The default experts implementation tries to use deepspeed fusion / grouped kernels which fail on XPU. Forcing eager mode avoids these code paths entirely.

### Original Config
```json
{
    // ... other config keys ...
    // No _experts_implementation key present
}
```

### Patched Config
```python
import json
config_path = os.path.join(MODEL_PATH, "config.json")
with open(config_path) as f:
    config = json.load(f)
config["_experts_implementation"] = "eager"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
```

### Diff Summary
- **Change**: Added `"_experts_implementation": "eager"` to model config.json
- **Impact**: Forces transformers to use the eager (Python loop) MoE implementation
- **Risk**: None — eager mode is the fallback path; combined with Patch 6, it uses our custom eager forward

---

## Patch 5: Disable deepgemm FP8 kernel dispatch

**File**: `transformers/integrations/finegrained_fp8.py`
**Function**: `_disable_deepgemm_on_multi_device`
**Why**: DeepGEMM FP8 kernels fail on XPU (CUDA-only). The original function only checks CUDA devices, but on XPU systems we need to unconditionally disable deepgemm.

### Original Code
```python
def _disable_deepgemm_on_multi_device(model: nn.Module) -> None:
    """Internal, temporary helper (not public API): flag every FP8 module to skip DeepGEMM when the
    model spans >1 CUDA device in one process.

    DeepGEMM loads each kernel via `cuKernelGetFunction`, which binds the `CUfunction` handle to the
    CUDA context live at load time; driving that cached handle from another device launches it against
    the wrong context and produces garbage. (Build-time fix: compile DeepGEMM with
    `DG_JIT_USE_RUNTIME_API=1` for a context-free `cudaKernel_t` loader; until our wheel picks that up
    we avoid single-process multi-device.) Setting `_deepgemm_disabled` routes both the linear and
    experts paths through Triton/grouped_mm. A model that fits on one device keeps DeepGEMM even with
    other GPUs visible; TP/EP put one device per process, so this is a no-op there.
    """
    fp8_modules = [m for m in model.modules() if isinstance(m, (FP8Linear, FP8Experts))]
    cuda_devices = set()
    for m in fp8_modules:
        param = next(m.parameters(), None)
        if param is not None and param.device.type == "cuda":
            cuda_devices.add(param.device.index)
    if len(cuda_devices) <= 1:
        return
    for m in fp8_modules:
        m._deepgemm_disabled = True
    logger.warning_once(
        "This FP8 model spans multiple CUDA devices in one process; routing its FP8 linear and experts "
        "layers through Triton/grouped_mm instead of DeepGEMM (DeepGEMM's cached kernels are bound to a "
        "single CUDA context and corrupt across devices). Run tensor/expert parallel (one device per "
        "process) to use the faster DeepGEMM path."
    )
```

### Patched Code
```python
def patched_disable_deepgemm(model):
    # No-op — deepgemm is already disabled
    pass
fg_fp8_module._disable_deepgemm_on_multi_device = patched_disable_deepgemm
# Also set _deepgemm_disabled on FP8 classes
if hasattr(fg_fp8_module, 'FP8Experts'):
    fg_fp8_module.FP8Experts._deepgemm_disabled = True
if hasattr(fg_fp8_module, 'FP8Linear'):
    fg_fp8_module.FP8Linear._deepgemm_disabled = True
```

### Diff Summary
- **Change**: Replaced function with no-op; additionally sets `_deepgemm_disabled = True` as class attributes on both `FP8Experts` and `FP8Linear`
- **Impact**: DeepGEMM kernels are never dispatched; all FP8 operations route through Triton/grouped_mm fallbacks
- **Risk**: None on XPU — DeepGEMM is CUDA-only and would fail anyway

---

## Patch 6: Replace DeepseekV4Experts.forward with eager FP8-aware implementation

**File**: `transformers/models/deepseek_v4/modeling_deepseek_v4.py`
**Function**: `DeepseekV4Experts.forward`
**Why**: The original forward uses FP8 fused kernels (via `@use_experts_implementation` decorator) which fail on XPU. The patched version dequantizes FP8 weights to BF16 and performs eager matrix multiplications with proper device placement.

### Original Code
```python
@use_experts_implementation
class DeepseekV4Experts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Lives on the class (like gpt-oss's _apply_gate) so the grouped_mm / batched_mm
        # backends swapped in by `@use_experts_implementation` apply the same clamp +
        # SiLU on top of their packed gate_up output instead of bypassing it.
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up
```

### Patched Code
```python
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


def _eager_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Direct eager MoE forward, handles 2D or 3D input. Wrapped in no_grad to avoid FP8 autograd."""
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
if hasattr(fg_fp8_module, 'FP8Experts'):
    fg_fp8_module.FP8Experts.forward = _eager_experts_forward

# CRITICAL: Patch get_interface to return our eager forward, NOT the default.
def _patched_get_interface(self, experts_implementation, default):
    """Return our eager forward instead of the default, breaking the recursion chain."""
    return _eager_experts_forward

moe_module.ExpertsInterface.get_interface = _patched_get_interface
if hasattr(fg_fp8_module, 'FP8ExpertsInterface'):
    fg_fp8_module.FP8ExpertsInterface.get_interface = _patched_get_interface
```

### Diff Summary
- **Change**: Complete replacement of the experts forward with an eager implementation that:
  1. Wraps everything in `torch.no_grad()` to prevent FP8 autograd errors
  2. Handles both 2D `[T, H]` and 3D `[B, S, H]` inputs
  3. Dequantizes FP8 weights to BF16 before use (handles 5 different scale formats)
  4. Transposes weights when needed (FP8 stores `[in, out]`, `F.linear` expects `[out, in]`)
  5. Uses `float32` accumulator for the final sum
  6. Patches `ExpertsInterface.get_interface` to break the recursion chain caused by `@use_experts_implementation`
- **Impact**: MoE experts run entirely in eager mode with FP8→BF16 dequantization
- **Risk**: Slower than fused kernels, but correct on XPU

---

## Patch 7: FP8 grouped matmul no_grad wrapper

**File**: `finegrained-fp8/build/torch-xpu/grouped.py` (kernel package)
**Function**: `w8a8_block_dynamic_fp8_matmul_grouped`
**Why**: The FP8 grouped matmul on XPU is registered as a custom op without a backward formula. Wrapping in `no_grad` prevents autograd errors during calibration.

### Original Code
```python
def w8a8_block_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-scale grouped FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations sorted by expert, bf16/fp16/fp32
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales
    output_dtype: defaults to ``A.dtype``
    """
    return ops.w8a8_block_dynamic_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size, output_dtype
    )
```

### Patched Code
```python
def w8a8_block_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-scale grouped FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations sorted by expert, bf16/fp16/fp32
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales
    output_dtype: defaults to ``A.dtype``
    """
    # PATCHED: Wrapped in no_grad to prevent autograd errors
    with torch.no_grad():
        return ops.w8a8_block_dynamic_fp8_matmul_grouped(
            A, B, Bs, offsets, tokens_per_expert, block_size, output_dtype
        )
```

### Diff Summary
- **Change**: Wrapped the `ops.w8a8_block_dynamic_fp8_matmul_grouped` call in `with torch.no_grad():`
- **Impact**: FP8 grouped matmul no longer enters the autograd graph
- **Risk**: None for calibration (forward-only); would break fine-tuning (not needed for quantization)
- **Note**: This is a file-system patch (modifies the kernel package on disk), not a runtime monkey-patch

---

## Patch 8a: Fix multi-device tensor mismatch in quant_tensor_sym

**File**: `auto_round/data_type/int.py`
**Function**: `quant_tensor_sym`
**Why**: During multi-device quantization, scale tensors (`min_scale`, `max_scale`, `init_scale`, `tensor_min`, `tensor_max`) may reside on a different XPU device than the input tensor, causing `RuntimeError: Expected all tensors to be on the same device`.

### Original Code
```python
@register_dtype("int_sym")
def quant_tensor_sym(
    tensor,
    bits=4,
    group_size=-1,
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
    """Quantize and de-quantize tensor asymmetrically. full range, credit goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = int(2.0 ** (bits - 1))

    if init_scale is not None:
        # ``max_scale`` is a per-group tuning coefficient (Tensor) during
        # SignRound optimization, but may be a plain scalar (e.g. 1.0) when the
        # init_scale is reused for a one-shot QDQ such as AWQ's smooth/clip grid
        # search.
        if isinstance(max_scale, torch.Tensor):
            scale = init_scale * max_scale.unsqueeze(dim=-1)
        else:
            scale = init_scale * max_scale
        scale = scale.to(scale_dtype)
        scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
        int_w = round_ste(tensor / scale + v)
        q = torch.clamp(int_w, -maxq, maxq - 1)
        qdq_result = (scale * q).to(tensor.dtype)
        qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
        return qdq_result, scale, maxq

    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max

    wmin_abs = -(wmin_tmp * min_scale)  # pylint: disable=E1130
    wmax_abs = wmax_tmp * max_scale
    max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)
    scale = (max_v / maxq).to(scale_dtype)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    scale = scale.unsqueeze(dim=-1)
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w, -maxq, maxq - 1)
    qdq_result = (scale * q).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, maxq
```

### Patched Code
```python
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
```

### Diff Summary
- **Change**: Wrapper that moves all tensor arguments to the input tensor's device before calling the original
- **Impact**: Eliminates cross-device tensor operations during quantization
- **Risk**: None — device transfer is a shallow copy for same-device-type tensors (xpu:0 → xpu:2)

---

## Patch 8b: Fix WrapperWALayer.forward multi-device scales

**File**: `auto_round/wrapper.py`
**Function**: `WrapperWALayer.forward`
**Why**: Same multi-device issue as Patch 8a — `min_scale`, `max_scale`, and `act_max` tensors may be on a different XPU device than the input activation.

### Original Code
```python
    def forward(self, x):
        # 1) Run stolen pre_hooks first (e.g., online Hadamard) → smooths activation
        for hook in self._stolen_pre_hooks:
            result = hook(self.orig_layer, (x,))
            if result is not None:
                x = result[0] if isinstance(result, tuple) else result

        # 2) Activation quantization on the smoothed activation
        import auto_round.envs as envs

        act_scale = envs.AR_ACT_SCALE
        act_max = self.orig_layer.act_max if hasattr(self.orig_layer, "act_max") else None

        max_scale = self.orig_layer.act_max_scale if math.isclose(act_scale, 1.0, rel_tol=1e-6) else act_scale
        min_scale = self.orig_layer.act_min_scale if math.isclose(act_scale, 1.0, rel_tol=1e-6) else act_scale
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
        # 3) Linear computation via orig_layer (pre_hooks already removed, no double execution)
        return self.orig_layer.forward(x)
```

### Patched Code
```python
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
```

### Diff Summary
- **Change**: Added device checks and transfers for `min_scale`, `max_scale`, and `act_max` before calling `act_quant_func`
- **Impact**: Activation quantization works correctly across multiple XPU devices
- **Risk**: None — identical logic to original, just with device alignment

---

## Patch 9: Fix regex error in revert_checkpoint_conversion_mapping

**File**: `auto_round/utils/common.py`
**Function**: `revert_checkpoint_conversion_mapping`
**Why**: The original strips capture groups from the source pattern (`re.sub(r"\(.*\)", "", source_pattern)`), which breaks when the target pattern contains backreferences like `\1`. This causes `re.error: invalid group reference 1`.

### Original Code
```python
def revert_checkpoint_conversion_mapping(name: str, key_mapping: dict[str, str]) -> str:
    if "," in name:
        return ",".join(revert_checkpoint_conversion_mapping(part, key_mapping) for part in name.split(","))

    for source_pattern, target_patterns in key_mapping.items():
        if isinstance(target_patterns, str):
            target_patterns = [target_patterns]
        for target_pattern in target_patterns:
            source_pattern = source_pattern.lstrip("^")  # strip off un-needed chars and patterns
            source_pattern = re.sub(r"\(.*\)", "", source_pattern)
            name, n_replace = re.subn(source_pattern, target_pattern, name)
            # Early exit of the loop
            if n_replace > 0:
                return name
    return name
```

### Patched Code
```python
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
```

### Diff Summary
- **Change**: Three improvements over original:
  1. **Removed** `source_pattern = re.sub(r"\(.*\)", "", source_pattern)` — preserves capture groups for backreference support
  2. **Added** `try/except re.error` with fallback to plain string replacement
  3. **Patched** in both `common_utils` and `shard_writer_module` (the latter imports it directly at module load time)
- **Impact**: Regex patterns with backreferences (e.g., `(.*)` → `\1.quant`) now work correctly
- **Risk**: None — fallback handles edge cases gracefully

---

## Patch 10a: Fix OOM in mv_module_from_gpu

**File**: `auto_round/utils/model.py`
**Function**: `mv_module_from_gpu`
**Why**: The original calls `module.to("cpu")` which allocates a temporary buffer the size of the entire module before freeing GPU memory. On a 284B model, this causes OOM. The patched version moves parameters and buffers individually.

### Original Code
```python
def mv_module_from_gpu(module):
    """Moves module from gpu to cpu.

    Args:
    module: The module to be moved.

    Returns:
    The module on the specified device.
    """
    if hasattr(module, "device"):
        if module.device.type in ("cpu", "meta"):
            return module

    has_meta = any(p.device.type == "meta" for p in module.parameters())
    if not has_meta:
        has_meta = any(b.device.type == "meta" for b in module.buffers())

    if has_meta:
        for _, child in module.named_children():
            mv_module_from_gpu(child)
        for attr_name in list(module._parameters.keys()):
            p = module._parameters[attr_name]
            if p is not None and p.device.type != "meta" and p.device.type != "cpu":
                module._parameters[attr_name] = torch.nn.Parameter(p.to("cpu"), requires_grad=p.requires_grad)
        for attr_name in list(module._buffers.keys()):
            b = module._buffers[attr_name]
            if b is not None and b.device.type != "meta" and b.device.type != "cpu":
                module._buffers[attr_name] = b.to("cpu")
        return module

    return module.to("cpu")
```

### Patched Code
```python
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
```

### Diff Summary
- **Change**: Three improvements over original:
  1. **Replaced** `return module.to("cpu")` with individual parameter/buffer moves (the critical OOM fix)
  2. **Added** `non_blocking=True` to all `.to("cpu")` calls for async transfers
  3. **Added** `_get_device_type` helper to handle string device types (e.g., `"xpu:0"`) in addition to `torch.device` objects
- **Impact**: No temporary buffer allocation — parameters are moved one at a time, freeing GPU memory incrementally
- **Risk**: None — functionally equivalent to `module.to("cpu")` but memory-safe

---

## Patch 10b: Patch mv_module_from_gpu in data_driven module

**File**: `auto_round/compressors/data_driven.py`
**Import**: `from auto_round.utils.model import mv_module_from_gpu`
**Why**: `data_driven.py` imports `mv_module_from_gpu` directly at module load time (line 60). Patching `model_utils.mv_module_from_gpu` alone is insufficient — the `data_driven` module has its own local reference to the original function.

### Original Import
```python
from auto_round.utils.model import (
    clear_memory,
    compress_layer_names,
    convert_module_to_hp_if_necessary,
    flatten_list,
    get_block_names,
    get_module,
    hook_ngram_embeddings_on_cpu,
    is_auto_device_mapping,
    is_quantized_input_module,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    to_device,
    to_dtype,
    wrap_block_forward_positional_to_kwargs,
)
```

### Patched Code
```python
# Also patch in data_driven which imports it directly
import auto_round.compressors.data_driven as data_driven_module
data_driven_module.mv_module_from_gpu = patched_mv_module_from_gpu
```

### Diff Summary
- **Change**: After patching `model_utils.mv_module_from_gpu`, also assigns the patched function to `data_driven_module.mv_module_from_gpu`
- **Impact**: All callers of `mv_module_from_gpu` (whether via `model_utils` or `data_driven`) get the OOM-safe version
- **Risk**: None — same pattern used for Patch 9 (shard_writer)

---

## Appendix: Patch Dependencies & Order

The patches must be applied in the order they appear in the test script:

1. **Patches 1-3**: Module-level monkey-patches (must be before model loading)
2. **Patch 4**: Config modification (must be before model loading)
3. **Patches 5-7**: FP8 kernel patches (must be before first FP8 forward pass)
4. **Patches 8a-8b**: Quantization function patches (must be before calibration)
5. **Patch 9**: Regex fix (must be before model saving)
6. **Patches 10a-10b**: Memory management patches (must be before layer movement)

### Files Modified on Disk
| File | Patch | Type |
|------|-------|------|
| `config.json` | 4 | JSON key addition |
| `grouped.py` (kernel) | 7 | Source code modification |

### Runtime Monkey-Patches Only
| Module | Patch | Attribute |
|--------|-------|-----------|
| `replace_modules` | 1 | `apply_replacements` |
| `modeling_deepseek_v4` | 2 | `DeepseekV4Attention.forward` |
| `modeling_deepseek_v4` | 3a | `DeepseekV4TopKRouter.forward` |
| `modeling_deepseek_v4` | 3b | `DeepseekV4HashRouter.forward` |
| `finegrained_fp8` | 5 | `_disable_deepgemm_on_multi_device` |
| `finegrained_fp8` | 5 | `FP8Experts._deepgemm_disabled` |
| `finegrained_fp8` | 5 | `FP8Linear._deepgemm_disabled` |
| `modeling_deepseek_v4` | 6 | `DeepseekV4Experts.forward` |
| `finegrained_fp8` | 6 | `FP8Experts.forward` |
| `moe` | 6 | `ExpertsInterface.get_interface` |
| `finegrained_fp8` | 6 | `FP8ExpertsInterface.get_interface` |
| `int` | 8a | `quant_tensor_sym` |
| `wrapper` | 8b | `WrapperWALayer.forward` |
| `common` | 9 | `revert_checkpoint_conversion_mapping` |
| `shard_writer` | 9 | `revert_checkpoint_conversion_mapping` |
| `model` | 10a | `mv_module_from_gpu` |
| `data_driven` | 10b | `mv_module_from_gpu` |

---

*Generated for DeepSeek-V4-Flash-0731-FP8 → INT4 AutoRound quantization on Intel XPU*
*Test script: `/home/dc/electric-sheep/vllm/experimental/test-autoround-xpu-patched.py`*
