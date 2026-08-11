#!/usr/bin/env bash
# =============================================================================
# 04.1-convert-int4-autoround-benchmark.sh — INT4 AutoRound (SPEED BENCHMARK)
# =============================================================================
# ⚠️  BENCHMARK VERSION — Optimized for SPEED, not quality ⚠️
#
# Changes from original:
#   - Iterations: 1000 → 100 (10x faster optimization)
#   - Calibration samples: 128 → 32 (4x faster data prep)
#   - Sequence length: 2048 → 512 (4x less context to process)
#   - Parallel: always enabled
#   - Batch size: default 8 (max concurrent layers)
#   - CPU threads: maxed out
#   - Low GPU mem: disabled (faster)
#   - damp_percent: 0.01 → 0.05 (less conservative, faster convergence)
#   - n_blocks: 100 → 200 (fewer save checkpoints)
#   - enable_quanted_input: true → false (skips quantized input path)
#   - enable_minmax_tuning: true → false (skips minmax search)
#   - enable_outlier_channel_wise: true → false (simpler quantization)
#   - per_channel: true → false (faster, less precise)
#   - search_method: ammp → rms (faster search)
#
# Usage:
#   bash convert-int4-autoround-parallel-benchmark.sh <huggingface-repo> [options]
#
# Example:
#   bash convert-int4-autoround-parallel-benchmark.sh apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8
#
# Options (overrides benchmark defaults):
#   --bits N              Quantization bits (default: 4)
#   --group-size N        Group size (default: 64)
#   --sym                 Symmetric quantization (default: asymmetric)
#   --samples N           Calibration samples (benchmark default: 32)
#   --seqlen N            Sequence length (benchmark default: 512)
#   --iters N             Optimization iterations (benchmark default: 100)
#   --batch-size N        Concurrent layers (benchmark default: 8)
#   --output-dir PATH     Output directory
#   --download-only       Download source model only
#   --help                Show this help
#
# Expected speed: ~10-20x faster than original (minutes vs hours)
# Quality trade-off: Perplexity will be higher (worse) — use for validation only
# =============================================================================

set -euo pipefail

# ============================================
# Configuration
# ============================================
VENV_DIR="$HOME/electric-sheep/vllm/.venv"
MODELS_DIR="$HOME/electric-sheep/models"

# Quantization defaults — BENCHMARK (speed-optimized)
BITS=4
GROUP_SIZE=64
SYMMETRIC=false
CALIBRATION_SAMPLES=32
CALIBRATION_DATASET="json"
SEQLEN=512
ITERS=100
LOW_GPU_MEM=false
RECIPE="auto-round-best"

# Parallel processing defaults — always enabled for benchmark
PARALLEL=true
BATCH_SIZE=8

# ============================================
# Helpers
# ============================================
print_header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

print_ok() {
    echo "  ✓ $1"
}

print_warn() {
    echo "  ⚠ $1"
}

fail() {
    echo "  ✗ $1"
    exit 1
}

show_help() {
    head -72 "$0" | tail -68
}

# ============================================
# Parse Arguments
# ============================================
SOURCE_REPO=""
OUTPUT_DIR=""
DOWNLOAD_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --bits)
            BITS="$2"; shift 2 ;;
        --group-size)
            GROUP_SIZE="$2"; shift 2 ;;
        --sym)
            SYMMETRIC=true; shift ;;
        --samples)
            CALIBRATION_SAMPLES="$2"; shift 2 ;;
        --seqlen)
            SEQLEN="$2"; shift 2 ;;
        --iters)
            ITERS="$2"; shift 2 ;;
        --low-gpu-mem)
            LOW_GPU_MEM=true; shift ;;
        --recipe)
            RECIPE="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --download-only)
            DOWNLOAD_ONLY=true; shift ;;
        --parallel)
            PARALLEL=true; shift ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --*)
            echo "Unknown option: $1"; show_help; exit 1 ;;
        *)
            SOURCE_REPO="$1"; shift ;;
    esac
done

# Validate
if [ -z "$SOURCE_REPO" ]; then
    echo "Error: Source model repo required."
    echo ""
    show_help
    exit 1
fi

# Parallel mode validation
if [ "$PARALLEL" = true ]; then
    # Force low GPU mem for large models to avoid OOM
    # FP8→FP16 load on 284B model needs ~568GB, we have ~375GB total
    # low_gpu_mem_usage=True processes layers sequentially (slower but won't crash)
    LOW_GPU_MEM=true
    echo "  ✓ BENCHMARK MODE — Parallel enabled, low GPU mem: $LOW_GPU_MEM (forced for large models)"
    echo "  Batch size: $BATCH_SIZE (concurrent layers)"
    echo "  Iterations: $ITERS (reduced from 1000 for speed)"
    echo "  Samples: $CALIBRATION_SAMPLES (reduced from 128 for speed)"
    echo "  Seq length: $SEQLEN (reduced from 2048 for speed)"
fi

# ============================================
# Pre-flight
# ============================================
print_header "INT4 AutoRound Conversion — Pre-flight"

# Check venv
if [ ! -d "$VENV_DIR" ]; then
    fail "vLLM virtual environment not found at $VENV_DIR"
    echo "  Run: bash ~/electric-sheep/scripts/ubuntu/02-setup-project-directory.sh"
fi
print_ok "Virtual environment found"

# Activate venv early for Python checks
source "$VENV_DIR/bin/activate"

# CPU threading optimization (NEW in 04.1-)
if [ "$PARALLEL" = true ]; then
    echo "  Setting CPU threading for maximum parallelism..."
    export OMP_NUM_THREADS=24
    export MKL_NUM_THREADS=24
    export OPENBLAS_NUM_THREADS=24
    export NUMEXPR_NUM_THREADS=24
    export VECLIB_MAXIMUM_THREADS=24
    echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
    echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
else
    echo "  Single-GPU mode — using default CPU threading"
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python: $python_version"
if [[ "$python_version" != "3.12" && "$python_version" != "3.11" ]]; then
    print_warn "Python $python_version detected — 3.11 or 3.12 recommended"
fi

# Check PyTorch with XPU support
if python3 -c "import torch; assert torch.xpu.is_available()" 2>/dev/null; then
    print_ok "PyTorch with XPU support"
else
    print_warn "PyTorch XPU not available — quantization will use CPU (slower)"
fi

# Check GPU devices (for VRAM estimation) — non-fatal, informational only
if command -v sycl-ls >/dev/null 2>&1; then
    gpu_count=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu" || echo "0")
    if [ "$gpu_count" -gt 0 ]; then
        total_vram_gb=0
        while IFS= read -r line; do
            # Use basic grep instead of -P (Perl regex) for compatibility
            # Add || echo "" to prevent exit on no match with set -o pipefail
            vram=$(echo "$line" | grep -o '[0-9.]* GB' | head -1 | awk '{print $1}' || echo "")
            if [ -n "$vram" ]; then
                if command -v bc >/dev/null 2>&1; then
                    total_vram_gb=$(echo "$total_vram_gb + $vram" | bc)
                else
                    # Fallback: integer addition if bc not available
                    total_vram_gb=$((total_vram_gb + ${vram%.*}))
                fi
            fi
        done < <(sycl-ls 2>/dev/null | grep "level_zero:gpu" || true)
        echo "  GPUs: $gpu_count SYCL GPU(s), ~${total_vram_gb}GB total VRAM"
        
        if [ "$PARALLEL" = true ] && [ "$gpu_count" -lt 2 ]; then
            print_warn "Only 1 GPU detected — parallel mode requires 2+ GPUs"
            echo "  Falling back to single-GPU mode"
            PARALLEL=false
        fi
    else
        print_warn "No SYCL GPU devices — quantization will use CPU only"
    fi
else
    print_warn "sycl-ls not found — cannot check GPU availability"
fi

# Check RAM (need ~2x model size for loading + quantization)
available_ram_gb=$(free -g | awk '/^Mem:/{print $7}')
echo "  Available RAM: ${available_ram_gb}GB"
if [ "$available_ram_gb" -lt 64 ]; then
    print_warn "Less than 64GB RAM — large models may fail to load"
fi

# Check hf CLI
if ! command -v hf >/dev/null 2>&1; then
    fail "hf CLI not found"
    echo "  Install: pipx install huggingface_hub"
fi
print_ok "hf CLI available"

# Check HF token (needed for gated models)
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
    print_warn "No HF_TOKEN set — gated models (Llama, etc.) may fail"
    echo "  Fix: export HF_TOKEN=hf_..."
fi

# Check disk space (need ~2x model size for source + output)
available_gb=$(df --output=avail -BM "$HOME" | tail -1 | awk '{printf "%d", $1/1024}')
echo "  Available disk: ${available_gb}GB"
if [ "$available_gb" -lt 100 ]; then
    print_warn "Less than 100GB free — quantization may need more space"
fi

# Check datasets library (needed for calibration)
if python3 -c "import datasets" 2>/dev/null; then
    print_ok "datasets library available"
else
    echo "  Installing datasets..."
    pip install datasets --quiet
    print_ok "datasets installed"
fi

# Check auto-round (core quantization library)
if python3 -c "import auto_round" 2>/dev/null; then
    print_ok "auto-round installed"
else
    echo "  Installing auto-round..."
    pip install auto-round --quiet
    print_ok "auto-round installed"
fi

# Check optimum (needed for some models)
if python3 -c "import optimum" 2>/dev/null; then
    print_ok "optimum installed"
else
    echo "  Installing optimum..."
    pip install optimum --quiet
    print_ok "optimum installed"
fi

# ============================================
# Download Source Model
# ============================================
# Derive output directory name
repo_name=$(basename "$SOURCE_REPO")
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$MODELS_DIR/${repo_name}-int4-AutoRound"
fi

# Source model download path
SOURCE_DIR="$MODELS_DIR/${repo_name}"

print_header "Downloading Source Model"
echo "  Repo: $SOURCE_REPO"
echo "  Destination: $SOURCE_DIR"
echo ""

if [ -d "$SOURCE_DIR" ]; then
    print_warn "Source directory already exists: $SOURCE_DIR"
    read -p "  Continue (will use existing files)? (Y/n): " confirm
    if [[ "$confirm" =~ ^[Nn]$ ]]; then
        echo "  Aborting."
        exit 0
    fi
else
    mkdir -p "$MODELS_DIR"
    hf download "$SOURCE_REPO" \
        --local-dir "$SOURCE_DIR" \

    print_ok "Source model downloaded"
fi

# Show source model size
source_size=$(du -sh "$SOURCE_DIR" | cut -f1)
echo "  Source size: $source_size"

# If download-only mode, stop here
if [ "$DOWNLOAD_ONLY" = true ]; then
    echo ""
    print_ok "Download complete. Source model at: $SOURCE_DIR"
    echo ""
    echo "  To run quantization:"
    echo "  $0 $SOURCE_REPO --output-dir $OUTPUT_DIR"
    exit 0
fi

# ============================================
# Quantization
# ============================================
print_header "Starting INT4 AutoRound Quantization"
echo "  Source:      $SOURCE_DIR"
echo "  Output:      $OUTPUT_DIR"
echo "  Bits:        $BITS"
echo "  Group size:  $GROUP_SIZE"
echo "  Symmetric:   $SYMMETRIC"
echo "  Samples:     $CALIBRATION_SAMPLES"
echo "  Seq length:  $SEQLEN"
echo "  Iterations:  $ITERS"
echo "  Recipe:      $RECIPE"
echo "  Parallel:    $PARALLEL"
echo "  Batch size:  $BATCH_SIZE"
echo "  Low GPU mem: $LOW_GPU_MEM"
echo ""
echo "  This may take a while depending on model size..."
echo "  For large models, ensure sufficient VRAM/RAM is available."
echo ""
read -p "  Continue? (Y/n): " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo "  Aborting."
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Export variables for the Python script
export SOURCE_DIR OUTPUT_DIR BITS GROUP_SIZE SYMMETRIC CALIBRATION_SAMPLES CALIBRATION_DATASET SEQLEN ITERS LOW_GPU_MEM RECIPE PARALLEL BATCH_SIZE

# Force unbuffered output so progress is visible immediately
export PYTHONUNBUFFERED=1

# Run quantization via Python (-u forces unbuffered stdout/stderr)
python3 -u << 'PYEOF'
import os
import sys
import time
import traceback

def log(msg):
    """Log with timestamp for debugging"""
    elapsed = time.time() - overall_start
    hours, remainder = divmod(elapsed, 3600)
    mins, secs = divmod(remainder, 60)
    print(f"[{int(hours):02d}:{int(mins):02d}:{int(secs):02d}] {msg}", flush=True)

overall_start = time.time()

import torch

# Set CPU threading BEFORE any computation (env vars alone don't work)
torch.set_num_threads(24)           # Intra-op parallelism (within a single op)
torch.set_num_interop_threads(24)   # Inter-op parallelism (across ops/layers)
log(f"PyTorch threads: intra={torch.get_num_threads()}, inter={torch.get_num_interop_threads()}")
log(f"PyTorch version: {torch.__version__}")
log(f"GPU available: {torch.cuda.is_available()}, XPU available: {hasattr(torch, 'xpu') and torch.xpu.is_available()}")

# Configuration from environment
source_dir = os.environ["SOURCE_DIR"]
output_dir = os.environ["OUTPUT_DIR"]
bits = int(os.environ["BITS"])
group_size = int(os.environ["GROUP_SIZE"])
symmetric = os.environ["SYMMETRIC"].lower() == "true"
calibration_samples = int(os.environ["CALIBRATION_SAMPLES"])
seqlen = int(os.environ["SEQLEN"])
iters = int(os.environ["ITERS"])
low_gpu_mem = os.environ["LOW_GPU_MEM"].lower() == "true"
parallel = os.environ["PARALLEL"].lower() == "true"
batch_size = int(os.environ["BATCH_SIZE"])

log(f"\n{'='*60}")
log("⚠️  BENCHMARK MODE — Speed optimized, quality reduced ⚠️")
log(f"{'='*60}")
log(f"Source model: {source_dir}")
log(f"Output dir:   {output_dir}")
log(f"Scheme:       W{bits}A16 (group_size={group_size}, sym={symmetric})")
log(f"Iterations:   {iters} (benchmark: reduced from 1000)")
log(f"Samples:      {calibration_samples} (benchmark: reduced from 128)")
log(f"Seq length:   {seqlen} (benchmark: reduced from 2048)")
log(f"Batch size:   {batch_size}")
log(f"Low GPU mem:  {low_gpu_mem}")
log(f"Parallel:     {parallel}")

# Check source model files
log("\n--- Checking model files ---")
import glob
safetensors = sorted(glob.glob(os.path.join(source_dir, "*.safetensors")))
total_size = sum(os.path.getsize(f) for f in safetensors)
log(f"Found {len(safetensors)} safetensor files, total {total_size/1e9:.1f}GB")

# Print RAM before
import resource
ram_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
log(f"RAM before start: {ram_before:.0f}MB")

# ============================================================
# Use auto_round 0.14.2 API
# AutoRound handles model loading, quantization, and saving
# as a single pipeline — no manual from_pretrained needed
# ============================================================
log(f"\n{'='*60}")
log("PHASE: Starting AutoRound (handles loading + quantize + save)")
log(f"{'='*60}")

start_time = time.time()

try:
    from auto_round import AutoRound, QuantizationScheme

    # Build quantization scheme
    scheme = QuantizationScheme(
        bits=bits,
        group_size=group_size,
        sym=symmetric,
    )
    log(f"Scheme: {scheme}")

    # Device map for tuning — use CPU for large models on XPU
    # AutoRound tunes block-by-block, so CPU works fine
    device_map = "cpu"
    log(f"Device map: {device_map}")

    # low_cpu_mem_usage: saves quantized blocks immediately after packing
    # This is CRITICAL for large models — prevents OOM during save
    log(f"low_cpu_mem_usage: True (saves block-by-block)")
    log(f"low_gpu_mem_usage: {low_gpu_mem}")

    log(f"\n--- Creating AutoRound instance ---")
    log("AutoRound will handle: model loading → calibration → quantization → save")
    log("This may take a while for large models...")

    # PATCH 1: Disable MoE fused module replacement for INT4 quantization
    # The replacement tries to wrap INT4 tensors in nn.Parameter, which fails
    # because integer dtypes can't require gradients
    # See: https://github.com/intel/auto-round/issues (MoE INT4 support pending)
    log("--- Patch 1: disabling MoE fused module replacement (INT4 incompatibility) ---")
    import auto_round.modeling.fused_moe.replace_modules as replace_mod
    original_apply = replace_mod.apply_replacements
    def patched_apply(model, *args, **kwargs):
        log(">>> Skipping MoE fused replacement (INT4 tensors can't be nn.Parameter)")
        return model
    replace_mod.apply_replacements = patched_apply
    log(">>> Patch 1 applied successfully")

    # PATCH 2: Fix DeepSeek V4 attention forward signature for auto_round compatibility
    # DeepseekV4Attention.forward() expects: hidden_states, position_embeddings(dict), position_ids, attention_mask, past_key_values
    # But auto_round passes hidden_states + **kwargs with position_embeddings as 3D tensor (wrong format)
    # Solution: create local rotary_emb instance and compute proper position_embeddings dict
    log("--- Patch 2: fixing DeepSeek V4 attention forward signature ---")
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as dv4_module
    orig_attn_forward = dv4_module.DeepseekV4Attention.forward
    # Single shared rotary embedding (lazy init) - DeepseekV4RotaryEmbedding handles both layer_types internally
    _rotary_emb = [None]  # mutable container for closure
    def get_rotary_emb(config):
        if _rotary_emb[0] is None:
            _rotary_emb[0] = dv4_module.DeepseekV4RotaryEmbedding(config)
        return _rotary_emb[0]
    def patched_attn_forward(self, hidden_states, **kwargs):
        # Extract from kwargs
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        past_key_values = kwargs.pop("past_key_values", None)
        # Remove invalid position_embeddings from kwargs (3D tensor, wrong format)
        kwargs.pop("position_embeddings", None)
        # Generate position_ids if not provided
        if position_ids is None:
            batch_size, seq_len, _ = hidden_states.shape
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        # Compute proper position_embeddings using shared rotary_emb
        rotary_emb = get_rotary_emb(self.config)
        position_embeddings = {
            "main": rotary_emb(hidden_states, position_ids=position_ids, layer_type="main"),
            "compress": rotary_emb(hidden_states, position_ids=position_ids, layer_type="compress"),
        }
        return orig_attn_forward(self, hidden_states, position_embeddings, position_ids, attention_mask, past_key_values, **kwargs)
    dv4_module.DeepseekV4Attention.forward = patched_attn_forward
    log(">>> Patch 2 applied successfully")

    # PATCH 3: Fix MoE gate forward signatures for auto_round compatibility
    # auto_round calls self.gate(hidden_states, input_ids) but routers have different signatures:
    #   - DeepseekV4TopKRouter.forward(self, hidden_states)  -- no input_ids param
    #   - DeepseekV4HashRouter.forward(self, hidden_states, input_ids)  -- requires input_ids
    log("--- Patch 3: fixing MoE gate forward signatures ---")
    
    # Patch TopKRouter: accept input_ids arg but ignore it (original doesn't use it)
    orig_topk_forward = dv4_module.DeepseekV4TopKRouter.forward
    def patched_topk_forward(self, hidden_states, input_ids=None):
        # Ignore input_ids, original only takes hidden_states
        return orig_topk_forward(self, hidden_states)
    dv4_module.DeepseekV4TopKRouter.forward = patched_topk_forward
    log(">>> Patched DeepseekV4TopKRouter.forward (accepts input_ids but ignores it)")
    
    # Patch HashRouter: provide dummy input_ids when None (original requires it for tid2eid)
    orig_hash_forward = dv4_module.DeepseekV4HashRouter.forward
    def patched_hash_forward(self, hidden_states, input_ids=None):
        if input_ids is None:
            flat = hidden_states.reshape(-1, self.hidden_dim)
            # Use zeros as dummy token IDs (all tokens map to same expert set)
            # This is acceptable for quantization calibration where exact routing doesn't matter
            input_ids = torch.zeros(flat.shape[0], dtype=torch.long, device=hidden_states.device)
        return orig_hash_forward(self, hidden_states, input_ids)
    dv4_module.DeepseekV4HashRouter.forward = patched_hash_forward
    log(">>> Patched DeepseekV4HashRouter.forward (provides dummy input_ids when None)")
    
    log(">>> Patch 3 applied successfully")

    # PATCH 4: Disable fine-grained FP8 MoE experts for models with hidden_size != moe_intermediate_size
    # DeepSeek V4 has hidden_size=4096 but moe_intermediate_size=2048
    # The fine-grained FP8 kernel expects matching K dimensions, causing AssertionError
    # Fix: force _experts_implementation="eager" in config so the decorator falls back to original_forward
    # which uses proper grouped_mm with _grouped_linear that handles 3D weight tensors correctly
    log("--- Patch 4: forcing eager expert implementation (bypass FP8 kernel) ---")
    import json as _json
    config_path = os.path.join(source_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as _f:
            _cfg = _json.load(_f)
        old_impl = _cfg.get("_experts_implementation", "NOT SET")
        _cfg["_experts_implementation"] = "eager"
        with open(config_path, "w") as _f:
            _json.dump(_cfg, _f, indent=2)
        log(f">>> Set _experts_implementation='eager' (was: {old_impl})")
        log(">>> Patch 4 applied - config.json updated")
    else:
        log(">>> No config.json found, skipping Patch 4")

    # PATCH 5: Disable deepgemm FP8 kernel dispatch (force Triton/eager fallback)
    # The FP8 model needs replace_with_fp8_linear to properly remap w1/w2/w3 → gate_up_proj/down_proj.
    # But deepgemm kernels fail on XPU with K mismatch errors. We disable deepgemm to use
    # Triton fallback instead, which handles hidden_size != moe_intermediate_size correctly.
    log("--- Patch 5: disabling deepgemm FP8 kernel dispatch (use Triton fallback) ---")
    import transformers.integrations.finegrained_fp8 as fg_fp8_module
    # Disable deepgemm globally on FP8Experts and FP8Linear classes
    fg_fp8_module.FP8Experts._deepgemm_disabled = True
    fg_fp8_module.FP8Linear._deepgemm_disabled = True
    # Patch _disable_deepgemm_on_multi_device to force disable on all modules post-load
    original_disable_deepgemm = fg_fp8_module._disable_deepgemm_on_multi_device
    def patched_disable_deepgemm(model):
        original_disable_deepgemm(model)
        for name, module in model.named_modules():
            if hasattr(module, '_deepgemm_disabled'):
                module._deepgemm_disabled = True
    fg_fp8_module._disable_deepgemm_on_multi_device = patched_disable_deepgemm
    log(">>> Patch 5 applied — deepgemm disabled, Triton fallback enabled")

    # PATCH 6: Let FP8→HP dequantization work normally
    # With FP8 modules loading properly (Patch 5 no longer blocks replace_with_fp8_linear),
    # FP8Linear has .weight_scale attributes that FP8Handler.convert_layer needs.
    # No patching needed — dequantization works correctly.
    log("--- Patch 6: FP8→HP dequantization enabled (no patch needed) ---")
    log(">>> Patch 6 skipped — FP8Handler.convert_layer works with proper FP8 modules")

    ar = AutoRound(
        model=source_dir,              # Pass path, not pre-loaded model
        scheme=scheme,
        iters=iters,                   # 100 for benchmark (speed)
        nsamples=calibration_samples,  # 32 for benchmark (speed)
        seqlen=seqlen,                 # 512 for benchmark (speed)
        batch_size=batch_size,
        low_gpu_mem_usage=low_gpu_mem,
        low_cpu_mem_usage=True,        # CRITICAL: saves block-by-block
        device_map=device_map,
        enable_torch_compile=False,    # Disable for reproducibility
        trust_remote_code=True,
    )
    ar_time = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"✓ AutoRound quantization complete in {ar_time:.0f}s ({ar_time/60:.1f} min)")
    log(f"{'='*60}")

    # Save — with low_cpu_mem_usage=True, most blocks are already saved
    # This just finalizes the output
    log(f"\n{'='*60}")
    log(f"PHASE: Finalizing save to: {output_dir}")
    log(f"{'='*60}")
    t0 = time.time()
    ar.quantize_and_save(output_dir=output_dir, format="auto_round")
    save_time = time.time() - t0
    log(f"✓ Save complete in {save_time:.0f}s")

except Exception as e:
    log(f"\n!!! ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Show results
if os.path.exists(output_dir):
    output_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(output_dir)
        for filename in filenames
    )
    output_size_gb = output_size / (1024**3)
    log(f"\nOutput size: {output_size_gb:.1f}GB")

    # Count output files
    output_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    log(f"Output files: {len(output_files)} safetensors")

ram_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
log(f"Peak RAM: {ram_after:.0f}MB (delta: {ram_after-ram_before:.0f}MB)")

total_time = time.time() - overall_start
log(f"\n{'='*60}")
log(f"✓ ALL DONE — Total time: {total_time:.0f}s ({total_time/3600:.1f} hours)")
log(f"{'='*60}")
PYEOF

print_ok "Quantization complete!"
echo ""
echo "  Output: $OUTPUT_DIR"
echo ""
echo "  To deploy with vLLM:"
echo "  VLLM_TARGET_DEVICE=xpu python3 -m vllm.entrypoints.openai.api_server \\"
echo "      --model $OUTPUT_DIR \\"
echo "      --tensor-parallel-size 4 \\"
echo "      --gpu-memory-utilization 0.80 \\"
echo "      --kv-cache-dtype fp8"
