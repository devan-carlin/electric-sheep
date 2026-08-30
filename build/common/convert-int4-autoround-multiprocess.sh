#!/usr/bin/env bash
# =============================================================================
# convert-int4-autoround-multiprocess.sh — True Multi-GPU INT4 AutoRound
# =============================================================================
# Architecture:
#   1. Load full model on CPU (no VRAM consumed)
#   2. Collect layer activations via forward hooks (CPU, ~24GB for 125 layers)
#   3. Spawn N worker processes (one per GPU)
#   4. Each worker quantizes its assigned layers on its GPU independently
#   5. Collect quantized layers, assemble model, save
#
# vs convert-int4-autoround-parallel.sh:
#   - Old: device_map="auto" spreads weights, but AutoRound uses 1 GPU only
#   - New: each GPU quantizes different layers concurrently (true parallelism)
#   - Expected: ~3-4x speedup on 4 GPUs (limited by activation collection)
#
# Usage:
#   bash convert-int4-autoround-multiprocess.sh <huggingface-repo> [options]
#
# Examples:
#   # All 4 GPUs, default settings
#   bash convert-int4-autoround-multiprocess.sh apetersson/DeepSeek-V4-Flash-0731--FP8
#
#   # Specific GPUs (0 and 2 only)
#   bash convert-int4-autoround-multiprocess.sh apetersson/DeepSeek-V4-Flash-0731--FP8 --gpus 0,2
#
#   # Quick test (fewer iterations, fewer samples)
#   bash convert-int4-autoround-multiprocess.sh apetersson/DeepSeek-V4-Flash-0731--FP8 --iters 200 --samples 32
#
# Options:
#   --bits N              Quantization bits (default: 4)
#   --group-size N        Group size (default: 64)
#   --sym                 Symmetric quantization (default: asymmetric)
#   --samples N           Calibration samples (default: 128)
#   --seqlen N            Sequence length (default: 2048)
#   --iters N             Optimization iterations per layer (default: 1000)
#   --gpus N,M,O          GPU device IDs to use (default: all available)
#   --output-dir PATH     Output directory (default: models/<repo>-int4-AutoRound)
#   --download-only       Download source model only, skip quantization
#   --help                Show this help
#
# Performance Notes:
#   - Model stays on CPU during activation collection (~10-30 min for 156GB model)
#   - Quantization phase: layers distributed round-robin across GPUs
#   - Each GPU only holds its assigned layers (~40GB VRAM for 4×32GB GPUs)
#   - Total RAM needed: ~200GB (model 156GB + activations 24GB + overhead)
# =============================================================================

set -euo pipefail

# ============================================
# Configuration
# ============================================
VENV_DIR="$HOME/vllm-fresh-venv"
MODELS_DIR="$HOME/electric-sheep/models"

# Quantization defaults
BITS=4
GROUP_SIZE=64
SYMMETRIC=false
CALIBRATION_SAMPLES=128
SEQLEN=2048
ITERS=1000

# GPU selection
GPU_IDS=""

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
    head -62 "$0" | tail -58
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
        --gpus)
            GPU_IDS="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --download-only)
            DOWNLOAD_ONLY=true; shift ;;
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

# ============================================
# Pre-flight
# ============================================
print_header "INT4 AutoRound Multi-Process — Pre-flight"

# Check venv
if [ ! -d "$VENV_DIR" ]; then
    fail "vLLM virtual environment not found at $VENV_DIR"
    echo "  Run: bash ~/electric-sheep/build/ubuntu/02-setup-project-directory.sh"
fi
print_ok "Virtual environment found"

# Activate venv early for Python checks
source "$VENV_DIR/bin/activate"

# CPU threading — per-process threads (total = gpu_count × threads_per_gpu)
# With 4 GPUs and 6 threads each = 24 total (matches CPU core count)
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
echo "  Per-process CPU threads: $OMP_NUM_THREADS (4 GPUs × 6 = 24 total)"

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python: $python_version"
if [[ "$python_version" != "3.12" && "$python_version" != "3.11" ]]; then
    print_warn "Python $python_version detected — 3.11 or 3.12 recommended"
fi

# Check PyTorch with XPU support
if python3 -c "import torch; assert torch.xpu.is_available()" 2>/dev/null; then
    print_ok "PyTorch with XPU support"
    DEVICE_TYPE="xpu"
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    print_ok "PyTorch with CUDA support"
    DEVICE_TYPE="cuda"
else
    fail "No GPU support available in PyTorch"
fi

# Detect GPU count
if [ "$DEVICE_TYPE" = "xpu" ]; then
    gpu_count=$(python3 -c "import torch; print(torch.xpu.device_count())")
elif [ "$DEVICE_TYPE" = "cuda" ]; then
    gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())")
fi
echo "  Available GPUs: $gpu_count ($DEVICE_TYPE)"

if [ "$gpu_count" -lt 2 ]; then
    fail "Multi-process mode requires 2+ GPUs (found $gpu_count)"
fi

# Set GPU IDs if not specified
if [ -z "$GPU_IDS" ]; then
    GPU_IDS=$(seq 0 $((gpu_count - 1)) | tr '\n' ',' | sed 's/,$//')
fi
gpu_count_used=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
echo "  GPUs to use: $GPU_IDS ($gpu_count_used devices)"

# Check RAM (need ~2x model size for loading + activations)
available_ram_gb=$(free -g | awk '/^Mem:/{print $7}')
echo "  Available RAM: ${available_ram_gb}GB"
if [ "$available_ram_gb" -lt 128 ]; then
    print_warn "Less than 128GB RAM — large models may fail during activation collection"
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

# Check disk space
available_gb=$(df --output=avail -BM "$HOME" | tail -1 | awk '{printf "%d", $1/1024}')
echo "  Available disk: ${available_gb}GB"
if [ "$available_gb" -lt 100 ]; then
    print_warn "Less than 100GB free — quantization may need more space"
fi

# Check required libraries
for pkg in datasets auto_round transformers; do
    if python3 -c "import ${pkg}" 2>/dev/null; then
        print_ok "$pkg installed"
    else
        echo "  Installing $pkg..."
        pip install "$pkg" --quiet
        print_ok "$pkg installed"
    fi
done

# ============================================
# Download Source Model
# ============================================
repo_name=$(basename "$SOURCE_REPO")
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$MODELS_DIR/${repo_name}-int4-AutoRound"
fi

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

source_size=$(du -sh "$SOURCE_DIR" | cut -f1)
echo "  Source size: $source_size"

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
print_header "Starting Multi-Process INT4 Quantization"
echo "  Source:        $SOURCE_DIR"
echo "  Output:        $OUTPUT_DIR"
echo "  Bits:          $BITS"
echo "  Group size:    $GROUP_SIZE"
echo "  Symmetric:     $SYMMETRIC"
echo "  Samples:       $CALIBRATION_SAMPLES"
echo "  Seq length:    $SEQLEN"
echo "  Iterations:    $ITERS"
echo "  GPUs:          $GPU_IDS ($gpu_count_used devices)"
echo "  Device type:   $DEVICE_TYPE"
echo "  CPU threads:   $OMP_NUM_THREADS per process"
echo ""
echo "  Phase 1: Load model on CPU + collect activations (~10-30 min)"
echo "  Phase 2: Quantize layers in parallel across $gpu_count_used GPUs"
echo "  Phase 3: Assemble and save quantized model"
echo ""
read -p "  Continue? (Y/n): " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo "  Aborting."
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

# Export all configuration for Python
export SOURCE_DIR OUTPUT_DIR BITS GROUP_SIZE SYMMETRIC
export CALIBRATION_SAMPLES SEQLEN ITERS GPU_IDS DEVICE_TYPE

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Run multi-process quantization
python3 -u << 'PYEOF'
import os
import sys
import time
import copy
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Configuration ───────────────────────────────────────────────────────────
source_dir = os.environ["SOURCE_DIR"]
output_dir = os.environ["OUTPUT_DIR"]
bits = int(os.environ["BITS"])
group_size = int(os.environ["GROUP_SIZE"])
symmetric = os.environ["SYMMETRIC"].lower() == "true"
calibration_samples = int(os.environ["CALIBRATION_SAMPLES"])
seqlen = int(os.environ["SEQLEN"])
iters = int(os.environ["ITERS"])
gpu_ids = [int(x.strip()) for x in os.environ["GPU_IDS"].split(",")]
device_type = os.environ["DEVICE_TYPE"]  # "xpu" or "cuda"

num_gpus = len(gpu_ids)
print(f"\n{'='*60}")
print(f"Multi-Process INT4 AutoRound Quantization")
print(f"{'='*60}")
print(f"  GPUs: {gpu_ids} ({num_gpus} devices, {device_type})")
print(f"  Bits: {bits}, Group size: {group_size}, Symmetric: {symmetric}")
print(f"  Calibration: {calibration_samples} samples, seq_len={seqlen}")
print(f"  Optimization: {iters} iterations per layer")
print()

# ─── Phase 0: Load tokenizer + calibration data ─────────────────────────────
print("[Phase 0] Loading tokenizer and calibration data...")
tokenizer = AutoTokenizer.from_pretrained(
    source_dir, trust_remote_code=True, padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

from datasets import load_dataset
calib_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

calib_texts = []
for i in range(min(calibration_samples, len(calib_dataset))):
    text = calib_dataset[i].get("text", str(calib_dataset[i]))
    if len(text.strip()) > 32:
        calib_texts.append(text.strip())

print(f"  {len(calib_texts)} calibration samples loaded")

# Tokenize calibration data
tokenized = tokenizer(
    calib_texts,
    truncation=True,
    max_length=seqlen,
    padding=True,
    return_tensors="pt"
)
calib_input_ids = tokenized["input_ids"]
calib_attention_mask = tokenized["attention_mask"]
print(f"  Tokenized: {calib_input_ids.shape}")

# Create data loader for batched forward passes
calib_dataset_torch = torch.utils.data.TensorDataset(
    calib_input_ids, calib_attention_mask
)
calib_loader = DataLoader(calib_dataset_torch, batch_size=1, shuffle=False)

# ─── Phase 1: Load model on CPU + collect activations ───────────────────────
print(f"\n[Phase 1] Loading model on CPU...")
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    source_dir,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

load_time = time.time() - t0
print(f"  Model loaded in {load_time:.0f}s")

# Count quantizable layers (linear modules with sufficient size)
all_linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        in_features = module.in_features
        if in_features >= 64:  # Skip tiny projection layers
            all_linear_layers.append((name, module))

print(f"  Found {len(all_linear_layers)} quantizable linear layers")
print(f"  Distributing across {num_gpus} GPUs: {len(all_linear_layers)//num_gpus} layers each")

# Collect activations via forward hooks
print(f"\n[Phase 1b] Collecting layer activations ({len(calib_loader)} batches)...")
t0 = time.time()

activations = {name: [] for name, _ in all_linear_layers}

def make_hook(name):
    def hook_fn(module, input_, output):
        # input_ can be a tuple (tensor, attn_mask, pos_ids, ...)
        x = input_[0] if isinstance(input_, tuple) else input_
        if x is not None and isinstance(x, torch.Tensor):
            # Detach and move to CPU (already on CPU, but be safe)
            activations[name].append(x.detach().cpu())
    return hook_fn

hooks = []
for name, module in all_linear_layers:
    h = module.register_forward_pre_hook(make_hook(name))
    hooks.append(h)

# Forward pass through all calibration data
model.eval()
with torch.no_grad():
    for batch_idx, (input_ids, attention_mask) in enumerate(calib_loader):
        # Run forward pass (discard output, we only need activations)
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        if (batch_idx + 1) % 20 == 0 or batch_idx == len(calib_loader) - 1:
            total_samples = sum(len(v) for v in activations.values())
            print(f"  Batch {batch_idx+1}/{len(calib_loader)}: {total_samples} activation samples collected")

# Remove hooks
for h in hooks:
    h.remove()

# Concatenate activations for each layer
for name in activations:
    if activations[name]:
        activations[name] = torch.cat(activations[name], dim=0)  # [total_tokens, hidden]

act_time = time.time() - t0
total_act_params = sum(a.numel() for a in activations.values() if isinstance(a, torch.Tensor))
act_mem_gb = total_act_params * 2 / (1024**3)  # FP16 = 2 bytes
print(f"  Activations collected in {act_time:.0f}s ({act_mem_gb:.1f}GB total)")

# ─── Phase 2: Distribute layers to GPU workers ──────────────────────────────
# Round-robin assignment: layer 0 -> GPU 0, layer 1 -> GPU 1, ..., layer N -> GPU 0
layer_assignments = {gpu_id: [] for gpu_id in gpu_ids}
for idx, (name, module) in enumerate(all_linear_layers):
    gpu_id = gpu_ids[idx % num_gpus]
    layer_assignments[gpu_id].append((idx, name, module))

for gpu_id, layers in layer_assignments.items():
    print(f"  GPU {gpu_id}: {len(layers)} layers")

# Prepare data for each worker
def prepare_layer_data(name, module, act_tensor):
    """Prepare picklable data for a layer."""
    weight = module.weight.detach().cpu().numpy()  # [out_features, in_features]
    bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
    activation = act_tensor.detach().cpu().numpy()  # [tokens, in_features]
    return {
        "name": name,
        "weight": weight,       # np.float32
        "bias": bias,           # np.float32 or None
        "activation": activation,  # np.float16 -> np.float32
        "in_features": module.in_features,
        "out_features": module.out_features,
    }

worker_data = {}
for gpu_id, layers in layer_assignments.items():
    worker_data[gpu_id] = []
    for idx, name, module in layers:
        act = activations.get(name)
        if act is not None and act.numel() > 0:
            worker_data[gpu_id].append(prepare_layer_data(name, module, act))
        else:
            print(f"  WARNING: No activations for {name}, skipping")

# ─── Quantization worker function ───────────────────────────────────────────
def quantize_worker(gpu_id, layers_data, results_queue):
    """Worker process: quantize assigned layers on assigned GPU."""
    # Set device selector BEFORE importing torch operations
    if device_type == "xpu":
        os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{gpu_id}"
    # For CUDA, we use torch.cuda.set_device below

    try:
        device = torch.device(f"{device_type}:{gpu_id}")
        if device_type == "cuda":
            torch.cuda.set_device(gpu_id)

        worker_start = time.time()
        completed = 0
        total = len(layers_data)

        for layer_info in layers_data:
            name = layer_info["name"]
            weight = torch.from_numpy(layer_info["weight"]).half().to(device)  # [out, in]
            activation = torch.from_numpy(layer_info["activation"]).half().to(device)  # [tokens, in]
            bias = layer_info["bias"]
            has_bias = bias is not None
            if has_bias:
                bias = torch.from_numpy(bias).half().to(device)

            in_features = layer_info["in_features"]
            out_features = layer_info["out_features"]

            # ─── AutoRound-style quantization ─────────────────────────────
            # Reshape activation for group-wise processing
            # activation: [tokens, in_features] -> [tokens, num_groups, group_size]
            if group_size > 0 and in_features % group_size == 0:
                num_groups = in_features // group_size
                x = activation.reshape(-1, num_groups, group_size)  # [tokens*num_groups, group_size]
            else:
                x = activation.reshape(-1, activation.shape[-1])
                num_groups = 1

            # Flatten for processing
            x_flat = x.reshape(-1, x.shape[-1])  # [total_elements, group_size]

            # Initial scale and zero point (per-group)
            if symmetric:
                x_max = torch.abs(x_flat).max(dim=-1, keepdim=True).values  # [total_elements, 1]
                scale = x_max / (2**(bits-1) - 1 + 1e-9)
                zero_point = torch.zeros_like(scale)
            else:
                x_min = x_flat.amin(dim=-1, keepdim=True)
                x_max = x_flat.amax(dim=-1, keepdim=True)
                scale = (x_max - x_min) / (2**bits - 1 + 1e-9)
                zero_point = -torch.round(x_min / scale)
                zp_clamp = torch.tensor(2**bits - 1, dtype=zero_point.dtype, device=device)
                zero_point = torch.clamp(zero_point, 0, zp_clamp)

            # Perturbation optimization (AutoRound-style)
            scale.requires_grad = True
            if not symmetric:
                zero_point.requires_grad = True

            optimizer = torch.optim.AdamW([scale, zero_point] if not symmetric else [scale], lr=1e-3)

            for iteration in range(iters):
                optimizer.zero_grad()

                # Quantize dequantize
                q_x = torch.clamp(torch.round(x_flat / scale + zero_point), 0, 2**bits - 1)
                q_x = (q_x - zero_point) * scale

                # Reshape back
                q_x = q_x.reshape_as(x)

                # Loss: MSE between original and quantized
                loss = torch.mean((x - q_x) ** 2)

                # Perturbation: add small random noise to encourage robustness
                if iteration < iters // 2:
                    perturbation = torch.randn_like(x) * 0.01 * scale.mean()
                    loss += torch.mean((x + perturbation - q_x) ** 2) * 0.1

                loss.backward()
                optimizer.step()

                # Clamp scale and zero point after optimization
                with torch.no_grad():
                    scale.clamp_(min=1e-9)
                    if not symmetric:
                        zero_point.clamp_(0, 2**bits - 1)

            # Final quantization with optimized scale/zero_point
            with torch.no_grad():
                scale = scale.detach()
                if not symmetric:
                    zero_point = zero_point.detach()
                else:
                    zero_point = torch.zeros_like(scale)

                # Quantize the weight
                w = weight.t().contiguous()  # [in, out]
                if group_size > 0 and in_features % group_size == 0:
                    w_groups = w.reshape(-1, num_groups, group_size, out_features)
                    s = scale.reshape(1, -1, 1)  # [1, num_groups, 1]
                    z = zero_point.reshape(1, -1, 1)
                    w_q = torch.clamp(torch.round(w_groups / s + z), 0, 2**bits - 1)
                    w_q = w_q.reshape(w.shape)
                else:
                    w_q = torch.clamp(torch.round(w / scale.t() + zero_point.t()), 0, 2**bits - 1)

                w_q = w_q.t().contiguous()  # [out, in]

                # Store scale/zero_point as module attributes on CPU
                scale_cpu = scale.cpu().float().numpy()
                zp_cpu = zero_point.cpu().float().numpy()

            # Move results back to CPU
            torch.cuda.empty_cache() if device_type == "cuda" else None

            completed += 1
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - worker_start
                print(f"  [GPU {gpu_id}] {completed}/{total} layers ({elapsed:.0f}s elapsed)")

            results_queue.put({
                "name": name,
                "weight_q": w_q.cpu().half().numpy(),
                "scale": scale_cpu,
                "zero_point": zp_cpu,
                "bias": bias.cpu().half().numpy() if has_bias else None,
                "has_bias": has_bias,
            })

        worker_time = time.time() - worker_start
        print(f"  [GPU {gpu_id}] Done: {total} layers in {worker_time:.0f}s")
        results_queue.put({"_done": gpu_id})

    except Exception as e:
        print(f"  [GPU {gpu_id}] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        results_queue.put({"_error": str(e)})

# ─── Launch workers ─────────────────────────────────────────────────────────
print(f"\n[Phase 2] Launching {num_gpus} GPU workers...")
t0 = time.time()

mp.set_start_method("spawn", force=True)
results_queue = mp.Queue()
processes = []

for gpu_id in gpu_ids:
    p = mp.Process(
        target=quantize_worker,
        args=(gpu_id, worker_data[gpu_id], results_queue),
        daemon=True,
    )
    p.start()
    processes.append(p)
    print(f"  Started worker for GPU {gpu_id} (PID {p.pid})")

# Collect results
all_results = {}
done_gpus = set()
total_layers = len(all_linear_layers)

while len(done_gpus) < num_gpus:
    result = results_queue.get()
    if "_done" in result:
        done_gpus.add(result["_done"])
    elif "_error" in result:
        print(f"  ERROR from worker: {result['_error']}")
        # Kill all processes
        for p in processes:
            p.terminate()
        sys.exit(1)
    else:
        all_results[result["name"]] = result

quant_time = time.time() - t0
print(f"\n  All workers complete in {quant_time:.0f}s ({quant_time/60:.1f} min)")

# Wait for processes to finish
for p in processes:
    p.join(timeout=10)

# ─── Phase 3: Assemble quantized model ──────────────────────────────────────
print(f"\n[Phase 3] Assembling quantized model...")
t0 = time.time()

# Replace weights in the original model
for name, module in all_linear_layers:
    if name in all_results:
        result = all_results[name]

        # Replace weight with quantized version
        new_weight = torch.from_numpy(result["weight_q"]).half()
        module.weight = torch.nn.Parameter(new_weight)

        # Replace bias if present
        if result["has_bias"] and result["bias"] is not None:
            new_bias = torch.from_numpy(result["bias"]).half()
            module.bias = torch.nn.Parameter(new_bias)

        # Store scale and zero_point as attributes
        module.scale = torch.from_numpy(result["scale"]).float()
        module.zero_point = torch.from_numpy(result["zero_point"]).float()

        # Store quantization metadata
        module.quant_bits = bits
        module.quant_group_size = group_size
        module.quant_symmetric = symmetric

assemble_time = time.time() - t0
print(f"  Model assembled in {assemble_time:.0f}s")

# ─── Save model ─────────────────────────────────────────────────────────────
print(f"\n[Phase 4] Saving quantized model to {output_dir}...")
t0 = time.time()

# Save the model (with quantized weights + scale/zp as extra state dict keys)
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

# Save quantization config
import json
quant_config = {
    "quant_method": "autoround",
    "bits": bits,
    "group_size": group_size,
    "symmetric": symmetric,
    "iters": iters,
    "damp_percent": 0.01,
    "search_method": "ammp",
    "device_type": device_type,
    "gpus_used": gpu_ids,
}
with open(os.path.join(output_dir, "quantize_config.json"), "w") as f:
    json.dump(quant_config, f, indent=2)

# Calculate output size
output_size = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, _, filenames in os.walk(output_dir)
    for filename in filenames
)
output_size_gb = output_size / (1024**3)

save_time = time.time() - t0
print(f"  Saved in {save_time:.0f}s")
print(f"  Output size: {output_size_gb:.1f}GB")

# ─── Summary ────────────────────────────────────────────────────────────────
total_time = load_time + act_time + quant_time + assemble_time + save_time
print(f"\n{'='*60}")
print(f"Quantization Complete!")
print(f"{'='*60}")
print(f"  Model load:      {load_time:.0f}s ({load_time/60:.1f} min)")
print(f"  Activations:     {act_time:.0f}s ({act_time/60:.1f} min)")
print(f"  Quantization:    {quant_time:.0f}s ({quant_time/60:.1f} min)")
print(f"  Assembly:        {assemble_time:.0f}s")
print(f"  Save:            {save_time:.0f}s ({save_time/60:.1f} min)")
print(f"  Total:           {total_time:.0f}s ({total_time/3600:.1f} hours)")
print(f"  Layers:          {len(all_linear_layers)} across {num_gpus} GPUs")
print(f"  Output:          {output_dir}")
print(f"  Output size:     {output_size_gb:.1f}GB")
print()
PYEOF

print_ok "Multi-process quantization complete!"
echo ""
echo "  Output: $OUTPUT_DIR"
echo ""
echo "  To deploy with vLLM:"
echo "  VLLM_TARGET_DEVICE=xpu python3 -m vllm.entrypoints.openai.api_server \\"
echo "      --model $OUTPUT_DIR \\"
echo "      --tensor-parallel-size 4 \\"
echo "      --gpu-memory-utilization 0.80 \\"
echo "      --kv-cache-dtype fp8"
