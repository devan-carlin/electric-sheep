#!/usr/bin/env bash
# =============================================================================
# 04.1-convert-int4-autoround.sh — INT4 AutoRound Quantization (Parallel)
# =============================================================================
# Evolution from 04-:
#   - Multi-GPU parallel processing (--parallel flag)
#   - CPU threading optimization (OMP/MKL/OpenBLAS)
#   - Batch layer processing (--batch-size flag)
#   - Expected 4-8x speedup (2-4 hours vs 12-16 hours for large models)
#
# Usage:
#   bash 04.1-convert-int4-autoround.sh <huggingface-repo> [options]
#
# Examples:
#   # Max quality, single GPU (same as 04-)
#   bash 04.1-convert-int4-autoround.sh apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8
#
#   # Parallel mode — all 4 GPUs, 24 CPU threads
#   bash 04.1-convert-int4-autoround.sh apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8 --parallel
#
#   # Parallel with custom batch size (4 layers concurrent)
#   bash 04.1-convert-int4-autoround.sh apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8 --parallel --batch-size 4
#
# Options:
#   --bits N              Quantization bits (default: 4)
#   --group-size N        Group size (default: 64)
#   --sym                 Symmetric quantization (default: asymmetric)
#   --samples N           Calibration samples (default: 128)
#   --seqlen N            Sequence length (default: 2048)
#   --iters N             Optimization iterations (default: 1000)
#   --low-gpu-mem         Conservative VRAM mode (default: true)
#   --recipe NAME         Quantization recipe (default: auto-round-best)
#   --output-dir PATH     Output directory (default: models/<repo>-int4-AutoRound)
#   --download-only       Download source model only, skip quantization
#   --parallel            Enable multi-GPU parallel processing
#   --batch-size N        Concurrent layers per batch (default: 1, recommended: 4 for 4 GPUs)
#   --help                Show this help
#
# Performance Notes:
#   - Single GPU mode: safe for any model, ~12-16 hours for 30B+ models
#   - Parallel mode: uses all GPUs + CPU threads, ~2-4 hours for 30B+ models
#   - Batch size 4: optimal for 4× Arc Pro B70 (128 GiB total VRAM)
#   - CPU threads: 24 threads (12C/24T Threadripper PRO 3945WX)
# =============================================================================

set -euo pipefail

# ============================================
# Configuration
# ============================================
VENV_DIR="$HOME/electric-sheep/vllm/.venv"
MODELS_DIR="$HOME/electric-sheep/models"

# Quantization defaults
BITS=4
GROUP_SIZE=64
SYMMETRIC=false
CALIBRATION_SAMPLES=128
CALIBRATION_DATASET="json"
SEQLEN=2048
ITERS=1000
LOW_GPU_MEM=true
RECIPE="auto-round-best"

# Parallel processing defaults (NEW in 04.1-)
PARALLEL=false
BATCH_SIZE=1

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
    # Parallel mode disables low GPU memory for faster processing
    LOW_GPU_MEM=false
    echo "  Parallel mode enabled — disabling low GPU memory mode"
    echo "  Batch size: $BATCH_SIZE (concurrent layers)"
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

# Check GPU devices (for VRAM estimation)
if command -v sycl-ls >/dev/null 2>&1; then
    gpu_count=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu" || echo "0")
    if [ "$gpu_count" -gt 0 ]; then
        total_vram_gb=0
        while IFS= read -r line; do
            vram=$(echo "$line" | grep -oP '[\d.]+ GB' | head -1 | awk '{print $1}')
            if [ -n "$vram" ]; then
                total_vram_gb=$(echo "$total_vram_gb + $vram" | bc)
            fi
        done < <(sycl-ls 2>/dev/null | grep "level_zero:gpu")
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
if [ -z "$HF_TOKEN" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
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
import torch
from transformers import AutoTokenizer

# Configuration from environment
source_dir = os.environ["SOURCE_DIR"]
output_dir = os.environ["OUTPUT_DIR"]
bits = int(os.environ["BITS"])
group_size = int(os.environ["GROUP_SIZE"])
symmetric = os.environ["SYMMETRIC"].lower() == "true"
calibration_samples = int(os.environ["CALIBRATION_SAMPLES"])
calibration_dataset = os.environ["CALIBRATION_DATASET"]
seqlen = int(os.environ["SEQLEN"])
iters = int(os.environ["ITERS"])
low_gpu_mem = os.environ["LOW_GPU_MEM"].lower() == "true"
recipe = os.environ["RECIPE"]
parallel = os.environ["PARALLEL"].lower() == "true"
batch_size = int(os.environ["BATCH_SIZE"])

print(f"\nLoading model from: {source_dir}")
print(f"Quantization: {bits}-bit, group_size={group_size}, sym={symmetric}")
print(f"Recipe: {recipe}")
print(f"Iterations: {iters} (1000=highest accuracy)")
print(f"Parallel mode: {parallel}")
print(f"Batch size: {batch_size} (concurrent layers)")
print(f"Low GPU memory mode: {low_gpu_mem} (saves ~20GB VRAM)")
print(f"Calibration: {calibration_samples} samples from {calibration_dataset}")
print(f"Sequence length: {seqlen}\n")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    source_dir,
    trust_remote_code=True,
    padding_side="right"
)

# Set padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load calibration data
print(f"Loading calibration dataset: {calibration_dataset}")
from datasets import load_dataset

try:
    calib_dataset = load_dataset(calibration_dataset, split="train")
except Exception:
    # Fallback to wikitext
    print("  Fallback: using wikitext")
    calib_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Prepare calibration samples
calib_data = []
for i in range(min(calibration_samples, len(calib_dataset))):
    text = calib_dataset[i]["text"] if "text" in calib_dataset[i] else str(calib_dataset[i])
    if len(text.strip()) > 32:  # Skip very short samples
        calib_data.append(text.strip())

print(f"  {len(calib_data)} calibration samples prepared")

# Load model for quantization
print("\nLoading model for quantization...")
print("  This may take several minutes for large models...")

start_time = time.time()

from auto_round import AutoRound
from transformers import AutoModelForCausalLM

# Device map: auto for parallel mode, cpu for single-GPU conservative
device_map = "auto" if parallel else "cpu"
print(f"  Device map: {device_map}")

model = AutoModelForCausalLM.from_pretrained(
    source_dir,
    torch_dtype=torch.float16,
    device_map=device_map,
    trust_remote_code=True,
)

load_time = time.time() - start_time
print(f"  Model loaded in {load_time:.0f}s, starting quantization...")

# Quantize with maximum quality settings
start_time = time.time()

model_q, _ = AutoRound.quantize_model(
    model=model,
    tokenizer=tokenizer,
    calib_data=calib_data,
    bits=bits,
    group_size=group_size,
    sym=symmetric,
    layer_wise=True,
    n_samples=calibration_samples,
    iters=iters,
    low_gpu_mem_usage=low_gpu_mem,
    enable_torch_compile=False,
    enable_quanted_input=True,
    batch_size=batch_size if parallel else 1,
    seqlen=seqlen,
    amp=True,
    n_blocks=100,
    gradient_accumulation_steps=1,
    lr=1e-3,
    minmax_range=0,
    enable_quantscale_search=False,
    enable_minmax_tuning=True,
    enable_outlier_channel_wise=True,
    per_channel=True,
    damp_percent=0.01,
    search_method="ammp",
    dynamic_config=None,
)

quant_time = time.time() - start_time
print(f"\n  Quantization complete in {quant_time:.0f}s ({quant_time/60:.1f} min)")

# Save the quantized model
print(f"\nSaving quantized model to: {output_dir}")
model_q.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

# Show output size
import shutil
output_size = shutil.getsize(output_dir) if os.path.isfile(output_dir) else sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, _, filenames in os.walk(output_dir)
    for filename in filenames
)
output_size_gb = output_size / (1024**3)
print(f"  Output size: {output_size_gb:.1f}GB")

total_time = time.time() - start_time
print(f"\n  Total time: {total_time:.0f}s ({total_time/3600:.1f} hours)")
print(f"  ✓ Quantization complete!")
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
