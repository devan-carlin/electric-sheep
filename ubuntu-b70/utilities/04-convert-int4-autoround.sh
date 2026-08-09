#!/usr/bin/env bash
# ============================================
# Convert Model to INT4 AutoRound for vLLM
# ============================================
# Downloads a source model from HuggingFace and
# quantizes it to INT4 AutoRound format using
# the auto-round library.
#
# Usage:
#   ./04-convert-int4-autoround.sh <huggingface-repo>
#   ./04-convert-int4-autoround.sh --help
#
# Examples:
#   ./04-convert-int4-autoround.sh apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8
#   ./04-convert-int4-autoround.sh meta-llama/Llama-3.3-70B-Instruct
#
# Requirements:
#   - vLLM virtual environment (~/electric-sheep/vllm/.venv)
#   - Sufficient VRAM/RAM for the source model
#   - auto-round package (auto-installed if missing)
#
# Output:
#   ~/electric-sheep/models/<repo-name>-int4-AutoRound/
# ============================================

set -e

# ============================================
# Configuration
# ============================================
VENV_DIR="$HOME/electric-sheep/vllm/.venv"
MODELS_DIR="$HOME/electric-sheep/models"
CALIBRATION_DATASET="timdettmers/openassistant-guanaco"
CALIBRATION_SAMPLES=128
BITS=4
GROUP_SIZE=128
SYMMETRIC=false
SEQLEN=2048

# ============================================
# Helpers
# ============================================
print_header() {
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

print_ok() {
    echo "  ✓ $1"
}

print_warn() {
    echo "  ⚠ $1"
}

print_err() {
    echo "  ✗ $1"
}

fail() {
    echo ""
    print_err "$1"
    echo ""
    echo "Terminal will stay open for investigation."
    echo "Press Enter to close..."
    read -r
    exit 1
}

# ============================================
# Help
# ============================================
show_help() {
    echo "Usage: $0 <huggingface-repo> [OPTIONS]"
    echo ""
    echo "Convert a HuggingFace model to INT4 AutoRound format for vLLM."
    echo ""
    echo "Arguments:"
    echo "  <huggingface-repo>   Model repo (e.g., apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8)"
    echo ""
    echo "Options:"
    echo "  --bits <n>           Quantization bits (default: 4)"
    echo "  --group-size <n>     Group size (default: 128)"
    echo "  --sym                Use symmetric quantization (default: asymmetric)"
    echo "  --samples <n>        Calibration samples (default: 128)"
    echo "  --seqlen <n>         Sequence length for calibration (default: 2048)"
    echo "  --output-dir <path>  Output directory (default: ~/electric-sheep/models/<name>-int4-AutoRound)"
    echo "  --download-only      Download source model only, skip quantization"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8"
    echo "  $0 apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8 --bits 4 --group-size 64"
    echo "  $0 apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8 --download-only"
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
print_header "INT4 AutoRound Conversion — Pre-flight"

errors=0

# Check venv
if [ ! -d "$VENV_DIR" ]; then
    fail "vLLM virtual environment not found at $VENV_DIR"
    echo "  Run: bash ~/ubuntu-b70/vllm/02-setup-project-directory.sh"
fi
print_ok "Virtual environment found"

# Activate venv early for Python checks
source "$VENV_DIR/bin/activate"

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
# Install auto-round
# ============================================
print_header "Checking Quantization Dependencies"

if python3 -c "import auto_round" 2>/dev/null; then
    print_ok "auto-round already installed"
else
    echo "  Installing auto-round..."
    pip install auto-round --quiet
    print_ok "auto-round installed"
fi

# Check for optimum (needed for some models)
if python3 -c "import optimum" 2>/dev/null; then
    print_ok "optimum installed"
else
    echo "  Installing optimum..."
    pip install optimum --quiet
    print_ok "optimum installed"
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
export SOURCE_DIR OUTPUT_DIR BITS GROUP_SIZE SYMMETRIC CALIBRATION_SAMPLES CALIBRATION_DATASET SEQLEN

# Run quantization via Python
python3 << 'PYEOF'
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

print(f"\nLoading model from: {source_dir}")
print(f"Quantization: {bits}-bit, group_size={group_size}, sym={symmetric}")
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
    calib_dataset = load_dataset(calibration_dataset, split="train", trust_remote_code=True)
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

model = AutoRound.from_pretrained(
    model_name_or_path=source_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    disable_gradient_checkpointing=True,
)

load_time = time.time() - start_time
print(f"  Model loaded in {load_time:.0f}s, starting quantization...")

# Quantize
start_time = time.time()

model_q, _ = AutoRound.quantize_model(
    model=model,
    tokenizer=tokenizer,
    calib_data=calib_data,
    bits=bits,
    group_size=group_size,
    sym=symmetric,
    enable_quanted_input=True,
    seqlen=seqlen,
    n_samples=len(calib_data),
    amp=True,
)

quant_time = time.time() - start_time
print(f"  Quantization complete in {quant_time:.0f}s")

print(f"\nSaving quantized model to: {output_dir}")
model_q.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

# Copy config files
import shutil
config_src = os.path.join(source_dir, "config.json")
if os.path.exists(config_src):
    shutil.copy2(config_src, os.path.join(output_dir, "config.json"))

print("\nQuantization complete!")

# Show output size
total_size = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, filenames in os.walk(output_dir)
    for f in filenames
)
print(f"  Output size: {total_size / (1024**3):.1f} GiB")
PYEOF

echo ""
print_ok "Quantization pipeline complete!"
echo ""
echo "  Output: $OUTPUT_DIR"
echo ""
echo "  To serve with vLLM:"
echo "    source $VENV_DIR/bin/activate"
echo "    vllm serve $OUTPUT_DIR --tensor-parallel-size 4 --kv-cache-dtype fp8"
