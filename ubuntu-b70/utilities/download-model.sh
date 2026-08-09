#!/usr/bin/env bash
# ============================================
# Download Models for vLLM (Native Format)
# ============================================
# Downloads models in vLLM's native HuggingFace
# safetensors format and places them in the
# electric-sheep model directory.
#
# Usage:
#   ./download-model.sh <huggingface-repo>              # Download by repo name
#   ./download-model.sh --list                          # List known models
#   ./download-model.sh --status                        # Check existing models
#
# Examples:
#   ./download-model.sh Intel/Qwen3.6-27B-int4-AutoRound
#   ./download-model.sh Intel/gemma-4-31B-it-int4-AutoRound-V2
#   ./download-model.sh --list
#
# Note on GGUF files:
#   vLLM can serve GGUF files directly (no conversion needed), but:
#   - GGUF support is experimental and under-optimized
#   - Requires vllm-gguf-plugin: pip install vllm-gguf-plugin
#   - Always pass --tokenizer with the base model repo (GGUF tokenizer is slow/buggy)
#   - Use --hf-config-path if model is not supported by HuggingFace
#
#   Example: vllm serve ./models/your-model.gguf --tokenizer base-model-repo
#
#   GGUF cannot be "converted" to native vLLM format because it's already
#   quantized (lossy). The only way to get native safetensors is to download
#   the original HuggingFace repo.
# ============================================

set -e

# ============================================
# Configuration
# ============================================
MODEL_DIR="${HOME}/electric-sheep/models"
HF_TOKEN="${HF_TOKEN:-}"  # Set HF_TOKEN env var if downloading gated models

# Known models (pre-configured for this hardware)
declare -A KNOWN_MODELS=(
    ["qwen3.6-27b"]="Intel/Qwen3.6-27B-int4-AutoRound"
    ["qwen3.6-35b-a3b"]="Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound"
    ["gemma-4-31b"]="Intel/gemma-4-31B-it-int4-AutoRound-V2"
    ["gemma-4-26b-a4b"]="Intel/gemma-4-26B-A4B-it-int4-AutoRound"
)

# ============================================
# Helper Functions
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

print_error() {
    echo "  ✗ $1"
}

# ============================================
# List Known Models
# ============================================
if [ "$1" = "--list" ]; then
    print_header "Known Models (Pre-configured)"
    echo "  Alias                    HuggingFace Repo                                    Est. Size"
    echo "  -----                    --------------                                    ---------"
    for alias in "qwen3.6-27b" "qwen3.6-35b-a3b" "gemma-4-31b" "gemma-4-26b-a4b"; do
        repo="${KNOWN_MODELS[$alias]}"
        printf "  %-24s %-45s\n" "$alias" "$repo"
    done
    echo ""
    echo "  Usage: ./download-model.sh <alias-or-repo>"
    echo "  Example: ./download-model.sh qwen3.6-27b"
    echo "           ./download-model.sh Intel/Qwen3.6-27B-int4-AutoRound"
    exit 0
fi

# ============================================
# Status Check
# ============================================
if [ "$1" = "--status" ]; then
    print_header "Model Directory Status"
    echo "  Model directory: $MODEL_DIR"
    echo ""

    if [ ! -d "$MODEL_DIR" ]; then
        print_error "Model directory does not exist"
        echo "  Create it: mkdir -p $MODEL_DIR"
        exit 1
    fi

    # List model directories
    model_count=0
    total_size=0
    for model_path in "$MODEL_DIR"/*/; do
        if [ -d "$model_path" ]; then
            model_name=$(basename "$model_path")
            # Check if it has safetensors (native format) or gguf
            has_safetensors=$(find "$model_path" -name "*.safetensors" 2>/dev/null | head -1)
            has_gguf=$(find "$model_path" -name "*.gguf" 2>/dev/null | head -1)
            size=$(du -sh "$model_path" 2>/dev/null | cut -f1)

            format="unknown"
            if [ -n "$has_safetensors" ]; then
                format="safetensors (native)"
            elif [ -n "$has_gguf" ]; then
                format="GGUF"
            fi

            printf "  %-40s %6s  %s\n" "$model_name" "$size" "$format"
            model_count=$((model_count + 1))
        fi
    done

    if [ "$model_count" -eq 0 ]; then
        echo "  No models found."
        echo ""
        echo "  Download a model:"
        echo "    ./download-model.sh qwen3.6-27b"
    else
        echo ""
        echo "  Total: $model_count model(s)"
    fi

    # Disk space
    echo ""
    echo "  Disk space:"
    df -h "$MODEL_DIR" | tail -1 | awk '{print "    Available: " $4 " (on " $6 ")"}'

    exit 0
fi

# ============================================
# Download Model
# ============================================
if [ -z "$1" ]; then
    print_header "Model Downloader"
    echo "  Usage: ./download-model.sh <model-alias-or-repo>"
    echo ""
    echo "  Examples:"
    echo "    ./download-model.sh qwen3.6-27b"
    echo "    ./download-model.sh Intel/Qwen3.6-27B-int4-AutoRound"
    echo ""
    echo "  Options:"
    echo "    --list      Show known model aliases"
    echo "    --status    Check existing models"
    echo ""
    echo "  GGUF files:"
    echo "    vLLM serves GGUF directly. Just copy to $MODEL_DIR/"
    exit 1
fi

MODEL_ARG="$1"

# Resolve alias to full repo name
if [ -z "${KNOWN_MODELS[$MODEL_ARG]}" ]; then
    MODEL_REPO="$MODEL_ARG"
else
    MODEL_REPO="${KNOWN_MODELS[$MODEL_ARG]}"
    print_ok "Resolved alias '$MODEL_ARG' → $MODEL_REPO"
fi

# Determine local directory name (last part of repo path)
MODEL_LOCAL_NAME=$(basename "$MODEL_REPO")
MODEL_LOCAL_PATH="$MODEL_DIR/$MODEL_LOCAL_NAME"

# Check if already downloaded
if [ -d "$MODEL_LOCAL_PATH" ]; then
    existing_size=$(du -sh "$MODEL_LOCAL_PATH" | cut -f1)
    print_warn "Model already exists at $MODEL_LOCAL_PATH ($existing_size)"
    echo ""
    read -p "  Re-download? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "  Skipped."
        exit 0
    fi
    echo "  Removing existing copy..."
    rm -rf "$MODEL_LOCAL_PATH"
fi

# Pre-flight checks
print_header "Downloading $MODEL_REPO"
echo "  Target: $MODEL_LOCAL_PATH"
echo ""

# Check hf CLI is available
if ! command -v hf &> /dev/null; then
    print_error "hf CLI not found"
    echo "  Install: pipx install huggingface-hub[hf_xet]"
    exit 1
fi

# Check disk space (rough estimate: most models are 10-20 GB)
available_space=$(df -BG "$MODEL_DIR" | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$available_space" -lt 15 ]; then
    print_warn "Low disk space: ${available_space}GB available (models typically need 10-20GB)"
    read -p "  Continue anyway? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "  Aborted."
        exit 0
    fi
fi

# Download
echo "  Downloading (this may take several minutes)..."
echo ""

# Build hf download command
HF_CMD="hf download --resume-download --local-dir \"$MODEL_LOCAL_PATH\""

# Add token if set
if [ -n "$HF_TOKEN" ]; then
    HF_CMD="$HF_CMD --token \"$HF_TOKEN\""
    print_ok "Using HuggingFace token (authenticated download)"
else
    print_warn "No HF_TOKEN set — gated models will fail"
fi

# Execute
HF_CMD="$HF_CMD \"$MODEL_REPO\""
echo "  Command: $HF_CMD"
echo ""

eval $HF_CMD

# Verify download
echo ""
if [ -d "$MODEL_LOCAL_PATH" ]; then
    size=$(du -sh "$MODEL_LOCAL_PATH" | cut -f1)
    file_count=$(find "$MODEL_LOCAL_PATH" -type f | wc -l)

    # Check format
    has_safetensors=$(find "$MODEL_LOCAL_PATH" -name "*.safetensors" 2>/dev/null | wc -l)
    has_gguf=$(find "$MODEL_LOCAL_PATH" -name "*.gguf" 2>/dev/null | wc -l)

    print_ok "Download complete"
    echo "  Location: $MODEL_LOCAL_PATH"
    echo "  Size: $size"
    echo "  Files: $file_count"

    if [ "$has_safetensors" -gt 0 ]; then
        echo "  Format: safetensors ($has_safetensors shards) — native vLLM format"
    elif [ "$has_gguf" -gt 0 ]; then
        echo "  Format: GGUF ($has_gguf files) — vLLM can serve directly"
    else
        echo "  Format: unknown (check manually)"
    fi

    echo ""
    echo "  To serve this model:"
    echo "    source ~/electric-sheep/vllm/.venv/bin/activate"
    echo "    source ~/electric-sheep/vllm/set-env-0123-gpu.sh"
    echo "    vllm serve $MODEL_LOCAL_PATH --tensor-parallel-size 4 --port 8030"
else
    print_error "Download failed — directory not created"
    exit 1
fi
