
#!/usr/bin/env bash
# ============================================
# Build All — vLLM + llama.cpp
# ============================================
# Orchestrates the full build pipeline for both
# vLLM (XPU) and llama.cpp (SYCL) on the B70 server.
#
# Usage:
#   ./01-build-all.sh              # Interactive (asks what to build)
#   ./01-build-all.sh --all        # Build everything
#   ./01-build-all.sh --vllm       # Build vLLM only
#   ./01-build-all.sh --llama      # Build llama.cpp only
#   ./01-build-all.sh --prereqs    # Install prerequisites only
#   ./01-build-all.sh --status     # Show current build state
# ============================================

set -e

# ============================================
# Paths (resolve relative to this script's location)
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(dirname "$SCRIPT_DIR")"
VLLM_SCRIPTS="$SCRIPTS/vllm"
LLAMA_SCRIPTS="$SCRIPTS/llama"
VLLM_DIR="$HOME/electric-sheep/vllm"
LLAMA_DIR="$HOME/electric-sheep/llama"
MODELS_DIR="$HOME/electric-sheep/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================
# Helpers
# ============================================
banner() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
    echo ""
}

step() {
    echo -e "${BLUE}▸ $1${NC}"
}

ok() {
    echo -e "${GREEN}✓ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

err() {
    echo -e "${RED}✗ $1${NC}"
}

run_script() {
    local script="$1"
    local label="$2"
    local script_dir="${2%%/*}"

    echo ""
    banner "$label"
    step "Running: $script"
    echo ""

    if bash "$script"; then
        echo ""
        ok "$label complete"
        return 0
    else
        echo ""
        err "$label failed"
        echo "  Terminal will stay open for investigation."
        echo "  Press Enter to continue (or Ctrl+D to exit)..."
        read -r
        return 1
    fi
}

# ============================================
# Status Check
# ============================================
show_status() {
    banner "Build Status"

    echo -e "${CYAN}vLLM:${NC}"
    if [ -d "$VLLM_DIR/.venv" ]; then
        ok "Virtual environment exists"
    else
        warn "Virtual environment not found"
    fi

    if [ -f "$VLLM_DIR/.venv/lib/python3.12/site-packages/vllm/__init__.py" ]; then
        ok "vLLM installed"
    else
        warn "vLLM not installed"
    fi

    if [ -f "$VLLM_DIR/.venv/lib/python3.12/site-packages/vllm_xpu_kernels/__init__.py" ]; then
        ok "vllm-xpu-kernels installed"
    else
        warn "vllm-xpu-kernels not installed"
    fi

    echo ""
    echo -e "${CYAN}llama.cpp:${NC}"
    if [ -d "$LLAMA_DIR/llama.cpp/.git" ]; then
        ok "Source cloned"
        cd "$LLAMA_DIR/llama.cpp" && commit=$(git rev-parse --short HEAD) && ok "Commit: $commit"
    else
        warn "Source not cloned"
    fi

    if [ -f "$LLAMA_DIR/llama.cpp/build/bin/llama-server" ]; then
        ok "llama-server built"
        binaries=$(ls -1 "$LLAMA_DIR/llama.cpp/build/bin/" 2>/dev/null | wc -l)
        ok "$binaries binaries total"
    else
        warn "llama-server not built"
    fi

    echo ""
    echo -e "${CYAN}Models:${NC}"
    if [ -d "$MODELS_DIR" ]; then
        model_count=$(find "$MODELS_DIR" -name "*.gguf" -o -name "*.safetensors" 2>/dev/null | wc -l)
        ok "$MODELS_DIR exists ($model_count model files)"
        for dir in "$MODELS_DIR"/*/; do
            if [ -d "$dir" ]; then
                size=$(du -sh "$dir" 2>/dev/null | cut -f1)
                echo "  $(basename "$dir") ($size)"
            fi
        done
    else
        warn "$MODELS_DIR not found"
    fi

    echo ""
    echo -e "${CYAN}Scripts:${NC}"
    for s in "$VLLM_SCRIPTS"/0*.sh "$LLAMA_SCRIPTS"/0*.sh; do
        if [ -f "$s" ]; then
            echo "  $(basename "$s")"
        fi
    done
}

# ============================================
# Interactive Mode
# ============================================
interactive_mode() {
    banner "Build All — vLLM + llama.cpp"

    echo "  What would you like to build?"
    echo ""
    echo "  1) Prerequisites only (system packages, Python 3.12, oneAPI)"
    echo "  2) vLLM only (setup + build + patch)"
    echo "  3) llama.cpp only (setup + build)"
    echo "  4) Both vLLM and llama.cpp"
    echo "  5) Download models (after builds)"
    echo "  6) Status check only"
    echo "  0) Exit"
    echo ""
    read -p "  Choose (1-6): " choice

    case "$choice" in
        1) build_prereqs ;;
        2) build_vllm ;;
        3) build_llama ;;
        4) build_both ;;
        5) download_models ;;
        6) show_status ;;
        0) echo "  Bye!"; exit 0 ;;
        *) warn "Invalid choice"; exit 1 ;;
    esac
}

# ============================================
# Build Functions
# ============================================
build_prereqs() {
    run_script "$VLLM_SCRIPTS/01-install-prerequisites.sh" "Prerequisites"
}

build_vllm() {
    # Setup project directory + venv
    run_script "$VLLM_SCRIPTS/02-setup-project-directory.sh" "vLLM Project Setup"

    # Build vLLM XPU
    run_script "$VLLM_SCRIPTS/03-build-vllm-xpu.sh" "vLLM XPU Build"

    # Patch MoE qzeros
    run_script "$VLLM_SCRIPTS/05-patch-vllm-moe-qzeros.sh" "MoE Qzeros Patch"

    echo ""
    ok "vLLM build pipeline complete!"
    echo ""
    echo "  Next: download models"
    echo "  bash $VLLM_SCRIPTS/04-download-models.sh"
}

build_llama() {
    # Setup project + clone source
    run_script "$LLAMA_SCRIPTS/01-setup-project.sh" "llama.cpp Project Setup"

    # Build
    run_script "$LLAMA_SCRIPTS/02-build-llama-cpp.sh" "llama.cpp SYCL Build"

    echo ""
    ok "llama.cpp build pipeline complete!"
    echo ""
    echo "  Next: download GGUF models"
    echo "  hf download <repo> --local-dir $MODELS_DIR/"
}

build_both() {
    # Prerequisites first (shared)
    build_prereqs

    echo ""
    banner "Building Both — vLLM + llama.cpp"

    # Build vLLM
    build_vllm

    # Build llama.cpp
    build_llama

    echo ""
    banner "All Builds Complete!"
    echo ""
    echo "  vLLM:     $VLLM_DIR/.venv/"
    echo "  llama.cpp: $LLAMA_DIR/llama.cpp/build/bin/"
    echo "  Models:   $MODELS_DIR/"
    echo ""
    echo "  Next steps:"
    echo "  - Download models:  bash $VLLM_SCRIPTS/04-download-models.sh"
    echo "  - Start vLLM:       source $VLLM_DIR/.venv/bin/activate && bash $VLLM_DIR/start-*.sh"
    echo "  - Start llama.cpp:  bash $LLAMA_DIR/deepseek/start-deepseek-v4-flash.sh"
}

download_models() {
    run_script "$VLLM_SCRIPTS/04-download-models.sh" "Download Models"
}

# ============================================
# Main
# ============================================
case "${1:-}" in
    --all)
        build_both
        ;;
    --vllm)
        build_vllm
        ;;
    --llama)
        build_llama
        ;;
    --prereqs)
        build_prereqs
        ;;
    --status)
        show_status
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "  (no args)          Interactive mode (asks what to build)"
        echo "  --all              Build both vLLM and llama.cpp"
        echo "  --vllm             Build vLLM only"
        echo "  --llama            Build llama.cpp only"
        echo "  --prereqs          Install prerequisites only"
        echo "  --status           Show current build state"
        echo "  --help             Show this help"
        echo ""
        echo "Pipeline order:"
        echo "  1. Prerequisites (system packages, Python 3.12, oneAPI)"
        echo "  2. vLLM: setup → build → patch"
        echo "  3. llama.cpp: setup → build"
        echo "  4. Download models (optional)"
        exit 0
        ;;
    *)
        interactive_mode
        ;;
esac
