#!/usr/bin/env bash
set -e

# ============================================
# llama.cpp SYCL Build Script
# ============================================
# Builds llama.cpp with SYCL backend for
# Intel Arc B70 GPUs using FP16.
# ============================================

# Detect if run via sudo — preserve original user's HOME
if [ -n "$SUDO_USER" ]; then
    export HOME=$(eval echo ~$SUDO_USER)
    echo "Note: Running via sudo, using HOME=$HOME for user $SUDO_USER"
fi

LLAMA_DIR="$HOME/electric-sheep/llama"
LLAMA_SRC="$LLAMA_DIR/llama.cpp"
BUILD_DIR="$LLAMA_SRC/build"

# Graceful error handler — keeps terminal open for investigation
fail() {
    echo ""
    echo "=========================================="
    echo "  ERROR: $1"
    echo "=========================================="
    echo ""
    echo "Terminal will stay open for investigation."
    echo "Press Enter to close..."
    read -r
    exit 1
}

# -------------------------------------------
# Pre-flight checks
# -------------------------------------------
echo "=========================================="
echo "  llama.cpp SYCL Build — Pre-flight"
echo "=========================================="

warnings=0

# --- [1/8] OS & Kernel ---
echo ""
echo "── [1/8] OS & Kernel ──"
if [ -f /etc/os-release ]; then
    os_name=$(. /etc/os-release && echo "$PRETTY_NAME")
    echo "  OS: $os_name"
else
    echo "  OS: unknown (could not read /etc/os-release)"
fi

kernel_version=$(uname -r)
echo "  Kernel: $kernel_version"

kernel_major=$(echo "$kernel_version" | cut -d. -f1)
kernel_minor=$(echo "$kernel_version" | cut -d. -f2)
if [ "$kernel_major" -lt 6 ] || ([ "$kernel_major" -eq 6 ] && [ "$kernel_minor" -lt 2 ]); then
    echo "  ⚠ Kernel < 6.2 — xe GPU driver may not work correctly"
    warnings=$((warnings + 1))
fi

# --- [2/8] CPU & Memory ---
echo ""
echo "── [2/8] CPU & Memory ──"
cpu_cores=$(nproc)
cpu_model=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)
echo "  CPU: $cpu_model ($cpu_cores threads)"

total_ram=$(free -g | awk '/^Mem:/ {print $2}')
available_ram=$(free -g | awk '/^Mem:/ {print $7}')
echo "  RAM: ${total_ram}GiB total, ${available_ram}GiB available"

if [ "$available_ram" -lt 16 ]; then
    echo "  ⚠ Less than 16GiB RAM available — build may be slow or OOM"
    warnings=$((warnings + 1))
fi

# --- [3/8] GPU Hardware ---
echo ""
echo "── [3/8] GPU Hardware ──"
if command -v lspci >/dev/null 2>&1; then
    gpu_lines=$(lspci | grep -i "VGA\|3D\|Display" | grep -i "Intel" || true)
    gpu_count=$(echo "$gpu_lines" | grep -c "Intel" || true)

    if [ "$gpu_count" -gt 0 ]; then
        echo "  Intel GPU(s) detected: $gpu_count"
        echo "$gpu_lines" | while read -r line; do
            echo "    $line"
        done

        if echo "$gpu_lines" | grep -qi "Battlemage\|Arc.*B"; then
            echo "  ✓ Battlemage architecture detected"
        fi
    else
        fail "No Intel GPU detected via lspci. Install with: sudo apt install pciutils"
    fi
else
    echo "  ⚠ lspci not found — skipping GPU hardware check"
    echo "    Install with: sudo apt install pciutils"
    warnings=$((warnings + 1))
fi

# --- [4/8] User Permissions ---
echo ""
echo "── [4/8] User Permissions ──"
current_user=$(whoami)
echo "  Running as: $current_user"

user_groups=$(groups "$current_user" 2>/dev/null || echo "")
missing_groups=()

if ! echo "$user_groups" | grep -qw "render"; then
    missing_groups+=("render")
fi
if ! echo "$user_groups" | grep -qw "video"; then
    missing_groups+=("video")
fi

if [ ${#missing_groups[@]} -gt 0 ]; then
    echo "  ⚠ User not in required group(s): ${missing_groups[*]}"
    echo "    Fix with: sudo usermod -aG ${missing_groups[*]} $current_user"
    echo "    (logout/login required after)"
    warnings=$((warnings + 1))
else
    echo "  ✓ User in 'render' and 'video' groups"
fi

# --- [5/8] Build Tools (auto-install if missing) ---
echo ""
echo "── [5/8] Build Tools ──"

missing_tools=()

if ! command -v git >/dev/null 2>&1; then
    missing_tools+=("git")
else
    git_version=$(git --version | awk '{print $3}')
    echo "  ✓ git $git_version"
fi

if ! command -v cmake >/dev/null 2>&1; then
    missing_tools+=("cmake")
else
    cmake_version=$(cmake --version | head -1 | awk '{print $3}')
    cmake_major=$(echo "$cmake_version" | cut -d. -f1)
    cmake_minor=$(echo "$cmake_version" | cut -d. -f2)
    if [ "$cmake_major" -lt 3 ] || ([ "$cmake_major" -eq 3 ] && [ "$cmake_minor" -lt 21 ]); then
        echo "  ⚠ cmake $cmake_version < 3.21 — build may fail"
        warnings=$((warnings + 1))
    else
        echo "  ✓ cmake $cmake_version"
    fi
fi

if ! command -v ninja >/dev/null 2>&1; then
    missing_tools+=("ninja-build")
else
    ninja_version=$(ninja --version)
    echo "  ✓ ninja $ninja_version"
fi

if ! command -v gcc >/dev/null 2>&1; then
    missing_tools+=("build-essential")
else
    echo "  ✓ gcc $(gcc --version | head -1 | awk '{print $4}')"
fi

if [ ${#missing_tools[@]} -gt 0 ]; then
    echo ""
    echo "  ⚠ Missing: ${missing_tools[*]}"
    echo "  Installing via apt..."
    echo ""

    if [ -n "$SUDO_USER" ]; then
        apt-get update -qq && apt-get install -y "${missing_tools[@]}" || {
            echo ""
            echo "  ✗ apt install failed — try running manually:"
            echo "    sudo apt install ${missing_tools[*]}"
            exit 1
        }
    else
        sudo apt-get update -qq && sudo apt-get install -y "${missing_tools[@]}" || {
            echo ""
            echo "  ✗ apt install failed — try running manually:"
            echo "    sudo apt install ${missing_tools[*]}"
            exit 1
        }
    fi

    echo ""
    echo "  ✓ Installed: ${missing_tools[*]}"
    echo ""

    command -v git >/dev/null 2>&1 && echo "  ✓ git $(git --version | awk '{print $3}')"
    command -v cmake >/dev/null 2>&1 && echo "  ✓ cmake $(cmake --version | head -1 | awk '{print $3}')"
    command -v ninja >/dev/null 2>&1 && echo "  ✓ ninja $(ninja --version)"
    command -v gcc >/dev/null 2>&1 && echo "  ✓ gcc $(gcc --version | head -1 | awk '{print $4}')"
fi

# --- [6/8] Disk Space ---
echo ""
echo "── [6/8] Disk Space ──"
build_target_dir="$HOME"
available_gb=$(df --output=avail -BM "$build_target_dir" | tail -1 | awk '{printf "%d", $1/1024}')
echo "  Available on $build_target_dir: ${available_gb}GB"

if [ "$available_gb" -lt 20 ]; then
    fail "Need at least 20GB free for build + source, have ${available_gb}GB"
elif [ "$available_gb" -lt 40 ]; then
    echo "  ⚠ Less than 40GB — build will complete but leaves little headroom"
    warnings=$((warnings + 1))
else
    echo "  ✓ Sufficient disk space"
fi

# --- [7/8] oneAPI Toolkit ---
echo ""
echo "── [7/8] oneAPI Toolkit ──"

if [ ! -f /opt/intel/oneapi/setvars.sh ]; then
    fail "oneAPI Toolkit not found at /opt/intel/oneapi/setvars.sh"
fi
echo "  ✓ oneAPI setvars.sh found"

oneapi_components=()
[ -d /opt/intel/oneapi/compiler ] && oneapi_components+=("compiler")
[ -d /opt/intel/oneapi/dnnl ] && oneapi_components+=("oneDNN")
[ -d /opt/intel/oneapi/mkl ] && oneapi_components+=("oneMKL")
[ -d /opt/intel/oneapi/dpl ] && oneapi_components+=("oneDPL")

if [ ${#oneapi_components[@]} -gt 0 ]; then
    echo "  ✓ Components: ${oneapi_components[*]}"
else
    echo "  ⚠ No recognized oneAPI components found — build may fail"
    warnings=$((warnings + 1))
fi

compiler_versions=$(ls -d /opt/intel/oneapi/compiler/*/latest 2>/dev/null | head -1 || true)
if [ -n "$compiler_versions" ]; then
    compiler_ver_label=$(basename $(dirname "$compiler_versions"))
    echo "  ✓ Compiler version: $compiler_ver_label"
fi

# --- [8/8] Load oneAPI Environment & Validate ---
echo ""
echo "── [8/8] oneAPI Environment Load ──"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
echo "  ✓ oneAPI environment loaded"

if command -v icx >/dev/null 2>&1; then
    icx_ver=$(icx --version 2>&1 | head -1 | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
    echo "  ✓ icx compiler: $icx_ver"
else
    fail "icx compiler not found on PATH after loading oneAPI"
fi

if command -v icpx >/dev/null 2>&1; then
    icpx_ver=$(icpx --version 2>&1 | head -1 | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
    echo "  ✓ icpx compiler: $icpx_ver"
else
    fail "icpx compiler not found on PATH after loading oneAPI"
fi

if command -v sycl-ls >/dev/null 2>&1; then
    echo "  ✓ sycl-ls tool available"
else
    fail "sycl-ls not found on PATH — SYCL runtime may not be installed"
fi

# --- SYCL Device Detection ---
echo ""
echo "── SYCL GPU Devices ──"
sycl_output=$(sycl-ls 2>/dev/null || true)
sycl_device_count=$(echo "$sycl_output" | grep -c "level_zero:gpu" || true)

if [ "$sycl_device_count" -lt 1 ]; then
    echo "  ✗ No SYCL GPU devices detected!"
    echo ""
    echo "  Troubleshooting:"
    echo "    1. Check GPU driver: lspci | grep -i Intel"
    echo "    2. Check user groups: groups $current_user (need 'render' and 'video')"
    echo "    3. Try with sudo: sudo sycl-ls"
    echo ""
    echo "  Full sycl-ls output:"
    echo "$sycl_output" | head -20
    fail "No SYCL GPU devices detected"
fi

echo "  ✓ $sycl_device_count SYCL GPU device(s) detected"
echo "$sycl_output" | grep "level_zero:gpu" | while read -r line; do
    device_name=$(echo "$line" | grep -oP 'Intel\(R\)[^]]+' || echo "$line")
    echo "    $device_name"
done

if echo "$sycl_output" | grep -qi "Battlemage\|Arc.*B70\|Arc.*B580"; then
    echo "  ✓ Battlemage architecture confirmed"
elif echo "$sycl_output" | grep -qi "Arc"; then
    echo "  ✓ Arc GPU detected (non-Battlemage)"
fi

# --- Summary ---
echo ""
echo "=========================================="
echo "  Pre-flight Summary"
echo "=========================================="
echo "  GPUs:     $sycl_device_count SYCL device(s)"
echo "  CPU:      $cpu_cores threads"
echo "  RAM:      ${available_ram}GiB available"
echo "  Disk:     ${available_gb}GB available"
echo "  Compiler: icpx $icpx_ver"
echo "  CMake:    $cmake_version"

if [ "$warnings" -gt 0 ]; then
    echo ""
    echo "  ⚠ $warnings warning(s) detected — review above"
    echo "  Continuing anyway (warnings are non-fatal)..."
else
    echo ""
    echo "  ✓ All checks passed"
fi

echo ""
echo "=========================================="

# -------------------------------------------
# Verify source exists
# -------------------------------------------
echo ""
echo "[1/4] Verifying llama.cpp source..."
if [ ! -d "$LLAMA_SRC/.git" ]; then
    fail "llama.cpp source not found at $LLAMA_SRC. Run 01-setup-project.sh first."
fi

cd "$LLAMA_SRC"
git pull --quiet || true
echo "✓ llama.cpp source ready at $LLAMA_SRC"

# Check if rebuild is needed (compare commit hash)
LAST_BUILT_HASH="$BUILD_DIR/.last_built_commit"
CURRENT_HASH=$(cd "$LLAMA_SRC" && git rev-parse HEAD)
if [ -f "$LAST_BUILT_HASH" ]; then
    PREV_HASH=$(cat "$LAST_BUILT_HASH")
    if [ "$PREV_HASH" = "$CURRENT_HASH" ]; then
        echo ""
        echo "  → Source unchanged (commit $CURRENT_HASH) — skipping rebuild"
        echo "  (Force rebuild: rm -rf $BUILD_DIR)"
        echo ""
        echo "[4/4] Verifying build..."
        [ -f "$BUILD_DIR/bin/llama-cli" ] || fail "llama-cli binary not found"
        [ -f "$BUILD_DIR/bin/llama-server" ] || fail "llama-server binary not found"
        echo "✓ llama-cli binary found"
        echo "✓ llama-server binary found"
        binary_count=$(ls -1 "$BUILD_DIR/bin/" 2>/dev/null | wc -l)
        echo "✓ $binary_count binaries in $BUILD_DIR/bin/"
        exit 0
    fi
fi

# -------------------------------------------
# Configure with CMake (SYCL + FP16)
# -------------------------------------------
echo ""
echo "[2/4] Configuring build (SYCL + FP16)..."
cd "$LLAMA_SRC"

if [ -d "$BUILD_DIR" ]; then
    echo "  -> Removing previous build directory..."
    rm -rf "$BUILD_DIR"
fi

cmake -B "$BUILD_DIR" \
    -G Ninja \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_F16=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF

echo "✓ CMake configuration complete"

# -------------------------------------------
# Build
# -------------------------------------------
echo ""
echo "[3/4] Building llama.cpp..."
cmake --build "$BUILD_DIR" -j $(nproc)

cd "$LLAMA_SRC" && git rev-parse HEAD > "$BUILD_DIR/.last_built_commit"
echo "✓ Build complete"

# -------------------------------------------
# Verification
# -------------------------------------------
echo ""
echo "[4/4] Verifying build..."

[ -f "$BUILD_DIR/bin/llama-cli" ] || fail "llama-cli binary not found"
[ -f "$BUILD_DIR/bin/llama-server" ] || fail "llama-server binary not found"
echo "✓ llama-cli binary found"
echo "✓ llama-server binary found"

binary_count=$(ls -1 "$BUILD_DIR/bin/" 2>/dev/null | wc -l)
echo "✓ $binary_count binaries built in $BUILD_DIR/bin/"

echo ""
echo "SYCL devices:"
if [ -f "$BUILD_DIR/bin/llama-ls-sycl-device" ]; then
    "$BUILD_DIR/bin/llama-ls-sycl-device" 2>/dev/null || true
elif [ -f "$BUILD_DIR/bin/llama-print-sycl-info" ]; then
    "$BUILD_DIR/bin/llama-print-sycl-info" 2>/dev/null || true
else
    sycl-ls 2>/dev/null | grep "level_zero:gpu" || echo "⚠ No SYCL device listing tool found"
fi

echo ""
echo "Build summary:"
echo "  SYCL backend: ON"
echo "  FP16 support: ON"
echo "  Build type:   Release"
echo "  Compiler:     icx/icpx (oneAPI)"
echo "  Binaries:     $BUILD_DIR/bin/"

echo ""
echo "=========================================="
echo "  llama.cpp Build Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download a GGUF model:"
echo "     hf download <repo> --local-dir ~/electric-sheep/models/"
echo ""
echo "  2. Run inference (single GPU):"
echo "     source ~/electric-sheep/llama/set-env.sh 0"
echo "     $BUILD_DIR/bin/llama-cli -m ~/electric-sheep/models/your-model.gguf -ngl 99 -n 256"
echo ""
echo "  3. Run inference (all 4 GPUs, layer split):"
echo "     source ~/electric-sheep/llama/set-env.sh"
echo "     $BUILD_DIR/bin/llama-cli -m ~/electric-sheep/models/your-model.gguf -ngl 99 -n 256 -sm layer"
echo ""
echo "  4. Start llama-server (OpenAI-compatible API):"
echo "     source ~/electric-sheep/llama/set-env.sh"
echo "     $BUILD_DIR/bin/llama-server -m ~/electric-sheep/models/your-model.gguf -ngl 99 -sm layer --port 8080"
echo ""
