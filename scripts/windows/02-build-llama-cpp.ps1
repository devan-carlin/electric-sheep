#Requires -Version 5.1
# ============================================
# llama.cpp + beellama.cpp Build Script
# ============================================
# Clones and builds both llama.cpp (upstream)
# and beellama.cpp (fork with KVarN, precision
# tail, adaptive draft-max) with CUDA +
# FlashAttention for RTX 5090.
# ============================================

$ErrorActionPreference = "Stop"

$WorkspaceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$BuildRoot = Join-Path $WorkspaceRoot "llama"
$ModelsDir = Join-Path $WorkspaceRoot "models"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "── $Title ──" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Msg)
    Write-Host "  ✓ $Msg" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Msg)
    Write-Host "  ⚠ $Msg" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Msg)
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "  ERROR: $Msg" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    exit 1
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  llama.cpp + beellama.cpp Build" -ForegroundColor Cyan
Write-Host "  RTX 5090 (CUDA + FlashAttention)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# --- Pre-flight checks ---
Write-Section "Pre-flight Checks"

# CUDA
try {
    $nvccVer = nvcc --version 2>$null | Select-String "release (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    if (-not $nvccVer) { Write-Fail "nvcc not found. Install CUDA Toolkit." }
    Write-Ok "CUDA $nvccVer"
} catch { Write-Fail "Failed to run nvcc." }

# CMake
try {
    $cmakeVer = cmake --version 2>$null | Select-String "cmake version (\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    if (-not $cmakeVer) { Write-Fail "CMake not found." }
    Write-Ok "CMake $cmakeVer"
} catch { Write-Fail "Failed to run cmake." }

# GPU
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    $parts = $gpuInfo -split ',\s*'
    $gpuName = $parts[0]
    $gpuMem = $parts[1]
    Write-Ok "$gpuName ($gpuMem)"

    # Detect CUDA architecture
    if ($gpuName -match "RTX 5090") {
        $cudaArch = "100"
        Write-Ok "Targeting sm_100 (Blackwell)"
    } elseif ($gpuName -match "RTX 40") {
        $cudaArch = "89"
        Write-Ok "Targeting sm_89 (Ada Lovelace)"
    } elseif ($gpuName -match "RTX 30") {
        $cudaArch = "86"
        Write-Ok "Targeting sm_86 (Ampere)"
    } else {
        $cudaArch = "100"
        Write-Warn "Unknown GPU — defaulting to sm_100. Adjust CMAKE_CUDA_ARCHITECTURES manually."
    }
} catch { Write-Fail "nvidia-smi not found. Install NVIDIA drivers." }

# --- Build Configuration ---
Write-Section "Build Configuration"
Write-Host "  Build root:    $BuildRoot"
Write-Host "  CUDA arch:     sm_$cudaArch"
Write-Host "  FlashAttn:     ON"
Write-Host "  Build type:    Release"

# Ask which to build
Write-Host ""
Write-Host "  [1] Build both llama.cpp and beellama.cpp" -ForegroundColor Yellow
Write-Host "  [2] Build llama.cpp only" -ForegroundColor Yellow
Write-Host "  [3] Build beellama.cpp only" -ForegroundColor Yellow
$buildChoice = Read-Host "What to build? (1-3) [default=1]"
$buildBoth = $buildChoice -eq "" -or $buildChoice -eq "1"
$buildLlama = $buildBoth -or $buildChoice -eq "2"
$buildBee = $buildBoth -or $buildChoice -eq "3"

# ============================================
# Build llama.cpp (upstream)
# ============================================
if ($buildLlama) {
    Write-Section "Building llama.cpp (upstream)"

    $llamaDir = Join-Path $BuildRoot "llama.cpp"

    # Clone or update
    if (Test-Path (Join-Path $llamaDir ".git")) {
        Write-Host "  -> llama.cpp already cloned, pulling latest..."
        Push-Location $llamaDir
        git pull
        Pop-Location
    } else {
        Write-Host "  -> Cloning llama.cpp..."
        New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
        git clone https://github.com/ggml-org/llama.cpp.git $llamaDir
    }
    Write-Ok "Source ready"

    # Configure
    Write-Host "  -> Configuring CMake..."
    cmake -S $llamaDir -B (Join-Path $llamaDir "build") `
        -DGGML_CUDA=ON `
        -DGGML_NATIVE=ON `
        -DGGML_CUDA_FA=ON `
        -DCMAKE_CUDA_ARCHITECTURES=$cudaArch `
        -DCMAKE_BUILD_TYPE=Release

    # Build
    Write-Host "  -> Building (this may take 5-15 minutes)..."
    cmake --build (Join-Path $llamaDir "build") --config Release --parallel

    # Verify
    $llamaServer = Join-Path $llamaDir "build\bin\Release\llama-server.exe"
    if (Test-Path $llamaServer) {
        Write-Ok "llama-server.exe built successfully"
        $binCount = (Get-ChildItem "build\bin\Release" -File).Count
        Write-Ok "$binCount binaries total"
    } else {
        Write-Fail "llama-server.exe not found after build"
    }
}

# ============================================
# Build beellama.cpp (fork with KVarN, etc.)
# ============================================
if ($buildBee) {
    Write-Section "Building beellama.cpp"

    $beeDir = Join-Path $BuildRoot "beellama.cpp"

    # Clone or update
    if (Test-Path (Join-Path $beeDir ".git")) {
        Write-Host "  -> beellama.cpp already cloned, pulling latest..."
        Push-Location $beeDir
        git pull
        Pop-Location
    } else {
        Write-Host "  -> Cloning beellama.cpp..."
        New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
        git clone https://github.com/Anbeeld/beellama.cpp.git $beeDir
    }
    Write-Ok "Source ready"

    # Configure
    Write-Host "  -> Configuring CMake..."
    cmake -S $beeDir -B (Join-Path $beeDir "build") `
        -DGGML_CUDA=ON `
        -DGGML_NATIVE=ON `
        -DGGML_CUDA_FA=ON `
        -DCMAKE_CUDA_ARCHITECTURES=$cudaArch `
        -DCMAKE_BUILD_TYPE=Release

    # Build
    Write-Host "  -> Building (this may take 5-15 minutes)..."
    cmake --build (Join-Path $beeDir "build") --config Release --parallel

    # Verify
    $beeServer = Join-Path $beeDir "build\bin\Release\llama-server.exe"
    if (Test-Path $beeServer) {
        Write-Ok "llama-server.exe built successfully"
        $binCount = (Get-ChildItem "build\bin\Release" -File).Count
        Write-Ok "$binCount binaries total"
    } else {
        Write-Fail "llama-server.exe not found after build"
    }
}

# ============================================
# Summary
# ============================================
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Build Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if ($buildLlama) {
    Write-Host "  llama.cpp (upstream):" -ForegroundColor Green
    Write-Host "    Server: $llamaServer"
}
if ($buildBee) {
    Write-Host "  beellama.cpp:" -ForegroundColor Green
    Write-Host "    Server: $beeServer"
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Download GGUF models to $ModelsDir"
Write-Host "  2. Run the launcher:"
Write-Host "     powershell -ExecutionPolicy Bypass -File `"$WorkspaceRoot\configs\llama\deepseek\start-deepseek-v4-flash.ps1`""
Write-Host ""
