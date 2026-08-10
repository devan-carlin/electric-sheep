#Requires -Version 5.1
# ============================================
# llama.cpp Prerequisites Check — Windows 5090
# ============================================
# Validates CUDA toolkit, CMake, MSVC build tools,
# GPU detection, and disk space for building
# llama.cpp and beellama.cpp from source.
# ============================================

$ErrorActionPreference = "Stop"

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
    $script:warnings++
}

function Write-Fail {
    param([string]$Msg)
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "  ERROR: $Msg" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$script:warnings = 0

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  llama.cpp Prerequisites — Windows 5090" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# --- A. OS ---
Write-Section "A. Operating System"
$os = Get-CimInstance Win32_OperatingSystem
Write-Host "  OS: $($os.Caption)"
Write-Host "  Build: $($os.BuildNumber)"
$cpuCores = (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors
Write-Host "  CPU Threads: $cpuCores"

# --- B. GPU Detection ---
Write-Section "B. GPU Hardware"
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($gpuInfo) {
        $parts = $gpuInfo -split ',\s*'
        $gpuName = $parts[0]
        $gpuMem = $parts[1]
        $driverVer = $parts[2]
        Write-Host "  GPU: $gpuName"
        Write-Host "  VRAM: $gpuMem"
        Write-Host "  Driver: $driverVer"
        Write-Ok "NVIDIA GPU detected"

        # Check for RTX 5090 specifically
        if ($gpuName -match "RTX 5090") {
            Write-Ok "RTX 5090 (Blackwell sm_100) confirmed"
        } elseif ($gpuName -match "RTX 4") {
            Write-Warn "RTX 40xx detected — adjust CMAKE_CUDA_ARCHITECTURES to 89"
        }
    } else {
        Write-Fail "No NVIDIA GPU detected. Install NVIDIA drivers first."
    }
} catch {
    Write-Fail "nvidia-smi not found. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx"
}

# --- C. CUDA Toolkit ---
Write-Section "C. CUDA Toolkit"
try {
    $nvccVer = nvcc --version 2>$null | Select-String "release (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    if ($nvccVer) {
        Write-Ok "nvcc $nvccVer found"
        $cudaMajor = [int]($nvccVer -split '\.')[0]
        if ($cudaMajor -lt 12) {
            Write-Warn "CUDA < 12.0 — some features may not work optimally"
        }
    } else {
        Write-Fail "nvcc not found. Install CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit"
    }
} catch {
    Write-Fail "Failed to run nvcc. Ensure CUDA is installed and PATH is set."
}

# --- D. CMake ---
Write-Section "D. CMake"
try {
    $cmakeVer = cmake --version 2>$null | Select-String "cmake version (\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    if ($cmakeVer) {
        $cmakeMajor = [int]($cmakeVer -split '\.')[0]
        $cmakeMinor = [int]($cmakeVer -split '\.')[1]
        if ($cmakeMajor -lt 3 -or ($cmakeMajor -eq 3 -and $cmakeMinor -lt 24)) {
            Write-Warn "CMake $cmakeVer < 3.24 — upgrade recommended"
        } else {
            Write-Ok "CMake $cmakeVer"
        }
    } else {
        Write-Fail "CMake not found. Install from https://cmake.org/download/"
    }
} catch {
    Write-Fail "Failed to run cmake."
}

# --- E. MSVC Build Tools ---
Write-Section "E. MSVC Build Tools"
$msvcFound = $false

# Check for Visual Studio 2022
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ($vsPath) {
        Write-Ok "Visual Studio 2022 found at $vsPath"
        $msvcFound = $true
    }
}

# Check for Build Tools (standalone)
if (-not $msvcFound) {
    $clExe = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($clExe) {
        Write-Ok "cl.exe found on PATH (Build Tools or Developer Command Prompt)"
        $msvcFound = $true
    }
}

if (-not $msvcFound) {
    Write-Warn "MSVC compiler not detected on PATH"
    Write-Host "    Install Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    Write-Host "    Select 'Desktop development with C++' workload"
}

# --- F. Git ---
Write-Section "F. Git"
try {
    $gitVer = git --version 2>$null | Select-String "git version (\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    if ($gitVer) {
        Write-Ok "Git $gitVer"
    } else {
        Write-Fail "Git not found. Install from https://git-scm.com/download/win"
    }
} catch {
    Write-Fail "Failed to run git."
}

# --- G. Python (for model conversion/quantization) ---
Write-Section "G. Python (Optional)"
try {
    $pyVer = python --version 2>$null | Select-String "Python (\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    if ($pyVer) {
        Write-Ok "Python $pyVer (needed for GGUF conversion/quantization)"
    } else {
        Write-Warn "Python not found — model conversion tools won't work"
        Write-Host "    Install from https://www.python.org/downloads/"
    }
} catch {
    Write-Warn "Python check failed"
}

# --- H. Disk Space ---
Write-Section "H. Disk Space"
$targetDrive = "C:\"
$drive = Get-PSDrive $targetDrive.TrimEnd('\') -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Write-Host "  Free on $targetDrive: ${freeGB}GB"
    if ($freeGB -lt 20) {
        Write-Fail "Need at least 20GB free for build + source, have ${freeGB}GB"
    } elseif ($freeGB -lt 50) {
        Write-Warn "Less than 50GB — build will complete but leaves little headroom"
    } else {
        Write-Ok "Sufficient disk space"
    }
} else {
    Write-Warn "Could not check disk space for $targetDrive"
}

# --- I. Summary ---
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Prerequisites Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  GPU:      $gpuName ($gpuMem)"
Write-Host "  CUDA:     $nvccVer"
Write-Host "  CMake:    $cmakeVer"
Write-Host "  MSVC:     $(if ($msvcFound) { 'Found' } else { 'Not found' })"
Write-Host "  Git:      $gitVer"
Write-Host "  Disk:     ${freeGB}GB free"

if ($script:warnings -gt 0) {
    Write-Host ""
    Write-Host "  ⚠ $script:warnings warning(s) detected — review above" -ForegroundColor Yellow
    Write-Host "  Continuing anyway (warnings are non-fatal)..."
} else {
    Write-Host ""
    Write-Host "  ✓ All checks passed" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
