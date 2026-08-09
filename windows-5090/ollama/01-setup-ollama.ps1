#Requires -Version 5.1
# ============================================
# Ollama Setup — Windows 5090 (RTX 5090)
# ============================================
# Installs/configures Ollama with RTX 5090
# optimizations: flash attention, KV cache
# quantization, parallel requests, and
# multi-model loading.
# ============================================

param(
    [switch]$Reset,
    [int]$NumParallel = 4,
    [int]$MaxLoadedModels = 3,
    [string]$KvCacheType = "q8_0"
)

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
Write-Host "  Ollama Setup — Windows 5090" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# --- Check if Ollama is installed ---
Write-Section "Ollama Installation"
$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaPath) {
    Write-Host "  Ollama not found on PATH." -ForegroundColor Yellow
    Write-Host "  Installing Ollama..." -ForegroundColor Yellow
    try {
        # Official Ollama installer
        Invoke-WebRequest -Uri "https://ollama.com/install.ps1" -UseBasicParsing | Invoke-Expression
        Write-Ok "Ollama installed"
    } catch {
        Write-Fail "Failed to install Ollama. Install manually from https://ollama.com"
    }
} else {
    Write-Ok "Ollama found at: $($ollamaPath.Source)"
}

# --- Check GPU ---
Write-Section "GPU Detection"
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    $parts = $gpuInfo -split ',\s*'
    $gpuName = $parts[0]
    $gpuMem = $parts[1]
    Write-Ok "$gpuName ($gpuMem)"

    if ($gpuName -match "RTX 5090") {
        Write-Ok "RTX 5090 detected — optimal for large models"
    }
} catch {
    Write-Warn "nvidia-smi not found — GPU check skipped"
}

# --- Reset mode ---
if ($Reset) {
    Write-Section "Resetting Ollama Configuration"
    [Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", $null, "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", $null, "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", $null, "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", $null, "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", $null, "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_HOST", $null, "User")
    Write-Ok "All Ollama environment variables reset to defaults"
    Write-Host ""
    Write-Host "IMPORTANT: Restart Ollama for changes to take effect:" -ForegroundColor Yellow
    Write-Host "  1. Right-click Ollama icon in system tray → Quit" -ForegroundColor White
    Write-Host "  2. Re-launch Ollama from Start menu" -ForegroundColor White
    exit 0
}

# --- Configure environment variables ---
Write-Section "Configuring Ollama"

[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", $NumParallel.ToString(), "User")
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", $MaxLoadedModels.ToString(), "User")
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", $KvCacheType, "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "24h", "User")

Write-Ok "Environment variables configured:"
Write-Host "    OLLAMA_NUM_PARALLEL      = $NumParallel" -ForegroundColor White
Write-Host "    OLLAMA_MAX_LOADED_MODELS = $MaxLoadedModels" -ForegroundColor White
Write-Host "    OLLAMA_FLASH_ATTENTION   = 1" -ForegroundColor White
Write-Host "    OLLAMA_KV_CACHE_TYPE     = $KvCacheType" -ForegroundColor White
Write-Host "    OLLAMA_KEEP_ALIVE        = 24h" -ForegroundColor White

# --- Verify Ollama is running ---
Write-Section "Ollama Service Status"
try {
    $ollamaVersion = ollama --version 2>$null
    if ($ollamaVersion) {
        Write-Ok "Ollama is running: $ollamaVersion"
    } else {
        Write-Warn "Ollama may not be running. Start from system tray."
    }
} catch {
    Write-Warn "Could not check Ollama status (may not be running)"
}

# --- Summary ---
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: Restart Ollama for changes to take effect:" -ForegroundColor Yellow
Write-Host "  1. Right-click Ollama icon in system tray → Quit" -ForegroundColor White
Write-Host "  2. Re-launch Ollama from Start menu" -ForegroundColor White
Write-Host ""
Write-Host "After restart, verify with:" -ForegroundColor Yellow
Write-Host "  ollama list" -ForegroundColor White
Write-Host ""
Write-Host "To reset to defaults later:" -ForegroundColor Yellow
Write-Host "  .\01-setup-ollama.ps1 -Reset" -ForegroundColor White
Write-Host ""
