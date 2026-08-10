# Ollama Deployment Guide — RTX 5090

Quick-start guide for running Ollama with RTX 5090 optimizations on Windows.

## Table of Contents

1. [Installation](#1-installation)
2. [Configuration](#2-configuration)
3. [Running Models](#3-running-models)
4. [Custom Modelfiles](#4-custom-modelfiles)
5. [KV Cache Quantization](#5-kv-cache-quantization)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Installation

### Quick Install

```powershell
# Official one-line installer
Invoke-WebRequest -Uri "https://ollama.com/install.ps1" -UseBasicParsing | Invoke-Expression
```

Or download manually from [ollama.com](https://ollama.com).

### Verify Installation

```powershell
ollama --version
ollama list
```

Ollama runs as a system tray process on Windows. After install, it starts automatically.

### Run Setup Script

```powershell
cd ~/electric-sheep/scripts/windows
.\01-setup-ollama.ps1
```

This configures:
- `OLLAMA_NUM_PARALLEL=4` — concurrent requests
- `OLLAMA_MAX_LOADED_MODELS=3` — keep multiple models in VRAM
- `OLLAMA_FLASH_ATTENTION=1` — enable flash attention
- `OLLAMA_KV_CACHE_TYPE=q8_0` — quantized KV cache (saves VRAM)
- `OLLAMA_KEEP_ALIVE=24h` — models stay loaded for 24 hours

---

## 2. Configuration

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `OLLAMA_NUM_PARALLEL` | `4` | Concurrent request handling |
| `OLLAMA_MAX_LOADED_MODELS` | `3` | Models kept in VRAM simultaneously |
| `OLLAMA_FLASH_ATTENTION` | `1` | Flash attention (faster, less VRAM) |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | KV cache quantization (saves ~50% VRAM) |
| `OLLAMA_KEEP_ALIVE` | `24h` | How long models stay loaded |
| `OLLAMA_HOST` | `0.0.0.0:11434` | Bind to all interfaces (optional) |
| `OLLAMA_LOAD_TIMEOUT` | `180` | Model load timeout in seconds |

### Apply Configuration

```powershell
# Run the setup script
.\01-setup-ollama.ps1

# Or set manually (user-level, persists across sessions)
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")

# IMPORTANT: Restart Ollama after changing settings
# Right-click system tray icon → Quit → Re-launch
```

### Reset to Defaults

```powershell
.\01-setup-ollama.ps1 -Reset
```

---

## 3. Running Models

### Pull and Run a Model

```powershell
# Pull a model
ollama pull qwen3.6:27b

# Run interactively
ollama run qwen3.6:27b

# Run with a prompt
ollama run qwen3.6:27b "Explain quantum computing in simple terms."
```

### Available Model Tags

| Model | Tag | Size | VRAM (with q8_0 KV) |
|-------|-----|------|---------------------|
| Qwen 3.6 27B | `qwen3.6:27b` | ~16 GB | ~20 GB |
| Qwen 3.6 35B-A3B | `qwen3.6:35b-a3b` | ~20 GB | ~25 GB |
| Gemma 4 26B-A4B | `gemma4:26b` | ~16 GB | ~20 GB |
| Gemma 4 31B | `gemma4:31b` | ~18 GB | ~22 GB |
| Llama 3.3 70B | `llama3.3:70b` | ~40 GB | ❌ Too large for 32 GB |

### API Access

Ollama exposes an OpenAI-compatible API on `http://localhost:11434`:

```powershell
# List models
curl http://localhost:11434/api/tags

# Chat completion
curl http://localhost:11434/api/chat `
    -H "Content-Type: application/json" `
    -d '{
        "model": "qwen3.6:27b",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "stream": false
    }'
```

---

## 4. Custom Modelfiles

For models not on Ollama's library, create a Modelfile pointing to local GGUF files.

### Gemma 4 Modelfile

```dockerfile
# Point to your primary Gemma 4 text model
FROM "C:\path\to\gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-NVFP4.gguf"

# Load the vision projector via ADAPTER
ADAPTER "C:\path\to\gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-mmproj-BF16.gguf"

# Set up the chat structure for Gemma 4
TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

# Configure Gemma 4 stop tokens and generation preferences
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"
PARAMETER temperature 0.7
```

Build and run:
```powershell
ollama create gemma4-custom -f Modelfile_gemma
ollama run gemma4-custom
```

### Qwen 3.6 Modelfile

```dockerfile
# Point to your primary 4-bit text model
FROM "C:\path\to\Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_S.gguf"

# Point to the high-fidelity BF16 vision projector
FROM "C:\path\to\Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf"

# Set up the chat structure for Qwen
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
"""

# Configure stop tokens and generation preferences
PARAMETER stop "<|end|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
```

Build and run:
```powershell
ollama create qwen35b-custom -f Modelfile_qwen
ollama run qwen35b-custom
```

### Modelfile Reference

| Directive | Purpose |
|-----------|---------|
| `FROM` | Path to GGUF model file (local or URL) |
| `ADAPTER` | Path to adapter/LoRA/mmproj file |
| `TEMPLATE` | Chat template (Go template syntax) |
| `PARAMETER` | Generation parameter (temperature, stop, etc.) |
| `SYSTEM` | Default system prompt |

---

## 5. KV Cache Quantization

Ollama supports quantized KV caches to reduce VRAM usage. Set via `OLLAMA_KV_CACHE_TYPE`.

### Available Types

| Type | Bytes/Element | VRAM Savings vs f16 | Quality |
|------|--------------|-------------------|---------|
| `f16` | 2.0 | 0% | Full precision |
| `q8_0` | 1.0 | 50% | Excellent |
| `q4_0` | 0.5 | 75% | Good |

### Recommended for RTX 5090

```powershell
# Best balance of quality and VRAM savings
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"

# Maximum VRAM savings (for very large context windows)
$env:OLLAMA_KV_CACHE_TYPE = "q4_0"
```

### Context Size vs VRAM (27B model, q8_0 KV)

| Context | Model | KV Cache | Total | Fits? |
|---------|-------|----------|-------|-------|
| 32K | ~16 GB | ~1 GB | ~17 GB | ✅ |
| 64K | ~16 GB | ~2 GB | ~18 GB | ✅ |
| 128K | ~16 GB | ~4 GB | ~20 GB | ✅ |
| 256K | ~16 GB | ~8 GB | ~24 GB | ✅ |
| 512K | ~16 GB | ~16 GB | ~32 GB | ⚠️ Tight |

---

## 6. Troubleshooting

### "Model not found"

```powershell
# Check loaded models
ollama list

# Pull the model
ollama pull qwen3.6:27b
```

### "CUDA out of memory"

The model exceeds 32 GB VRAM. Options:

1. **Use smaller model:** `qwen3.6:27b` instead of `qwen3.6:35b-a3b`
2. **Reduce context:** Set `num_ctx` in Modelfile: `PARAMETER num_ctx 32768`
3. **Use KV quantization:** `OLLAMA_KV_CACHE_TYPE=q4_0`
4. **Reduce loaded models:** `OLLAMA_MAX_LOADED_MODELS=1`

### Ollama not using GPU

```powershell
# Check Ollama logs (Windows Event Viewer or system tray)
# Ensure NVIDIA drivers are installed
nvidia-smi

# Restart Ollama after setting environment variables
# Right-click system tray → Quit → Re-launch
```

### "Connection refused" on port 11434

Ollama isn't running. Start from system tray or:
```powershell
# Start Ollama service
ollama serve
```

### Bind to all network interfaces

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "User")
# Restart Ollama
```

### Verify Configuration

```powershell
# Check environment variables
Get-ChildItem Env:OLLAMA_*

# Check GPU usage
nvidia-smi

# Test API
curl http://localhost:11434/api/tags
```

---

## Appendix: Comparison with llama.cpp

| Feature | Ollama | llama.cpp |
|---------|--------|-----------|
| **Ease of use** | Very easy (one command) | Moderate (CMake build) |
| **Model management** | Built-in (pull, create, rm) | Manual (download GGUF) |
| **API** | OpenAI-compatible | llama-server (OpenAI-compatible) |
| **KV cache quantization** | `q8_0`, `q4_0` | `q4_0`–`q8_0`, `kvarn3`–`kvarn8` (beellama) |
| **Flash attention** | Auto-enabled | Manual (`-fa on`) |
| **Multi-model loading** | Built-in (`OLLAMA_MAX_LOADED_MODELS`) | Manual (separate processes) |
| **Custom modelfiles** | Yes (Modelfile syntax) | N/A (command-line flags) |
| **Best for** | Quick setup, model management, API serving | Maximum control, advanced features, KVarN |

**Recommendation:** Use Ollama for quick model testing and API serving. Use llama.cpp/beellama.cpp for maximum performance tuning and advanced KV cache features (KVarN, precision tail).
