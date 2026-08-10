# Tailscale Setup for HP Omen 45L (Windows)

## Overview

Tailscale creates a secure mesh network between your machines. On the HP Omen 45L, this enables the Ubuntu B70 AI server to reach your RTX 5090 llama.cpp/Ollama instances over the Tailscale network.

## Current Network Topology

| Machine | Tailscale IP | Role |
|---|---|---|
| HP Omen 45L (Windows) | `100.64.238.80` | RTX 5090, llama.cpp server |
| AI Server (Ubuntu B70) | *(TBD — install on server)* | 4× Intel Arc B70, vLLM server |

---

## Installation

### 1. Download & Install

1. Go to [tailscale.com/download](https://tailscale.com/download)
2. Download the Windows installer (`.msi`)
3. Run the installer, accept defaults
4. Sign in with your Google, GitHub, or Microsoft account

### 2. Verify Installation

```powershell
# Check status
tailscale status

# Show your Tailscale IP
tailscale ip -ip
```

You should see `100.64.238.80` (or similar `100.x.x.x` address).

### 3. Check Firewall

Tailscale usually handles this automatically, but verify:

```powershell
# Check if Tailscale firewall rules exist
Get-NetFirewallRule | Where-Object { $_.DisplayName -like "*Tailscale*" }
```

If missing, add them:

```powershell
New-NetFirewallRule -DisplayName "Tailscale" -Direction Inbound -Action Allow -Protocol UDP -LocalPort 41641
```

---

## Configure llama.cpp for Remote Access

### Critical: Listen on All Interfaces

Your llama.cpp server must bind to `0.0.0.0`, not `127.0.0.1`:

```powershell
.\server.exe -m models\qwen3.6-27b.gguf --host 0.0.0.0 --port 8080
```

**Verify it's listening on all interfaces:**

```powershell
netstat -an | Select-String "8080"
```

Should show:
```
TCP    0.0.0.0:8080           0.0.0.0:0              LISTENING
```

**NOT:**
```
TCP    127.0.0.1:8080         0.0.0.0:0              LISTENING
```

### Firewall Rule for llama.cpp

```powershell
New-NetFirewallRule -DisplayName "llama.cpp Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8080
```

### Test from Ubuntu Server

Once Tailscale is installed on the Ubuntu server:

```bash
# From Ubuntu, test connectivity
curl http://100.64.238.80:8080/v1/models
```

---

## Configure Ollama for Remote Access (Optional)

If you also want to access Ollama from the Ubuntu server:

### 1. Set Environment Variable

```powershell
# Allow Ollama to listen on all interfaces
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "Machine")
```

### 2. Restart Ollama

```powershell
# Stop and restart Ollama service
ollama serve
```

### 3. Firewall Rule

```powershell
New-NetFirewallRule -DisplayName "Ollama Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
```

### 4. Test from Ubuntu

```bash
curl http://100.64.238.80:11434/api/tags
```

---

## VS Code Chat Configuration

Your `chatLanguageModels.json` already has the Tailscale profile:

```json
{
  "name": "Remote RTX 5090 (HP Omen via Tailscale)",
  "vendor": "customendpoint",
  "apiKey": "${input:chat.lm.secret.-99e3b06}",
  "apiType": "chat-completions",
  "models": [
    {
      "id": "qwen3.6-27b",
      "name": "Qwen 3.6 27B (HP Omen RTX 5090 - Tailscale)",
      "url": "http://100.64.238.80:8080/v1",
      "toolCalling": true,
      "vision": true,
      "maxInputTokens": 229376,
      "maxOutputTokens": 32768
    }
  ]
}
```

When SSH'd into the Ubuntu server, select **"Remote RTX 5090 (HP Omen via Tailscale)"** in VS Code's chat model selector.

---

## Advanced Configuration

### MagicDNS (Hostname Resolution)

Enable MagicDNS in the [Tailscale Admin Console](https://login.tailscale.com/admin/dns) to use hostnames:

```bash
# Instead of: http://100.64.238.80:8080/v1
# Use: http://omen-45l:8080/v1
```

### SSH from Ubuntu to Windows

```bash
# From Ubuntu, SSH to Windows (requires Windows OpenSSH server)
ssh dcarl@100.64.238.80
```

### File Sharing via Tailscale Serve

Share files without setting up SMB/NFS:

```powershell
# Serve a directory (requires Tailscale Serve on Windows)
tailscale serve models\
```

---

## Troubleshooting

### Ubuntu can't reach llama.cpp

```powershell
# On Windows, check if llama.cpp is listening on 0.0.0.0
netstat -an | Select-String "8080"

# Check Windows Firewall
Get-NetFirewallRule | Where-Object { $_.DisplayName -like "*llama*" }

# Test locally first
curl http://localhost:8080/v1/models
```

### Tailscale connection drops

```powershell
# Reconnect
tailscale up

# Check logs
tailscale bugreport
```

### Wrong Tailscale IP

```powershell
# Check current IP
tailscale ip -ip

# If it changed, update chatLanguageModels.json
```

### Firewall blocking Tailscale

```powershell
# Allow Tailscale
New-NetFirewallRule -DisplayName "Tailscale" -Direction Inbound -Action Allow -Protocol UDP -LocalPort 41641

# Allow llama.cpp
New-NetFirewallRule -DisplayName "llama.cpp Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8080

# Allow Ollama (if using)
New-NetFirewallRule -DisplayName "Ollama Server" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
```

---

## Useful Commands

| Command | Description |
|---|---|
| `tailscale status` | Show connected devices |
| `tailscale ip -ip` | Show your Tailscale IP |
| `tailscale ping <ip>` | Test connectivity |
| `tailscale netcheck` | Check network conditions |
| `tailscale logout` | Disconnect from tailnet |
| `tailscale up` | Reconnect to tailnet |
| `netstat -an \| Select-String "8080"` | Check if llama.cpp is listening |
