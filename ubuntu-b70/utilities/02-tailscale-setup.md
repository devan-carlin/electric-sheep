# Tailscale Setup for Electric Sheep

## Overview

Tailscale creates a secure mesh network between your machines, enabling seamless communication regardless of physical location. This is essential for accessing your Windows RTX 5090 llama.cpp server from the Ubuntu B70 AI server (and vice versa).

## Why Tailscale?

- **No firewall configuration needed** — works through NAT, firewalls, and corporate networks
- **Stable IPs** — Tailscale IPs (`100.x.x.x`) don't change like DHCP addresses
- **Encrypted by default** — all traffic is encrypted end-to-end
- **Simple auth** — sign in with Google, GitHub, or Microsoft account

## Current Network Topology

| Machine | Tailscale IP | Role |
|---|---|---|
| HP Omen 45L (Windows) | `100.64.238.80` | RTX 5090, llama.cpp server |
| AI Server (Ubuntu B70) | *(install below)* | 4× Intel Arc B70, vLLM server |

---

## Installation

### Ubuntu B70 Server

```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Start and enable
sudo systemctl enable --now tailscaled

# Authenticate (opens browser on remote machine, or use auth key)
sudo tailscale up

# Verify
tailscale status
```

**Headless setup (no browser):** Generate an auth key at [Tailscale Admin Console](https://login.tailscale.com/admin/settings/keys), then:

```bash
sudo tailscale up --auth-key=tskey-xxxxxxxxxxxxxxxx
```

### Windows HP Omen 45L

1. Download installer: [tailscale.com/download](https://tailscale.com/download)
2. Run the `.msi` installer
3. Sign in with your Tailscale account
4. Verify in PowerShell:
```powershell
tailscale status
```

---

## Configuration

### 1. Verify Connectivity

From the Ubuntu server, ping your Windows machine:
```bash
ping 100.64.238.80
```

From Windows, ping the Ubuntu server (replace with actual Tailscale IP):
```powershell
ping <ubuntu-tailscale-ip>
```

### 2. Configure llama.cpp to Listen on All Interfaces

Your llama.cpp server on Windows needs to bind to `0.0.0.0`, not `127.0.0.1`:

```powershell
.\server.exe -m models\qwen3.6-27b.gguf --host 0.0.0.0 --port 8080
```

**Verify it's listening:**
```powershell
netstat -an | Select-String "8080"
```

Should show `0.0.0.0:8080` (not `127.0.0.1:8080`).

### 3. Test from Ubuntu Server

```bash
curl http://100.64.238.80:8080/v1/models
```

Should return the available models.

### 4. Configure VS Code Chat Profile

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

Select this profile in VS Code's chat model selector when SSH'd into the Ubuntu server.

---

## Advanced Configuration

### MagicDNS (Hostname Resolution)

Enable MagicDNS in the [Tailscale Admin Console](https://login.tailscale.com/admin/dns) to use hostnames instead of IPs:

```bash
# Instead of: http://100.64.238.80:8080/v1
# Use: http://omen-45l:8080/v1
```

### SSH via Tailscale

Skip password/SSH key setup — use Tailscale SSH:

```bash
# Enable Tailscale SSH on Ubuntu
sudo tailscale ssh --setup

# SSH from any Tailscale machine
tailscale ssh dc@ai-server
```

### File Sharing (Serve)

Share files over Tailscale without setting up NFS/SMB:

```bash
# Serve a directory on Ubuntu
sudo tailscale serve ~/electric-sheep/models

# Access from Windows (browser or curl)
curl http://ai-server:8080/models/
```

### Exit Node (Route All Traffic)

If you need to access local resources through one machine:

```bash
# On Ubuntu, enable as exit node
sudo tailscale up --advertise-exit-node

# On Windows, route through Ubuntu
tailscale up --exit-node=<ubuntu-tailscale-ip>
```

---

## Troubleshooting

### Can't reach the other machine

```bash
# Check Tailscale is running
sudo systemctl status tailscaled

# Check both machines are on the same tailnet
tailscale status

# Check firewall (shouldn't be needed, but just in case)
sudo ufw status
```

### llama.cpp not reachable via Tailscale IP

```powershell
# On Windows, check if listening on all interfaces
netstat -an | Select-String "8080"

# If it shows 127.0.0.1:8080, restart with --host 0.0.0.0
```

### Connection drops intermittently

```bash
# Check Tailscale logs
sudo tailscale bugreport

# Force reconnection
sudo tailscale up
```

### Windows Firewall blocking

```powershell
# Allow Tailscale through Windows Firewall
New-NetFirewallRule -DisplayName "Tailscale" -Direction Inbound -Action Allow -Protocol UDP -LocalPort 41641
New-NetFirewallRule -DisplayName "llama.cpp" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8080
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
| `tailscale tail <ip>` | View another machine's logs |
