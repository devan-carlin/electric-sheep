#!/usr/bin/env bash
# One-time bootstrap (run WITH a password):
#   sudo bash install-power-caps-service.sh
# Does three things, all idempotent:
#   1. Adds a NOPASSWD sudoers drop-in so the apply-script + systemctl
#      can be run by dc without a password going forward.
#   2. Installs arc-b70-power-caps.service (applies 180 W to all xe GPUs at boot).
#   3. Enables the service and applies the caps immediately.
set -euo pipefail

SERVE_DIR="/home/dc/electric-sheep/serve"
APPLY="$SERVE_DIR/apply-gpu-power-caps.sh"
UNIT="/etc/systemd/system/arc-b70-power-caps.service"
SUDOERS="/etc/sudoers.d/dc-gpu-persist"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
[ -x "$APPLY" ] || { echo "missing $APPLY"; exit 1; }

# --- 1. NOPASSWD sudoers drop-in -------------------------------------------
cat > "$SUDOERS" <<EOF
# Allow dc to manage B70 power caps + the boot service without a password.
dc ALL=(root) NOPASSWD: $APPLY, /usr/bin/systemctl
EOF
chmod 440 "$SUDOERS"
if ! visudo -cf "$SUDOERS" >/dev/null; then
  echo "FATAL: sudoers validation failed, removing $SUDOERS"
  rm -f "$SUDOERS"
  exit 1
fi
echo "[1/3] sudoers drop-in installed + validated: $SUDOERS"

# --- 2. systemd unit --------------------------------------------------------
cat > "$UNIT" <<'EOF'
[Unit]
Description=Apply 180W power cap to all Intel Arc Pro B70 GPUs
After=sysinit.target
Before=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/home/dc/electric-sheep/serve/apply-gpu-power-caps.sh

[Install]
WantedBy=multi-user.target
EOF
echo "[2/3] unit installed: $UNIT"

# --- 3. enable + apply now --------------------------------------------------
systemctl daemon-reload
systemctl enable arc-b70-power-caps.service
"$APPLY"
echo "[3/3] service enabled; caps applied now"
echo
echo "Current caps:"
for h in /sys/class/hwmon/hwmon*; do
  [ -e "$h/name" ] || continue
  [ "$(cat "$h/name" 2>/dev/null)" = "xe" ] || continue
  echo "  $(basename "$h"): $(cat "$h/power1_cap") uW"
done
echo
echo "Done. Reboots now auto-apply 180 W. Manage later with:"
echo "  sudo systemctl status arc-b70-power-caps"
echo "  sudo $APPLY            # re-apply now (passwordless)"
