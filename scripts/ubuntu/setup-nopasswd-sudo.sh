#!/usr/bin/env bash
# Grant passwordless sudo to the current user.
# Run: bash setup-nopasswd-sudo.sh
set -euo pipefail

USER_NAME="$(whoami)"
SUDOERS_FILE="/etc/sudoers.d/${USER_NAME}"

echo "User: ${USER_NAME}"

# Write the NOPASSWD rule (needs one sudo with password)
echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" | sudo tee "${SUDOERS_FILE}" >/dev/null
sudo chmod 440 "${SUDOERS_FILE}"

# Validate the sudoers file so a bad rule can't lock you out
sudo visudo -cf "${SUDOERS_FILE}"

echo "Done. Test with: sudo -n whoami"
