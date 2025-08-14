#!/usr/bin/env bash
set -euo pipefail

# Minimal helper to run on Raspberry Pi to commit and push new images from database/
# First run: installs a systemd service + timer to auto-run nightly and after boot.
# Usage: ./scripts/edge_sync.sh "Add images from lab session"

MSG=${1:-"chore(data): edge sync new images"}

SCRIPT_FILE="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_FILE")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_ABS="$SCRIPT_DIR/$(basename "$SCRIPT_FILE")"

SERVICE_NAME="edge-sync.service"
TIMER_NAME="edge-sync.timer"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
TIMER_PATH="/etc/systemd/system/$TIMER_NAME"

need_sudo() {
  # Return 0 if sudo available and we are not root
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || {
      echo "sudo is required to install the system service. Please run as root." >&2
      exit 1
    }
    echo sudo
  else
    echo ""
  fi
}

install_systemd() {
  # Install systemd service + timer if missing
  if [ -f "$SERVICE_PATH" ] && [ -f "$TIMER_PATH" ]; then
    return 0
  fi

  echo "Installing systemd units for edge sync..."
  SUDO=$(need_sudo)

  # Service: one-shot execution of this script
  SERVICE_CONTENT="[Unit]\nDescription=Edge sync database images to GitHub\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=oneshot\nWorkingDirectory=$REPO_ROOT\nExecStart=$SCRIPT_ABS \"auto sync\"\nStandardOutput=append:/var/log/edge-sync.log\nStandardError=inherit\n\n[Install]\nWantedBy=multi-user.target\n"
  echo -e "$SERVICE_CONTENT" | $SUDO tee "$SERVICE_PATH" >/dev/null

  # Timer: run after boot and nightly at 02:00, catch up if missed
  TIMER_CONTENT="[Unit]\nDescription=Run edge-sync service daily and after boot\n\n[Timer]\nOnBootSec=5min\nOnCalendar=02:00\nPersistent=true\nUnit=$SERVICE_NAME\n\n[Install]\nWantedBy=timers.target\n"
  echo -e "$TIMER_CONTENT" | $SUDO tee "$TIMER_PATH" >/dev/null

  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now "$TIMER_NAME"

  # Optional: kick off a first run once network is up (best-effort)
  if $SUDO systemctl is-active --quiet network-online.target 2>/dev/null; then
    $SUDO systemctl start "$SERVICE_NAME" || true
  fi

  echo "Systemd service and timer installed. Nightly sync scheduled at 02:00."
}

main_sync() {
  # Ensure we're at repo root
  cd "$REPO_ROOT"

  # Only add database changes
  if git status --porcelain=1 database | grep -qE '^(\?\?| M|A |D )'; then
    git add database
    if git diff --cached --quiet; then
      echo "No staged changes."
    else
      git commit -m "$MSG" || true
      if git push; then
        echo "Pushed image updates to origin."
      else
        echo "Push failed. Ensure remote auth (SSH keys or token) is configured." >&2
        exit 1
      fi
    fi
  else
    echo "No database changes to push."
  fi
}

# Install systemd on first run if not present
install_systemd || true

# Perform a sync run now (manual run or timer/boot)
main_sync
