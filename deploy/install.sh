#!/usr/bin/env bash
# Install (or refresh) the user-level systemd units that run the workspace.
#   ./deploy/install.sh            install + enable + start
#   ./deploy/install.sh --remove   stop, disable and delete them
set -euo pipefail
cd "$(dirname "$0")/systemd"
DEST="$HOME/.config/systemd/user"
# Every *.timer in this directory is installed and enabled (single source of truth).
TIMERS=($(ls ./*.timer | xargs -n1 basename | sed 's/\.timer$//'))
# Units from earlier layouts that no longer exist in this repo.
RETIRED=(nse-paper-book nse-candidate-paper nse-learn nse-retrain nse-logrotate nse-scanner nse-context)

for name in "${RETIRED[@]}"; do
  systemctl --user disable --now "$name.timer" 2>/dev/null || true
  rm -f "$DEST/$name.timer" "$DEST/$name.service"
done

if [[ "${1:-}" == "--remove" ]]; then
  for name in "${TIMERS[@]}"; do systemctl --user disable --now "$name.timer" 2>/dev/null || true; done
  systemctl --user disable --now nse-signal-lab.service 2>/dev/null || true
  for f in *.service *.timer; do rm -f "$DEST/$f"; done
  systemctl --user daemon-reload
  echo "removed"
  exit 0
fi

mkdir -p "$DEST"
install -m 644 ./*.service ./*.timer "$DEST/"
systemctl --user daemon-reload
systemctl --user enable --now nse-signal-lab.service
for name in "${TIMERS[@]}"; do systemctl --user enable --now "$name.timer"; done
loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user list-timers 'nse-*' --no-pager
