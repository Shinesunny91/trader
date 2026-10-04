#!/usr/bin/env bash
# Stop the app and every scheduled job. `./scripts/start_terminal.sh` undoes it.
set -euo pipefail
cd "$(dirname "$0")/.."

TIMERS=($(ls deploy/systemd/*.timer | xargs -n1 basename | sed 's/\.timer$//'))

systemctl --user stop nse-signal-lab.service 2>/dev/null || true
for t in "${TIMERS[@]}"; do systemctl --user stop "$t.timer" 2>/dev/null || true; done
pkill -f "$PWD.*(streamlit|gap_reversal|learn)" 2>/dev/null || true

notify-send "NSE Gap-Reversal Book" "App and all scheduled jobs STOPPED." --icon=process-stop 2>/dev/null || true
