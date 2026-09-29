#!/usr/bin/env bash
# Stop the app and every scheduled job. `./scripts/start_terminal.sh` undoes it.
set -euo pipefail
cd "$(dirname "$0")/.."

TIMERS=(nse-scanner nse-context nse-health nse-gap-picks nse-gap-final nse-gap-levels nse-gap-record nse-gap-learn)

systemctl --user stop nse-signal-lab.service 2>/dev/null || true
for t in "${TIMERS[@]}"; do systemctl --user stop "$t.timer" 2>/dev/null || true; done
pkill -f "$PWD.*(streamlit|scanner_daemon|gap_reversal)" 2>/dev/null || true

notify-send "NSE Intraday Signal Lab" "App and all scheduled jobs STOPPED." --icon=process-stop 2>/dev/null || true
