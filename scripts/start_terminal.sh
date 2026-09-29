#!/usr/bin/env bash
# Start the app and every scheduled job (see deploy/systemd/), then open the app.
set -euo pipefail
cd "$(dirname "$0")/.."

TIMERS=(nse-scanner nse-context nse-health nse-gap-picks nse-gap-levels nse-gap-record)

systemctl --user daemon-reload
systemctl --user start nse-signal-lab.service
for t in "${TIMERS[@]}"; do systemctl --user start "$t.timer" 2>/dev/null || true; done

# Wait up to 10 seconds for Streamlit to answer.
STARTED=false
for _ in $(seq 1 10); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 2>/dev/null | grep -qE "200|302|304"; then
    STARTED=true
    break
  fi
  sleep 1
done

if [ "$STARTED" = true ]; then
  notify-send "NSE Intraday Signal Lab" "Running at http://localhost:8501 — scheduled jobs active." --icon=utilities-terminal 2>/dev/null || true
else
  notify-send "NSE Intraday Signal Lab" "Starting... opening http://localhost:8501" --icon=utilities-terminal 2>/dev/null || true
fi
xdg-open "http://localhost:8501" 2>/dev/null || true
