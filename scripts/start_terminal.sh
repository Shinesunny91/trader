#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/shine/trading-workspace"
cd "$WORKSPACE"

# 1. Reload daemon and start systemd services and timers
systemctl --user daemon-reload
systemctl --user start nse-signal-lab.service
systemctl --user start nse-scanner.timer nse-paper-book.timer nse-candidate-paper.timer nse-context.timer nse-health.timer 2>/dev/null || true

# 2. Wait up to 10 seconds for Streamlit to respond on port 8501
STARTED=false
for i in $(seq 1 10); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 2>/dev/null | grep -qE "200|302|304"; then
    STARTED=true
    break
  fi
  sleep 1
done

if [ "$STARTED" = true ]; then
  notify-send "NSE Quant Terminal" "🚀 Trading Terminal is RUNNING at http://localhost:8501\nBackground scanners active." --icon=utilities-terminal 2>/dev/null || true
else
  notify-send "NSE Quant Terminal" "⏳ Trading Terminal starting... Opening http://localhost:8501" --icon=utilities-terminal 2>/dev/null || true
fi

# 3. Open browser
xdg-open "http://localhost:8501" 2>/dev/null || true
