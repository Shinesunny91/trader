#!/usr/bin/env bash
set -euo pipefail

# 1. Stop systemd services and timers
systemctl --user stop nse-signal-lab.service 2>/dev/null || true
systemctl --user stop nse-scanner.timer nse-paper-book.timer nse-candidate-paper.timer nse-context.timer nse-health.timer 2>/dev/null || true

# 2. Kill any lingering process from trading-workspace
pkill -f "/home/shine/trading-workspace.*(streamlit|scanner_daemon)" 2>/dev/null || true

# 3. Send desktop notification
notify-send "NSE Quant Terminal" "🛑 Trading Terminal and all background scanners have been STOPPED." --icon=process-stop 2>/dev/null || true
