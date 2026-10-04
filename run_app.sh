#!/usr/bin/env bash
# Foreground launcher for the NSE Gap-Reversal Book app.
# Same command as deploy/systemd/nse-signal-lab.service (ExecStart).
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m streamlit run src/nse_intraday_ai/app.py --server.port 8501 "$@"
