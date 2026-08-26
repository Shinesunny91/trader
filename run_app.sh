#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

APP_URL="http://127.0.0.1:8501"
APP_PATTERN="[p]ython -m streamlit run src/nse_intraday_ai/app.py"

if pgrep -f "$APP_PATTERN" >/dev/null 2>&1; then
  echo "NSE & Commodity Signal Lab is already running at $APP_URL"
  xdg-open "$APP_URL" >/dev/null 2>&1 || true
  exit 0
fi

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q '127\.0\.0\.1:8501'; then
  echo "Port 8501 is already in use. Opening $APP_URL"
  xdg-open "$APP_URL" >/dev/null 2>&1 || true
  exit 0
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

missing_deps=0
python - <<'PY' || missing_deps=1
import importlib

for module in ("streamlit", "pandas", "numpy", "yfinance", "requests", "plotly"):
    importlib.import_module(module)
PY

if [ "$missing_deps" -ne 0 ]; then
  echo "Required Python packages are missing from .venv."
  echo "Internet/DNS is not available right now, so automatic install from PyPI may fail."
  echo "When internet works, run:"
  echo "  cd /home/hp/Documents/hobby/trading-workspace"
  echo "  source .venv/bin/activate"
  echo "  python -m pip install -e '.[dev]'"
  exit 1
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "Starting NSE & Commodity Signal Lab at $APP_URL"
xdg-open "$APP_URL" >/dev/null 2>&1 || true
exec python -m streamlit run src/nse_intraday_ai/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
