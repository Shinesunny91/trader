#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

APP_PORT="${APP_PORT:-8501}"
PHONE_HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
PHONE_URL="http://${PHONE_HOST:-YOUR_COMPUTER_IP}:$APP_PORT"
EMULATOR_URL="http://10.0.2.2:$APP_PORT"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

python - <<'PY'
import importlib
import sys

missing = []
for module in ("streamlit", "pandas", "numpy", "yfinance", "requests", "plotly"):
    try:
        importlib.import_module(module)
    except ImportError:
        missing.append(module)

if missing:
    print("Missing Python packages:", ", ".join(missing))
    print("Install them with: source .venv/bin/activate && python -m pip install -e '.[dev]'")
    sys.exit(1)
PY

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

API_PORT="${API_PORT:-8502}"

echo "Starting NSE Signal Lab for Android."
echo "Use this URL in the Android emulator: $EMULATOR_URL"
echo "Use this URL on a phone on the same Wi-Fi: $PHONE_URL"
echo "Ticket API (the app polls this for notifications): port $API_PORT"
echo

# The JSON API the phone polls for new tickets. Streamlit serves a UI, not
# data, so without this the app can only render — it cannot notify.
python scripts/ticket_api.py --port "$API_PORT" &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM

exec python -m streamlit run src/nse_intraday_ai/app.py \
    --server.address 0.0.0.0 \
    --server.port "$APP_PORT" \
    --server.headless true
