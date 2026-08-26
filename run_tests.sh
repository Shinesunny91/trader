#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo ".venv is missing. Start the app once after internet is available, then run tests again."
  exit 1
fi

source .venv/bin/activate

missing_deps=0
python - <<'PY' || missing_deps=1
import importlib

for module in ("pytest", "streamlit", "pandas", "numpy", "yfinance", "requests", "plotly"):
    importlib.import_module(module)
PY

if [ "$missing_deps" -ne 0 ]; then
  echo "Test dependencies are missing from .venv."
  echo "When internet works, run:"
  echo "  cd /home/hp/Documents/hobby/trading-workspace"
  echo "  source .venv/bin/activate"
  echo "  python -m pip install -e '.[dev]'"
  exit 1
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
exec pytest "$@"
