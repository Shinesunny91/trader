"""Streamlit Cloud and Local Entrypoint for Quant Terminal 2.0.

Streamlit Community Cloud runs this file directly (not via __main__).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nse_intraday_ai.app import main

# Streamlit runs top-level, not via __main__
main()
