"""NSE Gap-Reversal Book — Streamlit app.

Run:  .venv/bin/python -m streamlit run src/nse_intraday_ai/app.py --server.port 8501

Read-only view of what the scheduled jobs publish: today's short list, the
forward paper book, and the ML ranker's evidence and live monitoring.
"""
from __future__ import annotations

import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import streamlit as st

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:          # `streamlit run` does not put src/ on the path
    sys.path.insert(0, str(_SRC))

from nse_intraday_ai import gap_reversal_ui as ui  # noqa: E402
from nse_intraday_ai import intraday_ui  # noqa: E402

ROOT = _SRC.parent
APP_NAME = "NSE Trading Desk"
DEVELOPER = "Shine"
PAGES = {"📋 Gap book — Today": ui.page_today, "📡 Intraday monitor": intraday_ui.page_intraday,
         "📈 Performance": ui.page_performance, "🧠 Model": ui.page_model, "ℹ️ About": None}


@st.cache_data(ttl=3600, show_spinner=False)
def build_info() -> tuple[str, str]:
    """(version from pyproject.toml, build date = last git commit, else file mtime)."""
    try:
        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        version = "unknown"
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%cI"],
                             capture_output=True, text=True, timeout=3, check=True).stdout.strip()
        built = datetime.fromisoformat(out).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError, subprocess.SubprocessError):
        built = datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return version, built


def page_about() -> None:
    version, built = build_info()
    c = st.columns(4)
    c[0].metric("Application", APP_NAME)
    c[1].metric("Version", version)
    c[2].metric("Build date", built)
    c[3].metric("Developer", DEVELOPER)
    st.markdown(
        "**Strategy**\n\n"
        "- Every trading day, rank the 300 most liquid NIFTY 500 names by **yesterday's overnight gap** "
        "(or by the promoted ML ranker) and **short the top 8** at the 09:15 open, equal ₹ weight, MIS.\n"
        "- Each short carries a **buy-stop at fill + 0.75 × 14-day ATR** and is **covered at 15:15**.\n"
        "- The edge is the intraday give-back of overnight gains: +24.7 bps/trade net over ten years of daily "
        "bars (t = 9.2); the walk-forward ML ranker added ~20 bps/trade over the rule (2024-10 → 2026-10).\n\n"
        "**Intraday monitor** — a live NIFTY 500 view every 5 minutes (stocks in play, circuits, "
        "announcements, gap-book stop alerts) and a paper *shadow board* of three unvalidated intraday "
        "strategies. About 100 published intraday strategy variants were tested on 5-minute and hourly data; "
        "none beat the ~13 bps round-trip cost out of sample (see docs/research-log.md).")
    st.caption("Research and paper-trading tool — not investment advice. Backtests and paper results do "
               "not guarantee future returns; shortability (ASM/T2T), slippage and leverage change outcomes.")


def login_gate() -> bool:
    """True once this browser session has entered the right password."""
    from nse_intraday_ai import auth

    if st.session_state.get("authed"):
        return True
    st.title(f"🔒 {APP_NAME}")
    if not auth.is_configured():
        st.error("No dashboard password is set. On the server run:  \n"
                 "`PYTHONPATH=src .venv/bin/python -m nse_intraday_ai.auth --set-password`")
        return False
    with st.form("login", clear_on_submit=True):
        pw = st.text_input("Password", type="password", key="pw")
        ok = st.form_submit_button("Log in")
    if ok:
        if auth.verify(pw):
            st.session_state["authed"] = True
            st.session_state["fails"] = 0
            st.rerun()
        fails = st.session_state.get("fails", 0) + 1
        st.session_state["fails"] = fails
        time.sleep(min(2 ** fails, 30))              # make guessing slow
        st.error("Wrong password.")
    return False


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="📉", layout="wide",
                       initial_sidebar_state="expanded")
    if not login_gate():
        return
    version, built = build_info()
    with st.sidebar:
        st.markdown(f"### 📉 {APP_NAME}")
        page = st.radio("Page", list(PAGES), key="page", label_visibility="collapsed")
        st.divider()
        st.caption(f"v{version} · built {built}  \nDeveloper: {DEVELOPER}")
        if st.button("↻ Reload data", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        if st.button("🔒 Log out", width="stretch"):
            st.session_state["authed"] = False
            st.rerun()
    st.title(page.split(" ", 1)[1])
    try:
        (PAGES[page] or page_about)()
    except Exception as exc:                             # noqa: BLE001 — show a message, not a traceback
        st.error(f"This page could not be rendered ({type(exc).__name__}: {exc}). "
                 "The data files may be mid-update — try **Reload data**.")


main()
