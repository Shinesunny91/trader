"""Smoke test: the Streamlit app renders every page without an exception."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "nse_intraday_ai" / "app.py"
UI = ROOT / "src" / "nse_intraday_ai" / "gap_reversal_ui.py"
FORBIDDEN = ("strategies scanner scan_service scan_config learning signal_model meta_model backtest "
             "simulator news_feed entry_quality event_risk risk models signals_log correlation horizons "
             "recommend_ui market_context macro_context context_series data indicators swing "
             "portfolio_sim execution_plan ml liquidity").split()

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not [e for e in at.error if "could not be rendered" in e.value], [e.value for e in at.error]


@pytest.fixture
def password(tmp_path, monkeypatch) -> str:
    from nse_intraday_ai import auth
    path = tmp_path / "auth.json"
    monkeypatch.setenv("NSE_AUTH_FILE", str(path))
    auth.set_password("test-pass-123", path)
    return "test-pass-123"


def _login(pw: str) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60).run(timeout=60)
    at.text_input(key="pw").input(pw)
    return at.button[0].click().run(timeout=60)


def test_pages_are_hidden_until_login(password) -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run(timeout=60)
    assert not at.sidebar.radio, "pages visible without logging in"
    at = _login("wrong-password")
    assert not at.sidebar.radio and any("Wrong password" in e.value for e in at.error)


def test_every_page_renders(password) -> None:
    at = _login(password)
    _no_exceptions(at)
    radio = at.sidebar.radio(key="page")
    for page in radio.options:
        at = radio.set_value(page).run(timeout=60)
        _no_exceptions(at)
        assert at.title[0].value in page
        radio = at.sidebar.radio(key="page")


def test_no_forbidden_imports() -> None:
    for path in (APP, UI):
        src = path.read_text()
        mods = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, re.M)
        mods += re.findall(r"^\s*from\s+nse_intraday_ai\s+import\s+([\w, ]+)", src, re.M)
        names = {part.strip() for m in mods for part in re.split(r"[.,]", m)}
        assert not names & set(FORBIDDEN), (path.name, names & set(FORBIDDEN))
        assert "scripts" not in names
