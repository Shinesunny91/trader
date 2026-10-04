"""Script-level smoke tests: every CLI parses, and the 08:45 message renders.

A SyntaxError in scripts/gap_reversal.py once slipped past the unit tests
(they only import src/); these tests exercise the scripts themselves.
"""
from __future__ import annotations

import importlib.util
import py_compile
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
# daily_monitor writes a report when run; everything else must be side-effect free on --help.
HELP_SAFE = [p for p in SCRIPTS if p.name != "daily_monitor.py"]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("path", HELP_SAFE, ids=lambda p: p.name)
def test_script_help(path):
    out = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT, capture_output=True,
                         text=True, timeout=120, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin",
                                                      "HOME": str(Path.home())})
    assert out.returncode == 0, out.stderr[-2000:]


def _load_gap_script():
    spec = importlib.util.spec_from_file_location("gap_script", ROOT / "scripts" / "gap_reversal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_picks_message_renders_limit_orders(monkeypatch, capsys):
    gs = _load_gap_script()
    G = gs.G
    picks = [G.Pick(rank=i, symbol=f"S{i}.NS", side="SHORT", gap_prev_pct=3.0, prev_close=100.0,
                    atr=2.0, stop_distance=1.5, quantity=1000, position_value=1e5, turnover_cr=50.0,
                    reserve=i > 8, limit_price=99.25) for i in range(1, 11)]
    info = {"ranked_by": "ranker", "guard_t": 1.2, "drift_alarm": False}
    monkeypatch.setattr(gs, "wait_for_clock", lambda *a, **k: None)
    monkeypatch.setattr(gs, "_publish", lambda s, fetch: (date(2026, 10, 5), date(2026, 10, 1), picks, info))
    gs.cmd_picks(SimpleNamespace(date="2026-10-05", push=False, no_fetch=True))
    out = capsys.readouterr().out
    assert "LIMIT ₹99.25" in out
    assert "CANCEL any unfilled order" in out
    assert "use the next: S9, S10" in out
    assert "Ranked by the trained model" in out
