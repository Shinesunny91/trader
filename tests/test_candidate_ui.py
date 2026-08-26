"""The candidate panel must never read as a green light."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai import candidate_ui as cu
from nse_intraday_ai.execution_plan import GO_LIVE_GATE


def _log(rows):
    return pd.DataFrame(rows)


def test_the_seeded_forward_record_does_not_clear_the_gate():
    d = _log([
        {"day": "2026-08-18", "traded": 1, "net_bps": 283.59, "rupees": 8508},
        {"day": "2026-08-19", "traded": 1, "net_bps": -50.69, "rupees": -1521},
        {"day": "2026-08-20", "traded": 0, "net_bps": None, "rupees": 0},
        {"day": "2026-08-21", "traded": 1, "net_bps": -231.33, "rupees": -6940},
        {"day": "2026-08-24", "traded": 1, "net_bps": -95.63, "rupees": -2869},
        {"day": "2026-08-25", "traded": 0, "net_bps": None, "rupees": 0},
    ])
    ok, checks, v = cu._gate_from_log(d)
    assert not ok
    assert v.size == 4
    assert not checks["sessions >= 60"]


def test_no_trades_yet_is_handled_without_crashing():
    """A log of no-trade sessions is a legitimate state, not an error."""
    d = _log([{"day": "2026-08-20", "traded": 0, "net_bps": None, "rupees": 0}])
    ok, checks, v = cu._gate_from_log(d)
    assert not ok and v.size == 0 and checks == {}


def test_even_a_flawless_short_record_is_held_back_by_sample_size():
    rows = [{"day": f"2026-09-{i:02d}", "traded": 1, "net_bps": 30.0, "rupees": 900}
            for i in range(1, 21)]
    ok, checks, _ = cu._gate_from_log(_log(rows))
    assert not ok
    assert not checks[f"sessions >= {GO_LIVE_GATE['min_sessions']}"]
    assert checks[f"net edge >= {GO_LIVE_GATE['min_net_bps']} bps"]


def test_the_rule_text_states_the_threshold_is_negative():
    """The screen must not let "top 1%" read as "predicted profit"."""
    assert "negative" in cu.RULE
