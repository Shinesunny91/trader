import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai.macro_context import (
    build_overnight_context,
    calendar_context,
    is_nse_monthly_expiry,
    is_nse_weekly_expiry,
    minutes_since_close,
    nse_expiry_weekday,
    overnight_change_pct,
    phase_weighted_global_risk,
    region_weight,
    session_phase,
)

IST = "Asia/Kolkata"


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=IST)


# ── sessions ──────────────────────────────────────────────────────────────────

def test_tokyo_is_live_in_the_nse_morning_and_shut_by_afternoon():
    assert region_weight("japan", ts("2026-08-11 10:00")) == 1.0
    assert region_weight("japan", ts("2026-08-11 14:00")) == 0.0


def test_europe_is_shut_at_the_nse_open_and_live_after_1230():
    assert region_weight("europe", ts("2026-08-11 10:00")) == 0.0
    assert region_weight("europe", ts("2026-08-11 14:00")) == 1.0


def test_region_weight_decays_rather_than_cliff_edging():
    """A market that closed 45 minutes ago is half-informative, not worthless."""
    weight = region_weight("japan", ts("2026-08-11 12:15"))   # Tokyo shut 11:30
    assert 0.0 < weight < 1.0
    assert weight == pytest.approx(0.5, abs=0.01)


def test_minutes_since_close_is_zero_while_open():
    assert minutes_since_close("japan", ts("2026-08-11 10:00")) == 0.0
    assert minutes_since_close("japan", ts("2026-08-11 12:00")) == 30.0


def test_session_phase_names_the_nse_day():
    assert session_phase(ts("2026-08-11 09:00")).name == "pre_open"
    assert session_phase(ts("2026-08-11 10:00")).name == "nse_asia"
    assert session_phase(ts("2026-08-11 12:00")).name == "nse_gap"
    assert session_phase(ts("2026-08-11 14:00")).name == "nse_europe"
    assert session_phase(ts("2026-08-11 16:00")).name == "post_close"


def test_phase_weighting_ignores_a_market_that_has_closed():
    """The whole point: a stale Nikkei print must not move the score."""
    morning = phase_weighted_global_risk(
        ts("2026-08-11 10:00"), es_change_pct=0.0, japan_change_pct=1.0
    )
    afternoon = phase_weighted_global_risk(
        ts("2026-08-11 14:00"), es_change_pct=0.0, japan_change_pct=1.0
    )
    assert morning > afternoon
    assert afternoon == pytest.approx(0.0, abs=1e-9)


def test_phase_weighted_risk_stays_bounded():
    score = phase_weighted_global_risk(
        ts("2026-08-11 14:00"), es_change_pct=99.0, europe_change_pct=99.0
    )
    assert -1.0 <= score <= 1.0


def test_no_inputs_is_neutral():
    assert phase_weighted_global_risk(ts("2026-08-11 10:00"), es_change_pct=None) == 0.0


# ── calendar ──────────────────────────────────────────────────────────────────

def test_expiry_weekday_follows_the_2025_rule_change():
    assert nse_expiry_weekday(date(2025, 1, 1)) == 3     # Thursday
    assert nse_expiry_weekday(date(2026, 1, 1)) == 1     # Tuesday


def test_weekly_and_monthly_expiry_detection():
    assert is_nse_weekly_expiry(ts("2026-08-11 14:00"))       # a Tuesday
    assert not is_nse_weekly_expiry(ts("2026-08-13 14:00"))   # a Thursday
    # 2026-08-25 is the last Tuesday of August
    assert is_nse_monthly_expiry(ts("2026-08-25 14:00"))
    assert not is_nse_monthly_expiry(ts("2026-08-11 14:00"))


def test_calendar_features_are_all_numeric_and_bounded():
    features = calendar_context(ts("2026-08-25 14:00")).as_features()
    assert features["cal_monthly_expiry"] == 1.0
    assert all(0.0 <= v <= 1.0 for v in features.values())


# ── overnight ─────────────────────────────────────────────────────────────────

def _frame(values, start, freq="5min"):
    index = pd.date_range(start, periods=len(values), freq=freq, tz=IST)
    return pd.DataFrame(
        {"open": values, "high": values, "low": values, "close": values, "volume": 1},
        index=index,
    )


def test_overnight_change_measures_from_yesterdays_close():
    frame = _frame([100.0] * 12 + [102.0] * 12, "2026-08-10 15:00")
    change = overnight_change_pct(frame, ts("2026-08-11 09:20"))
    assert change == pytest.approx(2.0, abs=0.01)


def test_overnight_change_is_none_without_prior_data():
    frame = _frame([100.0] * 5, "2026-08-11 09:15")
    assert overnight_change_pct(frame, ts("2026-08-11 09:35")) is None


def test_gap_and_gap_fill_detection():
    index = _frame([100.0] * 6, "2026-08-10 15:00")
    today = _frame([102.0, 103.0, 101.0, 99.5], "2026-08-11 09:15")
    combined = pd.concat([index, today])
    context = build_overnight_context(index_frame=combined, ts=ts("2026-08-11 09:35"))
    assert context.prev_close == 100.0
    assert context.today_open == 102.0
    assert context.gap_pct == pytest.approx(2.0)
    assert context.gap_filled is True     # low reached 99.5, below prev close


def test_missing_feeds_stay_none_not_zero():
    """A missing feed must be distinguishable from a flat market."""
    context = build_overnight_context(index_frame=None, ts=ts("2026-08-11 09:35"))
    assert context.gap_pct is None
    assert context.as_features()["on_gap_pct"] == 0.0     # encoded, but flagged
    assert context.as_features()["on_gap_filled"] == -1.0
