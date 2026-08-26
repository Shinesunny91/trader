"""The live build must be able to score the newest *closed* bar.

`build_dataset.build_symbol` is shared between training and live scoring, and
its loop was written for labelling: it needs one bar after the signal (the
fill) and a window after that (the barriers), so it stops two bars short of the
data.  That is right offline and wrong live — it made the freshest signal a
scan could produce 10-15 minutes old, and `sim_today.write_tickets` discards
anything older than 10 minutes, so the screen showed "no signal in the last 10
minutes" on every cycle regardless of what the market did.

These tests pin the two halves of the fix: a live build reaches the last closed
bar, and it never scores a forming one.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_dataset as bd  # noqa: E402
from nse_intraday_ai.models import Side  # noqa: E402

IST = "Asia/Kolkata"
BARRIER = "net_bps_1.5_3.0_12"


class _Vote:
    strategy, side, is_trade = "stub", Side.LONG, True


class _Plan:
    is_actionable, side, confidence, reward_risk = True, Side.LONG, 80.0, 2.0
    strategy_votes = [_Vote()]

    def __init__(self, entry):
        self.entry = entry


class _AlwaysFires:
    """Every bar is a signal, so the test measures the loop bounds and nothing else."""

    def __init__(self, *_, **__):
        pass

    def analyze_precomputed(self, symbol, history, *_, **__):
        return _Plan(float(history["close"].iloc[-1]))


def _frame(sessions=3, bars=75, last_bars=48, seed=5):
    """Two full sessions plus a partial one — the shape a live scan sees.

    The last session stops at 13:10 deliberately: `build_symbol` only scores
    bars between 09:15 and 14:50, so a fixture that runs to 15:25 would test
    the session filter rather than the live horizon.
    """
    rng = np.random.default_rng(seed)
    index, day = [], pd.Timestamp("2026-08-19 09:15", tz=IST)
    for session in range(sessions):
        count = last_bars if session == sessions - 1 else bars
        index += list(pd.date_range(day, periods=count, freq="5min"))
        day += pd.Timedelta(days=1)
    index = pd.DatetimeIndex(index)
    close = 1000 * np.exp(np.cumsum(rng.normal(0.0004, 0.003, len(index))))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": rng.lognormal(11, 0.5, len(index)),
        },
        index=index,
    )


@pytest.fixture
def wired(monkeypatch):
    frame = _frame()
    monkeypatch.setattr(bd, "load_frame", lambda symbol, since: frame)
    monkeypatch.setattr(bd, "VotingSignalEngine", _AlwaysFires)
    return frame


def test_training_build_stops_two_bars_short(wired):
    rows = pd.DataFrame(bd.build_symbol(("X.NS", "2026-08-01")))
    assert not rows.empty
    # Every training row is labelled: that is the contract the model trains on.
    assert rows[BARRIER].notna().all()
    assert rows["ts"].max() <= wired.index[-3]


def test_live_build_reaches_the_last_closed_bar(wired):
    # A scan three minutes into the bar that opened at index[-1].
    now = wired.index[-1] + pd.Timedelta(minutes=3)
    rows = pd.DataFrame(bd.build_symbol(("X.NS", "2026-08-01", now)))
    assert not rows.empty
    # index[-1] is still forming and must never be scored; index[-2] is the
    # newest closed bar and must be.
    assert rows["ts"].max() == wired.index[-2]
    assert (rows["ts"] < wired.index[-1]).all()


def test_live_tail_rows_carry_features_but_no_labels(wired):
    now = wired.index[-1] + pd.Timedelta(minutes=3)
    rows = pd.DataFrame(bd.build_symbol(("X.NS", "2026-08-01", now)))
    tail = rows[rows["ts"] >= wired.index[-3]]
    assert len(tail) == 2                      # index[-3] and index[-2]
    assert tail[BARRIER].isna().all()          # nothing may train on these
    assert tail["vol_z"].notna().all()         # but they are fully featured
    assert tail["atr"].gt(0).all()
    # The newest row has no fill bar yet, so the signal bar's own close stands
    # in — ExecutionPlan quotes stop and target relative to the real fill.
    newest = tail[tail["ts"] == wired.index[-2]].iloc[0]
    assert newest["fill"] == pytest.approx(float(wired["close"].loc[wired.index[-2]]))


def test_live_and_training_agree_where_they_overlap(wired):
    now = wired.index[-1] + pd.Timedelta(minutes=3)
    train = pd.DataFrame(bd.build_symbol(("X.NS", "2026-08-01"))).set_index("ts")
    live = pd.DataFrame(bd.build_symbol(("X.NS", "2026-08-01", now))).set_index("ts")
    shared = train.index.intersection(live.index)
    assert len(shared) > 50
    for column in ("vol_z", "run6", "ext_vwap", "clv", "atr", "fill"):
        pd.testing.assert_series_equal(train.loc[shared, column], live.loc[shared, column])
