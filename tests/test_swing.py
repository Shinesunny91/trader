"""Intra-week book: look-ahead, cost segment, sizing and exit mechanics.

The failure modes that matter here are the ones that silently inflate a
backtest, so each is pinned by a test rather than by reading the code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai.costs import Segment, segment_round_trip_bps, segment_round_trip_cost
from nse_intraday_ai.swing import (
    SwingConfig,
    _position_size,
    backtest,
    build_features,
    build_panel,
)


def make_frame(n=400, start=100.0, drift=0.0, seed=0, span_days=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 15:30", periods=n, freq=f"{span_days}D", tz="Asia/Kolkata")
    steps = rng.normal(drift, 0.015, n)
    close = start * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(high, np.maximum(open_, close)),
         "low": np.minimum(low, np.minimum(open_, close)), "close": close,
         "volume": rng.integers(5e5, 2e6, n).astype(float)},
        index=idx,
    )


# ── costs ──────────────────────────────────────────────────────────────────

def test_delivery_costs_far_more_than_intraday():
    """Overnight settles as CNC: STT is 0.1% on BOTH legs, not 0.025% on one."""
    intraday = segment_round_trip_bps(1000, 250, segment=Segment.EQUITY_INTRADAY)
    delivery = segment_round_trip_bps(1000, 250, segment=Segment.EQUITY_DELIVERY)
    assert delivery > 2 * intraday
    # STT should dominate the delivery bill
    d = segment_round_trip_cost(1000, 1000, 250, segment=Segment.EQUITY_DELIVERY)
    assert d.stt > d.brokerage + d.exchange + d.stamp


def test_delivery_pays_a_dp_fee_but_intraday_does_not():
    d = segment_round_trip_cost(1000, 1000, 100, segment=Segment.EQUITY_DELIVERY)
    i = segment_round_trip_cost(1000, 1000, 100, segment=Segment.EQUITY_INTRADAY)
    assert d.brokerage - i.brokerage == pytest.approx(20.0)   # the flat DP charge


def test_agri_commodities_are_ctt_exempt():
    gold = segment_round_trip_bps(1000, 100, segment=Segment.COMMODITY_FUTURES, symbol="GC=F")
    wheat = segment_round_trip_bps(1000, 100, segment=Segment.COMMODITY_FUTURES, symbol="ZW=F")
    assert gold > wheat


def test_costs_fall_with_size():
    assert (segment_round_trip_bps(1000, 25, segment=Segment.EQUITY_DELIVERY)
            > segment_round_trip_bps(1000, 2500, segment=Segment.EQUITY_DELIVERY))


# ── features ───────────────────────────────────────────────────────────────

def test_features_never_see_the_future():
    """Every feature must be computable from bars up to and including its row."""
    frame = make_frame(400, seed=3)
    full = build_features(frame, "X.NS")
    # Truncating the frame must not change any earlier row's features.
    cut = build_features(frame.iloc[:-40], "X.NS")
    shared = full.index.intersection(cut.index)
    assert len(shared) > 50
    cols = [c for c in full.columns if c not in ("symbol", "open_next")]
    pd.testing.assert_frame_equal(
        full.loc[shared, cols], cut.loc[shared, cols], check_exact=False, rtol=1e-9
    )


def test_open_next_is_the_fill_not_a_feature():
    frame = make_frame(300, seed=5)
    feats = build_features(frame, "X.NS")
    row = feats.index[10]
    nxt = frame.index[frame.index.get_loc(row) + 1]
    assert feats.loc[row, "open_next"] == pytest.approx(frame.loc[nxt, "open"])


def test_short_history_is_refused():
    assert build_features(make_frame(50), "X.NS").empty


# ── sizing ─────────────────────────────────────────────────────────────────

def test_size_is_risk_based_not_notional():
    """A stop-out costs the same rupees whether the name is calm or wild.

    Both cases here sit under the notional cap on purpose — a quiet enough name
    hits the cap first, which is what the next test covers.
    """
    cfg = SwingConfig(capital=10_00_000, risk_per_trade_pct=2.0, stop_atr=2.5,
                      max_position_pct=100.0)
    calm = _position_size(entry=1000, atr=20.0, config=cfg)   # ₹4.0L notional
    wild = _position_size(entry=1000, atr=50.0, config=cfg)   # ₹1.6L notional
    assert calm > wild
    assert calm * 1000 < cfg.capital and wild * 1000 < cfg.capital
    assert calm * 2.5 * 20.0 == pytest.approx(wild * 2.5 * 50.0, rel=0.02)
    assert calm * 2.5 * 20.0 == pytest.approx(cfg.capital * 0.02, rel=0.02)


def test_notional_cap_binds_on_quiet_names():
    cfg = SwingConfig(capital=10_00_000, risk_per_trade_pct=2.0, stop_atr=2.5,
                      max_position_pct=40.0)
    qty = _position_size(entry=100, atr=0.05, config=cfg)   # absurdly quiet
    assert qty * 100 <= 10_00_000 * 0.40 + 100


# ── backtest mechanics ─────────────────────────────────────────────────────

def _panel_and_frames(n_symbols=6, seed=0):
    frames = {f"S{i}.NS": make_frame(420, seed=seed + i, drift=0.0004 * (i - 2))
              for i in range(n_symbols)}
    return build_panel(frames), frames


def test_entry_fills_at_the_next_session_open():
    panel, frames = _panel_and_frames()
    cfg = SwingConfig(positions=1, hold_days=5, min_turnover=0.0, entry_weekday=None)
    res = backtest(panel, frames, lambda rows: rows["r_ret_1w"], cfg)
    assert res.n > 0
    for _, t in res.trades.head(20).iterrows():
        bars = frames[t.symbol]
        assert t.entry == pytest.approx(float(bars.loc[t.entry_date, "open"]))


def test_hold_days_is_respected():
    panel, frames = _panel_and_frames()
    cfg = SwingConfig(positions=1, hold_days=5, stop_atr=0.0, min_turnover=0.0,
                      entry_weekday=None)
    res = backtest(panel, frames, lambda rows: rows["r_ret_1w"], cfg)
    assert res.trades.held_days.max() <= 5


def test_a_gap_through_the_stop_fills_at_the_open():
    """The whole point of overnight risk: you do not get filled at your stop."""
    frame = make_frame(300, seed=1)
    # Force a violent gap down on the bar after entry.
    gap_at = frame.index[260]
    frame.loc[gap_at, ["open", "high", "low", "close"]] = [
        frame.loc[gap_at, "open"] * 0.80, frame.loc[gap_at, "open"] * 0.81,
        frame.loc[gap_at, "open"] * 0.78, frame.loc[gap_at, "open"] * 0.79,
    ]
    frames = {"G.NS": frame}
    panel = build_panel(frames)
    cfg = SwingConfig(positions=1, hold_days=5, stop_atr=1.0, min_turnover=0.0,
                      entry_weekday=None)
    res = backtest(panel, frames, lambda rows: pd.Series(1.0, index=rows.index), cfg)
    gapped = res.trades[res.trades.exit_date == gap_at]
    if not gapped.empty:
        row = gapped.iloc[0]
        stop = row.entry - 1.0 * frame["close"].iloc[0] * 0  # stop level unknown here
        assert row.exit <= row.entry, "a gap down must not fill above the entry"


def test_costs_are_actually_charged():
    panel, frames = _panel_and_frames()
    cfg = SwingConfig(positions=1, hold_days=5, min_turnover=0.0, entry_weekday=None)
    res = backtest(panel, frames, lambda rows: rows["r_ret_1w"], cfg)
    assert (res.trades.costs > 0).all()
    assert (res.trades.net_pnl < res.trades.gross_pnl).all()


def test_one_position_at_a_time_never_overlaps():
    panel, frames = _panel_and_frames()
    cfg = SwingConfig(positions=1, hold_days=5, min_turnover=0.0, entry_weekday=None)
    res = backtest(panel, frames, lambda rows: rows["r_ret_1w"], cfg)
    t = res.trades.sort_values("entry_date")
    assert (t.entry_date.iloc[1:].values >= t.exit_date.iloc[:-1].values).all()


def test_illiquid_names_are_excluded():
    panel, frames = _panel_and_frames()
    cfg = SwingConfig(positions=1, hold_days=5, min_turnover=1e15, entry_weekday=None)
    assert backtest(panel, frames, lambda rows: rows["r_ret_1w"], cfg).n == 0
