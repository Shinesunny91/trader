"""Tests for advanced quantitative features: Corwin-Schultz spread, VWAP dispersion bands,
CVD flow, new institutional strategies, and correlation-diversified ranking."""
import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai.indicators import (
    add_indicators,
    corwin_schultz_spread,
    cvd_flow,
    vwap_dispersion_bands,
)
from nse_intraday_ai.models import Side
from nse_intraday_ai.strategies import (
    MultiTimeframeTrendAlignmentStrategy,
    OpeningRangeBreakoutExpansionStrategy,
    VwapBandMeanReversionStrategy,
)


def _make_candles(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2026-08-25 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    base = 1000.0 + np.cumsum(np.random.default_rng(42).normal(0.2, 1.5, n))
    df = pd.DataFrame(
        {
            "open": base - 0.5,
            "high": base + 2.0,
            "low": base - 2.0,
            "close": base + 0.5,
            "volume": np.random.default_rng(42).uniform(10_000, 50_000, n),
        },
        index=idx,
    )
    return add_indicators(df)


def test_corwin_schultz_spread():
    df = _make_candles(30)
    spread = corwin_schultz_spread(df)
    assert len(spread) == len(df)
    assert (spread >= 0.0).all()
    assert (spread <= 0.05).all()
    assert "effective_spread_bps" in df.columns


def test_vwap_dispersion_bands():
    df = _make_candles(30)
    u1, l1, u2, l2 = vwap_dispersion_bands(df)
    assert (u2 >= u1).all()
    assert (u1 >= df["vwap"]).all()
    assert (df["vwap"] >= l1).all()
    assert (l1 >= l2).all()
    assert "vwap_u2" in df.columns
    assert "vwap_l2" in df.columns


def test_cvd_flow():
    df = _make_candles(30)
    cvd = cvd_flow(df)
    assert len(cvd) == len(df)
    assert "cvd" in df.columns


def test_vwap_band_mean_reversion_strategy():
    strat = VwapBandMeanReversionStrategy()
    df = _make_candles(30)
    # Simulate extreme oversold stretch below Lower Band 2 with bullish wick rejection
    df.loc[df.index[-1], "close"] = df["vwap_l2"].iloc[-1] - 5.0
    df.loc[df.index[-1], "open"] = df["vwap_l2"].iloc[-1] - 8.0
    df.loc[df.index[-1], "low"] = df["vwap_l2"].iloc[-1] - 12.0
    df.loc[df.index[-1], "high"] = df["vwap_l2"].iloc[-1] - 4.5
    df.loc[df.index[-1], "lower_wick_ratio"] = 0.50
    df.loc[df.index[-1], "clv"] = 0.60
    df.loc[df.index[-1], "atr_14"] = 3.0

    sig = strat.evaluate(df)
    assert sig.side == Side.LONG
    assert sig.confidence >= 70.0
    assert sig.target == df["vwap"].iloc[-1]


def test_opening_range_expansion_strategy():
    strat = OpeningRangeBreakoutExpansionStrategy()
    df = _make_candles(20)
    or_high = float(df.iloc[:3]["high"].max())
    # Breakout above opening range on volume
    df.loc[df.index[-1], "close"] = or_high + 5.0
    df.loc[df.index[-1], "open"] = or_high + 1.0
    df.loc[df.index[-1], "high"] = or_high + 6.0
    df.loc[df.index[-1], "low"] = or_high + 0.5
    df.loc[df.index[-1], "volume_z"] = 1.8
    df.loc[df.index[-1], "clv"] = 0.40
    df.loc[df.index[-1], "atr_14"] = 2.0

    sig = strat.evaluate(df)
    assert sig.side == Side.LONG
    assert sig.confidence >= 70.0


def test_mtf_trend_alignment_strategy():
    strat = MultiTimeframeTrendAlignmentStrategy()
    df = _make_candles(35)
    p = 1100.0
    df.loc[df.index[-1], "close"] = p
    df.loc[df.index[-1], "ema_9"] = p - 2.0
    df.loc[df.index[-1], "ema_21"] = p - 5.0
    df.loc[df.index[-1], "ema_50"] = p - 10.0
    df.loc[df.index[-1], "vwap"] = p - 12.0
    df.loc[df.index[-1], "adx_14"] = 28.0
    df.loc[df.index[-1], "plus_di_14"] = 35.0
    df.loc[df.index[-1], "minus_di_14"] = 12.0
    df.loc[df.index[-1], "atr_14"] = 2.5

    sig = strat.evaluate(df)
    assert sig.side == Side.LONG
    assert sig.confidence >= 70.0
