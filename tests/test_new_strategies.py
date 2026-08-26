import numpy as np
import pandas as pd

from nse_intraday_ai.indicators import add_indicators
from nse_intraday_ai.models import Side
from nse_intraday_ai.strategies import (
    LiquiditySweepReversalStrategy,
    VolumeProfilePocStrategy,
    OpeningDriveMomentumStrategy,
)


def _make_candles(n: int = 50, trend: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2026-08-25 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    base = 500.0
    opens, highs, lows, closes, vols = [], [], [], [], []
    for i in range(n):
        o = base + i * trend + np.sin(i / 5.0) * 2.0
        c = o + 0.5 + (0.5 if trend > 0 else (-0.5 if trend < 0 else 0.0))
        h = max(o, c) + 1.0
        l = min(o, c) - 1.0
        v = 20_000 + i * 100
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        vols.append(v)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=idx)
    return add_indicators(df)


def test_liquidity_sweep_reversal_structure():
    strat = LiquiditySweepReversalStrategy()
    df = _make_candles(40)
    sig = strat.evaluate(df)
    assert sig.strategy == "liquidity_sweep_reversal"
    assert sig.side in (Side.LONG, Side.SHORT, Side.WAIT)


def test_liquidity_sweep_reversal_bullish_trigger():
    strat = LiquiditySweepReversalStrategy()
    df = _make_candles(40)
    prior_low = df["low"].iloc[-25:-1].min()
    df.loc[df.index[-1], "open"] = prior_low + 0.5
    df.loc[df.index[-1], "low"] = prior_low - 3.0
    df.loc[df.index[-1], "high"] = prior_low + 2.0
    df.loc[df.index[-1], "close"] = prior_low + 1.8
    df.loc[df.index[-1], "clv"] = 0.8
    df.loc[df.index[-1], "volume_z"] = 2.0

    sig = strat.evaluate(df)
    assert sig.side == Side.LONG
    assert sig.confidence >= 65.0
    assert sig.entry is not None
    assert sig.stop_loss < sig.entry
    assert sig.target > sig.entry


def test_volume_profile_poc_strategy():
    strat = VolumeProfilePocStrategy()
    df = _make_candles(40)
    sig = strat.evaluate(df)
    assert sig.strategy == "volume_profile_poc"
    assert sig.side in (Side.LONG, Side.SHORT, Side.WAIT)


def test_opening_drive_momentum_strategy():
    strat = OpeningDriveMomentumStrategy()
    idx = pd.date_range("2026-08-25 09:15", periods=16, freq="5min", tz="Asia/Kolkata")
    df = _make_candles(16)
    df.index = idx
    df.loc[df.index[-1], "open"] = 510.0
    df.loc[df.index[-1], "low"] = 509.8
    df.loc[df.index[-1], "high"] = 518.0
    df.loc[df.index[-1], "close"] = 517.5
    df.loc[df.index[-1], "atr_14"] = 2.5
    df.loc[df.index[-1], "volume_z"] = 2.5
    df.loc[df.index[-1], "body_ratio"] = 0.8
    df.loc[df.index[-1], "streak"] = 3.0
    df.loc[df.index[-1], "vwap"] = 512.0
    df.loc[df.index[-1], "ema_9"] = 511.0

    sig = strat.evaluate(df)
    assert sig.side == Side.LONG
    assert sig.confidence >= 70.0
