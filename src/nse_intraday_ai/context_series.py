"""Per-timestamp market context for backtesting.

The live scanner fetches one MarketContext per scan cycle.  Backtests replay
history, so they need the context *as it was* at each candle.  This module
precomputes regime / VIX / global-cue series from cached (or fetched) index
candles and serves a MarketContext snapshot for any timestamp, causally
(only data at or before the timestamp is used).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from nse_intraday_ai.indicators import add_indicators
from nse_intraday_ai.market_context import (
    SECTOR_TO_INDEX,
    MarketContext,
    _classify_vix,
    _index_adjs,
    global_risk_score,
)

# Context symbols used to build a series (all fetchable from Yahoo at 5m).
# Crude (CL=F) belongs in the *equity* panel too: India imports ~85% of its
# oil, so the crude tape is a first-order macro driver for Indian equities —
# it was previously only fetched for the commodity universe.
EQUITY_CONTEXT_SYMBOLS = ["^NSEI", "^INDIAVIX", *sorted(set(SECTOR_TO_INDEX.values())),
                          "ES=F", "DX-Y.NYB", "USDINR=X", "^N225", "^HSI", "^GDAXI",
                          "^VIX", "CL=F"]
COMMODITY_CONTEXT_SYMBOLS = ["ES=F", "DX-Y.NYB", "^N225", "^HSI", "^GDAXI",
                             "USDINR=X", "^TNX", "^VIX"]

_CHANGE_BARS = 12          # 12 × 5m ≈ 1 hour momentum window
_STALENESS = pd.Timedelta("45min")
# Levels (VIX) stay meaningful far longer than momentum readings — a fear
# gauge from a few hours ago still describes the day; a stale 1-hour return
# does not.
_LEVEL_STALENESS = pd.Timedelta("12h")


def _series_asof(series: pd.Series | None, ts: pd.Timestamp, staleness: pd.Timedelta = _STALENESS):
    """Latest value at or before ts, or None if absent/stale."""
    if series is None or series.empty:
        return None
    try:
        pos = series.index.searchsorted(ts, side="right") - 1
    except TypeError:
        return None
    if pos < 0:
        return None
    if ts - series.index[pos] > staleness:
        return None
    value = series.iloc[pos]
    return None if pd.isna(value) else value


def _regime_series(frame: pd.DataFrame) -> pd.Series | None:
    if frame is None or frame.empty or len(frame) < 20:
        return None
    df = add_indicators(frame)
    if df.empty or "regime" not in df.columns:
        return None
    return df["regime"]


def _change_series(frame: pd.DataFrame, bars: int = _CHANGE_BARS) -> pd.Series | None:
    if frame is None or frame.empty or len(frame) < 2:
        return None
    return frame["close"].pct_change(min(bars, len(frame) - 1)) * 100


@dataclass
class MarketContextSeries:
    """Precomputed causal context lookup: ``at(ts) -> MarketContext``."""

    index_regime: pd.Series | None = None
    vix: pd.Series | None = None
    sector_regimes: dict[str, pd.Series] = field(default_factory=dict)
    es_change: pd.Series | None = None
    asia_change: pd.Series | None = None
    hsi_change: pd.Series | None = None
    europe_change: pd.Series | None = None
    dxy_change: pd.Series | None = None
    usdinr_change: pd.Series | None = None
    us10y_change: pd.Series | None = None
    us_vix: pd.Series | None = None
    # Fraction of the equity universe above session VWAP, bucketed in time.
    vwap_breadth: pd.Series | None = None

    def at(self, ts: pd.Timestamp) -> MarketContext:
        ctx = MarketContext()
        regime = _series_asof(self.index_regime, ts)
        if regime:
            ctx.index_regime = str(regime)
            ctx.index_long_adj, ctx.index_short_adj = _index_adjs(ctx.index_regime)
        vix = _series_asof(self.vix, ts)
        if vix is not None:
            ctx.vix_value = float(vix)
            ctx.vix_level = _classify_vix(ctx.vix_value)
        for sector, series in self.sector_regimes.items():
            sect_regime = _series_asof(series, ts)
            if sect_regime:
                ctx.sector_index_regimes[sector] = str(sect_regime)
        es_chg = _series_asof(self.es_change, ts)
        europe_chg = _series_asof(self.europe_change, ts)
        # Asia component blends Nikkei and Hang Seng, matching the live fetch.
        asia_parts = [
            float(value)
            for value in (_series_asof(self.asia_change, ts), _series_asof(self.hsi_change, ts))
            if value is not None
        ]
        asia_chg = sum(asia_parts) / len(asia_parts) if asia_parts else None
        ctx.global_risk = global_risk_score(
            float(es_chg) if es_chg is not None else None,
            asia_chg,
            float(europe_chg) if europe_chg is not None else None,
        )
        for attr, series in (
            ("dxy_change_pct", self.dxy_change),
            ("usdinr_change_pct", self.usdinr_change),
            ("us10y_change_pct", self.us10y_change),
            ("hsi_change_pct", self.hsi_change),
        ):
            value = _series_asof(series, ts)
            if value is not None:
                setattr(ctx, attr, float(value))
        us_vix = _series_asof(self.us_vix, ts, _LEVEL_STALENESS)
        if us_vix is not None:
            ctx.us_vix_value = float(us_vix)
        breadth = _series_asof(self.vwap_breadth, ts)
        if breadth is not None:
            ctx.vwap_breadth_pct = float(breadth)
        return ctx


def build_context_series(frames: dict[str, pd.DataFrame]) -> MarketContextSeries:
    """Build a MarketContextSeries from OHLCV frames keyed by Yahoo symbol.

    Frames must have tz-aware DatetimeIndex and lowercase OHLCV columns.
    Missing symbols simply leave that component of the context empty.
    """
    series = MarketContextSeries()
    nsei = frames.get("^NSEI")
    if nsei is not None:
        series.index_regime = _regime_series(nsei)
    vix = frames.get("^INDIAVIX")
    if vix is not None and not vix.empty:
        series.vix = vix["close"]
    for sector, index_symbol in SECTOR_TO_INDEX.items():
        frame = frames.get(index_symbol)
        if frame is not None:
            regime = _regime_series(frame)
            if regime is not None:
                series.sector_regimes[sector] = regime
    if "ES=F" in frames:
        series.es_change = _change_series(frames["ES=F"])
    if "^N225" in frames:
        series.asia_change = _change_series(frames["^N225"])
    if "^HSI" in frames:
        series.hsi_change = _change_series(frames["^HSI"])
    if "^GDAXI" in frames:
        series.europe_change = _change_series(frames["^GDAXI"])
    if "DX-Y.NYB" in frames:
        series.dxy_change = _change_series(frames["DX-Y.NYB"])
    if "USDINR=X" in frames:
        series.usdinr_change = _change_series(frames["USDINR=X"])
    if "^TNX" in frames:
        series.us10y_change = _change_series(frames["^TNX"])
    if "^VIX" in frames and not frames["^VIX"].empty:
        series.us_vix = frames["^VIX"]["close"]
    return series


def build_vwap_breadth_series(
    frames: dict[str, pd.DataFrame], *, bucket: str = "5min", min_symbols: int = 10
) -> pd.Series | None:
    """Fraction of symbols above session VWAP per time bucket, causally.

    `frames` are OHLCV frames (tz-aware index, lowercase columns) — typically
    the same universe frames the backtest replays.
    """
    from nse_intraday_ai.indicators import vwap

    columns = {}
    for symbol, frame in frames.items():
        if frame is None or frame.empty or len(frame) < 5:
            continue
        try:
            above = (frame["close"] > vwap(frame)).astype(float)
        except Exception:
            continue
        columns[symbol] = above.resample(bucket).last()
    if len(columns) < min_symbols:
        return None
    matrix = pd.DataFrame(columns)
    breadth = matrix.mean(axis=1)
    counts = matrix.notna().sum(axis=1)
    return breadth[counts >= min_symbols].dropna()


def load_vwap_breadth_from_cache(
    db_path: Path | str,
    symbols: list[str],
    *,
    interval: str = "1m",
    since: str | None = None,
    bucket: str = "5min",
) -> pd.Series | None:
    """Build the breadth series from cached candles for a symbol universe."""
    import sqlite3

    frames: dict[str, pd.DataFrame] = {}
    con = sqlite3.connect(str(db_path))
    try:
        for symbol in symbols:
            df = pd.read_sql_query(
                "SELECT ts, close, high, low, volume FROM candles"
                " WHERE symbol=? AND interval=? AND ts>=? ORDER BY ts",
                con,
                params=(symbol, interval, since or "1970-01-01"),
            )
            if df.empty:
                continue
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
            frames[symbol] = df.set_index("ts")
    finally:
        con.close()
    return build_vwap_breadth_series(frames)


def fetch_context_series(
    *,
    period: str = "14d",
    interval: str = "5m",
    for_commodities: bool = False,
) -> MarketContextSeries:
    """Download context symbols from Yahoo and build a context series.

    Used by the app's backtest so replayed signals see the same market
    context the live scanner would have seen at each candle.
    """
    import warnings

    import yfinance as yf

    from nse_intraday_ai.indicators import normalize_ohlcv

    symbols = COMMODITY_CONTEXT_SYMBOLS if for_commodities else EQUITY_CONTEXT_SYMBOLS
    frames: dict[str, pd.DataFrame] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                symbols, period=period, interval=interval,
                group_by="ticker", auto_adjust=False, progress=False, threads=True,
            )
        for symbol in symbols:
            try:
                frame = normalize_ohlcv(raw[symbol])
            except Exception:
                continue
            if frame.empty:
                continue
            if frame.index.tz is None:
                frame.index = frame.index.tz_localize("UTC")
            frames[symbol] = frame.tz_convert("Asia/Kolkata")
    except Exception:
        pass
    return build_context_series(frames)


def build_context_series_from_cache(
    db_path: Path | str,
    *,
    interval: str = "5m",
    since: str | None = None,
    symbols: list[str] | None = None,
) -> MarketContextSeries:
    """Build a context series from the local candle cache."""
    import sqlite3

    wanted = symbols or EQUITY_CONTEXT_SYMBOLS
    frames: dict[str, pd.DataFrame] = {}
    con = sqlite3.connect(str(db_path))
    try:
        for symbol in wanted:
            df = pd.read_sql_query(
                "SELECT ts, open, high, low, close, volume FROM candles"
                " WHERE symbol=? AND interval=? AND ts>=? ORDER BY ts",
                con,
                params=(symbol, interval, since or "1970-01-01"),
            )
            if df.empty:
                continue
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
            frames[symbol] = df.set_index("ts")
    finally:
        con.close()
    return build_context_series(frames)
