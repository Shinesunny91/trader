"""Backfill the macro/context symbols into the candle cache.

Every equity context symbol in `data/candles.sqlite3` stops dead on
2026-07-07 — the day `scan_service` started calling
`fetch_index_vix_context(for_commodities=...)`, a keyword the function never
accepted.  The TypeError was swallowed by a bare `except Exception`, so the
scanner has been running with `market_context = None` ever since: no NIFTY
regime, no India VIX, no sector indices, no global cues, no USDINR, no VWAP
breadth.  Nothing logged it, but the cache recorded the outage precisely.

This refills the gap (yfinance serves 60 days of 5m bars) so context-aware
studies and the live scanner have data again.  Crude is added to the equity
panel: India imports ~85% of its oil, so crude in INR terms is a first-order
macro driver for Indian equities that the old panel simply omitted.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.candle_cache import CandleCache  # noqa: E402
from nse_intraday_ai.context_series import (  # noqa: E402
    COMMODITY_CONTEXT_SYMBOLS,
    EQUITY_CONTEXT_SYMBOLS,
)
from nse_intraday_ai.indicators import normalize_ohlcv  # noqa: E402

DB = ROOT / "data" / "candles.sqlite3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="60d", help="yfinance period (5m caps at 60d)")
    parser.add_argument("--interval", default="5m")
    args = parser.parse_args()

    import yfinance as yf

    symbols = sorted(set(EQUITY_CONTEXT_SYMBOLS) | set(COMMODITY_CONTEXT_SYMBOLS))
    cache = CandleCache(DB)
    print(f"backfilling {len(symbols)} context symbols, period={args.period} "
          f"interval={args.interval}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            symbols, period=args.period, interval=args.interval,
            group_by="ticker", auto_adjust=False, progress=False, threads=True,
        )

    for symbol in symbols:
        try:
            frame = normalize_ohlcv(raw[symbol])
        except Exception as exc:
            print(f"  {symbol:12s} FAILED ({exc})")
            continue
        if frame.empty:
            print(f"  {symbol:12s} empty")
            continue
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame = frame.tz_convert("Asia/Kolkata")
        saved = cache.save(symbol, args.interval, frame)
        print(f"  {symbol:12s} {saved:6d} bars  "
              f"{frame.index[0]:%Y-%m-%d} .. {frame.index[-1]:%Y-%m-%d %H:%M}")

    print("\ncoverage after backfill:")
    import sqlite3

    con = sqlite3.connect(DB)
    try:
        for symbol in symbols:
            row = con.execute(
                "SELECT COUNT(*), MIN(ts), MAX(ts) FROM candles WHERE symbol=? AND interval=?",
                (symbol, args.interval),
            ).fetchone()
            print(f"  {symbol:12s} {row[0]:6d}  {str(row[1])[:10]} .. {str(row[2])[:16]}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
