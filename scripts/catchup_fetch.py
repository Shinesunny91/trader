"""Catch the candle cache up to now — the gap-filler the workspace lacked.

`backfill_context.py` refills the macro symbols and the scanner daemon fetches
~1 day per cycle, so a workspace that has been switched off for a week has no
way to repair its own history: the daemon's `period="1d"` request cannot reach
back past yesterday, and every study downstream silently replays a stale cache.

This fetches a *period* of intraday bars for the whole tradable universe plus
the context symbols and upserts them into `data/candles.sqlite3`, in chunks,
with retries — the one operation you want before running any study on a cache
that has been sitting idle.

Usage:
    python scripts/catchup_fetch.py                     # nifty500 + context, 10d/5m
    python scripts/catchup_fetch.py --period 5d --chunk 40
    python scripts/catchup_fetch.py --only context
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.candle_cache import CandleCache, drop_synthetic_bars  # noqa: E402
from nse_intraday_ai.context_series import (  # noqa: E402
    COMMODITY_CONTEXT_SYMBOLS,
    EQUITY_CONTEXT_SYMBOLS,
)
from nse_intraday_ai.indicators import normalize_ohlcv  # noqa: E402

DB = ROOT / "data" / "candles.sqlite3"
UNIVERSE_CSV = ROOT / "data" / "nifty500_symbols.csv"


def universe() -> list[str]:
    frame = pd.read_csv(UNIVERSE_CSV)
    col = "Symbol" if "Symbol" in frame.columns else frame.columns[0]
    return sorted({f"{s.strip()}.NS" for s in frame[col].astype(str) if s.strip()})


def context() -> list[str]:
    return sorted(set(EQUITY_CONTEXT_SYMBOLS) | set(COMMODITY_CONTEXT_SYMBOLS))


def fetch_chunk(symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            symbols, period=period, interval=interval, group_by="ticker",
            auto_adjust=False, progress=False, threads=True,
        )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for symbol in symbols:
        try:
            sub = raw[symbol] if isinstance(raw.columns, pd.MultiIndex) else raw
        except Exception:
            continue
        frame = normalize_ohlcv(sub)
        if frame.empty:
            continue
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame = frame.tz_convert("Asia/Kolkata")
        # A snapshot row is a quote, not a bar; never let one into the cache.
        frame = drop_synthetic_bars(frame, interval)
        if not frame.empty:
            out[symbol] = frame
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="10d", help="yfinance period (5m caps at 60d)")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between chunks")
    parser.add_argument("--only", choices=["all", "universe", "context"], default="all")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    symbols: list[str] = []
    if args.only in ("all", "context"):
        symbols += context()
    if args.only in ("all", "universe"):
        symbols += universe()
    symbols = list(dict.fromkeys(symbols))

    cache = CandleCache(DB)
    print(f"catching up {len(symbols)} symbols  period={args.period} interval={args.interval}")
    started = time.time()
    saved = ok = 0
    missing: list[str] = []

    for start in range(0, len(symbols), args.chunk):
        chunk = symbols[start : start + args.chunk]
        frames: dict[str, pd.DataFrame] = {}
        for attempt in range(args.retries + 1):
            try:
                frames = fetch_chunk(chunk, args.period, args.interval)
            except Exception as exc:
                print(f"  chunk {start // args.chunk}: {type(exc).__name__}: {exc}")
                frames = {}
            if frames:
                break
            time.sleep(2 * (attempt + 1))
        for symbol, frame in frames.items():
            saved += cache.save(symbol, args.interval, frame)
            ok += 1
        missing += [s for s in chunk if s not in frames]
        print(f"  {min(start + args.chunk, len(symbols)):4d}/{len(symbols)}  "
              f"{ok} symbols, {saved:,} rows  ({time.time() - started:.0f}s)", flush=True)
        time.sleep(args.pause)

    print(f"\n{ok}/{len(symbols)} symbols, {saved:,} rows upserted in {time.time() - started:.0f}s")
    if missing:
        print(f"{len(missing)} returned nothing: {', '.join(missing[:15])}"
              f"{' ...' if len(missing) > 15 else ''}")

    latest = cache.latest_ts("^NSEI", args.interval)
    print(f"^NSEI newest cached bar: {latest}")


if __name__ == "__main__":
    main()
