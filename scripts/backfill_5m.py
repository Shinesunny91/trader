"""Backfill the 5-minute cache up to the latest completed session.

The scanner daemon stopped writing on 2026-08-17 when the workspace moved to
the laptop, so the cache has a hole.  Yahoo serves 5-minute bars for roughly
the trailing 60 days, which is wider than any gap the daemon can leave, so a
single `period` fetch closes it without needing to reason about dates.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.candle_cache import CandleCache  # noqa: E402

CACHE = CandleCache(ROOT / "data" / "candles.sqlite3")
CHUNK = 40


def cached_symbols() -> list[str]:
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "candles.sqlite3")
    try:
        rows = con.execute(
            "SELECT DISTINCT symbol FROM candles WHERE interval='5m' AND symbol LIKE '%.NS'"
        ).fetchall()
    finally:
        con.close()
    return sorted(r[0] for r in rows)


def frame_for(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(-1):
            return pd.DataFrame()
        raw = raw.xs(symbol, axis=1, level=-1)
    out = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return out.dropna(subset=["open", "high", "low", "close"])


def main() -> None:
    symbols = cached_symbols()
    print(f"{len(symbols)} symbols", flush=True)
    saved = 0
    for i in range(0, len(symbols), CHUNK):
        batch = symbols[i : i + CHUNK]
        try:
            raw = yf.download(
                batch, period="1mo", interval="5m", progress=False,
                auto_adjust=False, group_by="column", threads=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"batch {i} failed: {exc}", flush=True)
            continue
        for sym in batch:
            frame = frame_for(raw, sym)
            if not frame.empty:
                saved += CACHE.save(sym, "5m", frame)
        print(f"{i + len(batch)}/{len(symbols)} rows_saved={saved}", flush=True)
        time.sleep(1.0)
    print(f"DONE saved={saved}", flush=True)


if __name__ == "__main__":
    main()
