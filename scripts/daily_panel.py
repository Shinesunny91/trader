"""Build a daily cross-sectional research panel from the 10-year cache.

Every study in this repo so far has been conditioned on `VotingSignalEngine`
firing, which means every negative result has been a verdict on the engine's
signals rather than on the market.  It has also meant working with 48 sessions,
because that is all the 5-minute cache holds.  The daily cache holds 2,479
sessions of 500 names -- fifty times the statistical power -- and nothing has
ever been asked of it directly.

This builds the panel those questions need: one row per (day, symbol) with
returns computed over the standard formation windows, cross-sectional ranks
within each day, and forward returns at the horizons a swing book could trade.
No signal engine is involved anywhere.

**Survivorship bias is present and material.** The symbol list is today's NIFTY
500, so names delisted or demoted over the decade are absent, and the survivors
are exactly the ones that did well.  Every long-only result off this panel is
biased upward.  Long-short results are far less affected, because the bias hits
both legs, which is one reason the studies downstream are built long-short.

    python scripts/daily_panel.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DB = ROOT / "data" / "candles.sqlite3"
OUT = ROOT / "data" / "daily_panel.parquet"

# Formation windows in trading days.  1 and 5 are the short-term reversal
# horizons; 21 is a month; 252/21 is the classic 12-1 momentum window that skips
# the most recent month precisely because that month reverses.
LOOKBACKS = [1, 5, 21, 63, 252]
FORWARDS = [1, 5, 10, 21]
MIN_PRICE = 20.0          # penny names have costs and spreads this study cannot model
MIN_TURNOVER_CR = 1.0     # Rs 1 crore median daily turnover


def load() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    try:
        d = pd.read_sql(
            "SELECT symbol, ts, open, high, low, close, volume FROM candles "
            "WHERE interval='1d' AND symbol LIKE '%.NS'", con)
    finally:
        con.close()
    d["day"] = pd.to_datetime(d["ts"], utc=True).dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
    d = d.drop(columns=["ts"]).sort_values(["symbol", "day"])
    return d[d["close"] > 0]


def build(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("symbol", sort=False)
    d["turnover_cr"] = d["close"] * d["volume"] / 1e7
    # Liquidity is measured on a trailing window so the filter is causal; using
    # the full-sample median would let a name's later popularity qualify it for
    # trades taken years earlier.
    d["turnover_med"] = g["turnover_cr"].transform(lambda s: s.rolling(63, min_periods=21).median())

    for k in LOOKBACKS:
        d[f"ret_{k}d"] = g["close"].transform(lambda s, k=k: s.pct_change(k)) * 100
    # 12-1 momentum: a year of return excluding the most recent month.
    d["mom_12_1"] = ((1 + d["ret_252d"] / 100) / (1 + d["ret_21d"] / 100) - 1) * 100
    d["vol_21d"] = g["close"].transform(
        lambda s: s.pct_change().rolling(21, min_periods=10).std()) * 100
    d["gap"] = (d["open"] / g["close"].shift(1) - 1) * 100

    for k in FORWARDS:
        # Forward return is entered at the NEXT day's open, not today's close:
        # a rank computed from today's close cannot be traded until tomorrow.
        nxt_open = g["open"].shift(-1)
        fwd_close = g["close"].shift(-k)
        d[f"fwd_{k}d"] = (fwd_close / nxt_open - 1) * 100

    d = d[(d["close"] >= MIN_PRICE) & (d["turnover_med"] >= MIN_TURNOVER_CR)]
    # Cross-sectional ranks, computed within each day over the names that passed
    # the liquidity filter that day.
    by_day = d.groupby("day")
    for col in ["ret_1d", "ret_5d", "ret_21d", "mom_12_1", "vol_21d", "turnover_cr", "gap"]:
        d[f"z_{col}"] = by_day[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
        d[f"rank_{col}"] = by_day[col].rank(pct=True)
    d["n_names"] = by_day["symbol"].transform("size")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    raw = load()
    print(f"loaded {len(raw):,} daily bars, {raw['symbol'].nunique()} symbols, "
          f"{raw['day'].nunique():,} sessions")
    panel = build(raw)
    panel = panel.dropna(subset=["ret_5d", "fwd_5d"])
    print(f"panel {len(panel):,} rows after liquidity/price filters, "
          f"{panel['symbol'].nunique()} symbols, {panel['day'].nunique():,} sessions")
    print(f"names per day: median {panel.groupby('day').size().median():.0f}")
    panel.to_parquet(args.out, index=False)
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
