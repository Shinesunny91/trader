"""Fetch and cache 10 years of DAILY bars for NSE 500 + the commodity universe.

The intraday cache holds ~4 months of 5-minute bars, which is 16 weeks — far
too few independent observations to say anything about a strategy that holds
for a week.  Daily bars go back a decade, giving ~520 weekly holding periods
per symbol instead, and that is the difference between a backtest and an
anecdote.

Stored in the same `candles` table under `interval='1d'`, so everything that
already reads the cache keeps working.

    python scripts/fetch_daily.py                # NSE 500 + commodities
    python scripts/fetch_daily.py --years 5
    python scripts/fetch_daily.py --only commodity
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.candle_cache import CandleCache  # noqa: E402
from nse_intraday_ai.data import DEFAULT_COMMODITY_SYMBOLS  # noqa: E402

CACHE = CandleCache(ROOT / "data" / "candles.sqlite3")
SYMBOL_CSV = ROOT / "data" / "nifty500_symbols.csv"
# Index and macro series worth having daily for regime/benchmark work.
CONTEXT = ["^NSEI", "^NSEBANK", "^INDIAVIX", "USDINR=X", "DX-Y.NYB", "^VIX", "CL=F"]


def nse500() -> list[str]:
    if not SYMBOL_CSV.exists():
        raise SystemExit(f"missing {SYMBOL_CSV}")
    frame = pd.read_csv(SYMBOL_CSV)
    col = "Symbol" if "Symbol" in frame.columns else frame.columns[0]
    return [f"{s.strip()}.NS" for s in frame[col].dropna().astype(str)]


def normalise(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinance hands back either a flat frame or a (field, ticker) MultiIndex."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw
    if isinstance(frame.columns, pd.MultiIndex):
        lvl0 = frame.columns.get_level_values(0)
        if symbol in frame.columns.get_level_values(-1):
            frame = frame.xs(symbol, axis=1, level=-1)
        elif symbol in lvl0:
            frame = frame.xs(symbol, axis=1, level=0)
        else:
            frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.lower)
    need = ["open", "high", "low", "close"]
    if not all(c in frame.columns for c in need):
        return pd.DataFrame()
    out = frame[need + (["volume"] if "volume" in frame.columns else [])].copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out = out.dropna(subset=need)
    if out.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(out.index)
    # Daily bars are dates, not instants; stamp them at the IST close so the
    # cache's UTC canonicalisation round-trips to the same calendar day.
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    else:
        idx = idx.tz_convert("Asia/Kolkata")
    out.index = idx.normalize() + pd.Timedelta(hours=15, minutes=30)
    return out[~out.index.duplicated(keep="last")]


def fetch(symbols: list[str], years: int, batch: int, pause: float) -> tuple[int, list[str]]:
    import yfinance as yf

    saved, failed = 0, []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i : i + batch]
        try:
            raw = yf.download(
                chunk, period=f"{years}y", interval="1d",
                progress=False, auto_adjust=False, group_by="ticker", threads=True,
            )
        except Exception as exc:                      # noqa: BLE001
            print(f"  batch {i // batch + 1}: {type(exc).__name__}: {exc}")
            failed.extend(chunk)
            continue
        for symbol in chunk:
            frame = normalise(raw, symbol)
            if frame.empty:
                failed.append(symbol)
                continue
            saved += CACHE.save(symbol, "1d", frame)
        done = min(i + batch, len(symbols))
        print(f"  {done}/{len(symbols)} symbols, {saved:,} bars cached", flush=True)
        if done < len(symbols):
            time.sleep(pause)
    return saved, failed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--pause", type=float, default=1.0, help="seconds between batches")
    p.add_argument("--only", choices=["nse", "commodity", "context"], default=None)
    args = p.parse_args()

    groups = {
        "nse": nse500(),
        "commodity": list(DEFAULT_COMMODITY_SYMBOLS),
        "context": CONTEXT,
    }
    if args.only:
        groups = {args.only: groups[args.only]}

    total_failed: list[str] = []
    for name, symbols in groups.items():
        print(f"\n{name}: {len(symbols)} symbols, {args.years}y daily")
        saved, failed = fetch(symbols, args.years, args.batch, args.pause)
        print(f"  -> {saved:,} bars, {len(failed)} failed")
        total_failed.extend(failed)

    if total_failed:
        print(f"\nfailed ({len(total_failed)}): {', '.join(total_failed[:20])}"
              + (" ..." if len(total_failed) > 20 else ""))
    print("\n" + str(CACHE.stats()))


if __name__ == "__main__":
    main()
