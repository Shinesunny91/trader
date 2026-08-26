"""Add sector, peer-relative and information-proxy columns to the daily panel.

`overnight_drift_study.py` shorts the largest raw gaps.  Raw gap size is a crude
instrument, because it mixes two things that should behave in opposite ways:

  common information   the whole sector gapped, or the whole market did.  That
                       is real news arriving, and real news is not supposed to
                       revert -- prices moved because value changed.
  idiosyncratic noise  one name gapped and its peers did not, on no unusual
                       volume.  That is order-flow pressure, and pressure decays.

Splitting the gap into those parts is what this adds.  `gap_sector` is the
sector's median gap that morning, `gap_resid` is what is left of a name's gap
after removing it, and `gap_mkt` does the same for the whole market.

Real news headlines would be the direct instrument, but `data/news.sqlite3`
holds 222 rows -- three months of one feed -- against 2,479 sessions, so it
cannot support a decade-long test.  Abnormal volume is the standard stand-in and
is available for every row: a gap on three times normal volume is information
arriving, a gap on half normal volume is not.

    python scripts/enrich_panel.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PANEL = ROOT / "data" / "daily_panel.parquet"
SYMBOLS = ROOT / "data" / "nifty500_symbols.csv"
OUT = ROOT / "data" / "daily_panel_rich.parquet"
MIN_PEERS = 4      # a "sector median" over three names is not a sector


def sectors() -> pd.Series:
    m = pd.read_csv(SYMBOLS)
    m["symbol"] = m["Symbol"].astype(str).str.strip() + ".NS"
    return m.set_index("symbol")["Industry"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    d = pd.read_parquet(PANEL)
    sec = sectors()
    d["sector"] = d["symbol"].map(sec)
    matched = d["sector"].notna().mean()
    print(f"{len(d):,} rows   sector matched for {matched:.1%}   "
          f"{d['sector'].nunique()} sectors")
    d = d[d["sector"].notna()].copy()

    # ── common vs idiosyncratic gap ──────────────────────────────────────────
    by_ds = d.groupby(["day", "sector"])["gap"]
    d["gap_sector"] = by_ds.transform("median")
    d["sector_n"] = by_ds.transform("size")
    # A one-name "sector" would give gap_resid == 0 for that name and silently
    # remove it from every extreme, so thin sectors fall back to the market.
    d["gap_mkt"] = d.groupby("day")["gap"].transform("median")
    thin = d["sector_n"] < MIN_PEERS
    d.loc[thin, "gap_sector"] = d.loc[thin, "gap_mkt"]
    d["gap_resid"] = d["gap"] - d["gap_sector"]
    d["gap_ex_mkt"] = d["gap"] - d["gap_mkt"]

    # ── information proxy: was the gap accompanied by unusual volume? ────────
    g = d.sort_values(["symbol", "day"]).groupby("symbol")
    med_vol = g["volume"].transform(lambda s: s.rolling(21, min_periods=10).median().shift(1))
    d["vol_ratio"] = d["volume"] / med_vol.replace(0, np.nan)
    d["log_vol_ratio"] = np.log(d["vol_ratio"].clip(lower=0.01))

    # ── peer confirmation: how many sector peers gapped the same way ─────────
    same = d.assign(up=(d["gap"] > 0).astype(float)).groupby(["day", "sector"])["up"]
    d["sector_up_share"] = same.transform("mean")
    d["gap_agrees_sector"] = np.sign(d["gap"]) == np.sign(d["gap_sector"])

    keep = d.dropna(subset=["gap", "fwd_1d", "gap_resid", "vol_ratio"])
    print(f"panel {len(keep):,} rows, {keep['day'].nunique():,} sessions")
    print(f"  median sector size {keep['sector_n'].median():.0f}   "
          f"median vol_ratio {keep['vol_ratio'].median():.2f}")
    keep.to_parquet(args.out, index=False)
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
