"""Do the classic cross-sectional effects exist in the NIFTY 500, after costs?

The engine trades breakouts and moving-average votes.  The effects tested here
are the ones the academic literature says actually exist in equity
cross-sections, and none of them is what the engine looks for:

  short-term reversal   yesterday's biggest losers outperform its winners
  momentum (12-1)       a year of return, skipping the last month, persists
  low volatility        low-volatility names beat high-volatility ones
  gap reversal          overnight gaps partially fill

Each is measured as a long-short decile spread rebalanced daily and held for a
fixed horizon, which is how these effects are defined and the only form in which
the cost arithmetic is honest -- a long-short book pays costs on both legs, and
survivorship bias in the universe hits both legs too.

Results are reported gross first.  If a spread has no gross edge there is no
point discussing costs, and if it does, the cost line is applied at the real
Groww delivery rate for the holding period.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.costs import Segment, segment_round_trip_bps  # noqa: E402

PANEL = ROOT / "data" / "daily_panel.parquet"
# Round trip on a Rs 3L delivery position, both legs of a long-short pair.
DELIVERY_BPS = segment_round_trip_bps(1000.0, 300, segment=Segment.EQUITY_DELIVERY,
                                      slippage_bps_per_leg=2.5)

# (column, direction, label).  direction +1 means "high rank is the long leg".
SIGNALS = [
    ("ret_1d", -1, "1-day reversal"),
    ("ret_5d", -1, "5-day reversal"),
    ("ret_21d", -1, "21-day reversal"),
    ("mom_12_1", +1, "12-1 momentum"),
    ("vol_21d", -1, "low volatility"),
    ("gap", -1, "gap reversal"),
    ("turnover_cr", -1, "low turnover"),
]
DECILE = 0.1


def spread(panel: pd.DataFrame, col: str, direction: int, fwd: str) -> pd.Series:
    """Daily long-short decile spread in percent, one value per session."""
    d = panel.dropna(subset=[col, fwd])
    r = d.groupby("day")[col].rank(pct=True)
    top = d[r >= 1 - DECILE].groupby("day")[fwd].mean()
    bot = d[r <= DECILE].groupby("day")[fwd].mean()
    return (top - bot) * direction


def summarise(s: pd.Series, hold: int) -> dict:
    """Annualise a spread series that is *overlapping* by `hold` days.

    Overlapping windows make consecutive observations dependent, so the naive
    t-statistic is inflated by roughly sqrt(hold).  Dividing the standard error
    by that factor is the standard Newey-West-style correction, applied crudely
    but in the right direction.
    """
    v = s.dropna().to_numpy(float)
    if v.size < 100:
        return {}
    per_day = v.mean() / hold
    ann = per_day * 252
    se = v.std() / np.sqrt(v.size / hold)
    t = v.mean() / (se + 1e-12)
    return {"n": v.size, "mean_pct": v.mean(), "ann_pct": ann, "t": t,
            "hit": (v > 0).mean() * 100}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holds", default="1,5,10,21")
    args = ap.parse_args()

    panel = pd.read_parquet(PANEL)
    days = np.sort(panel["day"].unique())
    mid = days[len(days) // 2]
    print(f"{len(panel):,} rows   {len(days):,} sessions   "
          f"{str(days[0])[:10]} -> {str(days[-1])[:10]}")
    print(f"delivery round trip {DELIVERY_BPS:.1f} bps per leg-pair rebalance\n")

    holds = [int(x) for x in args.holds.split(",")]
    header = (f"{'signal':<18}{'hold':>5}{'gross/trade':>13}{'ann %':>9}{'t':>7}"
              f"{'hit%':>7}{'h1 ann':>9}{'h2 ann':>9}")
    print(header)
    print("-" * len(header))
    for col, direction, label in SIGNALS:
        for hold in holds:
            fwd = f"fwd_{hold}d"
            if fwd not in panel.columns:
                continue
            s = spread(panel, col, direction, fwd)
            st = summarise(s, hold)
            if not st:
                continue
            h1 = summarise(s[s.index < mid], hold)
            h2 = summarise(s[s.index >= mid], hold)
            print(f"{label:<18}{hold:>5}{st['mean_pct']:>13.3f}{st['ann_pct']:>9.1f}"
                  f"{st['t']:>7.1f}{st['hit']:>7.1f}"
                  f"{h1.get('ann_pct', float('nan')):>9.1f}"
                  f"{h2.get('ann_pct', float('nan')):>9.1f}")
        print()

    print("gross/trade is the % spread earned over the whole holding period.")
    print(f"A trade must clear {DELIVERY_BPS / 100:.2f}% ({DELIVERY_BPS:.0f} bps) "
          f"to be worth taking; both legs of the pair pay it.")


if __name__ == "__main__":
    main()
