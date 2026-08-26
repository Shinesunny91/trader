"""What the candidate's paper record would have paid at a given capital and leverage.

Answers the concrete question -- "I put in Rs X at Nx intraday margin, what would
I have made?" -- without the two flattering assumptions that make leverage look
free.

The first is cost.  Charges are not a flat bps: brokerage is 0.1% capped at Rs
20, so a larger position amortises the cap and the *percentage* cost falls
slightly as size rises.  That part genuinely favours size and is computed from
the real Groww schedule rather than assumed.

The second is impact, and it runs the other way.  The recorded bps came from a
~Rs 3.3L position, roughly 1.4% of a median liquid name's 5-minute bar.  Five
times that size is not filled at the same price.  Impact is charged under the
square-root law, `k * sigma * sqrt(Q / bar_volume)`, and because `k` is an
assumption rather than a measurement the answer is reported across a range of
`k` instead of at one convenient value.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.costs import round_trip_bps  # noqa: E402

LOG = ROOT / "data" / "candidate_paper.csv"
BASE_POSITION = 300_000.0
BASE_COST_BPS = round_trip_bps(1000.0, 300)   # what the recorded bps was charged
# Median 5-minute rupee turnover of the names the candidate actually picked,
# from data/intraday_oos.parquet: ~Rs 996 lakh across its 14 in-sample trades.
BAR_TURNOVER = 996 * 1e5
# Typical 5-minute move of those names, used as the volatility term in the
# square-root law.  ~30 bps is the ATR of the high-volatility names it selects.
SIGMA_BPS = 30.0


def impact_bps(position: float, k: float) -> float:
    """One-way impact under the square-root law, in bps."""
    if position <= 0 or k <= 0:
        return 0.0
    return k * SIGMA_BPS * np.sqrt(position / BAR_TURNOVER)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capital", type=float, default=200_000.0)
    ap.add_argument("--leverage", type=float, default=5.0)
    ap.add_argument("--price", type=float, default=1000.0,
                    help="representative share price, for the flat-fee component")
    ap.add_argument("--coefficients", default="0,0.5,1.0,1.5",
                    help="impact coefficients k to report across")
    args = ap.parse_args()

    if not LOG.exists():
        raise SystemExit(f"no paper log at {LOG} — run scripts/candidate_paper.py first")
    d = pd.read_csv(LOG)
    traded = d[d["traded"] == 1].dropna(subset=["net_bps"])
    if traded.empty:
        raise SystemExit("paper log has no completed trades yet")

    position = args.capital * args.leverage
    qty = max(1, int(position / args.price))
    # The recorded net already had BASE_COST_BPS deducted; add it back to get the
    # move itself, then charge this position's own costs.
    gross = traded["net_bps"].to_numpy(float) + BASE_COST_BPS
    cost_at_size = round_trip_bps(args.price, qty)

    print(f"capital Rs {args.capital:,.0f}  x{args.leverage:g} intraday margin  "
          f"-> position Rs {position:,.0f}")
    print(f"{len(gross)} trades over {len(d)} observed sessions\n")
    print(f"round trip at Rs {BASE_POSITION:,.0f}: {BASE_COST_BPS:.2f} bps"
          f"   at Rs {position:,.0f}: {cost_at_size:.2f} bps (before impact)\n")

    header = (f"  {'impact k':<10}{'impact/leg':>12}{'cost bps':>10}"
              f"{'net bps':>10}{'P&L Rs':>12}{'% of capital':>14}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in (float(x) for x in args.coefficients.split(",")):
        imp = impact_bps(position, k)
        total_cost = cost_at_size + 2 * imp
        net = gross - total_cost
        pnl = net.sum() / 1e4 * position
        print(f"  {k:<10.2f}{imp:>12.2f}{total_cost:>10.2f}{net.mean():>10.2f}"
              f"{pnl:>12,.0f}{pnl / args.capital * 100:>13.1f}%")

    worst = gross.min() - (cost_at_size + 2 * impact_bps(position, 1.0))
    print(f"\n  worst single trade at k=1.0: {worst:+.1f} bps = "
          f"Rs {worst / 1e4 * position:+,.0f} "
          f"({worst / 1e4 * position / args.capital * 100:+.1f}% of capital)")
    print(f"  a {abs(worst / 1e4 * position / args.capital * 100):.0f}% "
          f"single-trade drawdown is what {args.leverage:g}x does to a "
          f"{abs(worst):.0f} bps move.")


if __name__ == "__main__":
    main()
