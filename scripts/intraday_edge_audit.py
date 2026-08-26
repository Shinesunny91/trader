"""Is there an intraday edge to rank at all?  Measured before any model is fit.

Every ranking study on this dataset asks "which signals are best?".  This one
asks the prior question — "is the best one good enough to pay for itself?" —
because ranking cannot create edge, only concentrate it.  If the population's
*gross* edge is a rounding error next to the round-trip cost, then no ordering
of that population is profitable, and a model that appears to find one has
found the sampling noise instead.

Three measurements, in the order that matters:

1. **Gross versus cost, per barrier.**  Costs are fixed per round trip while
   the move being chased grows roughly with the square root of holding time.
   So widening the barrier is the one structural lever available.  The table
   shows whether pulling it closes the gap or merely nudges it.

2. **Directional drift by side and horizon.**  If longs earn early and lose by
   the close while shorts do the reverse, the signals are systematically late
   — the entry is arriving after the move rather than before it.

3. **Conditional edge, calibrate then validate.**  For each feature the best
   decile is chosen on the first half of the sessions and then *scored on the
   second half*.  The shrinkage between the two columns is the honest cost of
   having gone looking.  A decile that clears the cost bar in-sample and
   collapses out-of-sample is the normal outcome, and reporting only the first
   number is how a backtest lies.

    python scripts/intraday_edge_audit.py
    python scripts/intraday_edge_audit.py --universe 0     # every symbol
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BARRIERS = ["1.0_2.0_12", "1.5_3.0_12", "1.5_3.0_24", "2.0_4.0_36"]
DRIFT = ["ret6_bps", "ret12_bps", "ret24_bps", "ret_eod_bps"]

FEATURES = [
    "vol_z", "run3", "run6", "run12", "ext_vwap", "ext_ema9", "ext_ema21",
    "ext_ema50", "pos_in_range", "age_extreme", "rsi", "adx", "clv", "clv_flow",
    "body", "wick_with", "wick_against", "bar_size_atr", "streak", "or_position",
    "sess_range_atr", "d_prior_high", "d_prior_low", "gap_atr", "minute",
    "bars_since_open", "atr_bps", "turnover_lakh", "turnover_z",
    "xs_impulse_rank", "xs_volume_rank", "xs_turnover_rank", "xs_n_signals",
    "xs_with_crowd", "m_nifty", "m_inr", "m_crude", "rel_strength",
    "conf", "rr", "n_strategies",
]


def cost_table(d: pd.DataFrame) -> float:
    """Gross edge, net edge and round-trip cost at each barrier width."""
    print(f"\n{'barrier':16s}{'n':>9s}{'gross':>9s}{'net':>9s}{'cost':>8s}"
          f"{'bars':>7s}{'target%':>9s}{'stop%':>8s}")
    cost = float("nan")
    for b in BARRIERS:
        net_col = f"net_bps_{b}"
        if net_col not in d:
            continue
        s = d.dropna(subset=[net_col])
        g, n = s[f"gross_bps_{b}"], s[net_col]
        cost = float((g - n).mean())
        hit = s[f"hit_{b}"]
        print(f"{b:16s}{len(s):9,d}{g.mean():+9.2f}{n.mean():+9.2f}{cost:8.2f}"
              f"{s[f'bars_{b}'].mean():7.1f}"
              f"{(hit == 'TARGET').mean() * 100:9.1f}{(hit == 'STOP').mean() * 100:8.1f}")
    return cost


def drift_table(d: pd.DataFrame) -> None:
    """Where the move actually is, relative to the entry, by side."""
    print(f"\n{'forward return':16s}{'all':>9s}{'LONG':>9s}{'SHORT':>9s}")
    for c in DRIFT:
        if c not in d:
            continue
        by = d.groupby("side")[c].mean()
        print(f"{c:16s}{d[c].mean():+9.2f}{by.get('LONG', np.nan):+9.2f}"
              f"{by.get('SHORT', np.nan):+9.2f}")
    print(f"\n(mean bars from signal to close: {d['bars_to_eod'].mean():.1f})")


def conditional_table(d: pd.DataFrame, gross: str, cost: float, deciles: int) -> pd.DataFrame:
    """Best decile per feature, chosen in-sample and scored out-of-sample."""
    days = np.array(sorted(d["day"].unique()))
    mid = days[len(days) // 2]
    cal, val = d[d["day"] < mid], d[d["day"] >= mid]
    print(f"\ncalibrate on {cal['day'].nunique()} sessions, "
          f"validate on {val['day'].nunique()} — cost to beat is {cost:.1f} bps")

    rows = []
    for f in FEATURES:
        if f not in d or cal[f].notna().sum() < 1000:
            continue
        try:
            bins, edges = pd.qcut(cal[f], deciles, labels=False,
                                  duplicates="drop", retbins=True)
        except (ValueError, TypeError):
            continue
        cal_mean = cal.groupby(bins)[gross].mean()
        if cal_mean.empty:
            continue
        best = cal_mean.idxmax()
        val_bins = pd.cut(val[f], edges, labels=False, include_lowest=True)
        val_mean = val.groupby(val_bins)[gross].mean()
        rows.append({
            "feature": f,
            "cal_best": cal_mean.max(),
            "val_best": val_mean.get(best, np.nan),
            "n_val": int((val_bins == best).sum()),
        })
    table = pd.DataFrame(rows).sort_values("val_best", ascending=False)
    table["shrinkage_%"] = (1 - table["val_best"] / table["cal_best"]) * 100
    table["clears_cost"] = table["val_best"] > cost
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--barrier", default="1.5_3.0_12")
    p.add_argument("--universe", type=int, default=150,
                   help="restrict to the N most liquid symbols (0 = all); NSE only")
    p.add_argument("--market", choices=["nse", "commodity"], default="nse")
    p.add_argument("--deciles", type=int, default=10)
    args = p.parse_args()

    if args.market == "commodity":
        # Futures carry no STT and CTT on the sell side only, so their round
        # trip is cheaper than equity's.  The label in this dataset was built
        # with that cost, which is why the two markets can be compared here at
        # all: each is measured against the hurdle it actually faces.
        d = pd.read_parquet(ROOT / "data" / "dataset_commodity.parquet")
    else:
        d = pd.read_parquet(ROOT / "data" / "dataset.parquet")
        if args.universe:
            from intraday_sim import liquid_symbols
            d = d[d["symbol"].isin(liquid_symbols(args.universe))]
    d = d.dropna(subset=[f"net_bps_{args.barrier}"]).copy()
    d["day"] = d["ts"].dt.normalize()

    print("=" * 88)
    print(f"INTRADAY EDGE AUDIT — {len(d):,} signals, {d['day'].nunique()} sessions, "
          f"{d['symbol'].nunique()} symbols")
    print("=" * 88)

    cost = cost_table(d)
    drift_table(d)

    gross = f"gross_bps_{args.barrier}"
    table = conditional_table(d, gross, cost, args.deciles)
    print(f"\nbest decile per feature on {gross} "
          f"(chosen on calibrate, scored on validate)\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))

    survivors = table[table["clears_cost"]]
    print("\n" + "=" * 88)
    print(f"VERDICT: {len(survivors)} of {len(table)} features have a best decile that "
          f"still clears\nthe {cost:.1f} bps round trip out of sample.")
    if survivors.empty:
        print("\nNo conditioning variable in this set produces a subset whose gross edge\n"
              "pays for the trade. Ranking concentrates edge; it cannot create it, so a\n"
              "model trained on this population should be expected to lose money after\n"
              "costs regardless of how few signals it is allowed to take.")
    else:
        print(survivors.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))


if __name__ == "__main__":
    main()
