"""What profit target can actually clear a fixed cost?

The intuition under test: aim for a small move, win most of the time, collect a
steady trickle.  The trouble is that cost does not shrink with the target.  A
round trip costs the same 10.1 bps whether you are reaching for 20 bps or 200,
so the smaller the target, the larger the share of it the exchange takes, and
the higher the win rate has to be before the arithmetic works at all.

For a target T and stop S in bps against a cost C, expectancy is

    E = p*T - (1 - p)*S - C

so break-even needs  p = (S + C) / (T + S).  This script prints that required
win rate beside the win rate the population actually delivers, for each barrier
the dataset was labelled under, so the gap can be read directly rather than
argued about.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.costs import round_trip_bps  # noqa: E402

BARRIERS = ["1.0_2.0_12", "1.5_3.0_12", "1.5_3.0_24", "2.0_4.0_36"]
COST = round_trip_bps(1000.0, 300)


def main() -> None:
    d = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    d["atr_bps"] = d["atr"] / d["fill"] * 1e4
    print(f"{len(d):,} signals   cost {COST:.2f} bps   median ATR {d.atr_bps.median():.1f} bps\n")

    head = (f"{'barrier':<14}{'target':>9}{'stop':>8}{'win%':>8}{'stop%':>8}"
            f"{'time%':>8}{'need%':>8}{'gap':>8}{'net':>9}")
    print(head)
    print("-" * len(head))
    for b in BARRIERS:
        stop_atr, tgt_atr, _ = (float(x) for x in b.split("_"))
        hit, net = d[f"hit_{b}"], d[f"net_bps_{b}"]
        # Barriers are ATR multiples, so express them in bps at the median ATR
        # to compare like with like against a bps cost.
        tgt = tgt_atr * d.atr_bps.median()
        stp = stop_atr * d.atr_bps.median()
        win = (hit == "TARGET").mean() * 100
        stopped = (hit == "STOP").mean() * 100
        timed = (hit == "TIME").mean() * 100
        need = (stp + COST) / (tgt + stp) * 100
        print(f"{b:<14}{tgt:>9.0f}{stp:>8.0f}{win:>8.1f}{stopped:>8.1f}"
              f"{timed:>8.1f}{need:>8.1f}{win - need:>8.1f}{net.mean():>9.2f}")

    print("\nBreak-even win rate for a symmetric target/stop, at this cost:")
    print(f"{'target=stop':<16}{'need win%':>12}")
    print("-" * 28)
    for t in (10, 20, 30, 50, 75, 100, 150, 200, 300):
        print(f"{t:>6} bps      {(t + COST) / (2 * t) * 100:>11.1f}")


if __name__ == "__main__":
    main()
