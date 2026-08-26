"""Selectivity as a live book could actually trade it.

`selectivity_sweep.py` ranks each session's signals and keeps the best K, and
that is not a decision anyone can make: at 09:20 the book does not know what
will print at 14:00.  Ranking a whole session at once quietly grants the
strategy the day's signal list in advance, which is the same look-ahead the
retracted +27.62 bps top-1 result was built on.

The causal version calibrates a score threshold on *prior* sessions only, then
walks the test session forward in time and takes a signal the moment it prints
above that threshold, up to a daily cap.  Nothing is ranked against the future,
zero-trade days are allowed, and the threshold is chosen the way it would have
to be chosen live -- from history.

Run:
    python scripts/causal_selectivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = ["p_ridge", "p_hgb_shallow", "p_hgb_deep", "p_rf"]
# Percentile of the trailing score distribution to trade above.  99.0 means
# "only the top 1% of what this model has historically produced".
PERCENTILES = [90.0, 95.0, 99.0, 99.5]
CAPS = [1, 3, 5]
WARMUP = 8          # sessions of history before the first live threshold


def main() -> None:
    d = pd.read_parquet(ROOT / "data" / "intraday_oos.parquet")
    d["day"] = pd.to_datetime(d["ts"]).dt.normalize()
    d = d.sort_values("ts")
    days = sorted(d["day"].unique())
    cost = float((d["gross_bps"] - d["label_bps"]).iloc[0])
    print(f"{len(d):,} signals, {len(days)} sessions, cost {cost:.2f} bps")
    print(f"threshold calibrated on prior sessions only, {WARMUP}-session warmup\n")

    header = (f"{'model':<14}{'pctile':>8}{'cap':>5}{'trades':>8}{'days':>6}"
              f"{'net_bps':>9}{'up':>8}{'total_Rs':>10}")
    print(header)
    print("-" * len(header))
    best = None
    for model in MODELS:
        for pct in PERCENTILES:
            for cap in CAPS:
                per_day, trades = [], 0
                for i, day in enumerate(days):
                    if i < WARMUP:
                        continue
                    hist = d[d["day"] < day][model]
                    if hist.empty:
                        continue
                    thresh = np.percentile(hist, pct)
                    session = d[d["day"] == day]
                    taken = session[session[model] >= thresh].head(cap)
                    if taken.empty:
                        continue                      # a day with no trade is a real outcome
                    per_day.append(taken["label_bps"].mean())
                    trades += len(taken)
                if not per_day:
                    continue
                net = float(np.mean(per_day))
                up = sum(1 for x in per_day if x > 0)
                # What the book would have earned at the ~Rs 3L position it sizes.
                rupees = net / 1e4 * 300_000 * trades
                print(f"{model:<14}{pct:>8.1f}{cap:>5}{trades:>8}{len(per_day):>6}"
                      f"{net:>9.2f}{f'{up}/{len(per_day)}':>8}{rupees:>10,.0f}")
                if best is None or net > best[0]:
                    best = (net, model, pct, cap, trades, up, len(per_day))
        print()

    net, model, pct, cap, trades, up, ndays = best
    print(f"best cell: {model} p{pct} cap {cap} -> {net:+.2f} bps, "
          f"{trades} trades over {ndays} sessions, {up} up")
    print(f"cost line is {cost:.2f} bps; a cell has to clear 0 here, not {cost:.2f}, "
          f"because net_bps is already after cost.")


if __name__ == "__main__":
    main()
