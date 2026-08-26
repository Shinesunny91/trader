"""Does trading *less* make the intraday book profitable?

The intuition under test: take only the handful of signals the model is most
sure about, accept a small profit per trade, and the book turns positive.  It
is a reasonable thing to believe and it is cheap to check, because selectivity
is a single knob — the fraction of the ranked population you are willing to
touch.

The sweep walks that knob from "take everything" down to "take the top 0.1%"
and reports net bps per trade at each stop.  Two guards keep it honest:

  halves      every cell is re-reported on the first and second half of the
              held-out sessions.  A cell that is positive in one half and
              negative in the other is noise wearing a result's clothes.
  scan width  the number of (model, threshold) cells tested is printed next to
              the number that came out positive in both halves, so the reader
              can compare it against the ~25% that chance alone delivers.

Predictions come from `intraday_oos.parquet`, where every row was scored by a
model fit only on sessions strictly before that row's own session.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = ["p_ridge", "p_hgb_shallow", "p_hgb_deep", "p_rf"]
# Fractions of the ranked population to keep.  The tail is deliberately extreme:
# "a few very confident trades" means single-digit trades per week, which at
# ~2000 signals a session is the 0.1% cell.
QUANTILES = [1.0, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001]


def load() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "intraday_oos.parquet")
    d["day"] = pd.to_datetime(d["ts"]).dt.normalize()
    return d


def cell(frame: pd.DataFrame, model: str, q: float) -> tuple[float, int]:
    """Mean net bps over the top `q` fraction of each session, ranked by `model`.

    Selection is per session, and the average is over session means rather than
    over pooled trades.  Both choices matter and both were wrong in the first
    version of this script.  Pooling ranks the whole held-out period at once,
    which is not a decision anyone can make on the morning of a given session;
    and it weights a 2,400-signal session three times as heavily as an
    800-signal one, so a good month with heavy signal counts can carry the
    average on its own.  Equal-weighting sessions is what the account
    experiences: one day, one outcome, whatever the signal count happened to
    be.  The gap between the two is not cosmetic -- it is the whole result.
    """
    if frame.empty:
        return float("nan"), 0
    means, trades = [], 0
    for _, session in frame.groupby("day"):
        k = max(1, int(round(len(session) * q)))
        means.append(session.nlargest(k, model)["label_bps"].mean())
        trades += k
    return float(np.mean(means)), trades


def main() -> None:
    d = load()
    days = sorted(d["day"].unique())
    mid = days[len(days) // 2]
    h1, h2 = d[d["day"] < mid], d[d["day"] >= mid]
    print(f"{len(d):,} signals over {len(days)} held-out sessions")
    print(f"half 1: {(h1['day'].nunique())} sessions   half 2: {h2['day'].nunique()} sessions")
    print(f"cost per round trip: {(d['gross_bps'] - d['label_bps']).iloc[0]:.2f} bps\n")

    print(f"{'model':<14}{'keep':>8}{'trades':>9}{'net_all':>10}{'net_h1':>10}{'net_h2':>10}  stable")
    print("-" * 70)
    both_positive = 0
    tested = 0
    for model in MODELS:
        for q in QUANTILES:
            net, k = cell(d, model, q)
            n1, _ = cell(h1, model, q)
            n2, _ = cell(h2, model, q)
            tested += 1
            stable = n1 > 0 and n2 > 0
            both_positive += stable
            flag = "YES" if stable else ""
            print(f"{model:<14}{q:>8.3%}{k:>9,}{net:>10.2f}{n1:>10.2f}{n2:>10.2f}  {flag}")
        print()

    print(f"cells positive in BOTH halves: {both_positive} of {tested}"
          f"   (chance alone delivers ~{tested // 4})")


if __name__ == "__main__":
    main()
