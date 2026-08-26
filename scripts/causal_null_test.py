"""Is the best causal-selectivity cell better than the best cell chance provides?

`causal_selectivity.py` scans 48 (model, percentile, cap) combinations and the
best of them returns +67 bps.  A maximum over 48 noisy cells is an order
statistic, not a measurement, and the only honest way to read it is against the
distribution of maxima the same scan produces when there is nothing to find.

The null keeps everything about the data except the one thing under test: model
scores are permuted *within each session*, so signal counts, session structure,
the label distribution and the daily cap all survive intact while any relation
between score and outcome is destroyed.  Re-running the scan on that gives the
best cell chance alone would have handed us.

Run:
    python scripts/causal_null_test.py --draws 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = ["p_ridge", "p_hgb_shallow", "p_hgb_deep", "p_rf"]
PERCENTILES = [90.0, 95.0, 99.0, 99.5]
CAPS = [1, 3, 5]
WARMUP = 8


def scan(scores: np.ndarray, day_idx: np.ndarray, labels: np.ndarray,
         n_days: int) -> dict[tuple[float, int], np.ndarray]:
    """Every (percentile, cap) cell for one score vector, keyed by its settings.

    Thresholds are computed once per (day, percentile) from strictly prior
    sessions and reused across caps, which is what makes a few hundred null
    draws tractable at all.
    """
    out: dict[tuple[float, int], list[float]] = {(p, c): [] for p in PERCENTILES for c in CAPS}
    for day in range(WARMUP, n_days):
        prior = scores[day_idx < day]
        if prior.size == 0:
            continue
        here = day_idx == day
        s_here, l_here = scores[here], labels[here]
        for pct in PERCENTILES:
            thresh = np.percentile(prior, pct)
            hits = np.flatnonzero(s_here >= thresh)
            if hits.size == 0:
                continue
            for cap in CAPS:
                out[(pct, cap)].append(float(l_here[hits[:cap]].mean()))
    return {k: np.asarray(v) for k, v in out.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=300)
    args = ap.parse_args()

    d = pd.read_parquet(ROOT / "data" / "intraday_oos.parquet")
    d["day"] = pd.to_datetime(d["ts"]).dt.normalize()
    d = d.sort_values("ts").reset_index(drop=True)
    day_idx = pd.factorize(d["day"])[0]
    labels = d["label_bps"].to_numpy(float)
    n_days = day_idx.max() + 1

    observed: dict[str, float] = {}
    all_cells: dict[tuple[str, float, int], np.ndarray] = {}
    for m in MODELS:
        cells = scan(d[m].to_numpy(float), day_idx, labels, n_days)
        for k, v in cells.items():
            all_cells[(m, *k)] = v
        observed[m] = max(v.mean() for v in cells.values())

    width = len(all_cells)
    best_key = max(all_cells, key=lambda k: all_cells[k].mean())
    print(f"{width} cells scanned across {len(MODELS)} models")
    print(f"observed best: {best_key} -> {all_cells[best_key].mean():+.2f} bps "
          f"on {len(all_cells[best_key])} sessions")
    positive = sum(1 for v in all_cells.values() if v.mean() > 0)
    print(f"cells positive: {positive}/{width} ({positive / width:.0%})\n")

    rng = np.random.default_rng(1)
    order = np.argsort(day_idx, kind="stable")
    starts = np.searchsorted(day_idx[order], np.arange(n_days))
    ends = np.searchsorted(day_idx[order], np.arange(n_days), side="right")

    null_best = []
    for _ in range(args.draws):
        shuffled = np.empty_like(labels)
        for a, b in zip(starts, ends):
            block = order[a:b]
            shuffled[block] = labels[rng.permutation(block)]
        best = -np.inf
        for m in MODELS:
            cells = scan(d[m].to_numpy(float), day_idx, shuffled, n_days)
            best = max(best, max(v.mean() for v in cells.values()))
        null_best.append(best)
    null_best = np.asarray(null_best)

    obs = all_cells[best_key].mean()
    p = float(np.mean(null_best >= obs))
    print(f"null distribution of the best-of-{width} cell ({args.draws} draws):")
    print(f"  mean {null_best.mean():+.2f}   median {np.median(null_best):+.2f}"
          f"   95th pct {np.percentile(null_best, 95):+.2f}   max {null_best.max():+.2f}")
    print(f"\n  observed {obs:+.2f} bps   ->   p = {p:.3f}")
    print("  " + ("chance produces a winner this good routinely."
                  if p > 0.05 else "this would be worth a second look."))


if __name__ == "__main__":
    main()
