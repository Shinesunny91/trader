"""Stress-test the walk-forward model result before believing it.

`train_model.py` found a random forest whose top-3 picks per session averaged
+12.3 bps against a −9.4 bps baseline, positive in 20 of 34 held-out sessions.
That is the first encouraging number in this project, which is exactly why it
needs adversarial testing rather than celebration. A t-stat of 0.90 does not
clear any reasonable bar on its own.

This script asks the questions that would expose a false positive:

  1. **Is it one session?**  Drop the best session; does the result survive?
     Bootstrap the session means for a confidence interval.
  2. **Is it a hyperparameter fluke?**  Re-run across a grid of forest sizes
     and depths — a real effect should be broadly present, not perched on one
     setting.
  3. **Does it survive different barriers?**  Including longer holds, the one
     structural lever left after costs.
  4. **Is it just the top-3 slice?**  Check k = 1, 2, 3, 5, 10.
  5. **Would a label-shuffled model produce the same thing?**  Permutation
     test: shuffle the labels within each training fold and re-run. Whatever
     the shuffled model "earns" is the noise floor of this whole procedure.

Usage:
    python scripts/model_stress.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_model import DATA, prepare  # noqa: E402

RNG = np.random.default_rng(0)


def run_fold(X, y, day, sessions, i, factory, top_ks, shuffle=False):
    from sklearn.preprocessing import StandardScaler

    test_day = sessions[i]
    train_mask = day < test_day
    test_mask = day == test_day
    if train_mask.sum() < 5000 or test_mask.sum() < 10:
        return None
    y_train = y[train_mask]
    if shuffle:
        y_train = RNG.permutation(y_train)
    scaler = StandardScaler().fit(X[train_mask])
    model = factory()
    model.fit(scaler.transform(X[train_mask]), y_train)
    pred = model.predict(scaler.transform(X[test_mask]))
    actual = y[test_mask]
    order = np.argsort(-pred)
    row = {"session": pd.Timestamp(test_day).date(), "n": int(test_mask.sum()),
           "all_bps": float(actual.mean())}
    for k in top_ks:
        row[f"top{k}"] = float(actual[order[:k]].mean())
    return row


def evaluate(X, y, day, sessions, factory, *, min_train, top_ks, shuffle=False) -> pd.DataFrame:
    rows = [
        r for i in range(min_train, len(sessions))
        if (r := run_fold(X, y, day, sessions, i, factory, top_ks, shuffle)) is not None
    ]
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, iters: int = 20_000) -> tuple[float, float]:
    if len(values) < 3:
        return (float("nan"), float("nan"))
    draws = RNG.choice(values, size=(iters, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarise(table: pd.DataFrame, k: int) -> dict:
    col = f"top{k}"
    values = table[col].to_numpy(float)
    low, high = bootstrap_ci(values)
    return {
        "mean_bps": round(float(values.mean()), 2),
        "median_bps": round(float(np.median(values)), 2),
        "sessions_up": f"{int((values > 0).sum())}/{len(values)}",
        "ci95": f"[{low:+.1f}, {high:+.1f}]",
        "excl_best": round(float(np.sort(values)[:-1].mean()), 2),
        "excl_worst": round(float(np.sort(values)[1:].mean()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=int, default=150)
    parser.add_argument("--min-train", type=int, default=15)
    args = parser.parse_args()

    from intraday_sim import liquid_symbols
    from sklearn.ensemble import RandomForestRegressor

    raw = pd.read_parquet(DATA)
    if args.universe:
        raw = raw[raw["symbol"].isin(liquid_symbols(args.universe))]

    barriers = ["1.5_3.0_12", "1.5_3.0_24", "1.0_2.0_12", "2.0_4.0_36"]
    top_ks = [1, 2, 3, 5, 10]

    def baseline_factory():
        return RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=100, n_jobs=-1, random_state=0
        )

    print("=" * 100)
    print("1. BARRIER SWEEP — is the effect specific to one exit design?")
    print("=" * 100)
    per_barrier = {}
    for barrier in barriers:
        frame, X, y = prepare(raw, barrier)
        day = frame["ts"].dt.normalize().to_numpy()
        sessions = np.array(sorted(frame["ts"].dt.normalize().unique()))
        table = evaluate(X, y, day, sessions, baseline_factory,
                         min_train=args.min_train, top_ks=top_ks)
        per_barrier[barrier] = table
        stats = summarise(table, 3)
        print(f"  {barrier:14s} baseline {table['all_bps'].mean():+7.2f} | "
              f"top3 {stats['mean_bps']:+7.2f} (median {stats['median_bps']:+6.2f}) "
              f"{stats['sessions_up']:>7s} up  CI95 {stats['ci95']:>16s}  "
              f"excl-best {stats['excl_best']:+6.2f}", flush=True)

    best_barrier = max(per_barrier, key=lambda b: summarise(per_barrier[b], 3)["mean_bps"])
    print(f"\n  strongest barrier: {best_barrier}")

    frame, X, y = prepare(raw, best_barrier)
    day = frame["ts"].dt.normalize().to_numpy()
    sessions = np.array(sorted(frame["ts"].dt.normalize().unique()))
    table = per_barrier[best_barrier]

    print("\n" + "=" * 100)
    print(f"2. TOP-K SWEEP on {best_barrier} — is the lift only true at k=3?")
    print("=" * 100)
    for k in top_ks:
        stats = summarise(table, k)
        print(f"  top-{k:<3d} {stats['mean_bps']:+7.2f} bps  median {stats['median_bps']:+7.2f}  "
              f"{stats['sessions_up']:>7s} up  CI95 {stats['ci95']:>16s}  "
              f"excl-best {stats['excl_best']:+7.2f}  excl-worst {stats['excl_worst']:+7.2f}")

    print("\n" + "=" * 100)
    print("3. HYPERPARAMETER GRID — a real effect should not perch on one setting")
    print("=" * 100)
    grid = [
        ("rf 200/6/200", lambda: RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=200, n_jobs=-1, random_state=1)),
        ("rf 300/8/100", baseline_factory),
        ("rf 500/10/50", lambda: RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=50, n_jobs=-1, random_state=2)),
        ("rf 300/12/20", lambda: RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=3)),
        ("rf 300/4/500", lambda: RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=500, n_jobs=-1, random_state=4)),
    ]
    for name, factory in grid:
        t = evaluate(X, y, day, sessions, factory, min_train=args.min_train, top_ks=top_ks)
        stats = summarise(t, 3)
        print(f"  {name:16s} top3 {stats['mean_bps']:+7.2f}  {stats['sessions_up']:>7s} up  "
              f"CI95 {stats['ci95']:>16s}  excl-best {stats['excl_best']:+7.2f}", flush=True)

    print("\n" + "=" * 100)
    print("4. PERMUTATION TEST — what does the same procedure earn on shuffled labels?")
    print("   (this is the noise floor of the pipeline, not of the data)")
    print("=" * 100)
    shuffled = []
    for trial in range(5):
        t = evaluate(X, y, day, sessions, baseline_factory,
                     min_train=args.min_train, top_ks=top_ks, shuffle=True)
        stats = summarise(t, 3)
        shuffled.append(stats["mean_bps"])
        print(f"  shuffle {trial + 1}: top3 {stats['mean_bps']:+7.2f} bps  "
              f"{stats['sessions_up']:>7s} up  CI95 {stats['ci95']}", flush=True)
    real = summarise(table, 3)["mean_bps"]
    floor_mean, floor_sd = float(np.mean(shuffled)), float(np.std(shuffled))
    print(f"\n  real {real:+.2f} bps vs shuffled mean {floor_mean:+.2f} (sd {floor_sd:.2f})")
    print(f"  -> {(real - floor_mean) / (floor_sd + 1e-9):.1f} shuffle-sd above the noise floor")

    print("\n" + "=" * 100)
    print("5. SESSION DETAIL — where does the money come from?")
    print("=" * 100)
    detail = table[["session", "n", "all_bps", "top3"]].copy()
    detail["cum_top3"] = detail["top3"].cumsum()
    print(detail.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
