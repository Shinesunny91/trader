"""Build the honest out-of-sample prediction store for intraday signals.

`scripts/train_model.py` fits four model families, then writes the predictions
of whichever one scored best on the held-out sessions.  That last step is the
problem: picking the winner *after* seeing its out-of-sample score is a fifth
model choice made on the test set, and the number it reports is a maximum over
four draws rather than an expectation.  On this dataset the gap is not
academic — three of the four families lose money on their top pick and one
appears to make 27 bps.

This script fits the same families the same way and keeps **all** of them, one
column per model, alongside the label.  Nothing is selected here.  Selection,
if any, happens downstream where it can be measured honestly, and an
agreement rule across models can be tested instead of a winner-take-all pick.

    python scripts/intraday_oos.py                  # liquid-150, default barrier
    python scripts/intraday_oos.py --universe 0     # every symbol

Writes data/intraday_oos.parquet.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_model import prepare  # noqa: E402

OUT = ROOT / "data" / "intraday_oos.parquet"


def families():
    """The candidate models, spanning linear to deep trees.

    Kept deliberately diverse: agreement between models that see the data
    differently is worth more than agreement between two tunings of the same
    one.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    return {
        "ridge": lambda: Ridge(alpha=10.0),
        "hgb_shallow": lambda: HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=200, random_state=0,
        ),
        "hgb_deep": lambda: HistGradientBoostingRegressor(
            max_depth=6, max_iter=400, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=100, random_state=0,
        ),
        "rf": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=100,
            n_jobs=-1, random_state=0,
        ),
    }


def walk_forward(X: np.ndarray, y: np.ndarray, day: np.ndarray, factory,
                 *, min_train: int, embargo: int, min_rows: int) -> np.ndarray:
    """Fit on strictly prior sessions, predict one session, never look back.

    `embargo` drops the last N sessions before the test day from training.  A
    12-bar label opened near the close of session N-1 is still unresolved when
    session N opens, so without an embargo the most recent training rows carry
    outcomes that overlap the test window.
    """
    from sklearn.preprocessing import StandardScaler

    sessions = np.array(sorted(np.unique(day)))
    pred = np.full(len(y), np.nan)
    for i in range(min_train, len(sessions)):
        test_day = sessions[i]
        cutoff = sessions[i - embargo] if embargo else test_day
        train, test = day < cutoff, day == test_day
        if train.sum() < min_rows or test.sum() < 5:
            continue
        scaler = StandardScaler().fit(X[train])
        model = factory()
        model.fit(scaler.transform(X[train]), y[train])
        pred[test] = model.predict(scaler.transform(X[test]))
    return pred


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--barrier", default="1.5_3.0_12")
    p.add_argument("--universe", type=int, default=150,
                   help="restrict to the N most liquid symbols (0 = all)")
    p.add_argument("--min-train", type=int, default=15)
    p.add_argument("--embargo", type=int, default=1,
                   help="sessions dropped between train and test")
    p.add_argument("--min-rows", type=int, default=5000)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    raw = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    if args.universe:
        from intraday_sim import liquid_symbols
        raw = raw[raw["symbol"].isin(liquid_symbols(args.universe))]

    frame, X, y = prepare(raw, args.barrier)
    day = frame["ts"].dt.normalize().to_numpy()
    print(f"{len(frame):,} signals | {len(np.unique(day))} sessions | "
          f"{X.shape[1]} features | barrier {args.barrier}")
    print(f"population net edge {y.mean():+.2f} bps  "
          f"(P(profit) {(y > 0).mean():.1%})\n")

    out = frame[["ts", "symbol", "side", "fill", "atr", "conf",
                 "turnover_lakh", "minute"]].copy()
    out["label_bps"] = y
    gross = f"gross_bps_{args.barrier}"
    if gross in frame:
        out["gross_bps"] = frame[gross].to_numpy(float)

    for name, factory in families().items():
        t0 = time.time()
        out[f"p_{name}"] = walk_forward(
            X, y, day, factory,
            min_train=args.min_train, embargo=args.embargo, min_rows=args.min_rows,
        )
        scored = out[f"p_{name}"].notna().sum()
        print(f"  {name:14s} {scored:>7,} scored  ({time.time() - t0:5.1f}s)", flush=True)

    cols = [c for c in out.columns if c.startswith("p_")]
    out = out.dropna(subset=cols, how="all").reset_index(drop=True)
    out.to_parquet(args.out)
    print(f"\nwrote {args.out} — {len(out):,} rows, {len(cols)} model columns, "
          f"{out['ts'].dt.date.nunique()} held-out sessions")


if __name__ == "__main__":
    main()
