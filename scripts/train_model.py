"""Walk-forward model over the signal dataset — does *any* model beat the gate?

Discipline, because this dataset will happily manufacture an edge if allowed:

* **Splits are by session, never by row.** A 12-bar label straddles bars, so a
  random split leaks the answer from a neighbouring row of the same trade.
* **Every fold is strictly forward.** Fold N trains on sessions < N and is
  scored on session N only. Nothing is refitted on data it has seen scored.
* **The scaler is fitted on training folds only**, for the same reason.
* **The metric is economic, not statistical.** AUC on 400K rows will look fine
  and mean nothing. What matters is: if the book takes the top-K predicted
  signals per session, what does it earn net of costs? That is what the
  portfolio actually does, so that is what gets reported.
* A model is only interesting if its per-session edge is **positive in a
  majority of held-out sessions**, not merely on average — one runaway session
  can carry a mean.

Usage:
    python scripts/train_model.py --barrier 1.5_3.0_12 --top-k 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "dataset_live.parquet"
if not DATA.exists():
    DATA = ROOT / "data" / "dataset.parquet"
PRED_OUT = ROOT / "data" / "model_predictions.parquet"

FEATURES = [
    # timing
    "vol_z", "run3", "run6", "run12", "ext_vwap", "ext_ema9", "ext_ema21", "ext_ema50",
    "pos_in_range", "age_extreme", "rsi", "adx",
    # microstructure
    "clv", "clv_flow", "body", "wick_with", "wick_against", "bar_size_atr", "streak",
    # session structure
    "or_position", "sess_range_atr", "d_prior_high", "d_prior_low", "gap_atr",
    "minute", "bars_since_open", "atr_bps",
    # liquidity
    "turnover_lakh", "turnover_z",
    # cross-sectional
    "xs_impulse_rank", "xs_volume_rank", "xs_turnover_rank", "xs_n_signals",
    "xs_with_crowd",
    # macro
    "m_nifty", "m_inr", "m_crude", "rel_strength",
    # engine's own view (included so the model can use it if it is worth using)
    "conf", "rr", "n_strategies",
]
CATEGORICAL = ["side", "regime"]


def prepare(frame: pd.DataFrame, barrier: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = frame.dropna(subset=[f"net_bps_{barrier}"]).copy()
    X = frame[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X["is_long"] = (frame["side"] == "LONG").astype(float)
    for regime in ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"):
        X[f"regime_{regime}"] = (frame["regime"] == regime).astype(float)
    y = frame[f"net_bps_{barrier}"].to_numpy(float)
    return frame, X.to_numpy(float), y


def models():
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
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


def walk_forward(
    frame: pd.DataFrame, X: np.ndarray, y: np.ndarray, *,
    name: str, factory, min_train_sessions: int, top_k: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    sessions = np.array(sorted(frame["ts"].dt.normalize().unique()))
    day = frame["ts"].dt.normalize().to_numpy()
    predictions = np.full(len(frame), np.nan)
    rows = []

    for i in range(min_train_sessions, len(sessions)):
        test_day = sessions[i]
        train_mask = day < test_day
        test_mask = day == test_day
        if train_mask.sum() < 5000 or test_mask.sum() < 5:
            continue
        scaler = StandardScaler().fit(X[train_mask])
        model = factory()
        model.fit(scaler.transform(X[train_mask]), y[train_mask])
        pred = model.predict(scaler.transform(X[test_mask]))
        predictions[test_mask] = pred

        actual = y[test_mask]
        order = np.argsort(-pred)
        picked = actual[order[:top_k]]
        rows.append({
            "session": pd.Timestamp(test_day).date(),
            "n": int(test_mask.sum()),
            "all_bps": float(actual.mean()),
            f"top{top_k}_bps": float(picked.mean()) if len(picked) else np.nan,
            "top10pct_bps": float(actual[order[: max(1, len(order) // 10)]].mean()),
            "bot10pct_bps": float(actual[order[-max(1, len(order) // 10):]].mean()),
        })
    return pd.DataFrame(rows), predictions


def report(name: str, table: pd.DataFrame, top_k: int) -> dict:
    if table.empty:
        print(f"{name}: no evaluable sessions")
        return {}
    col = f"top{top_k}_bps"
    positive = int((table[col] > 0).sum())
    spread = table["top10pct_bps"] - table["bot10pct_bps"]
    summary = {
        "model": name,
        "sessions": len(table),
        "all_bps": round(table["all_bps"].mean(), 2),
        f"top{top_k}_bps": round(table[col].mean(), 2),
        "lift": round(table[col].mean() - table["all_bps"].mean(), 2),
        "sessions_up": f"{positive}/{len(table)}",
        "hit_rate": round(positive / len(table), 3),
        "decile_spread": round(spread.mean(), 2),
        "spread_up": f"{int((spread > 0).sum())}/{len(table)}",
        # A t-stat on session means: crude, but it names the noise floor.
        "t_stat": round(
            table[col].mean() / (table[col].std() / np.sqrt(len(table)) + 1e-9), 2
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barrier", default="1.5_3.0_12")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-train-sessions", type=int, default=15)
    parser.add_argument("--universe", type=int, default=150,
                        help="restrict to the N most liquid symbols (0 = all)")
    args = parser.parse_args()

    raw = pd.read_parquet(DATA)
    if args.universe:
        sys.path.insert(0, str(ROOT / "scripts"))
        from intraday_sim import liquid_symbols

        raw = raw[raw["symbol"].isin(liquid_symbols(args.universe))]
    frame, X, y = prepare(raw, args.barrier)
    print(f"{len(frame):,} signals | {frame['ts'].dt.date.nunique()} sessions | "
          f"{X.shape[1]} features | barrier {args.barrier}")
    print(f"unconditional net edge: {y.mean():+.2f} bps  "
          f"(P(profitable) {(y > 0).mean():.1%})\n")

    summaries = []
    best_name, best_predictions, best_score = None, None, -1e9
    for name, factory in models().items():
        table, predictions = walk_forward(
            frame, X, y, name=name, factory=factory,
            min_train_sessions=args.min_train_sessions, top_k=args.top_k,
        )
        summary = report(name, table, args.top_k)
        if summary:
            summaries.append(summary)
            print(f"  {name:12s} done — top{args.top_k} "
                  f"{summary[f'top{args.top_k}_bps']:+.2f} bps, "
                  f"{summary['sessions_up']} sessions up", flush=True)
            score = summary["hit_rate"] * 100 + summary[f"top{args.top_k}_bps"]
            if score > best_score:
                best_name, best_predictions, best_score = name, predictions, score

    print("\n" + "=" * 104)
    print(f"WALK-FORWARD RESULTS — mean net bps of the top-{args.top_k} picks per session")
    print("=" * 104)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(
        "\nRead `sessions_up` and `t_stat`, not the mean alone: with this much session\n"
        "variance a positive mean carried by a handful of sessions is not an edge."
    )

    if best_predictions is not None:
        out = frame[["ts", "symbol", "side", "fill", "atr"]].copy()
        out["pred_bps"] = best_predictions
        out["actual_bps"] = y
        out["model"] = best_name
        out = out.dropna(subset=["pred_bps"])
        out.to_parquet(PRED_OUT)
        print(f"\nbest model: {best_name} -> {PRED_OUT} ({len(out):,} scored signals)")


if __name__ == "__main__":
    main()
