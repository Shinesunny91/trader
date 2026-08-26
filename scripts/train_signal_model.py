"""Train and ship the signal-ranking model — but only if it earns it.

Mirrors the discipline of `train_meta_model.py`: the script runs a walk-forward
book with the model doing the ranking, and **refuses to write the model file**
unless that book beat the composite-ranked book it would replace. The file
existing is therefore the evidence that the check passed.

What "walk-forward" means here: for every session after a warm-up, a forest is
fit on strictly prior sessions only, used to rank that session's signals, and
the top-K are traded through the real portfolio simulator. Nothing scores a
session it was trained on.

Usage:
    python scripts/train_signal_model.py
    python scripts/train_signal_model.py --force   # write regardless (research)
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

from intraday_sim import liquid_symbols, load_frame  # noqa: E402
from nse_intraday_ai import execution_plan as EP  # noqa: E402
from nse_intraday_ai.portfolio_sim import IntradayPortfolioSimulator, SimConfig  # noqa: E402
from nse_intraday_ai.signal_model import feature_matrix, model_path, train  # noqa: E402

DATA = ROOT / "data" / "dataset_live.parquet"
if not DATA.exists():
    DATA = ROOT / "data" / "dataset.parquet"

# The book the evidence gate is judged on must be the book that runs, so it is
# sourced from execution_plan exactly as sim_today's LIVE_CONFIG is.  It used
# to be a hard-coded 3/day, 1.5 ATR stop, 3.0 ATR target copy — the defaults of
# 2026-08-11.  When the cap moved to 1 and the stop to 2.0/5.0, this file kept
# validating the old book, so `validation` in signal_model.json (and the
# expectancy line on every order ticket) described a configuration that no
# longer existed.
BOOK = SimConfig(
    starting_capital=10_00_000.0,
    max_concurrent_positions=EP.MAX_CONCURRENT,
    max_trades_per_day=EP.MAX_TRADES_PER_DAY,
    max_position_pct=EP.MAX_POSITION_PCT,
    risk_per_trade_pct=1.0,
    scale_out_fraction=0.0,
    stop_atr=EP.STOP_ATR,
    target_atr=EP.TARGET_ATR,
    breakeven_after_atr=EP.BREAKEVEN_ATR,
    trail_atr=0.0,
    max_hold_bars=EP.MAX_HOLD_MINUTES // 5,
    slippage_bps_per_leg=1.5,
)


def walk_forward_predictions(frame: pd.DataFrame, barrier: str, min_train: int) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    X = feature_matrix(frame)
    y = frame[f"net_bps_{barrier}"].to_numpy(float)
    day = frame["ts"].dt.normalize().to_numpy()
    sessions = np.array(sorted(frame["ts"].dt.normalize().unique()))
    predictions = np.full(len(frame), np.nan)

    for i in range(min_train, len(sessions)):
        test_day = sessions[i]
        train_mask, test_mask = day < test_day, day == test_day
        if train_mask.sum() < 5000 or not test_mask.any():
            continue

        # Recency weighting per fold
        sess_in_train = sorted(pd.Series(day[train_mask]).unique())
        sess_map = {s: idx for idx, s in enumerate(sess_in_train)}
        train_sess_idx = pd.Series(day[train_mask]).map(sess_map).to_numpy()
        max_idx = max(sess_map.values()) if sess_map else 0
        decay = np.exp((train_sess_idx - max_idx) / 25.0)
        decay = decay / (decay.mean() if decay.mean() > 0 else 1.0)

        scaler = StandardScaler().fit(X[train_mask])
        X_tr = scaler.transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])

        forest = RandomForestRegressor(
            n_estimators=250, max_depth=8, min_samples_leaf=80, n_jobs=-1, random_state=42
        )
        forest.fit(X_tr, y[train_mask], sample_weight=decay)
        p_rf = forest.predict(X_te)

        hgb = HistGradientBoostingRegressor(
            max_iter=150, max_depth=6, min_samples_leaf=80, l2_regularization=1.5, random_state=42
        )
        hgb.fit(X_tr, y[train_mask], sample_weight=decay)
        p_hgb = hgb.predict(X_te)

        predictions[test_mask] = 0.5 * p_rf + 0.5 * p_hgb
        print(f"  fold {i - min_train + 1}: {pd.Timestamp(test_day).date()} "
              f"({int(test_mask.sum())} signals)", flush=True)
    return pd.Series(predictions, index=frame.index)


def run_book(frame: pd.DataFrame, ranks: pd.Series, frames: dict) -> object:
    payload = frame[["ts", "symbol", "side", "atr"]].copy()
    payload["rank"] = ranks.to_numpy()
    payload["note"] = ""
    payload = payload.dropna(subset=["rank"])
    return IntradayPortfolioSimulator(BOOK).run(payload, frames)


def describe(result, label: str) -> dict:
    if result.trades.empty:
        return {"book": label, "trades": 0}
    trades = result.trades
    wins = trades[trades["net_pnl"] > 0]["net_pnl"].sum()
    losses = abs(trades[trades["net_pnl"] <= 0]["net_pnl"].sum())
    equity = result.equity["equity"]
    return {
        "book": label,
        "sessions": len(result.daily),
        "trades": len(trades),
        "net_pct": round(result.pnl_pct, 2),
        "win_pct": round((trades["net_pnl"] > 0).mean() * 100, 1),
        "profit_factor": round(wins / losses, 2) if losses else float("inf"),
        "max_dd_pct": round((equity.cummax() - equity).max() / BOOK.starting_capital * 100, 2),
        "sessions_up": f"{int((result.daily['pnl'] > 0).sum())}/{len(result.daily)}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barrier", default="1.5_3.0_12")
    parser.add_argument("--min-train", type=int, default=15)
    parser.add_argument("--universe", type=int, default=150)
    parser.add_argument("--data", type=Path, default=DATA, help="parquet dataset path")
    parser.add_argument("--force", action="store_true",
                        help="write the model even if it fails the check")
    args = parser.parse_args()

    raw = pd.read_parquet(args.data)
    raw = raw[raw["symbol"].isin(liquid_symbols(args.universe))].reset_index(drop=True)
    frame = raw.dropna(subset=[f"net_bps_{args.barrier}"]).reset_index(drop=True)

    # Never train on the session being traded.  A model fitted on today's
    # labels and then used to rank today's signals is scoring data it has
    # already seen — harmless for tomorrow, but it makes today's paper number
    # look better than a live book could have achieved.
    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    current = frame["ts"].dt.normalize() >= today
    if current.any():
        print(f"excluding {int(current.sum()):,} signals from the current session "
              f"({today.date()}) — a model must not be trained on the day it ranks")
        frame = frame[~current].reset_index(drop=True)

    print(f"{len(frame):,} signals | {frame['ts'].dt.date.nunique()} sessions "
          f"| barrier {args.barrier}\n")

    print("walk-forward folds:")
    predictions = walk_forward_predictions(frame, args.barrier, args.min_train)
    scored = predictions.notna()
    print(f"\n{int(scored.sum()):,} signals scored out of sample")

    # The incumbent: the composite the gate currently ranks by.
    composite = (
        frame["m_nifty"].fillna(0) + frame["m_inr"].fillna(0) + frame["m_crude"].fillna(0)
        + frame["vol_z"].clip(0, 5) + frame["run6"].clip(0, 4)
    ).where(scored)

    since = (frame.loc[scored, "ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    symbols = sorted(frame.loc[scored, "symbol"].unique())
    print(f"loading {len(symbols)} price frames...")
    frames = {s: f for s in symbols if not (f := load_frame(s, since)).empty}

    subset = frame[scored]
    model_book = run_book(subset, predictions[scored], frames)
    base_book = run_book(subset, composite[scored], frames)

    rows = [describe(base_book, "composite rank (incumbent)"),
            describe(model_book, "model rank (candidate)")]
    print("\n" + "=" * 92)
    print("WALK-FORWARD BOOK COMPARISON")
    print("=" * 92)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nmodel book recency:\n{model_book.recency_split()}")

    model_stats, base_stats = rows[1], rows[0]
    improved = (
        model_stats.get("net_pct", -99) > base_stats.get("net_pct", 99)
        and model_stats.get("net_pct", -99) > 0
        and model_stats.get("profit_factor", 0) > 1.0
    )
    print(f"\ncheck: model {model_stats.get('net_pct')}% vs composite "
          f"{base_stats.get('net_pct')}%, profit factor "
          f"{model_stats.get('profit_factor')} -> "
          f"{'PASS' if improved else 'FAIL'}")

    if not improved and not args.force:
        print("\nREFUSING to write the model — it did not beat the incumbent book.\n"
              "This is the evidence gate working, not an error.")
        return

    validation = {
        "net_pct": model_stats.get("net_pct"),
        "sessions": model_stats.get("sessions"),
        "profit_factor": model_stats.get("profit_factor"),
        "win_pct": model_stats.get("win_pct"),
        "max_dd_pct": model_stats.get("max_dd_pct"),
        "sessions_up": model_stats.get("sessions_up"),
        "incumbent_net_pct": base_stats.get("net_pct"),
        "forced": bool(args.force and not improved),
    }
    model = train(frame, barrier=args.barrier, validation=validation)
    path = model_path()
    model.save(path)
    print(f"\nwrote {path} (and {path.with_suffix('.forest.pkl').name}) — "
          f"trained on all {len(frame):,} signals")


if __name__ == "__main__":
    main()
