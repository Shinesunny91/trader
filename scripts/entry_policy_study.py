"""Which signal does the book actually take, and does the entry gate help?

Two questions the repo had answered separately, in ways that turn out not to
compose:

1. `model_stress.py` reports "top-1 = +27.62 bps", and that number is the whole
   justification for `MAX_TRADES_PER_DAY = 1`.  It is computed as
   `argsort(-pred)[:1]` over **a whole session at once** — the day's
   best-predicted signal, chosen with knowledge of signals that had not printed
   yet.  A live book cannot do that.  So the first thing measured here is the
   *causal* version: rank only what has printed, take the first one you are
   allowed to take, and see whether the result survives.

2. `train_signal_model.py` validated the ranking over **every** engine signal,
   and the README's conclusion was that "the gate is throwing away signal".
   But `sim_today.py` — the live path — still filters with
   `entry_quality.passes_entry_gate` *before* ranking.  So the book that runs
   is neither the book that was validated nor the gated books that were
   measured at -0.17%/-0.55%.  The second thing measured here is that
   difference, on the same sessions, with the same model scores.

Both are scored two ways: on the triple-barrier label (fast, exactly what the
model was trained to predict) and through the real portfolio simulator with the
live exit design (slower, but it is the book).

Usage:
    python scripts/entry_policy_study.py
    python scripts/entry_policy_study.py --no-sim     # labels only
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

from intraday_sim import load_frame  # noqa: E402
from nse_intraday_ai import execution_plan as EP  # noqa: E402
from nse_intraday_ai.entry_quality import (  # noqa: E402
    MIN_IMPULSE_ATR,
    MIN_VOLUME_Z,
    _NORM,
)
from nse_intraday_ai.portfolio_sim import IntradayPortfolioSimulator, SimConfig  # noqa: E402

PREDICTIONS = ROOT / "data" / "model_predictions.parquet"
DATASET = ROOT / "data" / "dataset.parquet"

GATE_FEATURES = ["vol_z", "run6", "m_nifty", "m_inr", "m_crude"]

LIVE_BOOK = SimConfig(
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


def macro_score(frame: pd.DataFrame) -> pd.Series:
    """`entry_quality.macro_alignment(...).score`, vectorised.

    The dataset's m_* columns are already signed by trade side, and `_NORM`
    centres each component at zero, so the score is the clipped standardised
    sum of the three survivors.
    """
    return sum(
        np.clip(frame[col] / _NORM[key][1], -3, 3)
        for col, key in (("m_nifty", "nifty"), ("m_inr", "inr"), ("m_crude", "crude"))
    )


def gate_verdicts(frame: pd.DataFrame) -> pd.Series:
    """`entry_quality.passes_entry_gate(...).allow`, vectorised.

    Kept here rather than looping the real function over 67K rows, and pinned
    to it by `tests/test_entry_policy_gate.py` — a study that measures a
    slightly different gate from the one that ships is worse than no study.
    """
    conviction = (frame["vol_z"] >= MIN_VOLUME_Z) & (frame["run6"] >= MIN_IMPULSE_ATR)
    complete = frame[["m_nifty", "m_inr", "m_crude"]].notna().all(axis=1)
    return conviction & complete & (macro_score(frame) >= 0.0)


def load() -> pd.DataFrame:
    predictions = pd.read_parquet(PREDICTIONS)
    features = pd.read_parquet(DATASET, columns=["ts", "symbol", "side", *GATE_FEATURES])
    frame = predictions.merge(features, on=["ts", "symbol", "side"], how="left")
    frame = frame.drop_duplicates(["ts", "symbol", "side"]).dropna(subset=GATE_FEATURES)
    frame["day"] = frame["ts"].dt.normalize()
    frame["macro_score"] = macro_score(frame)
    frame["gated"] = gate_verdicts(frame)
    return frame.sort_values("ts").reset_index(drop=True)


# ── causal entry policies ────────────────────────────────────────────────────
#
# Each returns the rows a book following that policy would have entered.  A
# policy may only look at rows whose `ts` is at or before the one it picks.


def policy_oracle(frame: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """NOT CAUSAL — the session's k best-predicted signals. The ceiling."""
    return frame.sort_values("pred_bps", ascending=False).groupby("day").head(k)


def policy_first_bar(frame: pd.DataFrame, *, gated: bool, k: int = 1) -> pd.DataFrame:
    """The best-predicted signal on the first bar that offers one."""
    pool = frame[frame["gated"]] if gated else frame
    if pool.empty:
        return pool
    first = pool.groupby("day")["ts"].transform("min")
    return pool[pool["ts"] == first].sort_values("pred_bps", ascending=False).groupby("day").head(k)


def policy_threshold(frame: pd.DataFrame, quantile: float, *, gated: bool = False,
                     k: int = 1) -> pd.DataFrame:
    """First signal whose predicted bps clears a bar set by *prior* sessions.

    The causal stand-in for "the day's best": you cannot know the maximum in
    advance, but you can know what an exceptional score looked like yesterday.
    """
    pool = frame[frame["gated"]] if gated else frame
    days = sorted(frame["day"].unique())
    picks = []
    for i, day in enumerate(days):
        if i == 0:
            continue
        prior = frame[frame["day"] < day]["pred_bps"]
        if len(prior) < 1000:
            continue
        bar = float(prior.quantile(quantile))
        today = pool[(pool["day"] == day) & (pool["pred_bps"] >= bar)]
        if not today.empty:
            picks.append(today.sort_values("ts").head(k))
    return pd.concat(picks) if picks else frame.iloc[:0]


def bootstrap_ci(values: np.ndarray, iters: int = 20_000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iters, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def score_labels(frame: pd.DataFrame, picks: pd.DataFrame, label: str) -> dict:
    if picks.empty:
        return {"policy": label, "trades": 0}
    per_session = picks.groupby("day")["actual_bps"].mean()
    lo, hi = bootstrap_ci(per_session.to_numpy())
    return {
        "policy": label,
        "trades": len(picks),
        "sessions": int(picks["day"].nunique()),
        "bps": round(float(picks["actual_bps"].mean()), 2),
        "sessions_up": f"{int((per_session > 0).sum())}/{len(per_session)}",
        "ci_low": round(lo, 1),
        "ci_high": round(hi, 1),
        "median_entry": picks["ts"].dt.strftime("%H:%M").mode().iloc[0],
    }


def score_book(picks: pd.DataFrame, frames: dict, label: str) -> dict:
    if picks.empty:
        return {"book": label, "trades": 0}
    payload = picks[["ts", "symbol", "side", "atr"]].copy()
    payload["rank"] = picks["pred_bps"].to_numpy()
    payload["note"] = ""
    result = IntradayPortfolioSimulator(LIVE_BOOK).run(payload, frames)
    if result.trades.empty:
        return {"book": label, "trades": 0}
    trades = result.trades
    wins = trades[trades["net_pnl"] > 0]["net_pnl"].sum()
    losses = abs(trades[trades["net_pnl"] <= 0]["net_pnl"].sum())
    equity = result.equity["equity"]
    daily = result.daily["pnl"]
    lo, hi = bootstrap_ci(daily.to_numpy())
    return {
        "book": label,
        "trades": len(trades),
        "net_%": round(result.pnl_pct, 2),
        "net": round(result.pnl),
        "win_%": round((trades["net_pnl"] > 0).mean() * 100, 1),
        "pf": round(wins / losses, 2) if losses else float("inf"),
        "sessions_up": f"{int((daily > 0).sum())}/{len(daily)}",
        "dd_%": round((equity.cummax() - equity).max() / LIVE_BOOK.starting_capital * 100, 2),
        "ci_low": round(lo),
        "ci_high": round(hi),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-sim", action="store_true", help="skip the portfolio simulator")
    args = parser.parse_args()

    frame = load()
    print(f"{len(frame):,} model-scored signals over {frame['day'].nunique()} held-out "
          f"sessions | {frame['gated'].mean():.1%} pass the entry gate\n")

    policies = {
        "every signal (population)": frame,
        "ORACLE best of session (not causal)": policy_oracle(frame),
        "first bar, best pred — no gate": policy_first_bar(frame, gated=False),
        "first gated bar, best pred (LIVE)": policy_first_bar(frame, gated=True),
        "pred >= prior p99, no gate": policy_threshold(frame, 0.99),
        "pred >= prior p99.5, no gate": policy_threshold(frame, 0.995),
        "pred >= prior p99.9, no gate": policy_threshold(frame, 0.999),
        "pred >= prior p99, gated": policy_threshold(frame, 0.99, gated=True),
    }

    print("=" * 118)
    print("ENTRY POLICY — scored on the triple-barrier label the model predicts")
    print("=" * 118)
    rows = [score_labels(frame, picks, label) for label, picks in policies.items()]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nci_low/ci_high: bootstrap 95% interval on the per-session mean. An interval\n"
          "spanning zero means the policy is not distinguishable from no edge.")

    if args.no_sim:
        return

    symbols = sorted(frame["symbol"].unique())
    since = (frame["ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\nloading {len(symbols)} price frames for the portfolio run...", flush=True)
    frames = {s: f for s in symbols if not (f := load_frame(s, since)).empty}

    print("\n" + "=" * 118)
    print(f"THE SAME POLICIES AS A BOOK — ₹{LIVE_BOOK.starting_capital:,.0f}, live exits "
          f"(stop {LIVE_BOOK.stop_atr} ATR / target {LIVE_BOOK.target_atr} ATR / "
          f"{LIVE_BOOK.max_hold_bars * 5}min), cap {LIVE_BOOK.max_trades_per_day}/day")
    print("=" * 118)
    book_rows = [score_book(picks, frames, label)
                 for label, picks in policies.items() if label != "every signal (population)"]
    book_rows.insert(0, score_book(frame, frames, "every signal, cap applies"))
    print(pd.DataFrame(book_rows).to_string(index=False))


if __name__ == "__main__":
    main()
