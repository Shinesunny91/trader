"""Meta-labeling study over candidate events (López de Prado style).

The primary system decides *what* to trade (the ensemble + gates); a
secondary classifier learns *which* of those signals to trust.  This script
trains a small logistic regression on week-1 candidate events — label: did
the trade cover round-trip costs? — and validates on week 2:

- rank quality: hit-rate by predicted-probability quintile (should rise),
- portfolio effect: P&L at the recommended gates with vs. without the filter.

It is a research tool, deliberately NOT wired into live scans: the July-2026
study showed real hit-rate ranking out of sample (commodities 26%→37%,
NSE 24%→39% across quintiles) but mixed magnitude effects and small
strict-gate samples.  Re-run it as more cached weeks accumulate; wire it in
only once the portfolio effect validates repeatedly.

Usage:
    python scripts/two_week_backtest.py commodity        # writes data/events_commodity.pkl
    python scripts/meta_label_study.py data/events_commodity.pkl --cost 6 --gates 85,1.5,1
    python scripts/meta_label_study.py data/events_nse.pkl --cost 18 --gates 75,1.5,2
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.backtest import BacktestConfig, run_backtest_from_events  # noqa: E402

STRATEGIES = [
    "trend_continuation", "vwap_mean_reversion", "opening_range_breakout",
    "volatility_compression_breakout", "ema_scalp", "vwap_bounce_scalp",
    "momentum_burst_scalp", "supertrend", "bb_kc_squeeze", "rsi_divergence",
    "fair_value_gap",
]
REGIMES = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]
FEATURE_NAMES = ["conf", "rr", "votes", "share", "side", "sin_hour", "cos_hour",
                 *REGIMES, *STRATEGIES]


def _regime_of(event) -> str:
    for reason in event.reasons:
        match = re.search(r"Market regime: (\w+)", reason)
        if match:
            return match.group(1)
    return "RANGING"


def _features(event) -> list[float]:
    hour = event.timestamp.hour + event.timestamp.minute / 60
    row = [
        event.confidence / 100,
        min(event.reward_risk, 5) / 5,
        event.agreeing_count / 4,
        event.vote_share,
        1.0 if event.side.value == "LONG" else -1.0,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    ]
    regime = _regime_of(event)
    row += [1.0 if regime == r else 0.0 for r in REGIMES]
    row += [1.0 if s in event.agreeing_strategies else 0.0 for s in STRATEGIES]
    return row


def _train_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                    iters: int = 800, lr: float = 0.5) -> np.ndarray:
    weights = np.zeros(X.shape[1] + 1)
    Xb = np.c_[np.ones(len(X)), X]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ weights))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[0, weights[1:]] / len(y)
        weights -= lr * grad
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_pkl")
    parser.add_argument("--cost", type=float, required=True,
                        help="round-trip cost in bps used for labels and simulation")
    parser.add_argument("--gates", default="75,1.5,2",
                        help="conf,rr,votes gates for the portfolio check")
    args = parser.parse_args()
    gate_conf, gate_rr, gate_votes = args.gates.split(",")

    with open(args.events_pkl, "rb") as f:
        events = pickle.load(f)
    if not events:
        print("No events in", args.events_pkl)
        return
    is_commodity = any("=" in e.symbol for e in events[:50])
    config = BacktestConfig(
        no_new_trade_after="" if is_commodity else "15:00",
        estimated_cost_bps=args.cost * 2 / 3, slippage_bps=args.cost / 3,
    )

    X = np.array([_features(e) for e in events])
    bps = np.array([e.per_share_pnl / e.entry * 10000 for e in events])
    y = (bps > args.cost).astype(float)

    days = sorted({e.timestamp.normalize() for e in events})
    mid = days[len(days) // 2]
    w1 = np.array([e.timestamp.normalize() < mid for e in events])
    mu, sd = X[w1].mean(0), X[w1].std(0) + 1e-9
    Xs = (X - mu) / sd
    weights = _train_logistic(Xs[w1], y[w1])
    prob = 1 / (1 + np.exp(-np.c_[np.ones(len(Xs)), Xs] @ weights))

    print(f"{Path(args.events_pkl).name}: n={len(events)}  split={mid.date()}  "
          f"base hit-rate w1={y[w1].mean():.3f} w2={y[~w1].mean():.3f}")
    order = np.argsort(prob[~w1])
    quintiles = np.array_split(y[~w1][order], 5)
    print("week-2 hit-rate by predicted-prob quintile:",
          [round(float(q.mean()), 3) for q in quintiles])
    bps_quintiles = np.array_split(bps[~w1][order], 5)
    print("week-2 mean gross bps by quintile:         ",
          [round(float(q.mean()), 1) for q in bps_quintiles])

    def portfolio(evts, label):
        summary, _, _ = run_backtest_from_events(
            events=evts, starting_capital=100_000, risk_per_trade_pct=0.5,
            max_position_pct=25.0, cooldown_minutes=15,
            min_confidence=float(gate_conf), min_reward_risk=float(gate_rr),
            min_agreeing_votes=int(gate_votes), min_vote_share=0.5, config=config)
        print(f"  {label:22s} trades={summary.trades:3d} "
              f"win_rate={summary.win_rate:5.1f}% pnl={summary.pnl:+8.0f} "
              f"max_dd={summary.max_drawdown:.0f}")

    # pick the filter threshold on week 1, validate on week 2
    w1_events = [e for e, m in zip(events, w1) if m]
    w2_events = [e for e, m in zip(events, w1) if not m]
    best_tau = None
    best_pnl = -1e18
    for tau in np.percentile(prob[w1], [30, 40, 50, 60, 70]):
        selected = [e for e, p in zip(w1_events, prob[w1]) if p >= tau]
        summary, _, _ = run_backtest_from_events(
            events=selected, starting_capital=100_000, risk_per_trade_pct=0.5,
            max_position_pct=25.0, cooldown_minutes=15,
            min_confidence=float(gate_conf), min_reward_risk=float(gate_rr),
            min_agreeing_votes=int(gate_votes), min_vote_share=0.5, config=config)
        if summary.trades >= 5 and summary.pnl > best_pnl:
            best_tau, best_pnl = float(tau), summary.pnl

    print(f"\nweek-2 portfolio at gates conf={gate_conf} rr={gate_rr} votes={gate_votes}"
          f" (tau={None if best_tau is None else round(best_tau, 3)}):")
    portfolio(w2_events, "no meta filter")
    if best_tau is not None:
        filtered = [e for e, p in zip(w2_events, prob[~w1]) if p >= best_tau]
        portfolio(filtered, "with meta filter")

    top = np.argsort(-np.abs(weights[1:]))[:8]
    print("strongest features:",
          [(FEATURE_NAMES[i], round(float(weights[1:][i]), 3)) for i in top])


if __name__ == "__main__":
    main()
