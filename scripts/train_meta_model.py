"""Train + validate the meta-label veto model from replayed candidate events.

Rolling weekly walk-forward: every validation week is scored by a model
trained only on prior weeks.  The model file is written ONLY if the veto
demonstrably helps out of sample:

  1. median top-vs-bottom quintile hit-rate lift across folds > +0.05, and
  2. OOS portfolio P&L at the production gates improves at veto-50%.

Otherwise the script prints the evidence and refuses (see --force).
July-2026 result: commodity PASSES (−554 → +681 over 9 OOS weeks);
NSE FAILS (no abstention level flips ≈0 gross edge − 18 bps costs positive).

Usage:
    python scripts/two_week_backtest.py commodity --since 2026-04-28   # events
    python scripts/train_meta_model.py commodity                       # model
    python scripts/train_meta_model.py nse        # expected to refuse
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.backtest import BacktestConfig, run_backtest_from_events  # noqa: E402
from nse_intraday_ai.meta_model import (  # noqa: E402
    features_from_event,
    model_path,
    train_logistic,
    train_meta_model,
)

# (cost_bps, production gates (conf, rr, votes), same-day-exit)
UNIVERSES = {
    "nse": (18.0, (70.0, 1.5, 2), True),
    "commodity": (6.0, (85.0, 1.5, 1), False),
}
MIN_MEDIAN_LIFT = 0.05


def walk_forward(events, cost_bps, gates, same_day, veto_fractions=(0.3, 0.5, 0.7)):
    conf_g, rr_g, votes_g = gates
    config = BacktestConfig(
        no_new_trade_after="15:00" if same_day else "",
        same_day_exit_only=same_day,
        estimated_cost_bps=cost_bps * 2 / 3,
        slippage_bps=cost_bps / 3,
    )
    X = np.array([features_from_event(e) for e in events])
    bps = np.array([e.per_share_pnl / e.entry * 1e4 for e in events])
    y = (bps > cost_bps).astype(float)
    weeks = pd.Series([e.timestamp.isocalendar().week for e in events])
    uniq = sorted(weeks.unique())

    def portfolio(evts):
        summary, _, _ = run_backtest_from_events(
            events=evts, starting_capital=100_000, risk_per_trade_pct=0.5,
            max_position_pct=25.0, cooldown_minutes=15,
            min_confidence=conf_g, min_reward_risk=rr_g,
            min_agreeing_votes=votes_g, min_vote_share=0.5, config=config)
        return summary

    lifts: list[float] = []
    kept: dict[float, list] = {frac: [] for frac in veto_fractions}
    oos_all: list = []
    for k in range(2, len(uniq)):
        train_mask = weeks.isin(uniq[:k]).to_numpy()
        val_mask = (weeks == uniq[k]).to_numpy()
        if train_mask.sum() < 500 or val_mask.sum() < 100:
            continue
        mu, sd = X[train_mask].mean(0), X[train_mask].std(0) + 1e-9
        w = train_logistic((X[train_mask] - mu) / sd, y[train_mask])

        def prob(mask):
            z = np.clip(np.c_[np.ones(mask.sum()), (X[mask] - mu) / sd] @ w, -30, 30)
            return 1 / (1 + np.exp(-z))

        p_va, p_tr = prob(val_mask), prob(train_mask)
        order = np.argsort(p_va)
        quintiles = [float(q.mean()) for q in np.array_split(y[val_mask][order], 5)]
        lifts.append(quintiles[-1] - quintiles[0])
        va_events = [e for e, m in zip(events, val_mask) if m]
        oos_all.extend(va_events)
        gate_tr = np.array([
            e.confidence >= conf_g and e.reward_risk >= rr_g
            and e.agreeing_count >= votes_g and e.vote_share >= 0.5
            for e, m in zip(events, train_mask) if m
        ])
        base = p_tr[gate_tr] if gate_tr.any() else p_tr
        for frac in veto_fractions:
            tau = np.quantile(base, frac)
            kept[frac].extend(e for e, p in zip(va_events, p_va) if p >= tau)
        print(f"  wk{uniq[k]}: n_val={val_mask.sum():5d} "
              f"quintile hit-rates={[round(q, 3) for q in quintiles]} "
              f"lift={quintiles[-1] - quintiles[0]:+.3f}")

    s0 = portfolio(oos_all)
    print(f"\n  OOS portfolio @ gates conf>={conf_g} rr>={rr_g} votes>={votes_g}:")
    print(f"    no veto        trades={s0.trades:3d} wr={s0.win_rate:5.1f}% "
          f"pnl={s0.pnl:+8.0f} dd={s0.max_drawdown:.0f}")
    results = {"no_veto": {"trades": s0.trades, "win_rate": round(s0.win_rate, 1),
                           "pnl": round(s0.pnl), "max_dd": round(s0.max_drawdown)}}
    for frac in veto_fractions:
        s1 = portfolio(kept[frac])
        print(f"    veto {int(frac * 100):2d}%       trades={s1.trades:3d} wr={s1.win_rate:5.1f}% "
              f"pnl={s1.pnl:+8.0f} dd={s1.max_drawdown:.0f}")
        results[f"veto_{int(frac * 100)}"] = {
            "trades": s1.trades, "win_rate": round(s1.win_rate, 1),
            "pnl": round(s1.pnl), "max_dd": round(s1.max_drawdown)}
    return lifts, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("universe", choices=sorted(UNIVERSES))
    parser.add_argument("--events", default=None, help="events pickle (default data/events_<universe>.pkl)")
    parser.add_argument("--force", action="store_true", help="write the model even if validation fails")
    args = parser.parse_args()

    cost_bps, gates, same_day = UNIVERSES[args.universe]
    events_path = Path(args.events) if args.events else ROOT / "data" / f"events_{args.universe}.pkl"
    with events_path.open("rb") as f:
        events = pickle.load(f)
    events.sort(key=lambda e: e.timestamp)
    print(f"{args.universe}: {len(events)} events from {events_path.name}, "
          f"cost={cost_bps} bps, gates={gates}\n\nWalk-forward validation:")

    lifts, results = walk_forward(events, cost_bps, gates, same_day)
    median_lift = float(np.median(lifts)) if lifts else 0.0
    pnl_gain = results.get("veto_50", {}).get("pnl", 0) - results["no_veto"]["pnl"]
    ok_lift = median_lift > MIN_MEDIAN_LIFT
    ok_pnl = pnl_gain > 0
    print(f"\n  median quintile lift {median_lift:+.3f} over {len(lifts)} folds "
          f"({'PASS' if ok_lift else 'FAIL'} > {MIN_MEDIAN_LIFT})")
    print(f"  veto-50% OOS pnl gain {pnl_gain:+.0f} ({'PASS' if ok_pnl else 'FAIL'} > 0)")

    if not (ok_lift and ok_pnl) and not args.force:
        print(f"\nREFUSED: validation failed for {args.universe}; no model written. "
              "(--force overrides, not recommended)")
        sys.exit(1)

    model = train_meta_model(events, universe=args.universe, cost_bps=cost_bps, gates=gates)
    model.validation = {"median_quintile_lift": round(median_lift, 3),
                        "folds": len(lifts), "oos_portfolio": results}
    out = model_path(args.universe)
    model.save(out)
    print(f"\nModel written: {out} (n={model.n_events}, taus={model.taus})")


if __name__ == "__main__":
    main()
