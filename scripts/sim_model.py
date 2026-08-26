"""Run the ₹10L book on the walk-forward model's rankings.

`train_model.py` reports mean net bps of the top-K picks. That is a useful
statistic but it is not a P&L: it ignores capital limits, concurrency, the
square-off, and the fact that costs depend on position size. This runs the
model's out-of-sample predictions through the same portfolio simulator the
rest of the study used, so the number is in rupees and comparable with every
other variant.

Every prediction here was produced by a model that never saw the session it is
scoring — `train_model.py` refits per session on strictly prior data.

Usage:
    python scripts/sim_model.py
    python scripts/sim_model.py --top-k 2 --slippage 2.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from intraday_sim import load_frame  # noqa: E402
from nse_intraday_ai.portfolio_sim import IntradayPortfolioSimulator, SimConfig  # noqa: E402

PREDICTIONS = ROOT / "data" / "model_predictions.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital", type=float, default=10_00_000.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--slippage", type=float, default=1.5)
    parser.add_argument("--min-pred", type=float, default=None,
                        help="only trade signals predicted above this net-bps threshold")
    args = parser.parse_args()

    predictions = pd.read_parquet(PREDICTIONS)
    print(f"{len(predictions):,} scored signals over "
          f"{predictions['ts'].dt.date.nunique()} held-out sessions "
          f"(model: {predictions['model'].iloc[0]})")
    if args.min_pred is not None:
        before = len(predictions)
        predictions = predictions[predictions["pred_bps"] >= args.min_pred]
        print(f"  {len(predictions):,} of {before:,} clear the "
              f"{args.min_pred:+.1f} bps prediction threshold")
    if predictions.empty:
        print("nothing left to trade")
        return

    since = (predictions["ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    symbols = sorted(predictions["symbol"].unique())
    print(f"loading {len(symbols)} price frames...")
    frames = {s: f for s in symbols if not (f := load_frame(s, since)).empty}

    payload = predictions[["ts", "symbol", "side", "atr"]].copy()
    payload["rank"] = predictions["pred_bps"].to_numpy()
    payload["note"] = ""

    rows = []
    for top_k in (1, 2, 3, 5):
        config = SimConfig(
            starting_capital=args.capital,
            max_concurrent_positions=min(3, top_k),
            max_trades_per_day=top_k,
            max_position_pct=33.0, risk_per_trade_pct=1.0,
            scale_out_fraction=0.0, stop_atr=1.5, target_atr=3.0,
            breakeven_after_atr=0.9, trail_atr=0.0, max_hold_bars=12,
            slippage_bps_per_leg=args.slippage,
        )
        result = IntradayPortfolioSimulator(config).run(payload, frames)
        if result.trades.empty:
            continue
        trades = result.trades
        equity = result.equity["equity"]
        turnover = (trades["entry"] * trades["quantity"]).sum()
        daily = result.daily["pnl"]
        rows.append({
            "top_k": top_k, "positions": len(trades),
            "net": round(result.pnl), "net_%": round(result.pnl_pct, 2),
            "gross_bps": round(trades["gross_pnl"].sum() / max(turnover, 1) * 1e4, 1),
            "cost_bps": round(trades["costs"].sum() / max(turnover, 1) * 1e4, 1),
            "win_%": round((trades["net_pnl"] > 0).mean() * 100, 1),
            "sessions_up": f"{int((daily > 0).sum())}/{len(daily)}",
            "dd_%": round((equity.cummax() - equity).max() / args.capital * 100, 2),
        })
        if top_k == args.top_k:
            best = result

    print("\n" + "=" * 92)
    print(f"MODEL-RANKED BOOK — ₹{args.capital:,.0f}, {args.slippage} bps/leg slippage")
    print("=" * 92)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 92)
    print(f"DETAIL at top-{args.top_k}")
    print("=" * 92)
    print(best.summary())
    print(best.recency_split())
    print("\n  exits:")
    print(best.trades.groupby("exit_reason").agg(
        n=("net_pnl", "size"), net=("net_pnl", "sum"), avg=("net_pnl", "mean")
    ).round(0).to_string())
    daily = best.daily.assign(cum=best.daily["pnl"].cumsum())
    print(f"\n  daily P&L:\n{daily.round(0).to_string()}")


if __name__ == "__main__":
    main()
