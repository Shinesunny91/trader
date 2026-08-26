"""Final configuration test: liquid universe, cost-aware sizing, honest slippage.

The design sweep left the book at gross 9.8 bps vs cost 10.4 bps — a dead heat.
Two levers remained, and both are about the *cost* side rather than inventing
more signal:

  1. **Liquidity.**  Slippage is the biggest single cost term and it is not
     uniform across a 500-name universe.  Restricting to the most liquid names
     makes a low per-leg slippage assumption defensible instead of hopeful.
  2. **Ranking.**  The book can only hold a few positions a day, so *which*
     few it picks is the whole game.  Ranking by the composite of the three
     out-of-sample survivors (macro alignment, volume expansion, impulse) beats
     ranking by macro alignment alone.

Slippage is swept rather than assumed, because the answer depends on it and
the reader deserves to see where the break-even sits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from intraday_sim import build_signals, liquid_symbols, load_frame  # noqa: E402
from nse_intraday_ai.portfolio_sim import IntradayPortfolioSimulator, SimConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["all", "train", "test"], default="test")
    parser.add_argument("--capital", type=float, default=10_00_000.0)
    parser.add_argument("--universe", type=int, default=150, help="most-liquid N symbols")
    parser.add_argument("--gate", default="full_gate",
                        choices=["full_gate", "quality_gate", "engine_gate"])
    args = parser.parse_args()

    signals = build_signals(args.split)
    liquid = liquid_symbols(args.universe)
    gated = signals[signals[args.gate] & signals["symbol"].isin(liquid)].copy()
    sessions = gated["ts"].dt.date.nunique()
    print(f"{len(gated):,} gated signals in the top-{args.universe} liquid universe "
          f"over {sessions} sessions ({len(gated) / sessions:.0f}/session)")

    since = (gated["ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    symbols = sorted(gated["symbol"].unique())
    frames = {s: f for s in symbols if not (f := load_frame(s, since)).empty}
    print(f"loaded {len(frames)} price frames\n")

    payload = gated[["ts", "symbol", "side", "atr", "rank_score", "note"]].rename(
        columns={"rank_score": "rank"}
    )

    rows = []
    for slippage in (0.5, 1.0, 1.5, 2.0, 2.5):
        for trades_per_day in (2, 3, 4, 6):
            config = SimConfig(
                starting_capital=args.capital,
                max_concurrent_positions=min(3, trades_per_day),
                max_trades_per_day=trades_per_day,
                max_position_pct=33.0,          # ₹3.3L positions amortise the flat fees
                risk_per_trade_pct=1.0,
                scale_out_fraction=0.0,         # an extra leg costs more than it saves
                stop_atr=1.5, target_atr=3.0,
                breakeven_after_atr=0.9, trail_atr=0.0,
                max_hold_bars=12,
                slippage_bps_per_leg=slippage,
            )
            result = IntradayPortfolioSimulator(config).run(payload, frames)
            if result.trades.empty:
                continue
            trades = result.trades
            equity = result.equity["equity"]
            turnover = (trades["entry"] * trades["quantity"]).sum()
            rows.append({
                "slip/leg": slippage, "tr/day": trades_per_day,
                "positions": len(trades),
                "net": round(result.pnl), "net_%": round(result.pnl_pct, 2),
                "gross_bps": round(trades["gross_pnl"].sum() / max(turnover, 1) * 1e4, 1),
                "cost_bps": round(trades["costs"].sum() / max(turnover, 1) * 1e4, 1),
                "win_%": round((trades["net_pnl"] > 0).mean() * 100, 1),
                "dd_%": round((equity.cummax() - equity).max() / args.capital * 100, 2),
            })

    table = pd.DataFrame(rows)
    print("=" * 92)
    print("SLIPPAGE SENSITIVITY — where does this book break even?")
    print("=" * 92)
    print(table.to_string(index=False))

    best = table.sort_values("net", ascending=False).iloc[0]
    print(f"\nbest: {best['tr/day']:.0f} trades/day at {best['slip/leg']} bps/leg slippage "
          f"→ ₹{best['net']:+,.0f} ({best['net_%']:+.2f}%)")

    # Full detail on the realistic-slippage configuration.
    print("\n" + "=" * 92)
    print("DETAIL — 3 trades/day at 1.5 bps/leg slippage (a defensible large-cap assumption)")
    print("=" * 92)
    config = SimConfig(
        starting_capital=args.capital, max_concurrent_positions=3, max_trades_per_day=3,
        max_position_pct=33.0, risk_per_trade_pct=1.0, scale_out_fraction=0.0,
        stop_atr=1.5, target_atr=3.0, breakeven_after_atr=0.9, trail_atr=0.0,
        max_hold_bars=12, slippage_bps_per_leg=1.5,
    )
    result = IntradayPortfolioSimulator(config).run(payload, frames)
    print(result.summary())
    print(result.recency_split())
    if not result.trades.empty:
        print("\n  exits:")
        print(result.trades.groupby("exit_reason").agg(
            n=("net_pnl", "size"), net=("net_pnl", "sum"), avg=("net_pnl", "mean")
        ).round(0).to_string())
        print("\n  daily P&L:")
        print(result.daily.assign(cum=result.daily["pnl"].cumsum()).round(0).to_string())
        print("\n  sample trades:")
        cols = ["entry_time", "symbol", "side", "entry", "exit", "quantity",
                "gross_pnl", "costs", "net_pnl", "exit_reason"]
        print(result.trades[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
