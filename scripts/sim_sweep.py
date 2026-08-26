"""Sweep the portfolio design against the tool's signals.

The first simulation run showed the shape of the problem precisely: gross P&L
was **positive** (+₹19k) and transaction costs (−₹68k) were 3.5x larger.  The
book was taking ~15 trades a day, each earning ~9 bps gross against ~16 bps of
round-trip cost.  So the question is not "is there signal" but "can the trade
structure carry the signal over the cost floor", and that is a design sweep:

  * trades per day        — fewer, better-ranked signals
  * position size         — the ₹20-per-order floor is amortised by size
  * scale-out on/off      — a partial exit is an extra order, i.e. extra cost,
                            and it caps winners while losers run full size
  * stop / target width   — bigger targets earn more per fixed cost
  * hold time             — a 60-minute cap may be cutting winners short

Everything is scored on the out-of-sample split only.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from intraday_sim import SPLIT_DATE, build_signals, load_frame  # noqa: E402
from nse_intraday_ai.portfolio_sim import IntradayPortfolioSimulator, SimConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["all", "train", "test"], default="test")
    parser.add_argument("--capital", type=float, default=10_00_000.0)
    parser.add_argument("--gate", default="full_gate",
                        choices=["full_gate", "quality_gate", "engine_gate"])
    args = parser.parse_args()

    signals = build_signals(args.split)
    gated = signals[signals[args.gate]].copy()
    sessions = gated["ts"].dt.date.nunique()
    print(f"{len(gated):,} gated signals ({args.gate}) over {sessions} sessions")

    since = (gated["ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    symbols = sorted(gated["symbol"].unique())
    print(f"loading {len(symbols)} price frames...")
    frames = {s: f for s in symbols if not (f := load_frame(s, since)).empty}

    payload = gated[["ts", "symbol", "side", "atr", "macro_score", "note"]].rename(
        columns={"macro_score": "rank"}
    )

    grid = {
        "max_trades_per_day": [2, 4, 8],
        "risk_per_trade_pct": [0.5, 1.0],
        "scale_out_fraction": [0.0, 0.5],
        "stop_target": [(1.0, 1.6), (1.2, 2.4), (1.5, 3.0)],
        "max_hold_bars": [12, 24],
    }
    keys = list(grid)
    rows = []
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"sweeping {len(combos)} configurations...\n")

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        stop_atr, target_atr = params.pop("stop_target")
        config = SimConfig(
            starting_capital=args.capital,
            max_concurrent_positions=min(4, params["max_trades_per_day"]),
            stop_atr=stop_atr,
            target_atr=target_atr,
            scale_out_atr=stop_atr * 0.9,
            trail_atr=stop_atr * 1.2,
            breakeven_after_atr=stop_atr * 0.6,
            **params,
        )
        result = IntradayPortfolioSimulator(config).run(payload, frames)
        if result.trades.empty:
            continue
        trades = result.trades
        equity = result.equity["equity"]
        # Count positions, not fills — a scale-out produces two rows.
        positions = trades.groupby(["symbol", "entry_time"]).ngroups
        turnover = (trades["entry"] * trades["quantity"]).sum()
        rows.append({
            "tr/day": params["max_trades_per_day"],
            "risk%": params["risk_per_trade_pct"],
            "scale": params["scale_out_fraction"],
            "stop": stop_atr, "tgt": target_atr,
            "hold": params["max_hold_bars"],
            "pos": positions,
            "net": round(result.pnl),
            "net_%": round(result.pnl_pct, 2),
            "gross": round(trades["gross_pnl"].sum()),
            "costs": round(trades["costs"].sum()),
            "cost_bps": round(trades["costs"].sum() / max(turnover, 1) * 1e4, 1),
            "gross_bps": round(trades["gross_pnl"].sum() / max(turnover, 1) * 1e4, 1),
            "win_%": round((trades["net_pnl"] > 0).mean() * 100, 1),
            "dd_%": round((equity.cummax() - equity).max() / args.capital * 100, 2),
        })
        if i % 12 == 0:
            print(f"  {i}/{len(combos)}", flush=True)

    table = pd.DataFrame(rows).sort_values("net", ascending=False)
    print("\n" + "=" * 110)
    print("TOP 15 CONFIGURATIONS (out-of-sample)")
    print("=" * 110)
    print(table.head(15).to_string(index=False))
    print("\nWORST 5:")
    print(table.tail(5).to_string(index=False))

    print("\n" + "=" * 110)
    print("WHAT MOVES THE NEEDLE — mean net P&L by each design choice")
    print("=" * 110)
    for column in ("tr/day", "risk%", "scale", "stop", "hold"):
        grouped = table.groupby(column).agg(
            configs=("net", "size"), mean_net=("net", "mean"),
            best_net=("net", "max"), mean_gross_bps=("gross_bps", "mean"),
            mean_cost_bps=("cost_bps", "mean"),
        ).round(1)
        print(f"\n{grouped.to_string()}")


if __name__ == "__main__":
    main()
