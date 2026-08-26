"""What does removing the daily trade cap actually do?

The cap was not an arbitrary conservatism — it came out of the design sweep,
where more trades a day was monotonically worse. But "fewer trades" and "more
trades" trade off against two different things, and the sweep only varied one:

  * a daily cap limits *how many* signals get taken;
  * `max_position_pct` limits *how many can be open at once*, because ₹10L at
    33% a position only fits three.

So "no cap" has two readings — keep taking trades as slots free up (same size,
more turnover through the day), or hold more, smaller positions at once. Both
are measured here against the capped baseline, on the same signals and the same
sessions, so the cost of the choice is visible rather than argued.

Usage:
    python scripts/sim_nocap.py
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
    parser.add_argument("--slippage", type=float, default=1.5)
    args = parser.parse_args()

    predictions = pd.read_parquet(PREDICTIONS)
    print(f"{len(predictions):,} model-scored signals over "
          f"{predictions['ts'].dt.date.nunique()} held-out sessions\n")

    since = (predictions["ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    symbols = sorted(predictions["symbol"].unique())
    frames = {s: f for s in symbols if not (f := load_frame(s, since)).empty}

    payload = predictions[["ts", "symbol", "side", "atr"]].copy()
    payload["rank"] = predictions["pred_bps"].to_numpy()
    payload["note"] = ""

    # (label, daily cap, concurrent, position %)
    variants = [
        ("capped: 3/day, 3 open @33%", 3, 3, 33.0),
        ("no cap, 3 open @33%", 0, 3, 33.0),
        ("no cap, 5 open @20%", 0, 5, 20.0),
        ("no cap, 8 open @12%", 0, 8, 12.0),
        ("no cap, 12 open @8%", 0, 12, 8.0),
        ("no cap, 20 open @5%", 0, 20, 5.0),
    ]

    rows = []
    for label, cap, concurrent, position_pct in variants:
        config = SimConfig(
            starting_capital=args.capital,
            max_concurrent_positions=concurrent,
            max_trades_per_day=cap,
            max_position_pct=position_pct,
            risk_per_trade_pct=1.0,
            scale_out_fraction=0.0, stop_atr=1.5, target_atr=3.0,
            breakeven_after_atr=0.9, trail_atr=0.0, max_hold_bars=12,
            slippage_bps_per_leg=args.slippage,
        )
        result = IntradayPortfolioSimulator(config).run(payload, frames)
        if result.trades.empty:
            rows.append({"variant": label, "positions": 0})
            continue
        trades = result.trades
        equity = result.equity["equity"]
        turnover = (trades["entry"] * trades["quantity"]).sum()
        daily = result.daily["pnl"]
        rows.append({
            "variant": label,
            "positions": len(trades),
            "per_day": round(len(trades) / max(len(daily), 1), 1),
            "net_%": round(result.pnl_pct, 2),
            "net": round(result.pnl),
            "gross_bps": round(trades["gross_pnl"].sum() / max(turnover, 1) * 1e4, 1),
            "cost_bps": round(trades["costs"].sum() / max(turnover, 1) * 1e4, 1),
            "costs": round(trades["costs"].sum()),
            "win_%": round((trades["net_pnl"] > 0).mean() * 100, 1),
            "sessions_up": f"{int((daily > 0).sum())}/{len(daily)}",
            "dd_%": round((equity.cummax() - equity).max() / args.capital * 100, 2),
        })

    print("=" * 118)
    print(f"REMOVING THE DAILY CAP — ₹{args.capital:,.0f}, {args.slippage} bps/leg slippage, "
          f"same signals and sessions throughout")
    print("=" * 118)
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\nRead the cost_bps column against gross_bps. Smaller positions do not change\n"
        "the signal — they change how much of it survives the ₹20-per-order floor."
    )


if __name__ == "__main__":
    main()
