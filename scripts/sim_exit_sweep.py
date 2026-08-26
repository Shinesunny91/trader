"""Were today's losses bad signals or bad exits?

On 2026-08-12 the book shorted CARTRADE, ANGELONE and KPITTECH at 09:20. NIFTY
closed the morning −0.54%, CARTRADE fell 1.79% and KPITTECH fell 1.88% — the
direction calls were right. The book still lost money, because two of the three
were stopped out on the *entry bar* before the move went their way.

That is a specific, testable claim: the 1.5 ATR stop is being taken out by
noise on trades whose direction was correct. This sweeps stop and target width
(and a no-entry-on-the-opening-bar variant) over the model-ranked book, on the
same held-out sessions, so the claim is checked rather than assumed.

The trap to avoid: a wider stop always looks better on a sample where the
direction happened to be right. Watch drawdown and the session-up count, not
just net.

Usage:
    python scripts/sim_exit_sweep.py
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
    since = (predictions["ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = {
        s: f for s in sorted(predictions["symbol"].unique())
        if not (f := load_frame(s, since)).empty
    }
    base = predictions[["ts", "symbol", "side", "atr"]].copy()
    base["rank"] = predictions["pred_bps"].to_numpy()
    base["note"] = ""
    # Variant that simply refuses the session's first bar.
    no_open = base[base["ts"].dt.time > pd.Timestamp("09:20").time()]

    print(f"{len(base):,} model-scored signals, "
          f"{predictions['ts'].dt.date.nunique()} held-out sessions")
    print(f"({len(base) - len(no_open):,} of them fire on the opening bar)\n")

    rows = []
    for label, payload in (("all bars", base), ("skip 09:15/09:20", no_open)):
        for stop_atr in (1.5, 2.0, 2.5, 3.0):
            for reward in (2.0, 2.5, 3.0):
                config = SimConfig(
                    starting_capital=args.capital,
                    max_concurrent_positions=3, max_trades_per_day=3,
                    max_position_pct=33.0, risk_per_trade_pct=1.0,
                    scale_out_fraction=0.0,
                    stop_atr=stop_atr, target_atr=stop_atr * reward,
                    breakeven_after_atr=stop_atr * 0.6, trail_atr=0.0,
                    max_hold_bars=12, slippage_bps_per_leg=args.slippage,
                )
                result = IntradayPortfolioSimulator(config).run(payload, frames)
                if result.trades.empty:
                    continue
                trades = result.trades
                equity = result.equity["equity"]
                daily = result.daily["pnl"]
                exits = trades["exit_reason"].value_counts()
                rows.append({
                    "entries": label,
                    "stop": stop_atr, "R": reward,
                    "target": round(stop_atr * reward, 2),
                    "pos": len(trades),
                    "net_%": round(result.pnl_pct, 2),
                    "win_%": round((trades["net_pnl"] > 0).mean() * 100, 1),
                    "stopped": int(exits.get("STOP", 0)),
                    "targets": int(exits.get("TARGET", 0)),
                    "sessions_up": f"{int((daily > 0).sum())}/{len(daily)}",
                    "dd_%": round((equity.cummax() - equity).max() / args.capital * 100, 2),
                })

    table = pd.DataFrame(rows)
    print("=" * 108)
    print("STOP / TARGET WIDTH — model-ranked book, same signals throughout")
    print("=" * 108)
    print(table.sort_values("net_%", ascending=False).to_string(index=False))

    print("\n" + "=" * 108)
    print("Does a wider stop help on its own?  (mean across targets, all-bars entries)")
    print("=" * 108)
    grouped = table[table["entries"] == "all bars"].groupby("stop").agg(
        mean_net=("net_%", "mean"), best_net=("net_%", "max"),
        mean_dd=("dd_%", "mean"), stops=("stopped", "mean"), targets=("targets", "mean"),
    ).round(2)
    print(grouped.to_string())

    print("\nSkipping the opening bar:")
    for label in ("all bars", "skip 09:15/09:20"):
        subset = table[table["entries"] == label]
        print(f"  {label:20s} mean {subset['net_%'].mean():+6.2f}%  "
              f"best {subset['net_%'].max():+6.2f}%  "
              f"mean dd {subset['dd_%'].mean():.2f}%")


if __name__ == "__main__":
    main()
