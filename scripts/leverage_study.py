"""What 5x intraday margin actually buys — priced with market impact, not without it.

The account this repo trades gets roughly 5x MIS leverage on liquid NSE
equities, and the reasonable-sounding conclusion is that a small percentage
edge is fine because leverage multiplies it.  The arithmetic of the existing
simulator agrees enthusiastically, and that agreement is the problem: every
cost in it is either a flat fee or a constant bps, so multiplying position size
multiplies gross P&L and cost by the same factor and the net return on equity
scales linearly, forever, with nothing pushing back.  Under that model 20x
would be better than 5x and 100x better still.

Reality pushes back through market impact.  The live book's ₹3,30,000 position
is about 1.4% of a median liquid name's 5-minute bar; the ₹16,50,000 position
5x margin allows is about 7%, and an order that large does not get the price
the backtest fills it at.  `liquidity.py` prices that with the square-root law
and `portfolio_sim` now charges it, so this study can ask the question honestly:

  1. Does the measured edge survive being scaled up?
  2. Is margin better spent on **size** (one bigger position) or on **breadth**
     (more names at today's size)?
  3. At what impact coefficient does leverage stop paying?  That is a
     sensitivity, not a forecast — the coefficient is an assumption, so the
     result is reported across a range rather than at one flattering value.

Usage:
    python scripts/leverage_study.py
    python scripts/leverage_study.py --coefficients 0,0.5,1.0,1.5
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
from nse_intraday_ai.portfolio_sim import IntradayPortfolioSimulator, SimConfig  # noqa: E402

PREDICTIONS = ROOT / "data" / "model_predictions.parquet"
CAPITAL = 10_00_000.0


def book(
    *,
    leverage: float = 1.0,
    concurrent: int = 1,
    trades_per_day: int = 1,
    impact: float = 0.0,
    participation_pct: float = 0.0,
) -> SimConfig:
    """The live book, scaled.

    `leverage` scales the three limits that decide size together — per-name
    cap, gross exposure and the risk budget — because scaling only some of them
    does not produce a 5x book, it produces whichever limit was left behind.
    That was the trap in the first version of this study: raising
    `max_position_pct` alone left `risk_per_trade_pct` binding at 1%, so
    "5x" quietly traded at 1.9x and looked wonderfully well-behaved.
    """
    return SimConfig(
        starting_capital=CAPITAL,
        max_concurrent_positions=concurrent,
        max_trades_per_day=trades_per_day,
        max_position_pct=EP.MAX_POSITION_PCT * leverage,
        max_gross_exposure_pct=100.0 * leverage,
        risk_per_trade_pct=1.0 * leverage,
        scale_out_fraction=0.0,
        stop_atr=EP.STOP_ATR,
        target_atr=EP.TARGET_ATR,
        breakeven_after_atr=EP.BREAKEVEN_ATR,
        trail_atr=0.0,
        max_hold_bars=EP.MAX_HOLD_MINUTES // 5,
        slippage_bps_per_leg=1.5,
        impact_coefficient=impact,
        max_participation_pct=participation_pct,
        # A levered book hits a percentage loss limit five times sooner on the
        # same price move, so the limit is scaled with it; leaving it at 2%
        # would halt the 5x book on a move the 1x book trades through, and the
        # comparison would be of two different strategies.
        daily_loss_limit_pct=2.0 * leverage,
    )


def bootstrap_ci(values: np.ndarray, iters: int = 20_000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iters, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def describe(result, label: str) -> dict:
    if result.trades.empty:
        return {"book": label, "trades": 0}
    trades = result.trades
    wins = trades[trades["net_pnl"] > 0]["net_pnl"].sum()
    losses = abs(trades[trades["net_pnl"] <= 0]["net_pnl"].sum())
    equity = result.equity["equity"]
    daily = result.daily["pnl"].to_numpy()
    lo, hi = bootstrap_ci(daily)
    turnover = (trades["entry"] * trades["quantity"]).sum()
    return {
        "book": label,
        "trades": len(trades),
        "net_%": round(result.pnl_pct, 2),
        "net_₹": round(result.pnl),
        "win_%": round((trades["net_pnl"] > 0).mean() * 100, 1),
        "pf": round(wins / losses, 2) if losses else float("inf"),
        "maxDD_%": round((equity.cummax() - equity).max() / CAPITAL * 100, 2),
        "peak_x": round(result.peak_exposure_x, 2),
        "sess_up": f"{int((daily > 0).sum())}/{len(daily)}",
        # The edge per rupee traded. Leverage cannot change this; if it falls
        # as size rises, impact is eating the strategy rather than the account
        # simply doing more of it.
        "bps/₹traded": round(result.pnl / turnover * 1e4, 2) if turnover else 0.0,
        "CI_₹/sess": f"[{lo:+,.0f}, {hi:+,.0f}]",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", default="0,0.5,1.0",
                        help="comma-separated impact coefficients to sweep")
    parser.add_argument("--leverages", default="1,2,3,5")
    args = parser.parse_args()

    coefficients = [float(c) for c in args.coefficients.split(",")]
    leverages = [float(x) for x in args.leverages.split(",")]

    frame = pd.read_parquet(PREDICTIONS)
    frame["day"] = frame["ts"].dt.normalize()
    print(f"{len(frame):,} out-of-sample scored signals over {frame['day'].nunique()} "
          f"held-out sessions ({frame['day'].min().date()} .. {frame['day'].max().date()})")

    symbols = sorted(frame["symbol"].unique())
    since = (frame["ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"loading {len(symbols)} price frames...", flush=True)
    frames = {s: f for s in symbols if not (f := load_frame(s, since)).empty}

    payload = frame[["ts", "symbol", "side", "atr"]].copy()
    payload["rank"] = frame["pred_bps"].to_numpy()
    payload["note"] = ""

    def run(cfg: SimConfig, label: str) -> dict:
        return describe(IntradayPortfolioSimulator(cfg).run(payload, frames), label)

    print("\n" + "=" * 132)
    print("1. SIZE — the same one-trade-a-day book, scaled.  Does the edge survive being")
    print("   made bigger, once the bar has to absorb it?")
    print("=" * 132)
    for coefficient in coefficients:
        rows = [run(book(leverage=lev, impact=coefficient), f"{lev:g}x")
                for lev in leverages]
        tag = ("constant slippage (what every earlier study assumed)"
               if coefficient == 0 else f"impact coefficient {coefficient:g}")
        print(f"\n  -- {tag} --")
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 132)
    print("2. BREADTH vs SIZE — spending the same 5x of margin two ways: one position at")
    print("   5x, or five positions at 1x each.")
    print("=" * 132)
    for coefficient in coefficients:
        rows = [
            run(book(leverage=1, concurrent=1, trades_per_day=1, impact=coefficient),
                "1 name @ 1x  (live book)"),
            run(book(leverage=5, concurrent=1, trades_per_day=1, impact=coefficient),
                "1 name @ 5x  (size)"),
            run(book(leverage=5, concurrent=3, trades_per_day=3, impact=coefficient),
                "3 names @ 1.7x each"),
            run(book(leverage=5, concurrent=5, trades_per_day=5, impact=coefficient),
                "5 names @ 1x each (breadth)"),
        ]
        tag = "constant slippage" if coefficient == 0 else f"impact coefficient {coefficient:g}"
        print(f"\n  -- {tag} --")
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 132)
    print("3. THE PARTICIPATION CAP — refusing to be more than 1% of the bar, at 5x.")
    print("=" * 132)
    rows = []
    for coefficient in coefficients:
        for cap in (0.0, 2.0, 1.0, 0.5):
            label = f"coef {coefficient:g}, cap {'off' if cap == 0 else f'{cap:g}%'}"
            rows.append(run(book(leverage=5, impact=coefficient, participation_pct=cap), label))
    print(pd.DataFrame(rows).to_string(index=False))

    print("""
Reading this table
------------------
`bps/₹traded` is the number leverage cannot flatter: it is the edge per rupee
put at risk, and it is identical across leverages only when impact is switched
off.  Where it falls as leverage rises, the account is buying less edge with
each additional rupee, and `net_%` overstates what more margin is worth.
`peak_x` is the margin the book actually required — if it is far below the
leverage requested, some other limit bound first and the row is not the book
its label claims.""")


if __name__ == "__main__":
    main()
