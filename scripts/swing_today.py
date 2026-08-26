"""Pick today's intra-week candidate from NSE 500 or the commodity universe.

This is the *option* the intra-week feature exposes: one name, a stop, a size,
and a hold horizon.  It deliberately prints the measured expectancy of the rule
it is using alongside the pick, because on the ten-year test the honest answer
for NSE equities at a five-session hold is that this loses money.  A picker that
shows the trade without the evidence is how a research tool turns into a
recommendation it has not earned.

    python scripts/swing_today.py nse
    python scripts/swing_today.py commodity --strategy rsi_oversold
    python scripts/swing_today.py nse --hold 40      # the horizon that measured positive
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nse_intraday_ai.costs import Segment, segment_round_trip_bps  # noqa: E402
from nse_intraday_ai.swing import SCORES, SwingConfig, _position_size, build_panel  # noqa: E402
from swing_backtest import config_for, load_frames  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
OUT = ROOT / "data" / "swing_tickets.json"

# Measured on 10 years of daily bars, Groww costs, risk-based sizing.
# scripts/swing_backtest.py reproduces every number.
EXPECTANCY = {
    ("nse", 5): "-14.8% over 10y (1 pos, trend_pullback). EVERY factor tested "
                "was negative at a 5-session hold; NIFTY buy & hold returned "
                "+176% over the same window. Do not trade this for profit.",
    ("nse", 40): "+77.3% over 10y (1 pos, trend_pullback, 40-session hold) — "
                 "but that is a two-month position, not intra-week, and it "
                 "still lost to NIFTY buy & hold (+176%).",
    ("commodity", 5): "+55.2% over 10y (1 pos, rsi_oversold), 8/10 years "
                      "positive, max drawdown 17.6%. Bootstrap CI on bps/trade "
                      "still spans zero.",
}


def expectancy_note(universe: str, hold: int, strategy: str) -> str:
    key = (universe, hold)
    if key in EXPECTANCY:
        return EXPECTANCY[key]
    return (f"No stored measurement for {universe} at a {hold}-session hold. "
            f"Run: python scripts/swing_backtest.py {universe} --hold {hold}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("universe", choices=["nse", "commodity"])
    p.add_argument("--strategy", default=None,
                   help=f"one of: {', '.join(k for k in SCORES if k != 'random')}")
    p.add_argument("--hold", type=int, default=5)
    p.add_argument("--capital", type=float, default=10_00_000.0)
    p.add_argument("--risk-pct", type=float, default=2.0)
    p.add_argument("--top", type=int, default=5, help="candidates to show")
    args = p.parse_args()

    strategy = args.strategy or ("rsi_oversold" if args.universe == "commodity"
                                 else "trend_pullback")
    if strategy not in SCORES:
        raise SystemExit(f"unknown strategy {strategy!r}")

    frames = load_frames(args.universe)
    panel = build_panel(frames)
    if panel.empty:
        raise SystemExit("no daily data — run scripts/fetch_daily.py first")

    latest = panel.date.max()
    rows = panel[panel.date == latest]
    cfg = config_for(args.universe, hold_days=args.hold, positions=1,
                     capital=args.capital, risk_per_trade_pct=args.risk_pct)
    rows = rows[rows.turnover_20d >= cfg.min_turnover]
    if rows.empty:
        raise SystemExit("nothing liquid enough on the latest bar")

    ranked = rows.assign(_score=SCORES[strategy](rows)).dropna(subset=["_score"])
    ranked = ranked.sort_values("_score", ascending=False).head(args.top)

    now = datetime.now(IST)
    print(f"\nintra-week candidates — {args.universe}, {strategy}, "
          f"{args.hold}-session hold")
    print(f"data through {latest.date()}   generated {now:%Y-%m-%d %H:%M} IST")
    print("─" * 78)
    print(f"{'symbol':16s}{'close':>10s}{'stop':>10s}{'qty':>7s}"
          f"{'value':>12s}{'risk':>9s}{'cost':>8s}")

    tickets = []
    for _, r in ranked.iterrows():
        entry = float(r["close"])          # reference; the fill is tomorrow's open
        atr = float(r["atr"])
        qty = _position_size(entry, atr, cfg)
        if qty <= 0:
            continue
        stop = entry - cfg.stop_atr * atr
        value = qty * entry
        risk = qty * cfg.stop_atr * atr
        cost_bps = segment_round_trip_bps(
            entry, qty, segment=cfg.segment,
            slippage_bps_per_leg=cfg.slippage_bps_per_leg, symbol=r["symbol"])
        print(f"{r['symbol']:16s}{entry:10.2f}{stop:10.2f}{qty:7d}"
              f"{value:12,.0f}{risk:9,.0f}{cost_bps:7.1f}b")
        tickets.append({
            "symbol": r["symbol"], "side": "LONG",
            "reference_close": round(entry, 2),
            "fill": "next session's open — the stop is quoted off the FILL, not this close",
            "stop_atr": cfg.stop_atr, "stop_distance": round(cfg.stop_atr * atr, 2),
            "quantity": qty, "position_value": round(value, 2),
            "risk_rupees": round(risk, 2), "cost_bps": round(cost_bps, 1),
            "hold_sessions": args.hold, "score": round(float(r["_score"]), 4),
        })

    payload = {
        "generated_at": now.isoformat(timespec="seconds"),
        "universe": args.universe, "strategy": strategy,
        "hold_sessions": args.hold, "data_through": str(latest.date()),
        "capital": args.capital, "segment": cfg.segment.value,
        "expectancy": expectancy_note(args.universe, args.hold, strategy),
        "tickets": tickets,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print("─" * 78)
    print(f"measured expectancy:\n  {payload['expectancy']}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
