"""Live Intraday Trading Assistant for Today's NSE Session.

Monitors the market, executes real-time scans on closed 5-minute bars,
and outputs high-conviction institutional trade tickets with exact entry,
stop loss, breakeven, profit lock, target, and position sizing.

Usage:
    python scripts/live_trader_today.py             # continuous live monitor
    python scripts/live_trader_today.py --scan-now   # run single live scan immediately
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nse_intraday_ai.scan_service import run_scan_cycle
from nse_intraday_ai.data import GoogleFinanceQuoteClient
from nse_intraday_ai import execution_plan as EP
from nse_intraday_ai.signal_model import load_if_available

IST = ZoneInfo("Asia/Kolkata")


def print_banner():
    now = datetime.now(IST)
    print("\n" + "═" * 80)
    print("  🚀 NSE INTRADAY QUANTITATIVE TRADING ENGINE — LIVE ASSISTANT")
    print(f"  📅 Date: {now.strftime('%A, %d %B %Y')} | ⏰ Current IST Time: {now.strftime('%H:%M:%S')}")
    print("  🧠 Model: Stacked Ensemble (RF + HGB) | Alpha Ranker: Profit Factor 2.52")
    print("  🛡️  Risk Ratchet: SL=2.0 ATR | Breakeven=+1.2 ATR | Profit Lock=+2.5 ATR | Target=+5.0 ATR")
    print("═" * 80 + "\n")


def display_tickets(cycle, min_confidence: float = 65.0):
    recs = [r for r in cycle.recommendations if r.plan.confidence >= min_confidence]
    if not recs:
        print(f"[{datetime.now(IST).strftime('%H:%M:%S')}] No high-conviction actionable tickets passing entry filters yet.")
        print(f"  Total raw signals scanned: {len(cycle.recommendations) if cycle.recommendations else 0}")
        return

    print("\n" + "─" * 80)
    print(f"🔥 TOP ACTIONABLE TRADE TICKETS ({len(recs)} signals found)")
    print("─" * 80)

    for i, r in enumerate(recs[:3], 1):
        plan = r.plan
        side_emoji = "🟢 BUY (LONG)" if plan.side.value == "LONG" else "🔴 SELL (SHORT)"
        atr = plan.atr if plan.atr and plan.atr > 0 else (plan.entry * 0.01)
        be_level = plan.entry + (1.2 * atr if plan.side.value == "LONG" else -1.2 * atr)
        lock_level = plan.entry + (2.5 * atr if plan.side.value == "LONG" else -2.5 * atr)

        print(f"\n【 TICKET #{i} 】 {r.symbol} — {side_emoji}")
        print(f"  • Regime:        {r.regime}")
        print(f"  • Confidence:    {plan.confidence:.1f}% | Reward/Risk: {plan.reward_risk:.2f}")
        print(f"  • Alpha Score:   {getattr(r, 'signal_model_bps', 0.0):+.2f} pts")
        print(f"  • Strategies:    {', '.join(v.strategy for v in plan.strategy_votes if v.is_trade and v.side == plan.side)}")
        print(f"  ──────────────────────────────────────────────────────────")
        print(f"  🎯 ENTRY PRICE:  ₹{plan.entry:.2f}")
        print(f"  🛑 STOP LOSS:    ₹{plan.stop_loss:.2f}  (Risk: ₹{abs(plan.entry - plan.stop_loss):.2f} / share)")
        print(f"  🔄 BREAKEVEN AT: ₹{be_level:.2f}  (Move SL to entry when price touches here)")
        print(f"  🔒 LOCK PROFIT:  ₹{lock_level:.2f}  (Lock +1.5 ATR when price touches here)")
        print(f"  🏁 TARGET:       ₹{plan.target:.2f}  (Reward: ₹{abs(plan.target - plan.entry):.2f} / share)")
        print(f"  📦 QUANTITY:     {plan.quantity or 100} shares (Position: ₹{((plan.quantity or 100) * plan.entry):,.0f})")
        print(f"  ⏱️ MAX HOLD:     {EP.MAX_HOLD_MINUTES} mins (or mandatory exit at 15:15 IST)")
    print("─" * 80 + "\n")


def live_loop(capital: float = 10_00_000.0, poll_interval_sec: int = 60):
    print_banner()
    print(f"Target Capital: ₹{capital:,.0f} | Risk per trade: 1.0% (₹{capital * 0.01:,.0f})")
    print("Starting automated 5-minute candle monitor loop...\n")

    while True:
        now = datetime.now(IST)
        m = now.hour * 60 + now.minute

        # Check market hours (09:15 to 15:30 IST)
        if m < 555:
            mins_left = 555 - m
            print(f"\r⏳ Market opens at 09:15 IST ({mins_left} mins remaining). Standing by...", end="", flush=True)
            time.sleep(15)
            continue
        elif m > 930:
            print(f"\n🔔 Market is closed for today ({now.strftime('%H:%M:%S IST')}). See you tomorrow!")
            break

        print(f"[{now.strftime('%H:%M:%S')}] Fetching latest market candles and running scan cycle...")
        try:
            cycle = run_scan_cycle("nse", source="cli_trader", quote_client_factory=GoogleFinanceQuoteClient)
            display_tickets(cycle)
        except Exception as e:
            print(f"⚠️ Scan error: {e}")

        # Sleep until next scan check
        time.sleep(poll_interval_sec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-now", action="store_true", help="Run a single scan cycle immediately")
    parser.add_argument("--capital", type=float, default=10_00_000.0, help="Trading capital in INR")
    parser.add_argument("--poll", type=int, default=60, help="Polling interval in seconds")
    args = parser.parse_args()

    if args.scan_now:
        print_banner()
        print(f"Running single real-time scan for ₹{args.capital:,.0f} capital...")
        cycle = run_scan_cycle("nse", source="cli_trader", quote_client_factory=GoogleFinanceQuoteClient)
        display_tickets(cycle)
    else:
        live_loop(capital=args.capital, poll_interval_sec=args.poll)


if __name__ == "__main__":
    main()
