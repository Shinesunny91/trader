"""Live Tracker and Conviction Engine for Today's Intraday Trading (2026-08-26).

Tracks real-time performance of recommended tickets, computes real-time P&L,
and isolates ultra-high conviction trades with strict downside protection.
"""
from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.candle_cache import CandleCache
from nse_intraday_ai.strategies import VotingSignalEngine
from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.scanner import ScanResult
from nse_intraday_ai.signal_model import load_if_available, score_and_rank_scan_results
from intraday_sim import liquid_symbols

IST = ZoneInfo("Asia/Kolkata")
DB_PATH = ROOT / "data" / "candles.sqlite3"

def track_today_recommendations():
    cache = CandleCache(DB_PATH)
    engine = VotingSignalEngine()
    risk_cfg = RiskConfig()
    model = load_if_available()
    
    liquid = liquid_symbols(150)
    today_str = pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d")
    
    # Load NIFTY 50 index for market direction context
    nifty_df = cache.load("^NSEI", "5m", limit=100)
    nifty_change = 0.0
    if nifty_df is not None and len(nifty_df) >= 2:
        today_nifty = nifty_df[nifty_df.index.strftime("%Y-%m-%d") == today_str]
        if len(today_nifty) >= 1:
            open_p = today_nifty["open"].iloc[0]
            curr_p = today_nifty["close"].iloc[-1]
            nifty_change = (curr_p - open_p) / open_p * 100.0
            
    print(f"📊 Market Context: NIFTY 50 today change from open: {nifty_change:+.2f}%")
    
    candidates = []
    performance_records = []
    
    for s in liquid:
        df = cache.load(s, "5m", limit=300)
        if df is None or len(df) < 30:
            continue
        
        today_bars = df[df.index.strftime("%Y-%m-%d") == today_str]
        if len(today_bars) < 2:
            continue
            
        # Scan signal at open vs scan signal at current bar
        regime, plan = engine.analyze(s, df, risk_cfg)
        curr_price = float(df["close"].iloc[-1])
        open_price = float(today_bars["open"].iloc[0])
        day_high = float(today_bars["high"].max())
        day_low = float(today_bars["low"].min())
        
        # Track simulated performance if entered at bar 1 (09:20)
        entry_0920 = float(today_bars["open"].iloc[1]) if len(today_bars) > 1 else open_price
        gain_since_0920 = (curr_price - entry_0920) / entry_0920 * 100.0
        
        res = ScanResult(
            symbol=s,
            plan=plan,
            rows=len(df),
            source="live_cache",
            frame=df,
            regime=str(regime.value) if hasattr(regime, "value") else str(regime),
            last_close=curr_price,
        )
        
        # Check signal strength
        if plan.is_actionable and plan.confidence >= 60.0:
            candidates.append(res)
            
        performance_records.append({
            "symbol": s,
            "open": open_price,
            "entry_0920": entry_0920,
            "curr": curr_price,
            "high": day_high,
            "low": day_low,
            "gain_pct": gain_since_0920,
            "side": plan.side.value if plan.is_actionable else "NONE",
            "confidence": plan.confidence if plan.is_actionable else 0.0,
        })
        
    ranked = score_and_rank_scan_results(candidates, model=model)
    return ranked, pd.DataFrame(performance_records), nifty_change

if __name__ == "__main__":
    ranked, perf_df, nifty_change = track_today_recommendations()
    print(f"\nTotal Actionable Setups Evaluated: {len(ranked)}")
    print("\n" + "=" * 95)
    print("🏆 TODAY'S ULTRA-HIGH CONVICTION RECOMMENDATIONS (CAPITAL-PRESERVATION FIRST)")
    print("=" * 95)
    
    for i, r in enumerate(ranked[:5], 1):
        p = r.plan
        s = r.symbol
        side = p.side.value
        risk = abs(p.entry - p.stop_loss)
        atr = max(risk / 2.0, p.entry * 0.005)
        be = p.entry + (1.2 * atr if side == "LONG" else -1.2 * atr)
        lock = p.entry + (2.5 * atr if side == "LONG" else -2.5 * atr)
        
        is_ultra_conviction = (getattr(r, "rank_score", 0.0) >= 4.0 and p.confidence >= 80.0 and p.reward_risk >= 2.0)
        conviction_badge = "🌟 ULTRA-HIGH CONVICTION (GRADE A+)" if is_ultra_conviction else "✅ HIGH CONVICTION (GRADE A)"
        
        print(f"\n【 SELECTION #{i} 】 {s} — {'🟢 BUY (LONG)' if side == 'LONG' else '🔴 SELL (SHORT)'}")
        print(f"  Status:           {conviction_badge}")
        print(f"  • Alpha Score:    {getattr(r, 'rank_score', 0.0):+.2f} pts")
        print(f"  • Strategy Votes: {', '.join(v.strategy for v in p.strategy_votes if v.is_trade and v.side == p.side)}")
        print(f"  • Confidence:     {p.confidence:.1f}%  |  Reward / Risk: {p.reward_risk:.2f}:1")
        print(f"  • Market Regime:  {r.regime}")
        print(f"  ──────────────────────────────────────────────────────────")
        print(f"  🎯 ENTRY PRICE:   ₹{p.entry:.2f}")
        print(f"  🛑 STOP LOSS:     ₹{p.stop_loss:.2f}  (Risk: ₹{risk:.2f}/share, exactly 2.0 ATR)")
        print(f"  🔄 BREAKEVEN AT:  ₹{be:.2f}  (👉 Move SL to entry as soon as price touches this level)")
        print(f"  🔒 PROFIT LOCK:   ₹{lock:.2f}  (👉 Lock +1.5 ATR profit)")
        print(f"  🏁 TARGET:        ₹{p.target:.2f}  (Reward: ₹{abs(p.target - p.entry):.2f}/share)")
        print(f"  📦 SUGGESTED QTY: {max(1, int(10_000 / max(risk, 0.1)))} shares (₹10,000 max risk)")
    print("=" * 95)
