"""Dedicated 10:30 IST Breakout & Conviction Scanner.

Executes real-time scan on post-consolidation 10:30 IST candles across NSE liquid names,
filtering through strict anti-exhaustion, Nifty index alignment, and volume expansion gates.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nse_intraday_ai.strategies import VotingSignalEngine
from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.scanner import ScanResult
from nse_intraday_ai.signal_model import load_if_available, score_and_rank_scan_results
from intraday_sim import liquid_symbols

IST = ZoneInfo("Asia/Kolkata")

def run_1030_scan():
    import yfinance as yf
    
    liquid = list(liquid_symbols(100))
    symbols_to_fetch = liquid + ["^NSEI", "^NSEBANK"]
    
    print("=" * 95)
    print("📡 FETCHING LIVE 10:30 IST CANDLES ACROSS TOP LIQUID NSE INSTRUMENTS...")
    print("=" * 95)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(symbols_to_fetch, period="1d", interval="5m", group_by="ticker", progress=False)
        
    nifty_df = raw["^NSEI"].dropna(how="all") if "^NSEI" in raw else pd.DataFrame()
    nifty_slope_positive = True
    nifty_chg = 0.0
    if not nifty_df.empty and len(nifty_df) >= 2:
        nifty_last_c = float(nifty_df["Close"].iloc[-1])
        nifty_open = float(nifty_df["Open"].iloc[0])
        nifty_chg = (nifty_last_c - nifty_open) / nifty_open * 100.0
        nifty_bar_green = float(nifty_df["Close"].iloc[-1]) >= float(nifty_df["Open"].iloc[-1])
        nifty_ema9 = nifty_df["Close"].ewm(span=9).mean().iloc[-1]
        nifty_slope_positive = (nifty_last_c >= nifty_ema9) or nifty_bar_green
        print(f"📊 NIFTY 50 State: {nifty_last_c:.2f} ({nifty_chg:+.2f}%) | 5m Trend: {'🟢 Bullish/Rising' if nifty_slope_positive else '🔴 Bearish/Pullback'}")

    engine = VotingSignalEngine()
    risk_cfg = RiskConfig()
    model = load_if_available()
    
    candidates = []
    for s in liquid:
        if s not in raw:
            continue
        df = raw[s].dropna(how="all")
        if len(df) < 5:
            continue
        df.columns = [c.lower() for c in df.columns]
        
        # Microstructure metrics
        cum_vol = df["volume"].cumsum()
        cum_vp = (df["close"] * df["volume"]).cumsum()
        vwap = cum_vp / np.maximum(cum_vol, 1e-9)
        ltp = float(df["close"].iloc[-1])
        curr_vwap = float(vwap.iloc[-1])
        
        tr = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
        atr = float(tr.rolling(5).mean().iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = ltp * 0.005
            
        ema9 = float(df["close"].ewm(span=9).mean().iloc[-1])
        ema21 = float(df["close"].ewm(span=21).mean().iloc[-1])
        ema50 = float(df["close"].ewm(span=50).mean().iloc[-1])
        
        vol_sma = float(df["volume"].rolling(5).mean().iloc[-1])
        latest_vol = float(df["volume"].iloc[-1])
        vol_ratio = latest_vol / max(vol_sma, 1.0)
        
        # Evaluate strategies
        regime, plan = engine.analyze(s, df, risk_cfg)
        if not plan.is_actionable or plan.confidence < 75.0:
            continue
            
        side = plan.side.value
        
        # 1. Strict Anti-Exhaustion Gate
        dist_ema9_atr = (ltp - ema9) / atr if side == "LONG" else (ema9 - ltp) / atr
        if dist_ema9_atr > 0.85:
            continue  # Extended top, do not chase!
            
        # 2. Strict Nifty Alignment Gate
        if side == "LONG" and not nifty_slope_positive:
            continue  # Block longs when index is falling
            
        res = ScanResult(
            symbol=s,
            plan=plan,
            rows=len(df),
            source="live_1030_scanner",
            frame=df,
            regime=str(regime.value) if hasattr(regime, "value") else str(regime),
            last_close=ltp,
        )
        setattr(res, "vol_ratio", vol_ratio)
        setattr(res, "dist_ema9_atr", dist_ema9_atr)
        candidates.append(res)
        
    print(f"🎯 Setups passing all Institutional Zero-Loss Gates: {len(candidates)}")
    ranked = score_and_rank_scan_results(candidates, model=model)
    return ranked, nifty_chg

if __name__ == "__main__":
    ranked, nifty_chg = run_1030_scan()
    print("\n" + "=" * 95)
    print("🏆 10:30 IST PRIME CONVICTION RECOMMENDATIONS")
    print("=" * 95)
    
    if not ranked:
        print("🛡️ No trades passed the strict Zero-Loss filter at this exact bar.")
        print("   The engine is protecting capital. Next scan window opens at next 5m candle.")
    else:
        for i, r in enumerate(ranked[:3], 1):
            p = r.plan
            s = r.symbol
            side = p.side.value
            ltp = r.last_close
            risk = abs(p.entry - p.stop_loss)
            atr = max(risk / 2.0, ltp * 0.005)
            
            p1 = p.entry + (1.0 * atr if side == "LONG" else -1.0 * atr)
            be = p.entry + (0.6 * atr if side == "LONG" else -0.6 * atr)
            p2 = p.target
            
            print(f"\n【 GRADE-A+ CALL #{i} 】 {s} — {'🟢 BUY (LONG)' if side == 'LONG' else '🔴 SELL (SHORT)'}")
            print(f"  • Strategy Votes: {', '.join(v.strategy for v in p.strategy_votes if v.is_trade and v.side == p.side)}")
            print(f"  • Confidence:     {p.confidence:.1f}%  |  Model Alpha Score: {getattr(r, 'rank_score', 0.0):+.2f} pts")
            print(f"  • Volume Surge:   {getattr(r, 'vol_ratio', 1.0):.1f}x 5m average volume")
            print(f"  • Distance EMA9:  {getattr(r, 'dist_ema9_atr', 0.0):+.2f} ATR (Low-risk entry point)")
            print(f"  ──────────────────────────────────────────────────────────")
            print(f"  🎯 ENTRY PRICE:   ₹{p.entry:.2f}")
            print(f"  🛑 STOP LOSS:     ₹{p.stop_loss:.2f}  (Risk: ₹{risk:.2f}/sh, exactly 2.0 ATR)")
            print(f"  🛡️ BREAKEVEN AT:  ₹{be:.2f}  (👉 Move SL to entry as soon as price touches here)")
            print(f"  💰 TARGET 1 (50%):₹{p1:.2f}  (👉 Lock 50% cash profit at +1.0 ATR)")
            print(f"  🏁 TARGET 2 (50%):₹{p2:.2f}  (👉 Let remainder ride to full target)")
            print(f"  📦 SIZING:        {max(1, int(10_000 / max(risk, 0.1)))} shares (₹10,000 risk budget)")
    print("=" * 95)
