#!/usr/bin/env python3
"""Combined signal + near-miss monitor. Called every 5 min by the cron loop."""
import warnings; warnings.filterwarnings("ignore")
import json, os, subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai.data import load_nifty100_symbols, YFinanceProvider
from nse_intraday_ai.scanner import scan_universe_batch, leading_trade_side
from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.strategies import EnsembleConfig
from nse_intraday_ai.learning import ShadowLearner, AdaptiveWeightEngine
from nse_intraday_ai.candle_cache import CandleCache
from nse_intraday_ai.models import Side
import nse_intraday_ai.scan_config as _sc

IST  = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
TRACK_FILE = ROOT / "data" / "today_signals.json"
NEAR_FILE  = ROOT / "data" / "today_near_misses.json"
LEARNER_DB = ROOT / "data" / "shadow_learner.sqlite3"
CACHE_DB   = ROOT / "data" / "candles.sqlite3"

now   = datetime.now(IST)
today = now.date().isoformat()
m     = now.hour * 60 + now.minute

PAUSE_FLAG = ROOT / "data" / "notifications_paused"

def notify(title, body, urgency="normal"):
    if PAUSE_FLAG.exists():
        print(f"  [notifications paused — delete {PAUSE_FLAG.name} to resume]")
        return
    env = os.environ.copy()
    env.setdefault("DBUS_SESSION_BUS_ADDRESS","unix:path=/run/user/1000/bus")
    env.setdefault("DISPLAY",":1")
    subprocess.run(["notify-send", title, body,
                    f"--urgency={urgency}", "--app-name=NSE Signal Lab",
                    "--expire-time=30000"], env=env, timeout=5, check=False)

def load_state(path, default):
    if path.exists():
        s = json.loads(path.read_text())
        return s if s.get("date") == today else default
    return default

def update_outcome(sig, curr):
    if sig["outcome"] != "OPEN" or not curr:
        return
    # actionable signals use "entry"; near-miss signals use "entry_price"
    e   = sig.get("entry") or sig.get("entry_price")
    if not e:
        return
    sl, tgt = sig.get("sl"), sig.get("target")
    if not sl or not tgt:
        bps = (curr-e)/e*10000 if sig["side"]=="LONG" else (e-curr)/e*10000
        sig["outcome_bps"] = round(bps,1); return
    if sig["side"] == "LONG":
        if curr>=tgt:  sig.update({"outcome":"TARGET ✅","outcome_bps":round((tgt-e)/e*10000,1)})
        elif curr<=sl: sig.update({"outcome":"STOP ❌","outcome_bps":round((sl-e)/e*10000,1)})
        else:          sig["outcome_bps"] = round((curr-e)/e*10000,1)
    else:
        if curr<=tgt:  sig.update({"outcome":"TARGET ✅","outcome_bps":round((e-tgt)/e*10000,1)})
        elif curr>=sl: sig.update({"outcome":"STOP ❌","outcome_bps":round((e-sl)/e*10000,1)})
        else:          sig["outcome_bps"] = round((e-curr)/e*10000,1)

# ── Load state ────────────────────────────────────────────────
sig_state = load_state(TRACK_FILE, {"date":today,"signals":{},"scan_count":0,"prev_regimes":{}})
nm_state  = load_state(NEAR_FILE,  {"date":today,"near_misses":{}})
sig_state["scan_count"] += 1

# ── Dual scan: tight (actionable) + wide (near-miss capture) ─
universe = load_nifty100_symbols(ROOT / "data/nifty100_symbols.csv")
cfg      = _sc.load()
symbols  = universe.symbols[:int(cfg.get("n_symbols", 30))]

learner = ShadowLearner(LEARNER_DB)
weights = AdaptiveWeightEngine(learner).weights()

# Fetch market context (index + VIX) before scanning
from nse_intraday_ai.market_context import fetch_index_vix_context
mctx = fetch_index_vix_context()

# Wide scan — captures actionable + near-misses + weak in one pass
wide_risk = RiskConfig(capital=int(cfg["capital"]), risk_per_trade_pct=float(cfg["risk_per_trade_pct"]),
                       max_position_pct=float(cfg["max_position_pct"]),
                       min_confidence=50, min_reward_risk=0.8,
                       estimated_cost_bps=float(cfg["estimated_cost_bps"]),
                       slippage_bps=float(cfg["slippage_bps"]))
wide_ens  = EnsembleConfig(min_agreeing_votes=1, min_vote_share=0.40, min_weighted_confidence=50)
_act_conf = float(cfg["min_confidence"])
_act_rr   = float(cfg["min_reward_risk"])

results = scan_universe_batch(
    symbols=symbols, provider_factory=YFinanceProvider,
    period=cfg.get("period","1d"), interval=cfg.get("interval","5m"), risk_config=wide_risk,
    ensemble_config=wide_ens, strategy_weights=weights, max_workers=10,
    market_context=mctx,
)
latest_prices = {r.symbol: r.last_close for r in results if r.last_close}

# ── Regimes ───────────────────────────────────────────────────
regimes = {}
for r in results:
    regimes[r.regime or "UNKNOWN"] = regimes.get(r.regime or "UNKNOWN", 0) + 1
prev_regimes = sig_state.get("prev_regimes", {})
shifts = [f"{k} {prev_regimes.get(k,0)}→{regimes.get(k,0)}"
          for k in set(list(regimes)+list(prev_regimes))
          if abs(regimes.get(k,0)-prev_regimes.get(k,0)) >= 3]
sig_state["prev_regimes"] = regimes
avg_rows = int(sum(r.rows for r in results)/max(len(results),1))

tod_labels = [(555,570,"Opening 0.88x"),(570,600,"Gap-fill 0.97x"),(600,660,"Momentum 1.05x"),
              (660,780,"Prime 1.10x"),(780,840,"Lunch 0.88x"),(840,885,"Afternoon 1.00x"),
              (885,915,"Late 0.88x"),(915,930,"Close 0.78x")]
tod = next((lbl for lo,hi,lbl in tod_labels if lo<=m<hi), "Closed")

print(f"\n{'='*64}")
print(f"  MONITOR #{sig_state['scan_count']}  {now.strftime('%H:%M IST')}  [{tod}]")
print(f"{'='*64}")
print(f"  {len(results)} symbols | avg {avg_rows} candles | regimes: {' '.join(f'{k}:{v}' for k,v in sorted(regimes.items(),key=lambda x:-x[1]))}")
if shifts: print(f"  ⚡ SHIFTS: {' | '.join(shifts)}")
# Market context line
_idx_sym = {"TRENDING_UP":"↑","TRENDING_DOWN":"↓","RANGING":"→","HIGH_VOL":"⚡"}.get(mctx.index_regime,"?")
_vix_str = f"{mctx.vix_value:.1f}" if mctx.vix_value else "N/A"
_sec_str = "  ".join(f"{s[:3]}={r[:4]}" for s,r in sorted(mctx.sector_regimes.items())[:4]) if mctx.sector_regimes else "—"
print(f"  NIFTY {_idx_sym}{mctx.index_regime}({mctx.index_long_adj:+.0f}L) | VIX={_vix_str}({mctx.vix_level}) | {mctx.breadth_signal}({mctx.breadth_up_pct:.0%}↑) | Sectors: {_sec_str}")

# ── Classify all results ──────────────────────────────────────
# Use exact same thresholds as the app sidebar
def _is_actionable(r):
    return (r.plan.is_actionable
            and r.plan.confidence >= _act_conf
            and r.plan.reward_risk >= _act_rr)

near_lo = max(50, _act_conf - 10)
actionable = [r for r in results if _is_actionable(r)]
near_band  = [r for r in results if not _is_actionable(r) and near_lo <= r.plan.confidence < _act_conf]
weak_band  = [r for r in results if not _is_actionable(r) and 50 <= r.plan.confidence < near_lo]

# ── Record + notify actionable ────────────────────────────────
print(f"\n  ACTIONABLE: {len(actionable)}")
for r in actionable:
    plan  = r.plan
    votes = [v for v in plan.strategy_votes if v.side==plan.side and v.is_trade]
    strats= ", ".join(v.strategy for v in votes)
    key   = f"{r.symbol}|{plan.side.value}|{round(plan.entry,2)}"
    print(f"  ▶ {r.symbol} {plan.side.value} conf={plan.confidence:.0f}% regime={r.regime}")
    print(f"    entry={plan.entry:.2f} SL={plan.stop_loss:.2f} T={plan.target:.2f} RR={plan.reward_risk:.2f} [{strats}]")
    if key not in sig_state["signals"]:
        sig_state["signals"][key] = {
            "symbol":r.symbol,"side":plan.side.value,"entry":plan.entry,
            "sl":plan.stop_loss,"target":plan.target,"rr":plan.reward_risk,
            "conf":plan.confidence,"regime":r.regime,"strategies":strats,
            "fired_at":now.strftime("%H:%M"),"outcome":"OPEN","outcome_bps":None,
        }
        notify(f"🚨 {r.symbol}",
               f"{plan.side.value} entry={plan.entry:.2f} SL={plan.stop_loss:.2f} T={plan.target:.2f} "
               f"RR={plan.reward_risk:.2f} conf={plan.confidence:.0f}%\nRegime:{r.regime} | {strats}",
               urgency="critical")

# ── Record near-misses ────────────────────────────────────────
for band, band_results in [("NEAR",near_band),("WEAK",weak_band)]:
    for r in band_results:
        side = leading_trade_side(r)
        if side == Side.WAIT: continue
        votes = [v for v in r.plan.strategy_votes if v.side==side and v.is_trade]
        if not votes: continue
        price = r.public_ltp or r.last_close
        if not price: continue
        strats = ", ".join(v.strategy for v in votes[:2])
        key = f"{r.symbol}|{side.value}|{now.strftime('%H:%M')}"
        if key not in nm_state["near_misses"]:
            stops   = [v.stop_loss for v in votes if v.stop_loss]
            targets = [v.target    for v in votes if v.target]
            sl  = (min(stops)   if side==Side.LONG else max(stops))   if stops   else None
            tgt = (max(targets) if side==Side.LONG else min(targets)) if targets else None
            nm_state["near_misses"][key] = {
                "symbol":r.symbol,"side":side.value,"conf":round(r.plan.confidence,1),
                "regime":r.regime,"entry_price":round(price,2),
                "sl":round(sl,2) if sl else None,"target":round(tgt,2) if tgt else None,
                "strategies":strats,"band":band,"recorded_at":now.strftime("%H:%M"),
                "outcome":"OPEN","exit_price":None,"outcome_bps":None,
            }

# ── Update all outcomes ───────────────────────────────────────
for sig in sig_state["signals"].values():
    update_outcome(sig, latest_prices.get(sig["symbol"]))
for sig in nm_state["near_misses"].values():
    update_outcome(sig, latest_prices.get(sig["symbol"]))

# ── Scoreboard ────────────────────────────────────────────────
def scoreboard(label, sigs, show_all=False):
    if not sigs: return
    with_bps = [s for s in sigs if s["outcome_bps"] is not None]
    closed   = [s for s in sigs if "✅" in s["outcome"] or "❌" in s["outcome"]]
    wins     = [s for s in closed if "✅" in s["outcome"]]
    avg_bps  = sum(s["outcome_bps"] for s in with_bps)/len(with_bps) if with_bps else None
    wr_str   = f"WR={len(wins)/len(closed)*100:.0f}%" if closed else "WR=n/a"
    avg_str  = f"avg={avg_bps:+.1f}bps" if avg_bps is not None else "avg=..."
    verdict  = ("✅ TRADING" if avg_bps and avg_bps>5 else
                "❌ BLOCKED" if avg_bps and avg_bps<-5 else "⚖️  WATCH")
    print(f"\n  ── {label} (n={len(sigs)}, closed={len(closed)}, {wr_str}, {avg_str}) {verdict}")
    items = sigs if show_all else sigs[:6]
    for s in items:
        bps = f"{s['outcome_bps']:+.1f}bps" if s['outcome_bps'] is not None else "..."
        print(f"     {s.get('recorded_at','?? ')or s.get('fired_at','?? ')} "
              f"{s['symbol']:14s} {s['side']:5s} {s['outcome']:12s} {bps:>8}  [{s.get('strategies','')[:40]}]")

all_signals = list(sig_state["signals"].values())
near_sigs   = [s for s in nm_state["near_misses"].values() if s["band"]=="NEAR"]
weak_sigs   = [s for s in nm_state["near_misses"].values() if s["band"]=="WEAK"]

print(f"\n  ── SIGNAL PERFORMANCE COMPARISON ──")
scoreboard("ACTIONABLE ≥65%  (we trade these)",       all_signals, show_all=True)
scoreboard("NEAR MISS  58–64% (just below threshold)", near_sigs)
scoreboard("WEAK       50–57% (well below threshold)", weak_sigs)

# ── Shadow learner ────────────────────────────────────────────
ev  = learner.evaluate_pending(latest_prices)
ins = learner.record_scan_results(results, min_confidence=50)
if ev or ins: learner.refresh_policy()
stats = learner.stats()
print(f"\n  SHADOW: evaluated={stats.evaluated:,} WR={stats.win_rate:.1f}% avg={stats.avg_reward_bps:+.1f}bps | +{ev} resolved +{ins} new")

# ── Strategy heat ─────────────────────────────────────────────
perf = learner.strategy_performance_frame(days=1, min_samples=3, limit=11, cost_bps=18)
if not perf.empty:
    print(f"\n  STRATEGY TODAY (net 18bps):")
    for _, row in perf.iterrows():
        flag = "🔥" if row["win_rate"]>60 else ("⚠️" if row["win_rate"]<35 else "  ")
        print(f"  {flag} {row['strategy']:<35} WR={row['win_rate']:.0f}% avg={row['avg_reward_bps']:+.1f}bps n={int(row['samples'])}")

# ── Near-miss: should we lower threshold? ────────────────────
near_with_bps = [s for s in near_sigs if s["outcome_bps"] is not None]
ac_with_bps   = [s for s in all_signals if s["outcome_bps"] is not None]
if len(near_with_bps)>=5 and len(ac_with_bps)>=2:
    nm_avg = sum(s["outcome_bps"] for s in near_with_bps)/len(near_with_bps)
    ac_avg = sum(s["outcome_bps"] for s in ac_with_bps)/len(ac_with_bps)
    print(f"\n  ⚖️  THRESHOLD CHECK: actionable={ac_avg:+.1f}bps  near-miss={nm_avg:+.1f}bps")
    if nm_avg > 5 and (ac_avg - nm_avg) < 8:
        print(f"  ⚡ Near-misses almost as good → consider lowering conf threshold 65→60")
    elif nm_avg < -5:
        print(f"  ✅ Filter working — near-misses are correctly blocked ({nm_avg:+.1f}bps avg)")

# ── End-of-day ────────────────────────────────────────────────
if m >= 935:
    print(f"\n{'='*64}")
    print("  END-OF-DAY — FULL ANALYSIS")
    print(f"{'='*64}")
    closed_all = [s for s in all_signals if "✅" in s["outcome"] or "❌" in s["outcome"]]
    wins_all   = [s for s in closed_all if "✅" in s["outcome"]]
    if closed_all:
        total = sum(s["outcome_bps"] or 0 for s in closed_all)
        print(f"\n  Actionable: {len(all_signals)} signals | {len(closed_all)} closed | WR={len(wins_all)/len(closed_all)*100:.0f}% | {total:+.1f}bps total")
    near_closed = [s for s in near_sigs if "✅" in s["outcome"] or "❌" in s["outcome"]]
    near_wins   = [s for s in near_closed if "✅" in s["outcome"]]
    if near_closed:
        near_total = sum(s["outcome_bps"] or 0 for s in near_closed)
        print(f"  Near-miss:  {len(near_sigs)} tracked | {len(near_closed)} closed | WR={len(near_wins)/len(near_closed)*100:.0f}% | {near_total:+.1f}bps total")
        if near_total/max(len(near_closed),1) > 5:
            print(f"\n  ⚡ IMPROVEMENT: Lower min_confidence from 65→60 in strategies.py / app sidebar")
            print(f"     Near-misses averaged {near_total/len(near_closed):+.1f}bps — that's profitable if traded")
        else:
            print(f"\n  ✅ Threshold is correct. Near-misses not worth trading ({near_total/len(near_closed):+.1f}bps avg)")
    bias = "BEARISH" if regimes.get("TRENDING_DOWN",0)>regimes.get("TRENDING_UP",0) else "BULLISH"
    print(f"\n  Tomorrow: {bias} | regime close: {dict(regimes)}")

# ── Save ──────────────────────────────────────────────────────
TRACK_FILE.write_text(json.dumps(sig_state, indent=2))
NEAR_FILE.write_text(json.dumps(nm_state, indent=2))
cc = CandleCache(CACHE_DB)
cs = cc.stats()
print(f"\n  Cache: {cs['candles']:,} candles | Near-miss tracker: {len(nm_state['near_misses'])} entries today")
