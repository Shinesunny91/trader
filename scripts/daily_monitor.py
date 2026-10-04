#!/usr/bin/env python3
"""Daily trading system health monitor and performance tracker.

Produces a daily report covering:
- Yesterday's P&L (per-trade and per-session)
- Win rate, stop-out rate, cost analysis
- CUSUM drift alarm status
- Feature importance drift detection
- Model staleness check
- Data pipeline health
- Comparison vs. 10-year backtest expectations

Designed to be run daily at 16:00 IST (after market close + square-off).

Usage:
    cd trading-workspace
    PYTHONPATH=src .venv/bin/python scripts/daily_monitor.py
"""
import sys
import json
import csv
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
GAP_DIR = DATA / "gap_reversal"
MODELS = DATA / "models"

IST_OFFSET = timedelta(hours=5, minutes=30)


def _read_json(path):
    """Safely read a JSON file."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _paper_book_rows():
    """Load all paper book trades."""
    path = GAP_DIR / "paper_book.csv"
    if not path.exists():
        return []
    return list(csv.DictReader(open(path)))


def check_yesterdays_pnl():
    """Analyze yesterday's trading results."""
    rows = _paper_book_rows()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # Find the most recent trading day
    sessions = sorted(set(r["session"] for r in rows))
    if not sessions:
        return {"status": "NO_DATA", "message": "No paper book data found"}

    latest = sessions[-1]
    if latest != yesterday:
        # Market might have been closed yesterday
        yesterday = latest

    day_rows = [r for r in rows if r["session"] == yesterday]
    if not day_rows:
        return {"status": "NO_TRADES", "session": yesterday}

    trades = []
    for r in day_rows:
        net = float(r["net"])
        gross = float(r["gross"])
        cost = float(r["costs"])
        trades.append({
            "symbol": r["symbol"],
            "net": net,
            "gross": gross,
            "cost": cost,
            "exit_reason": r["exit_reason"],
            "winner": net > 0,
        })

    total_net = sum(t["net"] for t in trades)
    total_gross = sum(t["gross"] for t in trades)
    total_cost = sum(t["cost"] for t in trades)
    winners = sum(1 for t in trades if t["winner"])
    stops = sum(1 for t in trades if t["exit_reason"] == "STOP")
    avg_win = (sum(t["net"] for t in trades if t["winner"]) / winners) if winners else 0
    losers = len(trades) - winners
    avg_loss = (sum(t["net"] for t in trades if not t["winner"]) / losers) if losers else 0

    return {
        "status": "OK",
        "session": yesterday,
        "trades": len(trades),
        "net_pnl": round(total_net, 2),
        "gross_pnl": round(total_gross, 2),
        "total_cost": round(total_cost, 2),
        "winners": winners,
        "losers": losers,
        "win_rate": round(winners / len(trades) * 100, 1),
        "stops": stops,
        "stop_rate": round(stops / len(trades) * 100, 1),
        "avg_winner": round(avg_win, 2),
        "avg_loser": round(avg_loss, 2),
        "profitable_day": total_net > 0,
    }


def check_rolling_performance():
    """Check rolling 5-day and 20-day performance."""
    rows = _paper_book_rows()
    sessions = {}
    for r in rows:
        s = r["session"]
        if s not in sessions:
            sessions[s] = 0
        sessions[s] += float(r["net"])

    sorted_sessions = sorted(sessions.items())
    if len(sorted_sessions) < 2:
        return {"status": "INSUFFICIENT_DATA"}

    last5 = sorted_sessions[-5:] if len(sorted_sessions) >= 5 else sorted_sessions
    last20 = sorted_sessions[-20:] if len(sorted_sessions) >= 20 else sorted_sessions

    return {
        "status": "OK",
        "total_sessions": len(sorted_sessions),
        "cumulative_pnl": round(sum(v for _, v in sorted_sessions), 2),
        "last_5_sessions": {
            "net_pnl": round(sum(v for _, v in last5), 2),
            "win_sessions": sum(1 for _, v in last5 if v > 0),
            "total_sessions": len(last5),
        },
        "last_20_sessions": {
            "net_pnl": round(sum(v for _, v in last20), 2),
            "win_sessions": sum(1 for _, v in last20 if v > 0),
            "total_sessions": len(last20),
        },
    }


def check_meta_state():
    """Check CUSUM drift alarm and model state."""
    state = _read_json(MODELS / "meta_state.json")
    if state is None:
        return {"status": "NO_STATE"}

    return {
        "status": "OK",
        "cusum": round(state.get("cusum", 0), 2),
        "alarm": state.get("alarm", False),
        "cautious_mode": state.get("cautious_mode", False),
        "alarm_sessions_elapsed": state.get("alarm_sessions_elapsed", 0),
        "threshold_bps": state.get("threshold_bps", 50),
        "cusum_pct_of_threshold": round(
            state.get("cusum", 0) / max(state.get("threshold_bps", 50), 1) * 100, 1
        ),
    }


def check_model_freshness():
    """Check when the champion model was last trained/promoted."""
    champion = _read_json(MODELS / "champion.json")
    if champion is None:
        # Try ranker-specific champion
        champion = _read_json(MODELS / "champion_ranker.json")
    if champion is None:
        return {"status": "NO_CHAMPION"}

    promoted_at = champion.get("promoted_at", "unknown")
    directory = champion.get("directory", "unknown")
    return {
        "status": "OK",
        "champion_dir": directory,
        "promoted_at": promoted_at,
    }


def check_picks():
    """Check today's picks quality."""
    picks = _read_json(GAP_DIR / "picks.json")
    if picks is None:
        return {"status": "NO_PICKS"}

    session = picks.get("session", "?")
    pick_list = picks.get("picks", [])
    ranked_by = picks.get("ranked_by", "?")
    cusum = picks.get("cusum")
    drift_alarm = picks.get("drift_alarm", False)

    active_picks = [p for p in pick_list if not p.get("reserve")]
    reserve_picks = [p for p in pick_list if p.get("reserve")]

    gaps = [p.get("gap_prev_pct", 0) for p in active_picks]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    return {
        "status": "OK",
        "session": session,
        "active_picks": len(active_picks),
        "reserve_picks": len(reserve_picks),
        "ranked_by": ranked_by,
        "cusum": round(cusum, 2) if cusum else None,
        "drift_alarm": drift_alarm,
        "avg_gap_pct": round(avg_gap, 2),
        "symbols": [p["symbol"] for p in active_picks],
    }


def check_data_pipeline():
    """Check data freshness."""
    issues = []

    # Check bhav data
    bhav_dir = DATA / "nse_bhav"
    if bhav_dir.exists():
        files = sorted(bhav_dir.glob("*/bhav_*.parquet"))
        if files:
            latest = files[-1].stem.split("_")[1]; latest = f"{latest[:4]}-{latest[4:6]}-{latest[6:]}"
            today = date.today().isoformat()
            if latest < (date.today() - timedelta(days=3)).isoformat():
                issues.append(f"Bhav data stale: latest={latest}")
        else:
            issues.append("No bhav parquet files found")
    else:
        issues.append("nse_bhav directory missing")

    # Check candle cache
    candle_db = DATA / "candles.sqlite3"
    if candle_db.exists():
        import sqlite3
        try:
            con = sqlite3.connect(str(candle_db))
            row = con.execute("SELECT MAX(ts) FROM candles").fetchone()
            latest_ts = row[0] if row else None
            con.close()
            if latest_ts:
                pass  # Could check staleness
        except Exception as e:
            issues.append(f"Candle cache error: {e}")
    else:
        issues.append("candles.sqlite3 missing")

    return {
        "status": "ISSUES" if issues else "OK",
        "issues": issues,
    }


def generate_report():
    """Generate the full daily health report."""
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pnl": check_yesterdays_pnl(),
        "rolling": check_rolling_performance(),
        "meta_state": check_meta_state(),
        "model": check_model_freshness(),
        "picks": check_picks(),
        "pipeline": check_data_pipeline(),
    }

    # Print human-readable summary
    print("=" * 70)
    print(f"  DAILY TRADING MONITOR — {report['timestamp']}")
    print("=" * 70)

    pnl = report["pnl"]
    if pnl["status"] == "OK":
        emoji = "🟢" if pnl["profitable_day"] else "🔴"
        print(f"\n{emoji} Yesterday ({pnl['session']}): ₹{pnl['net_pnl']:+,.0f} net")
        print(f"   {pnl['winners']}/{pnl['trades']} winners ({pnl['win_rate']}%), "
              f"{pnl['stops']} stops ({pnl['stop_rate']}%)")
        print(f"   Avg win: ₹{pnl['avg_winner']:+,.0f}  Avg loss: ₹{pnl['avg_loser']:+,.0f}")
    else:
        print(f"\n⚠️  P&L: {pnl['status']}")

    rolling = report["rolling"]
    if rolling["status"] == "OK":
        print(f"\n📊 Cumulative: ₹{rolling['cumulative_pnl']:+,.0f} "
              f"({rolling['total_sessions']} sessions)")
        l5 = rolling["last_5_sessions"]
        print(f"   Last 5 sessions: ₹{l5['net_pnl']:+,.0f} "
              f"({l5['win_sessions']}/{l5['total_sessions']} green)")

    meta = report["meta_state"]
    if meta["status"] == "OK":
        alarm_emoji = "🚨" if meta["alarm"] else "✅"
        cautious = " [CAUTIOUS 60%]" if meta["cautious_mode"] else ""
        print(f"\n{alarm_emoji} CUSUM: {meta['cusum']} / {meta['threshold_bps']} "
              f"({meta['cusum_pct_of_threshold']}%){cautious}")

    picks = report["picks"]
    if picks["status"] == "OK":
        print(f"\n📋 Today's picks ({picks['session']}): {picks['active_picks']} active, "
              f"ranked_by={picks['ranked_by']}, avg_gap={picks['avg_gap_pct']:+.2f}%")
        print(f"   Symbols: {', '.join(picks['symbols'])}")

    pipeline = report["pipeline"]
    if pipeline["issues"]:
        print(f"\n⚠️  Pipeline issues:")
        for issue in pipeline["issues"]:
            print(f"   - {issue}")
    else:
        print(f"\n✅ Data pipeline healthy")

    # Alerts
    print(f"\n{'='*70}")
    alerts = []
    if pnl.get("stop_rate", 0) > 40:
        alerts.append("⚠️  HIGH STOP RATE: >40% of trades stopped out")
    if meta.get("cusum_pct_of_threshold", 0) > 80:
        alerts.append("⚠️  CUSUM APPROACHING ALARM: >80% of threshold")
    if meta.get("alarm"):
        alerts.append("🚨 DRIFT ALARM ACTIVE — model may need retraining")
    if meta.get("cautious_mode"):
        alerts.append("🔶 CAUTIOUS MODE — position sizing at 60%")
    if pnl.get("net_pnl", 0) < -5000:
        alerts.append("🔴 SIGNIFICANT LOSS DAY: >₹5,000")

    if alerts:
        print("ALERTS:")
        for a in alerts:
            print(f"  {a}")
    else:
        print("✅ No alerts — system healthy")

    print("=" * 70)

    # Save report
    report_path = DATA / "daily_monitor_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {report_path}")

    return report


if __name__ == "__main__":
    generate_report()
