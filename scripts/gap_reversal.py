"""Run the gap-reversal book: pre-open picks, opening stops, paper record, backtest.

    python scripts/gap_reversal.py picks            # the next session's short list
    python scripts/gap_reversal.py levels           # after 09:15: exact stop prices
    python scripts/gap_reversal.py record           # after the close: paper result
    python scripts/gap_reversal.py backtest         # last 14 sessions + dev + decade

`--push` sends the result to the phone (ntfy).  Scheduled by the
nse-gap-{picks,levels,record} systemd timers (see deploy/systemd/).  Strategy
and evidence: src/nse_intraday_ai/gap_reversal.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai import gap_reversal as G  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
OUT = G.OUT_DIR
PAPER = OUT / "paper_book.csv"
BACKTEST = OUT / "backtest.json"
CONFIG = G.GapReversalConfig()
# The first session of the forward (never-backtested) paper record.
FORWARD_START = date(2026, 9, 29)


def wait_for_clock(timeout: int = 300) -> None:
    """Block until NTP has synchronised the clock, up to `timeout` seconds.

    On 2026-09-29 the hardware clock came up ~5.5 h fast.  Until NTP fixed it,
    TLS to Yahoo failed ("possibly delisted" for every symbol) and "today"
    meant the wrong session.  Timers fire right after boot, so they must wait.
    """
    import subprocess
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return                       # no timedatectl: nothing to wait for
        if out != "no":
            return
        time.sleep(10)
    print("warning: clock still not NTP-synchronised; continuing")


def push(title: str, body: str, *, enabled: bool, priority: str = "high") -> None:
    print(f"\n[{title}]\n{body}")
    if not enabled:
        return
    from nse_intraday_ai.alerts import send_ntfy

    if not send_ntfy(body, title=title, priority=priority, tags="chart_with_downwards_trend"):
        print("  (push failed)")


# ── commands ────────────────────────────────────────────────────────────────

def cmd_picks(args) -> None:
    wait_for_clock()
    session = date.fromisoformat(args.date) if args.date else None
    try:
        session, last, picks = G.publish_picks(session, CONFIG, fetch=not args.no_fetch)
    except G.StaleDataError as exc:
        push("⚠ Gap-reversal: NO LIST", f"{exc}.\nNo picks published — do not trade a list "
             "from an earlier message. Retry: python scripts/gap_reversal.py picks",
             enabled=args.push)
        raise SystemExit(2) from exc

    main = [p for p in picks if not p.reserve]
    lines = [f"{p.rank}. SHORT {p.quantity} {p.symbol.removesuffix('.NS'):<11} "
             f"gap yday {p.gap_prev_pct:+.1f}%  stop +₹{p.stop_distance:,.1f} ({p.stop_pct:.1f}%)"
             for p in main]
    reserves = ", ".join(p.symbol.removesuffix(".NS") for p in picks if p.reserve)
    now = datetime.now(IST)
    late = session == now.date() and now.strftime("%H:%M") > "09:08"
    if late:
        lines.insert(0, "⚠ LATE: the open has passed. The tested entry is the open itself — "
                        "entering now is a different, untested trade. Skip today.")
    body = "\n".join(lines) + (
        f"\nIf a name can't be shorted (ASM/T2T), use the next: {reserves}"
        f"\nEnter: MIS market SELL in pre-open 09:00-09:07 (or 09:15 sharp)."
        f"\nThen: BUY SL-M at your fill + the stop shown. Cover all at {CONFIG.square_off}."
        f"\n~₹{CONFIG.capital / CONFIG.picks:,.0f} each, based on {last} closes."
    )
    push(f"📉 Gap-reversal shorts for {session:%a %d %b}", body, enabled=args.push)


def cmd_levels(args) -> None:
    wait_for_clock()
    session = date.fromisoformat(args.date) if args.date else datetime.now(IST).date()
    picks = G.load_picks(session)
    if picks is None:
        print(f"no picks saved for {session}; run `picks` first")
        return
    symbols = [p.symbol for p in picks]
    if not args.no_fetch and G.missing_intraday(symbols, session):
        G.refresh(symbols, interval="5m", period="1d")
    opened = []
    for p in picks:
        bars = G.load_session_bars(p.symbol, session)
        if not bars.empty and 0 in bars.index:
            p.entry = round(float(bars.at[0, "open"]), 2)
            p.stop_price = round(p.entry + p.stop_distance, 2)
            opened.append(p)
    if not opened:
        push(f"Gap-reversal {session:%d %b}", "No opening prices — market holiday or no data yet. "
             "No trades today.", enabled=args.push, priority="default")
        return
    G.save_picks(session, picks, CONFIG, based_on=(G.read_picks() or {}).get("based_on"),
                 levels_at=datetime.now(IST).isoformat(timespec="seconds"))
    main = [p for p in opened if not p.reserve][:CONFIG.picks]
    body = "\n".join(
        f"{p.symbol.removesuffix('.NS'):<11} open ₹{p.entry:,.2f} → BUY SL-M ₹{p.stop_price:,.2f}"
        for p in main)
    body += f"\nCover everything at {CONFIG.square_off}. (Stops use the 09:15 open; use your own fill if different.)"
    push(f"🛑 Stop-loss levels {session:%d %b}", body, enabled=args.push)


def last_completed_session(now: datetime) -> date:
    day = now.date()
    if now.weekday() < 5 and now.strftime("%H:%M") >= "15:30":
        return day
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def record_session(session: date, *, fetch: bool) -> list[G.Trade]:
    """Paper-trade one session exactly as the backtest does; [] if no bars."""
    picks = G.load_picks(session)
    if picks is None:
        # The list is a pure function of the daily bars before the session, so
        # a session the timers missed is reconstructed rather than skipped —
        # but only from complete data, or it would record a list nobody got.
        universe = G.nifty500_symbols()
        try:
            G.check_fresh(universe, session)
        except G.StaleDataError as exc:
            print(f"{session}: cannot reconstruct the pick list ({exc})")
            return []
        daily = G.load_daily(universe, since=(session - timedelta(days=120)).isoformat())
        picks = G.select(daily, session, CONFIG)
    symbols = [p.symbol for p in picks]
    if fetch and G.missing_intraday(symbols, session, bar=G.SQUARE_OFF_BAR):
        G.refresh(symbols, interval="5m", period="1mo")
    trades = []
    for p in picks:
        if len(trades) >= CONFIG.picks:
            break
        trade = G.simulate_pick(p, G.load_session_bars(p.symbol, session), session, CONFIG)
        if trade is not None:
            trades.append(trade)
    return trades


def cmd_record(args) -> None:
    wait_for_clock()
    book = pd.read_csv(PAPER) if PAPER.exists() else pd.DataFrame()
    if args.date:
        sessions = [date.fromisoformat(args.date)]
    else:
        # Every forward session not yet in the book (Yahoo keeps 60 days of 5m
        # bars), so an outage day is retried on the next run instead of being
        # skipped forever once a later day has been recorded.
        last = last_completed_session(datetime.now(IST))
        start = max(FORWARD_START, last - timedelta(days=45))
        done = set(book["session"]) if not book.empty else set()
        sessions = [d.date() for d in pd.bdate_range(start, last)
                    if d.date().isoformat() not in done]

    coverage = G.session_coverage(G.nifty500_symbols(), min(sessions)) if sessions else {}
    newest = max(coverage) if coverage else None
    recorded = []
    for session in sessions:
        if newest and session < newest and coverage.get(session, 0) == 0:
            continue                     # a holiday: no real bars, later days have them
        trades = record_session(session, fetch=not args.no_fetch)
        if not trades:
            print(f"{session}: no bars for any pick — holiday or data outage, nothing recorded")
            continue
        rows = pd.DataFrame([{**asdict(t), "net": round(t.net, 2)} for t in trades])
        rows["session"] = session.isoformat()
        if not book.empty:
            book = book[book["session"] != session.isoformat()]
        book = pd.concat([book, rows], ignore_index=True)
        recorded.append((session, trades))
    if not recorded:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    book.sort_values(["session", "symbol"]).to_csv(PAPER, index=False)

    per_day = book.groupby("session")["net"].sum()
    for session, trades in recorded:
        net = sum(t.net for t in trades)
        body = "\n".join(f"{t.symbol.removesuffix('.NS'):<11} {t.exit_reason:<10} ₹{t.net:+,.0f}"
                         for t in trades)
        body += (f"\nDay: ₹{net:+,.0f} ({net / CONFIG.capital * 100:+.2f}%) · "
                 f"{sum(t.net > 0 for t in trades)}/{len(trades)} winners"
                 f"\nPaper book: {len(per_day)} sessions, ₹{per_day.sum():+,.0f} "
                 f"({per_day.sum() / CONFIG.capital * 100:+.2f}%), {(per_day > 0).sum()} up days")
        push(f"📒 Gap-reversal result {session:%d %b}", body,
             enabled=args.push and session == recorded[-1][0], priority="default")


def _window_summary(result: G.BacktestResult) -> dict:
    return {"summary": result.summary(),
            "daily": [{**r, "session": str(r["session"])}
                      for r in result.daily().round(2).to_dict("records")]}


def cmd_backtest(args) -> None:
    symbols = G.nifty500_symbols()
    daily = G.load_daily(symbols, since="2016-01-01" if args.decade else "2026-03-01")
    sessions = [d for d in daily["close"].index if d >= date.fromisoformat(args.since)] \
        if args.since else list(daily["close"].index[-args.sessions:])
    if args.fetch:
        for s in sessions:
            need = G.missing_intraday([p.symbol for p in G.select(daily, s, CONFIG)], s,
                                      bar=G.SQUARE_OFF_BAR)
            if need:
                G.refresh(need, interval="5m", period="60d")
    result = G.backtest(sessions, CONFIG, daily=daily)
    s = result.summary()
    print(f"\nGap-reversal book, {s['sessions']} sessions {s['first']} .. {s['last']} "
          f"on ₹{CONFIG.capital:,.0f}")
    daily_pnl = result.daily()
    for row in daily_pnl.itertuples():
        day_trades = [t for t in result.trades if t.session == row.session]
        names = " ".join(f"{t.symbol.removesuffix('.NS')}({t.gross_bps:+.0f})" for t in day_trades)
        print(f"  {row.session}  ₹{row.net:+10,.0f}  cum ₹{row.cum_net:+11,.0f}  {names}")
    print(f"\n  net ₹{s['net_rupees']:+,.0f} ({s['net_pct']:+.2f}%) | {s['trades']} trades, "
          f"win {s['win_rate_pct']}%, PF {s['profit_factor']} | up/down days "
          f"{s['up_days']}/{s['down_days']} | max DD {s['max_drawdown_pct']}% | "
          f"gross {s['avg_gross_bps']:+.1f} bps, net {s['avg_net_bps']:+.1f} bps/trade | "
          f"costs ₹{s['costs_rupees']:,.0f}")

    payload = {"generated_at": datetime.now(IST).isoformat(timespec="seconds"),
               "config": asdict(CONFIG), "window": _window_summary(result)}
    if args.decade:
        d = G.backtest_daily(daily, CONFIG)
        x = d["net_bps"]
        years = x.groupby([s.year for s in x.index]).mean().round(1)
        eq = (x / 100).cumsum()
        payload["decade"] = {
            "sessions": int(len(x)), "first": str(x.index[0]), "last": str(x.index[-1]),
            "net_bps_per_trade": round(float(x.mean()), 2),
            "t_stat": round(float(x.mean() / x.std() * len(x) ** 0.5), 2),
            "up_day_pct": round(float((x > 0).mean() * 100), 1),
            "max_drawdown_pct": round(float((eq.cummax() - eq).max()), 2),
            "by_year": {int(k): float(v) for k, v in years.items()},
        }
        dd = payload["decade"]
        print(f"\nDecade check (daily bars, open->close, 13 bps cost): {dd['sessions']} sessions, "
              f"{dd['net_bps_per_trade']:+.1f} bps/trade net, t={dd['t_stat']}, "
              f"max DD {dd['max_drawdown_pct']}%")
        print("  " + " ".join(f"{y}:{v:+.1f}" for y, v in dd["by_year"].items()))
    if args.save:
        OUT.mkdir(parents=True, exist_ok=True)
        BACKTEST.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n-> {BACKTEST}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("picks", "levels", "record"):
        p = sub.add_parser(name)
        p.add_argument("--date", help="ISO session date (default: the relevant session)")
        p.add_argument("--push", action="store_true", help="send to the phone via ntfy")
        p.add_argument("--no-fetch", action="store_true", help="use the candle cache as-is")
    b = sub.add_parser("backtest")
    b.add_argument("--sessions", type=int, default=14, help="replay the last N sessions")
    b.add_argument("--since", help="replay every session from this ISO date instead")
    b.add_argument("--decade", action="store_true", help="add the 10-year daily-bar check")
    b.add_argument("--fetch", action="store_true", help="download missing 5m bars (60-day limit)")
    b.add_argument("--save", action="store_true", help=f"write {BACKTEST.relative_to(ROOT)} for the app")
    args = parser.parse_args()
    {"picks": cmd_picks, "levels": cmd_levels, "record": cmd_record,
     "backtest": cmd_backtest}[args.cmd](args)


if __name__ == "__main__":
    main()
