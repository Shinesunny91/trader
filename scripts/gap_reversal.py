"""Run the gap-reversal book: pre-open picks, opening stops, paper record, backtest.

    python scripts/gap_reversal.py picks            # 08:45: the session's short list
    python scripts/gap_reversal.py final            # 09:09: re-rank on today's opening prices
    python scripts/gap_reversal.py levels           # after 09:15: exact stop prices
    python scripts/gap_reversal.py record           # after the close: paper result
    python scripts/gap_reversal.py backtest         # last 14 sessions + dev + decade

`--push` sends the result to the phone (ntfy).  Scheduled by the
nse-gap-{picks,levels,record} systemd timers (see deploy/systemd/).  Strategy
and evidence: src/nse_intraday_ai/gap_reversal.py.
"""
from __future__ import annotations

import argparse
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

def _band_filter(session: date, picks: list[G.Pick], info: dict, *, fetch: bool) -> list[G.Pick]:
    """Drop shorts that can't be placed/protected today (price band, series); never blocks a list."""
    from nse_intraday_ai import nse_bands

    try:
        table = nse_bands.load(session, fetch=fetch)
    except Exception as exc:                                  # noqa: BLE001 — optional input
        print(f"  price bands unavailable ({type(exc).__name__}); list not band-filtered")
        return picks
    if table.empty:
        print("  price bands unavailable; list not band-filtered")
    picks, dropped = G.drop_untradeable(picks, table, CONFIG)
    if dropped:
        print(f"  dropped: {'; '.join(dropped)}")
        info["band_excluded"] = dropped
    return picks


def _publish(session: date | None, *, fetch: bool) -> tuple[date, date, list[G.Pick], dict]:
    """The NSE-data pipeline (experts + Hedge blend); the Yahoo gap rule if it fails."""
    from nse_intraday_ai import pipeline as P

    target = session or G.next_session()
    try:
        prev, picks, info = P.morning(target, CONFIG, fetch=fetch)
        picks = _band_filter(target, picks, info, fetch=fetch)
        G.with_entry_limits(picks, CONFIG)
        G.save_picks(target, picks, CONFIG, based_on=prev.isoformat(), source="nse", **info)
        return target, prev, picks, info
    except Exception as exc:                                  # noqa: BLE001
        # A broken morning must still produce a correct list if one can be
        # made: the gap rule on Yahoo data has its own freshness guard.
        print(f"NSE pipeline failed ({type(exc).__name__}: {exc}); falling back to the gap rule")
        target, last, picks = G.publish_picks(target, CONFIG, fetch=fetch)
        info = {"fallback": f"{type(exc).__name__}: {exc}"[:200]}
        picks = _band_filter(target, picks, info, fetch=fetch)
        G.with_entry_limits(picks, CONFIG)
        payload = G.read_picks() or {}
        G.save_picks(target, picks, CONFIG, based_on=payload.get("based_on"), source="yahoo-rule", **info)
        return target, last, picks, info


def cmd_picks(args) -> None:
    wait_for_clock()
    session = date.fromisoformat(args.date) if args.date else None
    if session is None:
        from nse_intraday_ai.nse_calendar import is_trading_day
        today = datetime.now(IST).date()
        if today.weekday() < 5 and not is_trading_day(today):
            print(f"{today}: NSE holiday — no list published")
            return
    try:
        session, last, picks, info = _publish(session, fetch=not args.no_fetch)
    except G.StaleDataError as exc:
        push("⚠ Gap-reversal: NO LIST", f"{exc}.\nNo picks published — do not trade a list "
             "from an earlier message. Retry: python scripts/gap_reversal.py picks",
             enabled=args.push)
        raise SystemExit(2) from exc

    main = [p for p in picks if not p.reserve]
    lines = [f"{p.rank}. SHORT {p.quantity} {p.symbol.removesuffix('.NS'):<11} "
             + (f"LIMIT ₹{p.limit_price:,.2f}  " if p.limit_price else "")
             + f"gap yday {p.gap_prev_pct:+.1f}%  stop +₹{p.stop_distance:,.1f} ({p.stop_pct:.1f}%)"
             for p in main]
    reserves = ", ".join(p.symbol.removesuffix(".NS") for p in picks if p.reserve)
    now = datetime.now(IST)
    late = session == now.date() and now.strftime("%H:%M") > "09:08"   # auction entry can close from 09:08
    if late:
        lines.insert(0, "⚠ LATE: the open has passed. The tested entry is the open itself — "
                        "entering now is a different, untested trade. Skip today.")
    if info.get("drift_alarm") or info.get("cautious_mode"):
        mult = info.get("size_multiplier", 0.6)
        lines.insert(0, f"⚠ DRIFT ALARM / CAUTIOUS MODE: the live book has been running persistently "
                        f"below its backtest. Trade {mult:.0%} of the quantities shown until it clears.")
    if info.get("ranked_by") == "ranker":
        t = info.get("guard_t")
        ranked = ("\nRanked by the trained model"
                  + (f" (60-session edge over the gap rule: t={t:+.1f})." if t is not None else "."))
    elif info.get("ranked_by") == "rule" and "fallback" not in info:
        ranked = ("\nRanked by the gap rule — the model's guard has switched it off "
                  f"(t={info.get('guard_t')}) or no model is promoted yet.")
    else:
        ranked = f"\nRanked by the gap rule ({info.get('fallback', 'fallback')})."
    if info.get("band_excluded"):
        ranked += f"\nSkipped (can't be shorted/protected today): {'; '.join(info['band_excluded'])}."
    body = "\n".join(lines) + (
        f"\nIf a name can't be shorted (ASM/T2T), use the next: {reserves}"
        + ("\nEnter in pre-open 09:00-09:08: MIS SELL LIMIT at the price shown. It fills at the "
           "official open only if the stock opens at/above it. At 09:15 CANCEL any unfilled order "
           "— do not chase (names that open lower lose on average)."
           if any(p.limit_price for p in main) else
           "\nEnter: MIS market SELL in pre-open 09:00-09:05 (NSE rejects pre-open market orders "
           "after 09:05), or at 09:15 sharp.")
        + f"\nThen: BUY SL-M at your fill + the stop shown. Cover all at {CONFIG.square_off}."
        f"\n~₹{CONFIG.capital / CONFIG.picks:,.0f} each, based on {last} closes." + ranked
    )
    push(f"📉 Gap-reversal shorts for {session:%a %d %b}", body, enabled=args.push)


def cmd_final(args) -> None:
    """09:09 — re-rank with the auction's opening prices and push the FINAL list."""
    from nse_intraday_ai import pipeline as P

    wait_for_clock()
    session = date.fromisoformat(args.date) if args.date else datetime.now(IST).date()
    payload = G.read_picks()
    if payload is None or payload.get("session") != session.isoformat():
        print(f"no 08:45 list for {session}; nothing to re-rank")
        return
    try:
        result = P.open_rerank(session, CONFIG)
    except Exception as exc:                                  # noqa: BLE001
        print(f"09:09 re-rank failed ({type(exc).__name__}: {exc}); the 08:45 list stands")
        push(f"Gap-reversal {session:%d %b}: 08:45 list stands",
             f"The 09:09 re-rank could not run ({type(exc).__name__}). Trade the 08:45 list.",
             enabled=args.push, priority="default")
        return
    if result is None:
        # No open model promoted, or its guard is off: the 08:45 list already is
        # the final one, and a second message saying so would only be noise.
        print("the 08:45 list stands (no promoted open model, or its guard is off)")
        return
    picks, info = result
    picks = _band_filter(session, picks, info, fetch=True)
    preliminary = [p["symbol"] for p in payload.get("picks", []) if not p.get("reserve")]
    extra = {k: v for k, v in payload.items() if k not in ("session", "generated_at", "config", "picks")}
    G.save_picks(session, picks, CONFIG, **{**extra, **info, "preliminary": preliminary,
                                            "final_at": datetime.now(IST).isoformat(timespec="seconds")})
    main = [p for p in picks if not p.reserve]
    reserves = ", ".join(p.symbol.removesuffix(".NS") for p in picks if p.reserve)
    changed = len({p.symbol for p in main} - set(preliminary))
    lines = [f"{p.rank}. SHORT {p.quantity} {p.symbol.removesuffix('.NS'):<11} "
             f"stop +₹{p.stop_distance:,.1f} ({p.stop_pct:.1f}%)"
             for p in main]
    body = "\n".join(lines) + (
        f"\nReserves: {reserves}"
        f"\nEnter: MIS market SELL at 09:15:00 sharp (this list uses today's opening prices)."
        f"\nThen: BUY SL-M at your fill + the stop shown. Cover all at {CONFIG.square_off}."
        f"\n{changed} of 8 names differ from the 08:45 list. If you already entered the 08:45 "
        f"list in the pre-open, keep it — do not double up.")
    push(f"🎯 FINAL gap-reversal shorts {session:%a %d %b}", body, enabled=args.push)


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
    from nse_intraday_ai import nse_bands
    try:
        bands = nse_bands.load(session, fetch=False)
    except Exception:                                         # noqa: BLE001 — optional input
        bands = nse_bands.parse("Symbol,Series,Security Name,Band,Remarks\n")
    capped: dict[str, float] = {}
    for p in picks:
        bars = G.load_session_bars(p.symbol, session)
        if not bars.empty and 0 in bars.index:
            p.entry = round(float(bars.at[0, "open"]), 2)
            p.stop_price = round(p.entry + p.stop_distance, 2)
            cap = G.circuit_capped_stop(p, bands)
            if cap is not None:
                capped[p.symbol], p.stop_price = cap[1], cap[0]
            opened.append(p)
    if not opened:
        push(f"Gap-reversal {session:%d %b}", "No opening prices — market holiday or no data yet. "
             "No trades today.", enabled=args.push, priority="default")
        return
    previous = G.read_picks() or {}
    extra = {k: v for k, v in previous.items() if k not in ("session", "config", "picks")}
    G.save_picks(session, picks, CONFIG, **{**extra, "levels_at": datetime.now(IST).isoformat(timespec="seconds")})
    main = [p for p in opened if not p.reserve][:CONFIG.picks]
    body = "\n".join(
        f"{p.symbol.removesuffix('.NS'):<11} opened ₹{p.entry:,.2f} below limit ₹{p.limit_price:,.2f} "
        f"→ NOT FILLED, cancel the order"
        if p.limit_price and p.entry < p.limit_price else
        f"{p.symbol.removesuffix('.NS'):<11} open ₹{p.entry:,.2f} → BUY SL-M ₹{p.stop_price:,.2f}"
        + (f"  ⚠ capped below the ₹{capped[p.symbol]:,.2f} upper circuit" if p.symbol in capped else "")
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
        from nse_intraday_ai.atomic_io import atomic_write_json
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_write_json(BACKTEST, payload)
        print(f"\n-> {BACKTEST}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("picks", "final", "levels", "record"):
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
    {"picks": cmd_picks, "final": cmd_final, "levels": cmd_levels, "record": cmd_record,
     "backtest": cmd_backtest}[args.cmd](args)


if __name__ == "__main__":
    main()
