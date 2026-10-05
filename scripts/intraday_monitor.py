#!/usr/bin/env python3
"""Intraday NSE-500 monitor: one scan per run (systemd timer, every 5 minutes).

    python scripts/intraday_monitor.py scan [--push]          # live, during the session
    python scripts/intraday_monitor.py replay --date 2026-10-01   # past session, no fetch/record
    python scripts/intraday_monitor.py board                  # shadow-board statistics

A scan refreshes the NIFTY-500 5-minute bars, writes data/intraday/live.json for
the dashboard, alerts on the gap book's stops, and after 15:20 records the day's
paper shadow-board trades.  Signals are PAPER ONLY — nothing is pushed as a trade.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai import gap_reversal as G  # noqa: E402
from nse_intraday_ai import intraday_monitor as M  # noqa: E402
from nse_intraday_ai.atomic_io import atomic_read_json, atomic_write_json  # noqa: E402

IST = M.IST
APP = "nse-intraday-monitor"
DEVELOPER = "Shine"


def context(session: date, symbols: list[str], *, fetch_bands: bool) -> M.Context:
    """Daily context, cached once per session (baseline volume profile is the slow part)."""
    path = M.OUT / f"context_{session:%Y%m%d}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["symbols"]) == symbols:
            return M.Context(z["prev_close"], z["atr"], z["cum_base"], z["band"])
    pc, atr = M.daily_context(session, symbols)
    ctx = M.Context(pc, atr, M.volume_baseline(session, symbols), M.bands_for(session, symbols, fetch=fetch_bands))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, symbols=np.array(symbols), prev_close=ctx.prev_close, atr=ctx.atr,
             cum_base=ctx.cum_base, band=ctx.band)
    for old in sorted(M.OUT.glob("context_*.npz"))[:-5]:          # keep a few days
        old.unlink(missing_ok=True)
    return ctx


def run(session: date, nbar: int, *, fetch: bool, push: bool, record: bool, now: datetime) -> dict:
    picks = G.read_picks()
    book_names = ({str(p["symbol"]).removesuffix(".NS") for p in picks.get("picks", [])}
                  if picks and picks.get("session") == session.isoformat() else set())
    symbols = sorted(set(M.universe()) | book_names)
    if fetch:
        got = G.refresh([f"{s}.NS" for s in symbols], interval="5m", period="1d", batch=100, log=print)
        print(f"  5m bars: {len(got)}/{len(symbols)} symbols")
    ctx = context(session, symbols, fetch_bands=fetch)
    panel = M.load_day(session, symbols, nbar)
    ann = M.todays_announcements(session, set(symbols)) if fetch else pd.DataFrame(columns=["symbol", "ts", "desc", "text"])
    payload = M.build_live(panel, ctx, ann, picks, now)
    M.write_live(payload)
    print(f"  {session} bar {payload['last_bar']}: {len(payload['in_play'])} in play, "
          f"{len(payload['signals'])} paper signals, {len(payload['announcements'])} announcements")
    if push and payload["gap_book"]:
        alert_path = M.OUT / f"alerts_{session:%Y%m%d}.json"
        sent = set(atomic_read_json(alert_path, default=[]) or [])
        new = M.gap_alerts(payload["gap_book"], sent)
        if new:
            from nse_intraday_ai.alerts import send_ntfy
            for key, msg in new:
                if send_ntfy(msg, title="Gap book stop watch", priority="high", tags="warning"):
                    sent.add(key)
                print(f"  alert: {msg}")
            atomic_write_json(alert_path, sorted(sent))
    if record and nbar > M.SQ_BAR:
        n = M.record_day(session, payload["signals"])
        print(f"  shadow book: {n} paper trades recorded for {session}")
    return payload


def cmd_scan(args) -> None:
    now = datetime.now(IST)
    from nse_intraday_ai.nse_calendar import is_trading_day
    if not is_trading_day(now.date()):
        print(f"{now.date()}: not a trading day")
        return
    hm = now.hour * 60 + now.minute
    if not args.force and not (9 * 60 + 20 <= hm <= 15 * 60 + 45):
        print(f"{now:%H:%M}: outside the scan window (09:20-15:45)")
        return
    run(now.date(), M.completed_bars(now), fetch=True, push=args.push, record=True, now=now)


def cmd_replay(args) -> None:
    session = date.fromisoformat(args.date)
    payload = run(session, args.bars, fetch=False, push=False, record=False,
                  now=datetime.combine(session, datetime.min.time(), IST).replace(hour=15, minute=30))
    sig = pd.DataFrame(payload["signals"])
    if not sig.empty:
        print(sig[["strategy", "symbol", "side", "time", "entry", "stop", "target", "status", "exit", "net_bps"]]
              .to_string(index=False))
        print(sig.groupby("strategy").net_bps.agg(["count", "mean"]).round(1).to_string())


def cmd_board(_args) -> None:
    book = pd.read_csv(M.BOOK_PATH) if M.BOOK_PATH.exists() else pd.DataFrame()
    print(M.board_stats(book).to_string(index=False))


def main() -> None:
    print(f"{APP} | build {datetime.fromtimestamp(Path(__file__).stat().st_mtime):%Y-%m-%d} | developer {DEVELOPER}")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--push", action="store_true", help="phone alerts for the gap book's stops")
    s.add_argument("--force", action="store_true", help="scan outside 09:20-15:45")
    s.set_defaults(fn=cmd_scan)
    r = sub.add_parser("replay")
    r.add_argument("--date", required=True)
    r.add_argument("--bars", type=int, default=M.NBARS)
    r.set_defaults(fn=cmd_replay)
    sub.add_parser("board").set_defaults(fn=cmd_board)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
