"""Run the paper book for a single session — today by default.

This is the daily-operations version of `intraday_sim.py`: same engine, same
gate, same portfolio rules, but scoped to one session so it can be run live
(mid-session it reports the completed part of the day) or after the close for a
daily record.

It writes each session's result to `data/daily_sim.csv` so the paper track
record accumulates and can be read back later — a single day's number is
noise, and the only defence against reading it as a signal is having the
sequence sitting next to it.

Usage:
    python scripts/sim_today.py                 # today, live
    python scripts/sim_today.py --date 2026-08-11
    python scripts/sim_today.py --history       # print the accumulated record
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_dataset import add_cross_sectional, add_macro, build_symbol  # noqa: E402
from entry_timing_study import nse_symbols  # noqa: E402
from intraday_sim import liquid_symbols, load_frame, macro_panel  # noqa: E402
from nse_intraday_ai.entry_quality import (  # noqa: E402
    EntryQuality,
    GateConfig,
    macro_alignment,
    passes_entry_gate,
)
from nse_intraday_ai import execution_plan as EP  # noqa: E402
from nse_intraday_ai.execution_plan import build_execution_plan, expectancy_note  # noqa: E402
from nse_intraday_ai.portfolio_sim import IntradayPortfolioSimulator, SimConfig  # noqa: E402
from nse_intraday_ai.signal_model import expectancy_note as model_note  # noqa: E402
from nse_intraday_ai.signal_model import load_if_available  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
RECORD = ROOT / "data" / "daily_sim.csv"
TICKETS = ROOT / "data" / "today_tickets.json"

# The book configuration. Sourced from execution_plan so the daily run, the
# order tickets and the app cannot drift apart — there is one definition of
# "the validated trade", not three.
LIVE_CONFIG = SimConfig(
    starting_capital=10_00_000.0,
    max_concurrent_positions=EP.MAX_CONCURRENT,
    max_trades_per_day=EP.MAX_TRADES_PER_DAY,
    max_position_pct=EP.MAX_POSITION_PCT,
    risk_per_trade_pct=1.0,
    scale_out_fraction=0.0,
    stop_atr=EP.STOP_ATR,
    target_atr=EP.TARGET_ATR,
    breakeven_after_atr=EP.BREAKEVEN_ATR,
    trail_atr=0.0,
    max_hold_bars=EP.MAX_HOLD_MINUTES // 5,
    slippage_bps_per_leg=1.5,
)


def build_window(start: pd.Timestamp, end: pd.Timestamp, workers: int,
                 universe: int, *, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Engine signals for every session in [start, end], ranked.

    Features come from `build_dataset.build_symbol` — the *same* builder that
    produced the training set. An earlier version used a lighter feature
    extractor here, so the trained model could not score live signals at all
    and silently fell back to the composite rank. One builder, one feature
    contract.

    A *window* rather than a day because the per-symbol load is the expensive
    part on a cold cache: `build_symbol` reads and indicates the whole warmup
    frame regardless of how much of it is kept, so replaying a missed week one
    day at a time repeats that work once per session. Cross-sectional features
    also need the whole frame before the slice, not one session's rows.

    Signals are **ranked, not filtered**. `entry_quality.passes_entry_gate` is
    still evaluated, and its verdict rides along in `allow`/`note` for the
    screen and for diagnostics, but it no longer decides what the book may
    take — see `scripts/entry_policy_study.py`, where filtering first cost the
    live book 2.35 points of return over the 34 held-out sessions.
    """
    warmup_since = (start - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    liquid = liquid_symbols(universe)
    symbols = [s for s in nse_symbols(warmup_since, min_rows=100) if s in liquid]
    span = f"{start.date()}" if start.normalize() == end.normalize() else f"{start.date()}..{end.date()}"
    print(f"replaying {len(symbols)} liquid symbols for {span}...")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_symbol, (s, warmup_since, now)) for s in symbols]
        for future in as_completed(futures):
            rows.extend(future.result())
    if not rows:
        return pd.DataFrame(), {}

    frame = pd.DataFrame(rows)
    # Cross-sectional features are relative to everything trading at the same
    # instant, so they must be computed over the whole window before the day is
    # sliced out — not over one session's rows.
    frame = add_macro(add_cross_sectional(frame))
    in_window = (frame["ts"] >= start.normalize()) & (frame["ts"] < end.normalize() + pd.Timedelta(days=1))
    signals = frame[in_window].sort_values("ts").copy()
    print(f"{len(signals)} raw engine signals over {signals['ts'].dt.date.nunique() if len(signals) else 0} session(s)")
    if signals.empty:
        return signals, {}

    keep, scores, notes = [], [], []
    for row in signals.itertuples():
        macro = macro_alignment(
            row.side,
            nifty_change_pct=None if pd.isna(row.nifty) else row.nifty,
            usdinr_change_pct=None if pd.isna(row.usdinr) else row.usdinr,
            crude_change_pct=None if pd.isna(row.crude) else row.crude,
        )
        quality = EntryQuality(row.vol_z, row.run6, row.ext_vwap, row.rsi, row.age_extreme)
        verdict = passes_entry_gate(quality, macro, config=GateConfig())
        keep.append(verdict.allow)
        scores.append(macro.score)
        notes.append(verdict.reason)
    signals["allow"] = keep
    signals["note"] = notes

    # Ranking decides which three of ~50 signals the book takes, so it is the
    # whole decision. Prefer the trained model when one has passed its evidence
    # gate; fall back to the composite of the out-of-sample survivors.
    composite = [
        s + min(max(v, 0), 5) + min(max(r, 0), 4)
        for s, v, r in zip(scores, signals["vol_z"], signals["run6"])
    ]
    model = load_if_available()
    if model is not None:
        try:
            signals["rank"] = model.score(signals)
            signals["ranked_by"] = "model"
        except Exception as exc:
            print(f"  !! model scoring failed ({type(exc).__name__}: {exc}); "
                  f"falling back to the composite rank")
            signals["rank"] = composite
            signals["ranked_by"] = "composite"
    else:
        signals["rank"] = composite
        signals["ranked_by"] = "composite"
    print(f"  ranked by: {signals['ranked_by'].iloc[0]}")

    print(f"  {int(signals['allow'].sum())} of {len(signals)} would also pass the "
          f"entry gate (reported, not enforced)")

    # The gate used to double as the macro-outage alarm: when the context
    # symbols went stale in the cache on 2026-08-13/14 it refused all 1,552
    # signals and the empty screen was the only symptom.  Now that the gate no
    # longer decides anything, that canary has to be explicit — a stale panel
    # would otherwise feed the model three zeroed features in silence, which is
    # worse than an empty screen because it still produces a confident ticket.
    incomplete = signals[["nifty", "usdinr", "crude"]].isna().any(axis=1).mean()
    if incomplete > 0.5:
        print(f"  !! {incomplete:.0%} of signals have an incomplete macro panel — the "
              f"context symbols are stale in the candle cache.")
        print("     the model's m_nifty/m_inr/m_crude features are being zero-filled; "
              "the ranking is degraded, not merely uninformed.")
        print("     fix: python scripts/catchup_fetch.py --only context")

    frames = {}
    for symbol in signals["symbol"].unique():
        price = load_frame(symbol, (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        if not price.empty:
            frames[symbol] = price
    return signals, frames


def build_today(day: pd.Timestamp, workers: int, universe: int,
                *, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """One session — the single-day case of `build_window`."""
    return build_window(day, day, workers, universe, now=now)


def write_tickets(day: pd.Timestamp, signals: pd.DataFrame, result, capital: float) -> int:
    """Publish the session's *actionable* order tickets for the app to render.

    Two things this must get right, both learned from watching it live:

    * **Only fresh signals.** The first version published the session's
      top-ranked signals regardless of age, so at 14:40 the screen still showed
      a ticket derived from an 09:20 bar — an instruction to enter at a price
      five hours gone. A signal is actionable for the bar after it closes and
      then it is history.
    * **Respect what the book has already done.** The book caps at three trades
      a day; if it has taken them, the honest screen says so rather than
      offering a fourth.

    The app renders this file rather than building its own ticket, so the screen
    and the measured book cannot disagree.
    """
    import json

    now = pd.Timestamp.now(tz=IST)
    live = day.normalize() == now.normalize()
    # A 5m signal is tradable on the next bar; allow two bars of slack for the
    # scan/publish cycle, then it is stale.
    # For a historical replay there is no freshness constraint; anchor well
    # before the session rather than at pd.Timestamp.min, which overflows.
    cutoff = (now - pd.Timedelta(minutes=10)) if live else (day - pd.Timedelta(days=1))
    taken = int(result.trades.drop_duplicates(["symbol", "entry_time"]).shape[0]) if not result.trades.empty else 0
    # A cap of 0 means unlimited; the book is then bounded by concurrency and
    # capital instead, so the number of tickets to publish is however many free
    # position slots there are right now.
    uncapped = LIVE_CONFIG.max_trades_per_day <= 0
    # Which positions are STILL OPEN right now?
    #
    # Not `exit_time.isna()` — the simulator stamps an exit time on every row.
    # A position that never closed is the one it marked to market when the data
    # ran out, which mid-session means "now". That reason is the only reliable
    # open/closed signal in the frame, and getting it wrong matters in both
    # directions: counting every symbol touched today blocks re-entry forever
    # (0 tickets on 2026-08-17 despite 71 gated signals), while counting none
    # lets the screen offer a second ticket in a name the book already holds.
    if result.trades.empty or "exit_reason" not in result.trades:
        still_open = None
    else:
        still_open = result.trades[result.trades["exit_reason"] == "MARK_TO_MARKET"]

    open_now = int(still_open["symbol"].nunique()) if still_open is not None and not still_open.empty else 0
    remaining = (
        max(0, LIVE_CONFIG.max_concurrent_positions - open_now) if uncapped
        else max(0, LIVE_CONFIG.max_trades_per_day - taken)
    )

    fresh = signals[signals["ts"] >= cutoff].sort_values("rank", ascending=False)
    open_symbols = set()
    if still_open is not None and not still_open.empty:
        open_symbols = set(still_open["symbol"])

    tickets = []
    for row in fresh.itertuples():
        if len(tickets) >= remaining or row.symbol in open_symbols:
            continue
        open_symbols.add(row.symbol)
        ticket = build_execution_plan(
            symbol=row.symbol, side=row.side, signal_price=float(row.fill),
            atr=float(row.atr), capital=capital, taken_today=taken + len(tickets),
        )
        tickets.append({
            "symbol": row.symbol, "side": row.side,
            "signal_time": pd.Timestamp(row.ts).isoformat(),
            "age_minutes": round((now - pd.Timestamp(row.ts)).total_seconds() / 60, 1),
            "rank": float(row.rank), "ranked_by": str(row.ranked_by),
            "tradable": ticket.tradable,
            "ticket": ticket.order_ticket(),
            "quantity": ticket.quantity,
            "position_value": round(ticket.position_value, 2),
            "est_cost_rupees": round(ticket.est_cost_rupees, 2),
        })

    if not tickets:
        if remaining == 0:
            status = (f"The book has used its {LIVE_CONFIG.max_trades_per_day} trades for "
                      f"today. No further entries — taking more was measurably worse in "
                      f"the study, not better.")
        elif live and now.time() >= pd.Timestamp(LIVE_CONFIG.no_new_entry_after).time():
            status = f"Past the {LIVE_CONFIG.no_new_entry_after} entry cutoff. No new positions."
        else:
            status = ("No signal in the last 10 minutes clears the gate. A flat book is "
                      "a position.")
    else:
        status = f"{len(tickets)} actionable ticket(s); {remaining} slot(s) left today."

    model = load_if_available()
    payload = {
        "session": day.date().isoformat(),
        "generated_at": now.isoformat(timespec="seconds"),
        "capital": capital,
        "gated_signals": int(len(signals)),
        "trades_taken": taken,
        "slots_remaining": remaining,
        "daily_cap": LIVE_CONFIG.max_trades_per_day,
        "uncapped": uncapped,
        "status": status,
        "session_pnl": round(result.pnl, 2),
        "expectancy": expectancy_note(model_note(model)),
        "tickets": tickets,
    }
    TICKETS.write_text(json.dumps(payload, indent=2))
    return len(tickets)


def append_record(day: pd.Timestamp, result, signals: int) -> pd.DataFrame:
    row = {
        "date": day.date().isoformat(),
        "run_at": datetime.now(tz=IST).isoformat(timespec="seconds"),
        "gated_signals": signals,
        "trades": len(result.trades),
        "gross_pnl": round(result.trades["gross_pnl"].sum(), 2) if not result.trades.empty else 0.0,
        "costs": round(result.trades["costs"].sum(), 2) if not result.trades.empty else 0.0,
        "net_pnl": round(result.pnl, 2),
        "net_pct": round(result.pnl_pct, 3),
    }
    existing = pd.read_csv(RECORD) if RECORD.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing[existing["date"] != row["date"]]
    frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    frame = frame.sort_values("date").reset_index(drop=True)
    frame.to_csv(RECORD, index=False)
    return frame


def print_history(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("no sessions recorded yet")
        return
    frame = frame.copy()
    frame["cum_net"] = frame["net_pnl"].cumsum()
    print(frame[["date", "trades", "gross_pnl", "costs", "net_pnl", "cum_net"]].to_string(index=False))
    wins = (frame["net_pnl"] > 0).sum()
    print(f"\n  {len(frame)} sessions | {wins} up / {len(frame) - wins} down "
          f"| cumulative ₹{frame['net_pnl'].sum():+,.0f} "
          f"({frame['net_pnl'].sum() / LIVE_CONFIG.starting_capital * 100:+.2f}% of capital)")
    if len(frame) > 1:
        daily = frame["net_pnl"] / LIVE_CONFIG.starting_capital
        print(f"  mean ₹{frame['net_pnl'].mean():+,.0f}/session, "
              f"sd ₹{frame['net_pnl'].std():,.0f} — "
              f"a single session is {frame['net_pnl'].std() / max(abs(frame['net_pnl'].mean()), 1):.1f}x "
              f"the mean in noise, so read the cumulative column, not the last row")


def run_session(day: pd.Timestamp, signals: pd.DataFrame, frames: dict,
                *, write: bool) -> object:
    """Simulate one session's book from a pre-built signal window."""
    today = signals[signals["ts"].dt.normalize() == day.normalize()]
    day_frames = {}
    for symbol in today["symbol"].unique():
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            continue
        sliced = frame[frame.index.normalize() == day.normalize()]
        if not sliced.empty:
            day_frames[symbol] = sliced

    if today.empty or not day_frames:
        # A flat day still belongs in the record. Skipping it silently makes the
        # track record look denser than it is and hides outages like the one on
        # 2026-08-13, where the session simply never appeared.
        result = IntradayPortfolioSimulator(LIVE_CONFIG).run(pd.DataFrame(), {})
        if write:
            write_tickets(day, pd.DataFrame(columns=["ts", "symbol", "side", "atr",
                                                     "rank", "ranked_by", "fill", "note"]),
                          result, LIVE_CONFIG.starting_capital)
            append_record(day, result, 0)
        return result

    result = IntradayPortfolioSimulator(LIVE_CONFIG).run(
        today[["ts", "symbol", "side", "atr", "rank", "note"]], day_frames
    )
    if write:
        written = write_tickets(day, today, result, LIVE_CONFIG.starting_capital)
        print(f"{written} actionable ticket(s) -> {TICKETS}")
        append_record(day, result, len(today))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="ISO date; default = today (IST)")
    parser.add_argument("--since", default=None,
                        help="replay every session from this ISO date to --date/today")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--universe", type=int, default=150)
    parser.add_argument("--history", action="store_true", help="print the record and exit")
    parser.add_argument("--no-record", action="store_true",
                        help="print results without touching data/daily_sim.csv")
    args = parser.parse_args()

    if args.history:
        print_history(pd.read_csv(RECORD) if RECORD.exists() else pd.DataFrame())
        return

    day = pd.Timestamp(args.date, tz=IST) if args.date else pd.Timestamp.now(tz=IST).normalize()
    start = pd.Timestamp(args.since, tz=IST) if args.since else day
    now = pd.Timestamp.now(tz=IST)
    live = day.normalize() == now.normalize() and now.time() < pd.Timestamp("15:30").time()
    span = f"session {day.date()}" if start.normalize() == day.normalize() else \
           f"sessions {start.date()} .. {day.date()}"
    print(span + (f" (LIVE, {now:%H:%M} IST — partial day)" if live else ""))

    # Only a live run passes `now`: it truncates each frame to closed bars and
    # lets the newest one produce a signal, which is what makes a ticket fresh
    # enough to act on.  A historical replay must not — it would emit a tail
    # signal for a session whose next bar is already known.
    signals, frames = build_window(start, day, args.workers, args.universe,
                                   now=now if live else None)
    sessions = sorted(signals["ts"].dt.normalize().unique()) if not signals.empty else [day.normalize()]
    # The tickets file describes *now*, so only the last session in the window
    # may publish one; replaying history must never overwrite the live screen.
    last = sessions[-1]

    for session in sessions:
        session = pd.Timestamp(session)
        result = run_session(session, signals, frames,
                             write=not args.no_record and session == last)
        taken = signals[signals["ts"].dt.normalize() == session]
        print(f"\n{'─' * 78}\n{session.date()} — paper book on ₹{LIVE_CONFIG.starting_capital:,.0f}"
              f"  ({len(taken)} ranked signals)\n{'─' * 78}")
        print(result.summary())
        if not result.trades.empty:
            cols = ["entry_time", "exit_time", "symbol", "side", "entry", "exit",
                    "quantity", "gross_pnl", "costs", "net_pnl", "exit_reason"]
            print(f"\n{result.trades[cols].to_string(index=False)}")
        if not args.no_record and session != last:
            append_record(session, result, len(taken))

    print(f"\n{'─' * 78}\nrecord to date\n{'─' * 78}")
    print_history(pd.read_csv(RECORD) if RECORD.exists() else pd.DataFrame())


if __name__ == "__main__":
    main()
