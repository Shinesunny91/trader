"""Two-week event-study backtest over cached candles (NSE + commodities).

Generates candidate trade events for every symbol with recent candle-cache
coverage, replays them through the portfolio simulator across a grid of
confidence / reward-risk / vote gates, and reports the best configurations —
including a week1-calibrate → week2-validate walk-forward so in-sample gate
picks are kept honest.

Usage:
    python scripts/two_week_backtest.py nse         [--no-context]
    python scripts/two_week_backtest.py commodity   [--no-context]

Data comes from data/candles.sqlite3 (populated automatically by scans; the
context symbols in context_series.EQUITY_CONTEXT_SYMBOLS must also be cached —
one 14d/5m yfinance download does it).
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.backtest import BacktestConfig, run_backtest_from_events  # noqa: E402
from nse_intraday_ai.data import DEFAULT_COMMODITY_SYMBOLS, DataResult  # noqa: E402

DB = ROOT / "data" / "candles.sqlite3"

# Cost model per universe: (estimated_cost_bps, slippage_bps).
# NSE: discount-broker intraday equity round trip.  Commodity futures: flat
# per-lot commissions ≈ 2–6 bps of notional.
COSTS = {"nse": (15.0, 3.0), "commodity": (4.0, 2.0)}


def _read_candles(con, symbol: str, interval: str, since: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM candles"
        " WHERE symbol=? AND interval=? AND ts>=? ORDER BY ts",
        con,
        params=(symbol, interval, since),
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df.set_index("ts")


def _load_frame(symbol: str, interval: str, since: str) -> pd.DataFrame:
    # NSE candles were historically cached at 1m (resampled here to the
    # validated 5m timeframe); since 2026-07-07 the daemon scans at native 5m,
    # so stitch resampled-1m with native-5m and prefer native rows on overlap.
    con = sqlite3.connect(DB)
    try:
        if interval == "5m" and symbol.endswith(".NS"):
            legacy = _read_candles(con, symbol, "1m", since)
            if not legacy.empty:
                legacy = legacy.resample("5min").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna(subset=["open", "high", "low", "close"])
            native = _read_candles(con, symbol, "5m", since)
            if legacy.empty:
                return native
            if native.empty:
                return legacy
            return pd.concat([legacy[~legacy.index.isin(native.index)], native]).sort_index()
        return _read_candles(con, symbol, interval, since)
    finally:
        con.close()


def _nse_symbols(since: str, min_rows: int = 2000) -> list[str]:
    con = sqlite3.connect(DB)
    try:
        rows = con.execute(
            "SELECT symbol FROM candles WHERE interval='1m' AND ts>=? AND symbol LIKE '%.NS'"
            " GROUP BY symbol HAVING COUNT(*)>=? ORDER BY symbol",
            (since, min_rows),
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _events_for_symbol(args: tuple) -> list:
    symbol, interval, since, use_context, cost, slip, breadth = args
    from nse_intraday_ai.backtest import generate_candidate_events
    from nse_intraday_ai.context_series import (
        COMMODITY_CONTEXT_SYMBOLS,
        EQUITY_CONTEXT_SYMBOLS,
        build_context_series_from_cache,
    )

    frame = _load_frame(symbol, interval, since)
    if frame.empty:
        return []
    is_commodity = "=" in symbol
    config = BacktestConfig(
        no_new_trade_after="" if is_commodity else "15:00",
        same_day_exit_only=not is_commodity,
        estimated_cost_bps=cost,
        slippage_bps=slip,
    )
    context_series = None
    if use_context:
        context_symbols = COMMODITY_CONTEXT_SYMBOLS if is_commodity else EQUITY_CONTEXT_SYMBOLS
        context_series = build_context_series_from_cache(
            DB, interval="5m", since=since, symbols=context_symbols
        )
        context_series.vwap_breadth = breadth
    return generate_candidate_events(
        data={symbol: DataResult(symbol, frame, "cache", None)},
        config=config,
        risk_per_trade_pct=0.5,
        max_position_pct=25.0,
        context_series=context_series,
    )


def _sweep(events: list, config: BacktestConfig, label: str) -> pd.DataFrame:
    rows = []
    for conf in (65, 70, 75, 80, 85):
        for rr in (1.2, 1.5, 1.7, 2.0):
            for votes in (1, 2):
                summary, _, _ = run_backtest_from_events(
                    events=events,
                    starting_capital=100_000,
                    risk_per_trade_pct=0.5,
                    max_position_pct=25.0,
                    cooldown_minutes=15,
                    min_confidence=conf,
                    min_reward_risk=rr,
                    min_agreeing_votes=votes,
                    min_vote_share=0.5,
                    config=config,
                )
                rows.append(
                    {
                        "conf": conf, "rr": rr, "votes": votes,
                        "trades": summary.trades,
                        "win_rate": round(summary.win_rate, 1),
                        "pnl": round(summary.pnl),
                        "max_dd": round(summary.max_drawdown),
                    }
                )
    frame = pd.DataFrame(rows)
    print(f"\n[{label}] top gate combos by P&L:")
    print(frame.sort_values("pnl", ascending=False).head(8).to_string(index=False))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("universe", choices=["nse", "commodity"])
    parser.add_argument("--no-context", action="store_true", help="disable market-context scoring")
    parser.add_argument("--since", default=None, help="ISO date; default = 14 days ago")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    since = args.since or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    cost, slip = COSTS[args.universe]
    if args.universe == "nse":
        # 5m is the validated signal timeframe (resampled from cached 1m).
        symbols, interval = _nse_symbols(since), "5m"
    else:
        symbols, interval = list(DEFAULT_COMMODITY_SYMBOLS), "5m"
    if not symbols:
        print("No cached symbols with coverage since", since)
        return
    print(f"{args.universe}: {len(symbols)} symbols, interval={interval}, since={since}, "
          f"context={'off' if args.no_context else 'on'}, costs={cost}+{slip} bps")

    # VWAP breadth is cross-sectional — build it once in the parent and ship
    # the small series to each worker rather than recomputing 100+ frames per task.
    breadth = None
    if args.universe == "nse" and not args.no_context:
        from nse_intraday_ai.context_series import load_vwap_breadth_from_cache
        breadth = load_vwap_breadth_from_cache(DB, symbols, interval="1m", since=since)
        print(f"VWAP breadth series: {'unavailable' if breadth is None else f'{len(breadth)} buckets'}")

    started = time.time()
    events: list = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _events_for_symbol,
                (symbol, interval, since, not args.no_context, cost, slip, breadth),
            )
            for symbol in symbols
        ]
        for done, future in enumerate(as_completed(futures), 1):
            events.extend(future.result())
            if done % 20 == 0 or done == len(symbols):
                print(f"  {done}/{len(symbols)} symbols, {len(events)} events "
                      f"({time.time() - started:.0f}s)", flush=True)
    events.sort(key=lambda event: event.timestamp)
    out = ROOT / "data" / f"events_{args.universe}.pkl"
    with out.open("wb") as f:
        pickle.dump(events, f)
    print(f"{len(events)} candidate events -> {out}")
    if not events:
        return

    config = BacktestConfig(
        no_new_trade_after="" if args.universe == "commodity" else "15:00",
        estimated_cost_bps=cost,
        slippage_bps=slip,
    )
    _sweep(events, config, "full window")

    # Walk-forward: best gates on the first half must survive the second half.
    days = sorted({event.timestamp.normalize() for event in events})
    mid = days[len(days) // 2]
    week1 = [event for event in events if event.timestamp.normalize() < mid]
    week2 = [event for event in events if event.timestamp.normalize() >= mid]
    frame1 = _sweep(week1, config, f"week 1 (< {mid.date()})")
    viable = frame1[frame1["trades"] >= 5]
    if viable.empty:
        print("\nWalk-forward: no week-1 gates with >=5 trades.")
        return
    best = viable.sort_values(["pnl"], ascending=False).iloc[0]
    summary, _, _ = run_backtest_from_events(
        events=week2,
        starting_capital=100_000,
        risk_per_trade_pct=0.5,
        max_position_pct=25.0,
        cooldown_minutes=15,
        min_confidence=float(best["conf"]),
        min_reward_risk=float(best["rr"]),
        min_agreeing_votes=int(best["votes"]),
        min_vote_share=0.5,
        config=config,
    )
    print(f"\nWalk-forward validation (week-1 best gates conf={best['conf']} "
          f"rr={best['rr']} votes={int(best['votes'])} on week 2): "
          f"trades={summary.trades} win_rate={summary.win_rate:.1f}% "
          f"pnl={summary.pnl:.0f} max_dd={summary.max_drawdown:.0f}")


if __name__ == "__main__":
    main()
