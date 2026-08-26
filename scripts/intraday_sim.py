"""Simulated intraday trading of the tool's own recommendations, ₹10,00,000.

Every trade here comes from `VotingSignalEngine` — the same engine the app and
the scanner daemon run.  Nothing is hand-picked.  The only thing this script
adds on top of the engine is the *gate* (`entry_quality.passes_entry_gate`),
which is itself wired into the live scan path, so the simulated book and a live
book would receive the same recommendations.

Runs three configurations over the same window so the effect of each change is
attributable rather than asserted:

  engine-gates    the shipped gates (confidence ≥ 70, reward/risk ≥ 1.5,
                  ≥ 2 agreeing strategies) — the current product
  quality-gate    conviction filter only (volume expansion + impulse)
  full-gate       conviction + macro alignment (NIFTY / USDINR / crude)

Usage:
    python scripts/intraday_sim.py --split test
    python scripts/intraday_sim.py --split all --capital 1000000
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.candle_cache import drop_synthetic_bars  # noqa: E402
from nse_intraday_ai.entry_quality import (  # noqa: E402
    EntryQuality,
    GateConfig,
    macro_alignment,
    passes_entry_gate,
)
from nse_intraday_ai.portfolio_sim import IntradayPortfolioSimulator, SimConfig  # noqa: E402

DB = ROOT / "data" / "candles.sqlite3"
EVENTS = ROOT / "data" / "entry_timing_events.pkl"
CHANGE_BARS = 12
SPLIT_DATE = pd.Timestamp("2026-07-09", tz="Asia/Kolkata")


def load_frame(symbol: str, since: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    try:
        legacy = pd.read_sql_query(
            "SELECT ts, open, high, low, close, volume FROM candles"
            " WHERE symbol=? AND interval='1m' AND ts>=? ORDER BY ts",
            con, params=(symbol, since),
        )
        native = pd.read_sql_query(
            "SELECT ts, open, high, low, close, volume FROM candles"
            " WHERE symbol=? AND interval='5m' AND ts>=? ORDER BY ts",
            con, params=(symbol, since),
        )
    finally:
        con.close()

    def prep(df):
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
        return df.set_index("ts")

    legacy, native = prep(legacy), prep(native)
    if not legacy.empty:
        legacy = (
            legacy.resample("5min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["open", "high", "low", "close"])
        )
    if legacy.empty:
        frame = native
    elif native.empty:
        frame = legacy
    else:
        frame = pd.concat([legacy[~legacy.index.isin(native.index)], native]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    # Yahoo appends a snapshot row off the interval grid; it is a quote, not a
    # bar, and filling against it fabricates a price that never traded.
    return drop_synthetic_bars(frame, "5m")


def macro_panel(since: str) -> pd.DataFrame:
    panel = pd.DataFrame()
    con = sqlite3.connect(DB)
    try:
        for name, symbol in (("nifty", "^NSEI"), ("usdinr", "USDINR=X"), ("crude", "CL=F")):
            df = pd.read_sql_query(
                "SELECT ts, close FROM candles WHERE symbol=? AND interval='5m'"
                " AND ts>=? ORDER BY ts",
                con, params=(symbol, since),
            )
            if df.empty:
                raise SystemExit(f"missing {symbol}; run scripts/backfill_context.py")
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
            series = df.set_index("ts")["close"]
            panel[name] = series[~series.index.duplicated(keep="last")].pct_change(CHANGE_BARS) * 100
    finally:
        con.close()
    return panel.sort_index().ffill()


def build_signals(split: str) -> pd.DataFrame:
    """Engine signals joined to their macro context and scored by the gate."""
    events = pd.read_pickle(EVENTS).dropna(subset=["ret_6"]).sort_values("ts")
    if split == "test":
        events = events[events["ts"] >= SPLIT_DATE]
    elif split == "train":
        events = events[events["ts"] < SPLIT_DATE]

    since = (events["ts"].min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    panel = macro_panel(since)
    merged = pd.merge_asof(
        events, panel, left_on="ts", right_index=True,
        direction="backward", tolerance=pd.Timedelta("45min"),
    )

    macro_scores, macro_ok, notes = [], [], []
    for row in merged.itertuples():
        macro = macro_alignment(
            row.side,
            nifty_change_pct=None if pd.isna(row.nifty) else row.nifty,
            usdinr_change_pct=None if pd.isna(row.usdinr) else row.usdinr,
            crude_change_pct=None if pd.isna(row.crude) else row.crude,
        )
        quality = EntryQuality(
            volume_z=row.vol_z, impulse_atr=row.run6,
            extension_vwap_atr=row.ext_vwap, rsi=row.rsi,
            bars_since_extreme=row.age_extreme,
        )
        gate = passes_entry_gate(quality, macro, config=GateConfig())
        macro_scores.append(macro.score)
        macro_ok.append(gate.allow)
        notes.append(gate.reason if gate.allow else "")
    merged["macro_score"] = macro_scores
    merged["full_gate"] = macro_ok
    merged["note"] = notes
    merged["quality_gate"] = (merged["vol_z"] >= 2.0) & (merged["run6"] >= 1.5)
    merged["engine_gate"] = (
        (merged["conf"] >= 70) & (merged["rr"] >= 1.5)
        & merged["strategies"].str.contains(",", na=False)
    )
    # Composite rank for choosing between simultaneous signals.  Built only
    # from the features that survived the out-of-sample split — macro
    # alignment, volume expansion and impulse size — with volume and impulse
    # capped so one freak print cannot dominate the ranking.
    merged["rank_score"] = (
        merged["macro_score"]
        + merged["vol_z"].clip(0, 5)
        + merged["run6"].clip(0, 4)
    )
    return merged


def liquid_symbols(limit: int = 150) -> set[str]:
    """The most liquid cached symbols, by median 5m rupee turnover.

    Slippage is the largest single cost term, and it is not uniform: a 2.5 bps
    per-leg assumption is pessimistic for a top-100 name and optimistic for a
    thin mid-cap.  Restricting the tradable universe to names where the
    assumption is defensible is a real design choice, not a data filter.
    """
    con = sqlite3.connect(DB)
    try:
        rows = con.execute(
            "SELECT symbol, AVG(close * volume) t FROM candles"
            " WHERE interval='5m' AND symbol LIKE '%.NS' AND ts>='2026-07-01'"
            " GROUP BY symbol ORDER BY t DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


def run_variant(name: str, signals: pd.DataFrame, frames: dict, config: SimConfig,
                *, verbose: bool = False) -> None:
    sim = IntradayPortfolioSimulator(config)
    payload = signals[["ts", "symbol", "side", "atr", "macro_score", "note"]].copy()
    payload = payload.rename(columns={"macro_score": "rank"})
    result = sim.run(payload, frames, verbose=verbose)
    print(f"\n{'─' * 78}\n{name}  ({len(signals):,} gated signals)\n{'─' * 78}")
    print(result.summary())
    if not result.trades.empty:
        by_reason = result.trades.groupby("exit_reason").agg(
            n=("net_pnl", "size"), net=("net_pnl", "sum"), avg=("net_pnl", "mean")
        ).round(0).sort_values("net", ascending=False)
        print(f"\n  exits:\n{by_reason.to_string()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["all", "train", "test"], default="test")
    parser.add_argument("--capital", type=float, default=10_00_000.0)
    parser.add_argument("--max-positions", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"building signals ({args.split} split)...")
    signals = build_signals(args.split)
    sessions = signals["ts"].dt.date.nunique()
    print(f"{len(signals):,} raw engine signals over {sessions} sessions "
          f"({signals['ts'].min():%Y-%m-%d} .. {signals['ts'].max():%Y-%m-%d})")
    for column, label in (("engine_gate", "engine gates"), ("quality_gate", "quality gate"),
                          ("full_gate", "full gate")):
        passing = int(signals[column].sum())
        print(f"  {label:14s} {passing:7,} signals ({passing / sessions:6.1f}/session)")

    wanted = sorted(set(signals.loc[
        signals["engine_gate"] | signals["quality_gate"] | signals["full_gate"], "symbol"
    ]))
    since = (signals["ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\nloading price frames for {len(wanted)} symbols...")
    frames = {}
    for symbol in wanted:
        frame = load_frame(symbol, since)
        if not frame.empty:
            frames[symbol] = frame
    print(f"loaded {len(frames)} frames")

    config = SimConfig(
        starting_capital=args.capital,
        max_concurrent_positions=args.max_positions,
    )
    print(f"\ncapital ₹{args.capital:,.0f} | max {config.max_concurrent_positions} concurrent"
          f" | risk {config.risk_per_trade_pct}%/trade"
          f" | square-off {config.square_off_at}")

    results = {}
    for name, column in (
        ("A. engine gates (current product)", "engine_gate"),
        ("B. quality gate (volume + impulse)", "quality_gate"),
        ("C. full gate (quality + macro)", "full_gate"),
    ):
        subset = signals[signals[column]]
        results[name] = run_variant(name, subset, frames, config, verbose=args.verbose)

    print(f"\n{'═' * 78}\nHEAD TO HEAD\n{'═' * 78}")
    rows = []
    for name, result in results.items():
        if result is None or result.trades.empty:
            rows.append({"variant": name, "trades": 0})
            continue
        equity = result.equity["equity"]
        rows.append({
            "variant": name,
            "trades": len(result.trades),
            "net_pnl": round(result.pnl),
            "pnl_%": round(result.pnl_pct, 2),
            "win_%": round((result.trades["net_pnl"] > 0).mean() * 100, 1),
            "gross": round(result.trades["gross_pnl"].sum()),
            "costs": round(result.trades["costs"].sum()),
            "max_dd_%": round((equity.cummax() - equity).max() / args.capital * 100, 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
