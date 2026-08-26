"""Do the surviving macro features combine into a usable filter?

`macro_feature_study.py` found only three macro readings whose sign survives an
out-of-sample split — index momentum, USDINR and crude — while every foreign
equity market (S&P futures, Nikkei, DAX, Hang Seng) carried a large in-sample
correlation and *nothing* out of sample.

This script takes only the survivors, builds one alignment score, and grades it
with rolling weekly walk-forward: week N is scored by ranks fitted on weeks
< N only.  A score that is real should put the top bucket above the bottom
bucket in most weeks, not just on average.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DB = ROOT / "data" / "candles.sqlite3"
EVENTS = ROOT / "data" / "entry_timing_events.pkl"
CHANGE_BARS = 12


def load_close(symbol: str, since: str) -> pd.Series:
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT ts, close FROM candles WHERE symbol=? AND interval='5m' AND ts>=? ORDER BY ts",
            con, params=(symbol, since),
        )
    finally:
        con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
    series = df.set_index("ts")["close"]
    return series[~series.index.duplicated(keep="last")]


def main() -> None:
    events = pd.read_pickle(EVENTS).dropna(subset=["ret_6"]).sort_values("ts").reset_index(drop=True)
    since = (events["ts"].min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    panel = pd.DataFrame()
    for name, symbol in (("nifty", "^NSEI"), ("usdinr", "USDINR=X"), ("crude", "CL=F")):
        closes = load_close(symbol, since)
        if closes.empty:
            raise SystemExit(f"no cached data for {symbol}; run scripts/backfill_context.py")
        panel[name] = closes.pct_change(CHANGE_BARS) * 100
    panel = panel.sort_index().ffill()

    merged = pd.merge_asof(
        events, panel, left_on="ts", right_index=True,
        direction="backward", tolerance=pd.Timedelta("45min"),
    )
    sign = np.where(merged["side"] == "LONG", 1.0, -1.0)
    merged["a_nifty"] = merged["nifty"] * sign
    merged["a_inr"] = -merged["usdinr"] * sign      # INR strength helps equities
    merged["a_crude"] = -merged["crude"] * sign     # crude up hurts an oil importer
    merged["week"] = merged["ts"].dt.to_period("W")
    merged = merged.dropna(subset=["a_nifty", "a_inr", "a_crude"])

    print(f"{len(merged):,} events with full macro coverage, "
          f"{merged['week'].nunique()} weeks\n")

    # ── Rolling weekly walk-forward ──────────────────────────────────────────
    # Each week is standardised against the mean/sd of *prior* weeks only, so a
    # week's own distribution never informs its own score.
    weeks = sorted(merged["week"].unique())
    rows = []
    for i, week in enumerate(weeks):
        if i < 2:
            continue
        past = merged[merged["week"] < week]
        now = merged[merged["week"] == week].copy()
        if len(now) < 200 or len(past) < 2000:
            continue
        score = np.zeros(len(now))
        for column in ("a_nifty", "a_inr", "a_crude"):
            mu, sd = past[column].mean(), past[column].std()
            if sd > 0:
                score += ((now[column] - mu) / sd).clip(-3, 3).to_numpy()
        now["score"] = score
        top = now[now["score"] >= now["score"].quantile(0.9)]
        bot = now[now["score"] <= now["score"].quantile(0.1)]
        rows.append({
            "week": str(week)[:10], "n": len(now),
            "all_bps": round(now["ret6_bps"].mean(), 2),
            "top10_bps": round(top["ret6_bps"].mean(), 2),
            "bot10_bps": round(bot["ret6_bps"].mean(), 2),
            "spread": round(top["ret6_bps"].mean() - bot["ret6_bps"].mean(), 2),
            "top10_atr": round(top["ret_6"].mean(), 4),
        })
    table = pd.DataFrame(rows)
    print("=" * 88)
    print("ROLLING WEEKLY WALK-FORWARD — macro alignment score (nifty + INR + crude)")
    print("=" * 88)
    print(table.to_string(index=False))
    wins = (table["spread"] > 0).sum()
    print(f"\ntop-decile beats bottom-decile in {wins}/{len(table)} weeks")
    print(f"mean top-decile {table['top10_bps'].mean():+.2f} bps vs "
          f"all-signal {table['all_bps'].mean():+.2f} bps "
          f"(lift {table['top10_bps'].mean() - table['all_bps'].mean():+.2f} bps)")
    print(f"median weekly spread {table['spread'].median():+.2f} bps")

    # ── Does it survive being combined with the entry-quality survivors? ─────
    print("\n" + "=" * 88)
    print("MACRO SCORE x ENTRY QUALITY (vol_z>2 & run6>2.5), out-of-sample half")
    print("=" * 88)
    days = sorted(merged["ts"].dt.normalize().unique())
    mid = days[len(days) // 2]
    past = merged[merged["ts"] < mid]
    test = merged[merged["ts"] >= mid].copy()
    score = np.zeros(len(test))
    for column in ("a_nifty", "a_inr", "a_crude"):
        mu, sd = past[column].mean(), past[column].std()
        if sd > 0:
            score += ((test[column] - mu) / sd).clip(-3, 3).to_numpy()
    test["score"] = score
    quality = (test["vol_z"] > 2) & (test["run6"] > 2.5)
    for label, subset in (
        ("all test events", test),
        ("macro top 25%", test[test["score"] >= test["score"].quantile(0.75)]),
        ("quality only", test[quality]),
        ("quality & macro top 50%", test[quality & (test["score"] >= test["score"].median())]),
        ("quality & macro top 25%", test[quality & (test["score"] >= test["score"].quantile(0.75))]),
    ):
        if subset.empty:
            continue
        per_day = len(subset) / max(subset["ts"].dt.date.nunique(), 1)
        print(f"  {label:26s} n={len(subset):6d} ({per_day:6.1f}/day)  "
              f"ret6 {subset['ret6_bps'].mean():+6.2f} bps  "
              f"{subset['ret_6'].mean():+.4f} ATR  "
              f"MFE-MAE {subset['mfe_6'].mean() - subset['mae_6'].mean():+.3f}")

    print("\n" + "=" * 88)
    print("BY SIDE — is the macro score just long-bias in a rising tape?")
    print("=" * 88)
    for side in ("LONG", "SHORT"):
        sub = test[test["side"] == side]
        top = sub[sub["score"] >= sub["score"].quantile(0.75)]
        print(f"  {side:5s} all n={len(sub):6d} {sub['ret6_bps'].mean():+6.2f} bps | "
              f"macro-top n={len(top):6d} {top['ret6_bps'].mean():+6.2f} bps")


if __name__ == "__main__":
    main()
