"""Read scripts/entry_timing_study.py output and answer: are signals late?

Prints conditional forward-return tables for every candidate-timing feature.
Returns are in ATR units at the 6-bar (30-minute) horizon — the user's stated
scalp horizon — and in bps so they can be read against the ~18 bps round trip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "entry_timing_events.pkl"
COST_BPS = 18.0


def bucket_table(df: pd.DataFrame, column: str, bins, *, label: str | None = None) -> pd.DataFrame:
    cut = pd.cut(df[column], bins, duplicates="drop")
    grouped = df.groupby(cut, observed=True)
    out = pd.DataFrame({
        "n": grouped.size(),
        "ret1_atr": grouped["ret_1"].mean().round(4),
        "ret6_atr": grouped["ret_6"].mean().round(4),
        "ret12_atr": grouped["ret_12"].mean().round(4),
        "ret6_bps": grouped["ret6_bps"].mean().round(2),
        "mfe6": grouped["mfe_6"].mean().round(3),
        "mae6": grouped["mae_6"].mean().round(3),
        "adverse1_%": (grouped["ret_1"].apply(lambda s: (s < 0).mean()) * 100).round(1),
        "net6_win%": (grouped["ret6_bps"].apply(lambda s: (s > COST_BPS).mean()) * 100).round(1),
    })
    out.index.name = label or column
    return out


def main() -> None:
    df = pd.read_pickle(EVENTS)
    df = df.dropna(subset=["ret_6", "ret_1"])
    print(f"{len(df):,} candidate signals | {df['ts'].dt.date.nunique()} sessions "
          f"| {df['symbol'].nunique()} symbols")
    print(f"window {df['ts'].min()} .. {df['ts'].max()}\n")

    print("=" * 78)
    print("BASELINE — every signal the engine produces (pre-gate)")
    print("=" * 78)
    for horizon in (1, 3, 6, 12):
        col = f"ret_{horizon}"
        print(f"  +{horizon:2d} bars: mean {df[col].mean():+.4f} ATR   "
              f"median {df[col].median():+.4f}   "
              f"P(adverse) {(df[col] < 0).mean():.1%}")
    print(f"  +6 bars in bps: mean {df['ret6_bps'].mean():+.2f} "
          f"(round-trip cost ≈ {COST_BPS:.0f} bps)")
    print(f"  MFE6 {df['mfe_6'].mean():.3f} ATR vs MAE6 {df['mae_6'].mean():.3f} ATR "
          f"→ ratio {df['mfe_6'].mean() / max(df['mae_6'].mean(), 1e-9):.3f}")
    print(f"  P(first bar after entry already adverse) = {(df['ret_1'] < 0).mean():.1%}\n")

    print("=" * 78)
    print("Q1: IS THE SIGNAL LATE?  forward return vs how extended price already is")
    print("=" * 78)
    for column, bins, label in [
        ("ext_vwap", [-99, -1, 0, 0.5, 1, 1.5, 2, 3, 99], "entry distance from VWAP (ATR, signed toward trade)"),
        ("ext_ema21", [-99, -0.5, 0, 0.5, 1, 1.5, 2, 99], "entry distance from EMA-21 (ATR)"),
        ("run6", [-99, -0.5, 0, 0.5, 1, 1.5, 2, 3, 99], "6-bar run-up before entry (ATR)"),
        ("run12", [-99, -1, 0, 1, 2, 3, 5, 99], "12-bar run-up before entry (ATR)"),
        ("pos_in_range", [-9, 0.2, 0.4, 0.6, 0.8, 0.95, 9], "position in day's range (1.0 = at the extreme)"),
        ("ext_open", [-99, -2, -1, 0, 1, 2, 3, 99], "move from session open (ATR)"),
    ]:
        print(f"\n{bucket_table(df, column, bins, label=label).to_string()}")

    print("\n" + "=" * 78)
    print("Q2: IMPULSE AGE — how long has the move been running?")
    print("=" * 78)
    for column, bins, label in [
        ("age_ema21", [-1, 1, 2, 4, 8, 15, 30, 999], "bars since close crossed EMA-21 in trade direction"),
        ("age_vwap", [-1, 1, 2, 4, 8, 15, 30, 999], "bars since close crossed VWAP in trade direction"),
        ("age_extreme", [-1, 0, 1, 2, 4, 8, 20, 999], "bars since the day's extreme was set"),
    ]:
        print(f"\n{bucket_table(df, column, bins, label=label).to_string()}")

    print("\n" + "=" * 78)
    print("Q3: CONTEXT — does the engine's own confidence rank anything?")
    print("=" * 78)
    for column, bins, label in [
        ("conf", [0, 60, 65, 70, 75, 80, 85, 101], "engine confidence"),
        ("rsi", [0, 30, 40, 50, 60, 70, 101], "RSI at entry"),
        ("adx", [0, 15, 20, 25, 30, 40, 999], "ADX at entry"),
        ("vol_z", [-9, 0, 0.5, 1, 2, 3, 99], "volume z-score at entry"),
        ("minute", [554, 570, 600, 660, 720, 795, 870, 906], "minute of day (IST)"),
    ]:
        print(f"\n{bucket_table(df, column, bins, label=label).to_string()}")

    print("\n" + "=" * 78)
    print("Q4: PER-STRATEGY (single-strategy signals only)")
    print("=" * 78)
    solo = df[~df["strategies"].str.contains(",", na=False)]
    grouped = solo.groupby("strategies")
    table = pd.DataFrame({
        "n": grouped.size(),
        "ret6_atr": grouped["ret_6"].mean().round(4),
        "ret6_bps": grouped["ret6_bps"].mean().round(2),
        "adverse1_%": (grouped["ret_1"].apply(lambda s: (s < 0).mean()) * 100).round(1),
        "net6_win%": (grouped["ret6_bps"].apply(lambda s: (s > COST_BPS).mean()) * 100).round(1),
    }).sort_values("ret6_bps", ascending=False)
    print(table.to_string())

    print("\n" + "=" * 78)
    print("Q5: SIDE and REGIME")
    print("=" * 78)
    for key in ("side", "regime"):
        grouped = df.groupby(key)
        print(f"\n{pd.DataFrame({'n': grouped.size(), 'ret6_bps': grouped['ret6_bps'].mean().round(2), 'ret6_atr': grouped['ret_6'].mean().round(4), 'adverse1_%': (grouped['ret_1'].apply(lambda s: (s < 0).mean()) * 100).round(1)}).to_string()}")

    print("\n" + "=" * 78)
    print("Q6: LINEAR ATTRIBUTION — corr(feature, forward 6-bar ATR return)")
    print("=" * 78)
    features = ["ext_vwap", "ext_ema21", "ext_ema9", "ext_open", "run3", "run6", "run12",
                "pos_in_range", "age_ema21", "age_vwap", "age_extreme", "conf", "rr",
                "rsi", "adx", "vol_z", "minute", "atr_pct_price"]
    corr = df[features + ["ret_6"]].corr()["ret_6"].drop("ret_6").sort_values()
    print(corr.round(4).to_string())


if __name__ == "__main__":
    sys.exit(main())
