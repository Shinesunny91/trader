"""Follow-up to analyze_entry_timing.py: find the *joint* conditions that work.

The single-feature tables show momentum continuation pays only at the extremes
(big run-up, volume spike, deep RSI) while the bulk of the engine's signal flow
sits in a mushy, slightly-negative middle.  This script searches combinations,
and — crucially — checks every candidate rule on a *held-out* second half of
the window, because a conditional table on 280K in-sample events will always
find something.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "entry_timing_events.pkl"


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = sorted(df["ts"].dt.normalize().unique())
    mid = days[len(days) // 2]
    return df[df["ts"] < mid], df[df["ts"] >= mid]


def describe(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"rule": label, "n": 0}
    return {
        "rule": label,
        "n": len(frame),
        "per_day": round(len(frame) / max(frame["ts"].dt.date.nunique(), 1), 1),
        "ret6_bps": round(frame["ret6_bps"].mean(), 2),
        "ret6_atr": round(frame["ret_6"].mean(), 4),
        "ret12_atr": round(frame["ret_12"].mean(), 4),
        "mfe6": round(frame["mfe_6"].mean(), 3),
        "mae6": round(frame["mae_6"].mean(), 3),
        "edge": round(frame["mfe_6"].mean() - frame["mae_6"].mean(), 3),
    }


def main() -> None:
    df = pd.read_pickle(EVENTS).dropna(subset=["ret_6", "ret_12"])
    train, test = split(df)
    print(f"train {len(train):,} events to {train['ts'].max().date()} | "
          f"test {len(test):,} events from {test['ts'].min().date()}\n")

    # ── Candidate conditions, each a cheap boolean over the frame ────────────
    conditions = {
        "vol_z>1":        lambda d: d["vol_z"] > 1,
        "vol_z>2":        lambda d: d["vol_z"] > 2,
        "vol_z>3":        lambda d: d["vol_z"] > 3,
        "run6>1.5":       lambda d: d["run6"] > 1.5,
        "run6>2.5":       lambda d: d["run6"] > 2.5,
        "run12>3":        lambda d: d["run12"] > 3,
        "rsi_extreme":    lambda d: (d["rsi"] < 30) | (d["rsi"] > 70),
        "at_extreme":     lambda d: d["age_extreme"] <= 1,
        "afternoon":      lambda d: d["minute"].between(795, 870),
        "not_midday":     lambda d: ~d["minute"].between(600, 795),
        "long_only":      lambda d: d["side"] == "LONG",
        "wide_atr":       lambda d: d["atr_pct_price"] > 40,
        "breakout_strat": lambda d: d["strategies"].str.contains(
            "opening_range_breakout|volatility_compression_breakout", na=False),
        "multi_vote":     lambda d: d["strategies"].str.contains(",", na=False),
        "conf>=70":       lambda d: d["conf"] >= 70,
    }

    print("=" * 104)
    print("SINGLE CONDITIONS — in-sample (train) vs out-of-sample (test)")
    print("=" * 104)
    rows = []
    for name, fn in conditions.items():
        tr, te = describe(train[fn(train)], name), describe(test[fn(test)], name)
        rows.append({
            "rule": name, "n_tr": tr.get("n", 0), "bps_tr": tr.get("ret6_bps"),
            "n_te": te.get("n", 0), "bps_te": te.get("ret6_bps"),
            "atr6_te": te.get("ret6_atr"), "edge_te": te.get("edge"),
            "per_day_te": te.get("per_day"),
        })
    print(pd.DataFrame(rows).sort_values("bps_te", ascending=False).to_string(index=False))

    print("\n" + "=" * 104)
    print("PAIRS — ranked by out-of-sample bps, minimum 3 signals/day to be usable")
    print("=" * 104)
    rows = []
    for a, b in itertools.combinations(conditions, 2):
        mask_tr = conditions[a](train) & conditions[b](train)
        mask_te = conditions[a](test) & conditions[b](test)
        if mask_tr.sum() < 300 or mask_te.sum() < 150:
            continue
        tr, te = describe(train[mask_tr], f"{a} & {b}"), describe(test[mask_te], f"{a} & {b}")
        rows.append({
            "rule": f"{a} & {b}", "n_tr": tr["n"], "bps_tr": tr["ret6_bps"],
            "n_te": te["n"], "per_day_te": te["per_day"], "bps_te": te["ret6_bps"],
            "atr6_te": te["ret6_atr"], "atr12_te": te["ret12_atr"], "edge_te": te["edge"],
        })
    pairs = pd.DataFrame(rows).sort_values("bps_te", ascending=False)
    print(pairs.head(20).to_string(index=False))
    print("\nworst 5:")
    print(pairs.tail(5).to_string(index=False))

    print("\n" + "=" * 104)
    print("TRIPLES built on the best surviving pair components")
    print("=" * 104)
    rows = []
    for a, b, c in itertools.combinations(conditions, 3):
        mask_tr = conditions[a](train) & conditions[b](train) & conditions[c](train)
        mask_te = conditions[a](test) & conditions[b](test) & conditions[c](test)
        if mask_tr.sum() < 200 or mask_te.sum() < 100:
            continue
        tr, te = describe(train[mask_tr], ""), describe(test[mask_te], "")
        if tr["ret6_bps"] <= 0:
            continue   # must be positive in-sample too, or it is pure noise
        rows.append({
            "rule": f"{a} & {b} & {c}", "n_tr": tr["n"], "bps_tr": tr["ret6_bps"],
            "n_te": te["n"], "per_day_te": te["per_day"], "bps_te": te["ret6_bps"],
            "atr6_te": te["ret6_atr"], "edge_te": te["edge"],
        })
    triples = pd.DataFrame(rows).sort_values("bps_te", ascending=False)
    print(triples.head(20).to_string(index=False))

    # ── Stability: does the winner hold week by week? ────────────────────────
    print("\n" + "=" * 104)
    print("WEEKLY STABILITY of the headline rule (vol_z>2 & run6>1.5)")
    print("=" * 104)
    mask = (df["vol_z"] > 2) & (df["run6"] > 1.5)
    weekly = df[mask].groupby(pd.Grouper(key="ts", freq="W"))
    table = pd.DataFrame({
        "n": weekly.size(),
        "ret6_bps": weekly["ret6_bps"].mean().round(2),
        "ret6_atr": weekly["ret_6"].mean().round(4),
        "ret12_atr": weekly["ret_12"].mean().round(4),
    })
    baseline = df.groupby(pd.Grouper(key="ts", freq="W"))["ret6_bps"].mean().round(2)
    table["all_signals_bps"] = baseline
    print(table.to_string())


if __name__ == "__main__":
    main()
