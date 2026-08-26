"""Does macro context actually predict NSE intraday outcomes?

The user's hypotheses, tested directly rather than assumed:

  #2 foreign markets (US / Japan / Europe / Hong Kong) improve predictions
  #3 commodity markets (crude, gold, dollar) improve predictions
  #4 currency, date, time and season matter

Every candidate signal from `entry_timing_study.py` is joined — causally, as-of
its own timestamp — to the macro panel, then each feature is scored on its
ability to separate winners from losers, **in-sample and out-of-sample**.  The
out-of-sample column is the only one that means anything: the companion
entry-quality search showed almost every in-sample effect in this dataset
evaporating on held-out days.

Signed features are evaluated as *alignment* with the trade direction (a
positive global-risk reading is bullish for a long and bearish for a short),
which is how the engine would actually consume them.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.indicators import add_indicators  # noqa: E402
from nse_intraday_ai.macro_context import (  # noqa: E402
    calendar_context,
    phase_weighted_global_risk,
    session_phase,
)

DB = ROOT / "data" / "candles.sqlite3"
EVENTS = ROOT / "data" / "entry_timing_events.pkl"
CHANGE_BARS = 12          # 12 x 5m = 1 hour momentum window


def load(symbol: str, since: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT ts, open, high, low, close, volume FROM candles"
            " WHERE symbol=? AND interval='5m' AND ts>=? ORDER BY ts",
            con, params=(symbol, since),
        )
    finally:
        con.close()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df.set_index("ts")[~df.set_index("ts").index.duplicated(keep="last")]


def build_panel(since: str) -> pd.DataFrame:
    """One row per 5m stamp with every macro reading, forward-filled causally."""
    panel = pd.DataFrame()

    def add_change(name: str, symbol: str) -> None:
        frame = load(symbol, since)
        if frame.empty:
            print(f"  ! {symbol} has no cached data")
            return
        panel[name] = frame["close"].pct_change(CHANGE_BARS) * 100

    add_change("es", "ES=F")
    add_change("n225", "^N225")
    add_change("hsi", "^HSI")
    add_change("dax", "^GDAXI")
    add_change("dxy", "DX-Y.NYB")
    add_change("usdinr", "USDINR=X")
    add_change("crude", "CL=F")
    add_change("gold", "GC=F")
    add_change("nifty", "^NSEI")

    vix = load("^INDIAVIX", since)
    if not vix.empty:
        panel["india_vix"] = vix["close"]
    usvix = load("^VIX", since)
    if not usvix.empty:
        panel["us_vix"] = usvix["close"]

    nifty = load("^NSEI", since)
    if not nifty.empty:
        ind = add_indicators(nifty)
        panel["nifty_regime"] = ind["regime"]
        # Opening gap of the index, constant through the session.
        daily_open = nifty["open"].groupby(nifty.index.normalize()).first()
        daily_close = nifty["close"].groupby(nifty.index.normalize()).last()
        gap = (daily_open / daily_close.shift(1) - 1.0) * 100
        panel["index_gap"] = pd.Series(
            nifty.index.normalize().map(gap).to_numpy(), index=nifty.index
        )

    panel = panel.sort_index()
    # Forward-fill: a reading stays the best estimate until the next print.
    # No backward fill anywhere — that would leak the future.
    return panel.ffill()


def main() -> None:
    events = pd.read_pickle(EVENTS).dropna(subset=["ret_6", "ret_12"]).sort_values("ts")
    since = (events["ts"].min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    print(f"{len(events):,} events | building macro panel since {since}")
    panel = build_panel(since)
    print(f"panel: {len(panel):,} rows, {panel.index.min()} .. {panel.index.max()}")

    merged = pd.merge_asof(
        events, panel, left_on="ts", right_index=True, direction="backward",
        tolerance=pd.Timedelta("45min"),
    )
    sign = np.where(merged["side"] == "LONG", 1.0, -1.0)

    # ── Directional alignment features ───────────────────────────────────────
    feats: dict[str, pd.Series] = {}
    for name in ("es", "n225", "hsi", "dax", "nifty", "crude", "gold"):
        if name in merged:
            feats[f"align_{name}"] = merged[name] * sign
    if "dxy" in merged:
        feats["align_dxy_inv"] = -merged["dxy"] * sign      # dollar up = risk off
    if "usdinr" in merged:
        feats["align_inr_inv"] = -merged["usdinr"] * sign   # INR weak = headwind
    if "index_gap" in merged:
        feats["align_gap"] = merged["index_gap"] * sign
    feats["phase_global"] = pd.Series(
        [
            phase_weighted_global_risk(
                ts,
                es_change_pct=None if pd.isna(e) else e,
                japan_change_pct=None if pd.isna(j) else j,
                china_change_pct=None if pd.isna(c) else c,
                europe_change_pct=None if pd.isna(d) else d,
            )
            for ts, e, j, c, d in zip(
                merged["ts"], merged.get("es", np.nan), merged.get("n225", np.nan),
                merged.get("hsi", np.nan), merged.get("dax", np.nan),
            )
        ],
        index=merged.index,
    ) * sign
    # Level features (not directional)
    for name in ("india_vix", "us_vix"):
        if name in merged:
            feats[name] = merged[name]

    frame = pd.DataFrame(feats)
    frame["ret6_bps"] = merged["ret6_bps"].to_numpy()
    frame["ret_6"] = merged["ret_6"].to_numpy()
    frame["ret_12"] = merged["ret_12"].to_numpy()
    frame["ts"] = merged["ts"].to_numpy()

    days = sorted(pd.Series(frame["ts"]).dt.normalize().unique())
    mid = days[len(days) // 2]
    train, test = frame[frame["ts"] < mid], frame[frame["ts"] >= mid]
    print(f"train < {pd.Timestamp(mid).date()}: {len(train):,} | test: {len(test):,}\n")

    print("=" * 96)
    print("MACRO FEATURES — correlation with forward return, in-sample vs out-of-sample")
    print("(a feature only matters if the sign survives the split and the size is meaningful)")
    print("=" * 96)
    rows = []
    for name in feats:
        tr = train[[name, "ret_6"]].dropna()
        te = test[[name, "ret_6"]].dropna()
        if len(tr) < 500 or len(te) < 500:
            continue
        c_tr = tr[name].corr(tr["ret_6"])
        c_te = te[name].corr(te["ret_6"])
        # Top-vs-bottom decile spread in bps, out of sample: the economically
        # readable version of the correlation.
        te_full = test[[name, "ret6_bps"]].dropna()
        lo, hi = te_full[name].quantile([0.1, 0.9])
        spread = (te_full.loc[te_full[name] >= hi, "ret6_bps"].mean()
                  - te_full.loc[te_full[name] <= lo, "ret6_bps"].mean())
        rows.append({
            "feature": name, "n_te": len(te),
            "corr_train": round(c_tr, 4), "corr_test": round(c_te, 4),
            "sign_holds": "yes" if c_tr * c_te > 0 else "NO",
            "d9_minus_d1_bps_te": round(spread, 2),
        })
    table = pd.DataFrame(rows).sort_values("corr_test", ascending=False)
    print(table.to_string(index=False))

    print("\n" + "=" * 96)
    print("CALENDAR — forward return by date/time bucket (out-of-sample half)")
    print("=" * 96)
    cal = pd.DataFrame({
        "ts": merged["ts"], "ret6_bps": merged["ret6_bps"], "ret_6": merged["ret_6"],
    })
    cal["dow"] = cal["ts"].dt.day_name().str[:3]
    cal["phase"] = [session_phase(t).name for t in cal["ts"]]
    cal["expiry"] = [
        "monthly" if calendar_context(t).is_monthly_expiry
        else "weekly" if calendar_context(t).is_weekly_expiry
        else "-" for t in cal["ts"]
    ]
    cal["month_end"] = [calendar_context(t).is_month_end for t in cal["ts"]]
    cal_te = cal[cal["ts"] >= mid]
    for key in ("dow", "phase", "expiry", "month_end"):
        grouped = cal_te.groupby(key)
        out = pd.DataFrame({
            "n": grouped.size(),
            "ret6_bps": grouped["ret6_bps"].mean().round(2),
            "ret6_atr": grouped["ret_6"].mean().round(4),
        })
        # Same table on the training half, to see whether it is a stable effect.
        grouped_tr = cal[cal["ts"] < mid].groupby(key)
        out["ret6_bps_train"] = grouped_tr["ret6_bps"].mean().round(2)
        print(f"\n{out.to_string()}")


if __name__ == "__main__":
    main()
