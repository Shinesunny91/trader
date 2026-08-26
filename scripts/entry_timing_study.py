"""Entry-timing diagnostic: are the engine's signals *late*?

The user's field observation is that recommendations arrive after the move,
and what follows the signal is the reversal rather than the continuation.
This script tests that claim directly instead of arguing about it.

For every candidate signal the engine produces on cached 5m candles it
records, at signal time:

* how extended price already is (from session open, VWAP, EMA-9/21, in ATR)
* how much of a run already happened (returns over the last 3/6/12 bars, in ATR)
* how old the impulse is (bars since price crossed EMA-21 in the signal's
  direction, bars since the session extreme in that direction)
* microstructure/context: RSI, ADX, volume z, time of day, regime

and, from the future candles of the *same session*:

* MFE / MAE in ATR at +1, +3, +6, +12 bars
* the signed return at those horizons, in ATR and in bps
* whether the first bar after entry is already adverse ("immediate reversal")

Output is a set of conditional tables: forward return bucketed by extension,
by impulse age, by run-up.  If the "late signal" theory is right, forward
return must fall monotonically as pre-entry extension rises.

Usage:
    python scripts/entry_timing_study.py --workers 10 --limit 0
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.candle_cache import drop_synthetic_bars  # noqa: E402
from nse_intraday_ai.indicators import add_indicators  # noqa: E402
from nse_intraday_ai.models import Side  # noqa: E402
from nse_intraday_ai.risk import RiskConfig  # noqa: E402
from nse_intraday_ai.strategies import EnsembleConfig, VotingSignalEngine  # noqa: E402

DB = ROOT / "data" / "candles.sqlite3"
OUT = ROOT / "data" / "entry_timing_events.pkl"

# Strategies read at most ~70 bars of history, and the opening-range strategy
# needs the whole session (75 bars on a 5m NSE day).  A 200-bar tail therefore
# reproduces full-history behaviour while turning the per-bar replay from
# O(n^2) slicing into O(n).
TAIL = 200
WARMUP = 80
HORIZONS = (1, 3, 6, 12)


def read_candles(con, symbol: str, interval: str, since: str) -> pd.DataFrame:
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


def load_frame(symbol: str, since: str) -> pd.DataFrame:
    """5m frame: native 5m rows, backfilled with resampled legacy 1m rows."""
    con = sqlite3.connect(DB)
    try:
        legacy = read_candles(con, symbol, "1m", since)
        if not legacy.empty:
            legacy = (
                legacy.resample("5min")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna(subset=["open", "high", "low", "close"])
            )
        native = read_candles(con, symbol, "5m", since)
        if legacy.empty:
            return drop_synthetic_bars(native, "5m")
        if native.empty:
            return drop_synthetic_bars(legacy, "5m")
        stitched = pd.concat([legacy[~legacy.index.isin(native.index)], native]).sort_index()
        return drop_synthetic_bars(stitched, "5m")
    finally:
        con.close()


def nse_symbols(since: str, min_rows: int = 800) -> list[str]:
    con = sqlite3.connect(DB)
    try:
        rows = con.execute(
            "SELECT symbol, COUNT(*) c FROM candles WHERE ts>=? AND symbol LIKE '%.NS'"
            " GROUP BY symbol HAVING c>=? ORDER BY c DESC",
            (since, min_rows),
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _bars_since_cross(closes: np.ndarray, ref: np.ndarray, pos: int, side: int, cap: int = 60) -> int:
    """Bars since close last crossed `ref` in `side` direction (+1 long / -1 short)."""
    above = (closes - ref) * side > 0
    i = pos
    lo = max(0, pos - cap)
    while i > lo and above[i - 1]:
        i -= 1
    return pos - i


def study_symbol(args: tuple) -> list[dict]:
    symbol, since = args
    frame = load_frame(symbol, since)
    if frame.empty or len(frame) < WARMUP + 30:
        return []
    df = add_indicators(frame)
    if df.empty:
        return []

    engine = VotingSignalEngine(
        config=EnsembleConfig(min_agreeing_votes=1, min_vote_share=0.0, min_weighted_confidence=0.0)
    )
    lenient = RiskConfig(
        capital=1_000_000, risk_per_trade_pct=0.5, max_position_pct=25.0,
        min_confidence=0.0, min_reward_risk=0.0, estimated_cost_bps=15.0, slippage_bps=3.0,
    )

    idx = df.index
    day = idx.normalize()
    closes = df["close"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    opens = df["open"].to_numpy(float)
    atrs = df["atr_14"].to_numpy(float)
    vwaps = df["vwap"].to_numpy(float)
    ema9 = df["ema_9"].to_numpy(float)
    ema21 = df["ema_21"].to_numpy(float)
    rsis = df["rsi_14"].to_numpy(float)
    adxs = df["adx_14"].to_numpy(float)
    vz = df["volume_z"].to_numpy(float)
    regimes = df["regime"].to_numpy(object)
    minutes = np.asarray(idx.hour * 60 + idx.minute)

    # session start index for each bar
    day_codes = pd.factorize(day)[0]
    session_start = np.zeros(len(df), dtype=int)
    first_of_day: dict[int, int] = {}
    for i, code in enumerate(day_codes):
        if code not in first_of_day:
            first_of_day[code] = i
        session_start[i] = first_of_day[code]

    out: list[dict] = []
    for pos in range(WARMUP, len(df) - 1):
        # Only NSE session bars; no new entry in the last 25 minutes (square-off).
        if not (555 <= minutes[pos] <= 905):
            continue
        history = df.iloc[max(0, pos - TAIL + 1) : pos + 1]
        plan = engine.analyze_precomputed(symbol, history, lenient, {}, None)
        if not plan.is_actionable or plan.entry is None or plan.stop_loss is None:
            continue
        side = 1 if plan.side == Side.LONG else -1
        atr = atrs[pos]
        if not np.isfinite(atr) or atr <= 0:
            continue

        s0 = session_start[pos]
        sess_hi = highs[s0 : pos + 1].max()
        sess_lo = lows[s0 : pos + 1].min()
        entry = float(plan.entry)

        # ── pre-entry "how late are we" features ─────────────────────────────
        ext_open = side * (entry - opens[s0]) / atr          # from session open
        ext_vwap = side * (entry - vwaps[pos]) / atr         # from session VWAP
        ext_ema21 = side * (entry - ema21[pos]) / atr
        ext_ema9 = side * (entry - ema9[pos]) / atr
        run3 = side * (closes[pos] - closes[pos - 3]) / atr
        run6 = side * (closes[pos] - closes[pos - 6]) / atr
        run12 = side * (closes[pos] - closes[pos - 12]) / atr
        # Position inside the session range: 1.0 = entering long at the high of day.
        rng = max(sess_hi - sess_lo, atr * 0.1)
        pos_in_range = (entry - sess_lo) / rng if side > 0 else (sess_hi - entry) / rng
        age_ema21 = _bars_since_cross(closes, ema21, pos, side)
        age_vwap = _bars_since_cross(closes, vwaps, pos, side)
        # Bars since the session extreme in the trade direction was set.
        extreme_arr = highs[s0 : pos + 1] if side > 0 else lows[s0 : pos + 1]
        age_extreme = int(pos - s0 - (np.argmax(extreme_arr) if side > 0 else np.argmin(extreme_arr)))

        # ── forward path, same session only ──────────────────────────────────
        end = pos
        while end + 1 < len(df) and day_codes[end + 1] == day_codes[pos]:
            end += 1
        row: dict = {
            "symbol": symbol, "ts": idx[pos], "side": plan.side.value,
            "entry": entry, "stop": float(plan.stop_loss), "target": float(plan.target),
            "conf": float(plan.confidence), "rr": float(plan.reward_risk),
            "atr": float(atr), "atr_pct_price": float(atr / entry * 10_000),  # in bps
            "rsi": float(rsis[pos]), "adx": float(adxs[pos]), "vol_z": float(vz[pos]),
            "regime": str(regimes[pos]), "minute": int(minutes[pos]),
            "ext_open": float(ext_open), "ext_vwap": float(ext_vwap),
            "ext_ema21": float(ext_ema21), "ext_ema9": float(ext_ema9),
            "run3": float(run3), "run6": float(run6), "run12": float(run12),
            "pos_in_range": float(pos_in_range),
            "age_ema21": int(age_ema21), "age_vwap": int(age_vwap), "age_extreme": age_extreme,
            "strategies": ",".join(
                v.strategy for v in plan.strategy_votes if v.side == plan.side and v.is_trade
            ),
            "bars_left": end - pos,
        }
        for h in HORIZONS:
            j = min(pos + h, end)
            seg_hi = highs[pos + 1 : j + 1]
            seg_lo = lows[pos + 1 : j + 1]
            if len(seg_hi) == 0:
                row[f"ret_{h}"] = np.nan
                row[f"mfe_{h}"] = np.nan
                row[f"mae_{h}"] = np.nan
                continue
            row[f"ret_{h}"] = float(side * (closes[j] - entry) / atr)
            row[f"mfe_{h}"] = float(
                (seg_hi.max() - entry) / atr if side > 0 else (entry - seg_lo.min()) / atr
            )
            row[f"mae_{h}"] = float(
                (entry - seg_lo.min()) / atr if side > 0 else (seg_hi.max() - entry) / atr
            )
        # bps return at the 6-bar (30 min) horizon — the user's scalp horizon
        j6 = min(pos + 6, end)
        row["ret6_bps"] = float(side * (closes[j6] - entry) / entry * 10_000)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="2026-05-01")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="max symbols (0 = all)")
    args = ap.parse_args()

    symbols = nse_symbols(args.since)
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"{len(symbols)} symbols since {args.since}")

    started = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(study_symbol, (s, args.since)) for s in symbols]
        for done, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if done % 25 == 0 or done == len(symbols):
                print(f"  {done}/{len(symbols)} symbols, {len(rows)} events "
                      f"({time.time() - started:.0f}s)", flush=True)

    frame = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    frame.to_pickle(OUT)
    print(f"\n{len(frame)} candidate events -> {OUT}")
    print(f"window: {frame['ts'].min()} .. {frame['ts'].max()} "
          f"({frame['ts'].dt.date.nunique()} sessions)")


if __name__ == "__main__":
    main()
