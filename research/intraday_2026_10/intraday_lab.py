"""Intraday strategy lab: 5-minute NIFTY-500 panel, literature strategies, honest costs.

Run: PYTHONPATH=src .venv/bin/python <this file>
Splits: in-sample = sessions before SPLIT (parameter choice), out-of-sample after.
Every trade pays COST_BPS round trip (Groww MIS, 3 bps slippage per leg).
"""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/shine/trading-workspace")
DB = ROOT / "data" / "candles.sqlite3"
COST_BPS = 13.3
SPLIT = "2026-08-25"          # IS: Jul 1..Aug 24 (~38 sessions); OOS: Aug 25.. (~28+)
NBARS = 75                   # 09:15 .. 15:25 (5-minute)
SQ_BAR = 72                  # 15:15 bar -> exit at its open
OUT = Path(__file__).with_suffix(".json")


def load_panel():
    syms = sorted(pd.read_csv(ROOT / "data/nifty500_symbols.csv")["Symbol"].astype(str).str.strip() + ".NS")
    con = sqlite3.connect(DB)
    q = ("select symbol, ts, open, high, low, close, volume from candles where interval='5m' "
         f"and ts >= '2026-06-30' and symbol in ({','.join('?' * len(syms))})")
    df = pd.read_sql(q, con, params=syms)
    ts = pd.to_datetime(df.ts, utc=True).dt.tz_convert("Asia/Kolkata")
    df["day"] = ts.dt.date.astype(str)
    df["bar"] = ((ts.dt.hour * 60 + ts.dt.minute) - 555) // 5
    df = df[(df.bar >= 0) & (df.bar < NBARS)]
    good = df.groupby("day").symbol.nunique()
    days = sorted(good[good >= 400].index)
    df = df[df.day.isin(days)]
    symbols = sorted(df.symbol.unique())
    di = {d: i for i, d in enumerate(days)}
    si = {s: i for i, s in enumerate(symbols)}
    arr = {k: np.full((len(days), len(symbols), NBARS), np.nan, dtype=np.float64)
           for k in ("open", "high", "low", "close", "volume")}
    d_i, s_i, b_i = df.day.map(di).to_numpy(), df.symbol.map(si).to_numpy(), df.bar.to_numpy()
    for k in arr:
        arr[k][d_i, s_i, b_i] = df[k].to_numpy()
    # daily ATR14 / prev close from the 1d cache, point-in-time (prior sessions only)
    dq = ("select symbol, ts, high, low, close from candles where interval='1d' and ts >= '2026-03-01' "
          f"and symbol in ({','.join('?' * len(symbols))})")
    dd = pd.read_sql(dq, con, params=symbols)
    dd["day"] = pd.to_datetime(dd.ts, utc=True).dt.tz_convert("Asia/Kolkata").dt.date.astype(str)
    atr = np.full((len(days), len(symbols)), np.nan)
    pclose = np.full((len(days), len(symbols)), np.nan)
    for s, g in dd.groupby("symbol"):
        g = g.drop_duplicates("day").set_index("day").sort_index()
        tr = pd.concat([g.high - g.low, (g.high - g.close.shift()).abs(), (g.low - g.close.shift()).abs()], axis=1).max(axis=1)
        a = tr.rolling(14).mean().shift(1)            # known before the session
        pc = g.close.shift(1)
        j = si[s]
        for d, i in di.items():
            if d in a.index:
                atr[i, j] = a.loc[d]
                pclose[i, j] = pc.loc[d]
    return days, symbols, arr, atr, pclose


# ── generic simulator ───────────────────────────────────────────────────────
def simulate(arr, day, sym, entry_bar, side, entry_px, stop_px, target_px=None, exit_bar=SQ_BAR):
    """Trade from entry_bar (filled at entry_px inside that bar) to stop/target/exit_bar open.

    The entry bar itself can hit the stop only if price went through it AFTER the
    entry: conservative rule -> if the entry bar's range also contains the stop,
    count it as stopped (worst case)."""
    o, h, l = arr["open"][day, sym], arr["high"][day, sym], arr["low"][day, sym]
    for b in range(entry_bar, exit_bar):
        if np.isnan(h[b]):
            continue
        if side > 0:
            if l[b] <= stop_px:
                fill = stop_px if b == entry_bar else min(stop_px, o[b])
                return (fill - entry_px) / entry_px * 1e4, "STOP"
            if target_px is not None and h[b] >= target_px and b > entry_bar:
                fill = max(target_px, o[b])
                return (fill - entry_px) / entry_px * 1e4, "TARGET"
        else:
            if h[b] >= stop_px:
                fill = stop_px if b == entry_bar else max(stop_px, o[b])
                return (entry_px - fill) / entry_px * 1e4, "STOP"
            if target_px is not None and l[b] <= target_px and b > entry_bar:
                fill = min(target_px, o[b])
                return (entry_px - fill) / entry_px * 1e4, "TARGET"
    px = o[exit_bar]
    if np.isnan(px):
        closes = arr["close"][day, sym][:exit_bar]
        closes = closes[~np.isnan(closes)]
        if closes.size == 0:
            return 0.0, "NODATA"
        px = closes[-1]
    return side * (px - entry_px) / entry_px * 1e4, "EOD"


# ── strategies ──────────────────────────────────────────────────────────────
def rel_volume(arr, nbars, lookback=14):
    """Volume of the first `nbars` bars vs its mean over the previous `lookback` sessions."""
    v = np.nansum(arr["volume"][:, :, :nbars], axis=2)
    v[v == 0] = np.nan
    base = pd.DataFrame(v).rolling(lookback, min_periods=10).mean().shift(1).to_numpy()
    return v / base


def orb(arr, atr, *, nbars=1, top=20, rv_min=1.0, stop_atr=0.10, both=False, min_price=50, range_stop=False):
    """Zarattini, Barbon & Aziz (2024) 'Stocks in Play' opening-range breakout.

    Opening range = first `nbars` 5-min bars.  Stocks in play: the `top` names by
    relative opening-range volume with RV >= rv_min.  Direction = sign of the
    opening-range candle (long if close > open).  Entry: stop order at the range
    high (long) / low (short) from the next bar.  Stop: stop_atr x daily ATR14.
    Exit at 15:15."""
    rv = rel_volume(arr, nbars)
    trades = []
    D, S, _ = arr["open"].shape
    for d in range(D):
        o0 = arr["open"][d, :, 0]
        c_or = arr["close"][d, :, nbars - 1]
        hi = np.nanmax(arr["high"][d, :, :nbars], axis=1)
        lo = np.nanmin(arr["low"][d, :, :nbars], axis=1)
        ok = (rv[d] >= rv_min) & (o0 >= min_price) & ~np.isnan(atr[d]) & (atr[d] > 0)
        cand = np.where(ok)[0]
        cand = cand[np.argsort(-rv[d, cand])][:top]
        for s in cand:
            side = 1 if c_or[s] > o0[s] else -1 if c_or[s] < o0[s] else 0
            if side == 0:
                continue
            for sd in ((side, -side) if both else (side,)):
                level = hi[s] if sd > 0 else lo[s]
                # first bar that trades through the level
                for b in range(nbars, SQ_BAR):
                    hb, lb, ob = arr["high"][d, s, b], arr["low"][d, s, b], arr["open"][d, s, b]
                    if np.isnan(hb):
                        continue
                    if (sd > 0 and hb > level) or (sd < 0 and lb < level):
                        entry = max(level, ob) if sd > 0 else min(level, ob)
                        stop = (lo[s] if sd > 0 else hi[s]) if range_stop else entry - sd * stop_atr * atr[d, s]
                        r, why = simulate(arr, d, s, b, sd, entry, stop)
                        trades.append((d, s, sd, b, r, why))
                        break
    return trades


def first_hour_momentum(arr, atr, *, n=10, rv_min=1.5, stop_atr=0.5, at_bar=12):
    """Continuation of the first-hour move among high-volume names (Gao, Han, Li & Zhou
    2018 find intraday momentum; stock-level variants in Indian studies).  At 10:15,
    long the n strongest / short the n weakest (by return since the open, in ATR
    units) among names with RV(first hour) >= rv_min.  Stop stop_atr x ATR; exit 15:15."""
    rv = rel_volume(arr, at_bar)
    trades = []
    D = arr["open"].shape[0]
    for d in range(D):
        o0 = arr["open"][d, :, 0]
        px = arr["close"][d, :, at_bar - 1]
        move = (px - o0) / atr[d]
        ok = (rv[d] >= rv_min) & ~np.isnan(move) & (o0 >= 50)
        idx = np.where(ok)[0]
        if len(idx) < 2 * n:
            continue
        order = idx[np.argsort(move[idx])]
        for s, sd in [(s, -1) for s in order[:n]] + [(s, 1) for s in order[-n:]]:
            entry = arr["open"][d, s, at_bar]
            if np.isnan(entry):
                continue
            stop = entry - sd * stop_atr * atr[d, s]
            r, why = simulate(arr, d, s, at_bar, sd, entry, stop)
            trades.append((d, s, sd, at_bar, r, why))
    return trades


def gap_fade_today(arr, atr, pclose, *, min_gap_atr=1.0, n=10, stop_atr=0.5, target_frac=1.0):
    """Fade today's opening gap toward the previous close (gap-fill).  Shorts gap-ups,
    buys gap-downs, the n largest |gap| in ATR units above min_gap_atr; entry at the
    09:20 open (bar 1), stop stop_atr x ATR, target = target_frac of the gap filled."""
    trades = []
    D = arr["open"].shape[0]
    for d in range(D):
        o0 = arr["open"][d, :, 0]
        gap = (o0 - pclose[d]) / atr[d]
        ok = (np.abs(gap) >= min_gap_atr) & (o0 >= 50)
        idx = np.where(ok)[0]
        idx = idx[np.argsort(-np.abs(gap[idx]))][:n]
        for s in idx:
            sd = -1 if gap[s] > 0 else 1
            entry = arr["open"][d, s, 1]
            if np.isnan(entry):
                continue
            stop = entry - sd * stop_atr * atr[d, s]
            target = entry + sd * target_frac * abs(entry - pclose[d, s]) if target_frac else None
            r, why = simulate(arr, d, s, 1, sd, entry, stop, target)
            trades.append((d, s, sd, 1, r, why))
    return trades


def vwap_reversion(arr, atr, *, k=1.0, start_bar=12, stop_atr=0.5, n_max=10):
    """Fade stretches from VWAP: after 10:15, when close deviates more than k x ATR
    from session VWAP, enter against it at the next open; target VWAP at entry time;
    stop stop_atr x ATR beyond; exit 15:15.  One trade per name per day."""
    trades = []
    D, S, _ = arr["open"].shape
    tp = (arr["high"] + arr["low"] + arr["close"]) / 3
    v = np.nan_to_num(arr["volume"])
    vwap = np.cumsum(np.nan_to_num(tp) * v, axis=2) / np.where(np.cumsum(v, axis=2) > 0, np.cumsum(v, axis=2), np.nan)
    for d in range(D):
        taken = 0
        dev = (arr["close"][d] - vwap[d]) / atr[d][:, None]
        for b in range(start_bar, SQ_BAR - 6):
            hits = np.where(np.abs(dev[:, b]) >= k)[0]
            for s in hits:
                if taken >= n_max:
                    break
                sd = -1 if dev[s, b] > 0 else 1
                entry = arr["open"][d, s, b + 1]
                if np.isnan(entry):
                    continue
                stop = entry - sd * stop_atr * atr[d, s]
                r, why = simulate(arr, d, s, b + 1, sd, entry, stop, vwap[d, s, b])
                trades.append((d, s, sd, b + 1, r, why))
                taken += 1
                dev[s, :] = 0                       # one trade per name per day
    return trades


def day_high_breakout(arr, atr, *, after_bar=12, rv_min=2.0, stop_atr=0.3, n_max=10):
    """Momentum ignition: after 10:15, the first break of the session high (low) by a
    name whose volume so far is >= rv_min x normal; enter at the breakout level,
    stop stop_atr x ATR, exit 15:15.  Long highs / short lows."""
    trades = []
    D, S, _ = arr["open"].shape
    cumv = np.nancumsum(arr["volume"], axis=2)
    base = {}
    for b in range(after_bar - 1, SQ_BAR):
        vb = cumv[:, :, b].copy()
        vb[vb == 0] = np.nan
        base[b] = vb / pd.DataFrame(vb).rolling(14, min_periods=10).mean().shift(1).to_numpy()
    for d in range(D):
        taken, done = 0, set()
        runhi = np.fmax.accumulate(np.nan_to_num(arr["high"][d], nan=-np.inf), axis=1)
        runlo = np.fmin.accumulate(np.nan_to_num(arr["low"][d], nan=np.inf), axis=1)
        for b in range(after_bar, SQ_BAR - 3):
            rvb = base[b - 1][d]
            for sd, brk in ((1, arr["high"][d, :, b] > runhi[:, b - 1]), (-1, arr["low"][d, :, b] < runlo[:, b - 1])):
                for s in np.where(brk & (rvb >= rv_min))[0]:
                    if s in done or taken >= n_max or np.isnan(atr[d, s]):
                        continue
                    level = runhi[s, b - 1] if sd > 0 else runlo[s, b - 1]
                    ob = arr["open"][d, s, b]
                    entry = max(level, ob) if sd > 0 else min(level, ob)
                    stop = entry - sd * stop_atr * atr[d, s]
                    r, why = simulate(arr, d, s, b, sd, entry, stop)
                    trades.append((d, s, sd, b, r, why))
                    taken += 1
                    done.add(s)
    return trades


# ── evaluation ──────────────────────────────────────────────────────────────
def evaluate(trades, days):
    if not trades:
        return None
    t = pd.DataFrame(trades, columns=["d", "s", "side", "bar", "gross", "why"])
    t["net"] = t.gross - COST_BPS
    t["day"] = [days[i] for i in t.d]
    daily = t.groupby("day").net.mean()                  # equal-weight book per day
    out = {}
    for name, mask in (("IS", daily.index < SPLIT), ("OOS", daily.index >= SPLIT), ("ALL", daily.index == daily.index)):
        x = daily[mask]
        tt = t[t.day.isin(x.index)]
        if len(x) < 5:
            continue
        out[name] = {"days": int(len(x)), "trades": int(len(tt)), "net_bps_trade": round(float(tt.net.mean()), 2),
                     "gross_bps_trade": round(float(tt.gross.mean()), 2),
                     "daily_mean": round(float(x.mean()), 2),
                     "t": round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std() > 0 else 0.0, 2),
                     "hit": round(float((tt.net > 0).mean()), 3),
                     "pos_days": round(float((x > 0).mean()), 3),
                     "stops": round(float((tt.why == "STOP").mean()), 3),
                     "long_net": round(float(tt[tt.side > 0].net.mean()), 2) if (tt.side > 0).any() else None,
                     "short_net": round(float(tt[tt.side < 0].net.mean()), 2) if (tt.side < 0).any() else None}
    return out


def main():
    days, symbols, arr, atr, pclose = load_panel()
    print(f"panel: {len(days)} sessions {days[0]}..{days[-1]}, {len(symbols)} symbols; "
          f"IS {sum(d < SPLIT for d in days)} / OOS {sum(d >= SPLIT for d in days)}", flush=True)
    grid = {}
    for nb in (1, 3, 6):
        for top in (10, 20):
            for st in (0.05, 0.10, 0.25):
                grid[f"orb{nb*5}_top{top}_stop{st}"] = lambda nb=nb, top=top, st=st: orb(arr, atr, nbars=nb, top=top, stop_atr=st)
    for nb in (3, 6, 12):
        for top in (10, 20):
            grid[f"orbR{nb*5}_top{top}"] = lambda nb=nb, top=top: orb(arr, atr, nbars=nb, top=top, range_stop=True)
    for rv in (1.0, 2.0):
        for st in (0.25, 0.5):
            grid[f"fhmom_rv{rv}_stop{st}"] = lambda rv=rv, st=st: first_hour_momentum(arr, atr, rv_min=rv, stop_atr=st)
    for g in (0.5, 1.0):
        for st in (0.5, 1.0):
            grid[f"gapfade_g{g}_stop{st}"] = lambda g=g, st=st: gap_fade_today(arr, atr, pclose, min_gap_atr=g, stop_atr=st)
    for k in (0.75, 1.25):
        grid[f"vwaprev_k{k}"] = lambda k=k: vwap_reversion(arr, atr, k=k)
    for rv in (1.5, 3.0):
        for st in (0.2, 0.4):
            grid[f"dhb_rv{rv}_stop{st}"] = lambda rv=rv, st=st: day_high_breakout(arr, atr, rv_min=rv, stop_atr=st)
    only = sys.argv[1:] or None
    results = {}
    for name, fn in grid.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        res = evaluate(fn(), days)
        results[name] = res
        if res:
            row = "  ".join(f"{k}: {v['net_bps_trade']:+6.1f}bps t={v['t']:+5.2f} n={v['trades']}"
                            for k, v in res.items() if k in ("IS", "OOS"))
            print(f"{name:<28} {row}", flush=True)
    OUT.write_text(json.dumps({"days": [days[0], days[-1]], "split": SPLIT, "cost_bps": COST_BPS,
                               "results": results}, indent=1))


if __name__ == "__main__":
    main()
