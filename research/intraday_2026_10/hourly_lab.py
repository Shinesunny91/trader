"""Hourly intraday lab: ~2 years of Yahoo 1h bars for NIFTY-500, same cost model.

Bars (IST): 09:15, 10:15, 11:15, 12:15, 13:15, 14:15, 15:15 (partial).  Exit = open of
the 15:15 bar.  IS = first 60% of sessions, OOS = last 40%.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from intraday_lab import COST_BPS, ROOT, DB  # noqa: E402

NB, SQ = 7, 6
OUT = Path(__file__).with_suffix(".json")


def load():
    syms = sorted(pd.read_csv(ROOT / "data/nifty500_symbols.csv")["Symbol"].astype(str).str.strip() + ".NS")
    con = sqlite3.connect(DB)
    ph = ",".join("?" * len(syms))
    df = pd.read_sql(f"select symbol, ts, open, high, low, close, volume from candles where interval='1h' and symbol in ({ph})", con, params=syms)
    ts = pd.to_datetime(df.ts, utc=True).dt.tz_convert("Asia/Kolkata")
    df["day"] = ts.dt.date.astype(str)
    df["bar"] = ts.dt.hour - 9
    df = df[(df.bar >= 0) & (df.bar < NB) & (ts.dt.minute == 15)]
    good = df.groupby("day").symbol.nunique()
    days = sorted(good[good >= 300].index)
    days = [d for d in days if d < pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()]
    df = df[df.day.isin(days)]
    symbols = sorted(df.symbol.unique())
    di = {d: i for i, d in enumerate(days)}
    si = {s: i for i, s in enumerate(symbols)}
    arr = {k: np.full((len(days), len(symbols), NB), np.nan) for k in ("open", "high", "low", "close", "volume")}
    for k in arr:
        arr[k][df.day.map(di).to_numpy(), df.symbol.map(si).to_numpy(), df.bar.to_numpy()] = df[k].to_numpy()
    dd = pd.read_sql(f"select symbol, ts, high, low, close from candles where interval='1d' and ts >= '2024-01-01' and symbol in ({','.join('?'*len(symbols))})", con, params=symbols)
    dd["day"] = pd.to_datetime(dd.ts, utc=True).dt.tz_convert("Asia/Kolkata").dt.date.astype(str)
    atr = np.full((len(days), len(symbols)), np.nan)
    pclose = np.full((len(days), len(symbols)), np.nan)
    dayidx = pd.Index(days)
    for s, g in dd.groupby("symbol"):
        g = g.drop_duplicates("day").set_index("day").sort_index()
        tr = pd.concat([g.high - g.low, (g.high - g.close.shift()).abs(), (g.low - g.close.shift()).abs()], axis=1).max(axis=1)
        a = tr.rolling(14).mean().shift(1).reindex(dayidx)
        atr[:, si[s]] = a.to_numpy()
        pclose[:, si[s]] = g.close.shift(1).reindex(dayidx).to_numpy()
    return days, symbols, arr, atr, pclose


def sim(arr, d, s, b0, sd, entry, stop, target=None):
    o, h, l = arr["open"][d, s], arr["high"][d, s], arr["low"][d, s]
    for b in range(b0, SQ):
        if np.isnan(h[b]):
            continue
        if (sd > 0 and l[b] <= stop) or (sd < 0 and h[b] >= stop):
            fill = stop if b == b0 else (min(stop, o[b]) if sd > 0 else max(stop, o[b]))
            return sd * (fill - entry) / entry * 1e4, "STOP"
        if target is not None and b > b0 and ((sd > 0 and h[b] >= target) or (sd < 0 and l[b] <= target)):
            fill = max(target, o[b]) if sd > 0 else min(target, o[b])
            return sd * (fill - entry) / entry * 1e4, "TARGET"
    px = o[SQ]
    if np.isnan(px):
        c = arr["close"][d, s][:SQ]
        c = c[~np.isnan(c)]
        if not c.size:
            return 0.0, "NODATA"
        px = c[-1]
    return sd * (px - entry) / entry * 1e4, "EOD"


def relvol(arr, upto):
    v = np.nansum(arr["volume"][:, :, :upto], axis=2)
    v[v == 0] = np.nan
    return v / pd.DataFrame(v).rolling(20, min_periods=10).mean().shift(1).to_numpy()


def orb60(arr, atr, *, top=10, rv_min=1.5, stop="range", contra=False):
    rv = relvol(arr, 1)
    out = []
    for d in range(arr["open"].shape[0]):
        o0, c0, h0, l0 = (arr[k][d, :, 0] for k in ("open", "close", "high", "low"))
        ok = (rv[d] >= rv_min) & (o0 >= 50) & (atr[d] > 0)
        cand = np.where(ok)[0]
        cand = cand[np.argsort(-rv[d, cand])][:top]
        for s in cand:
            sd = 1 if c0[s] > o0[s] else -1 if c0[s] < o0[s] else 0
            if contra:
                sd = -sd
            if sd == 0:
                continue
            level = h0[s] if sd > 0 else l0[s]
            for b in range(1, SQ):
                hb, lb, ob = arr["high"][d, s, b], arr["low"][d, s, b], arr["open"][d, s, b]
                if np.isnan(hb):
                    continue
                if (sd > 0 and hb > level) or (sd < 0 and lb < level):
                    entry = max(level, ob) if sd > 0 else min(level, ob)
                    st = (l0[s] if sd > 0 else h0[s]) if stop == "range" else entry - sd * stop * atr[d, s]
                    r, why = sim(arr, d, s, b, sd, entry, st)
                    out.append((d, s, sd, b, r, why))
                    break
    return out


def rank_at(arr, atr, *, at=1, n=10, rv_min=1.0, stop_atr=0.5, follow=True, since="open", pclose=None):
    """At the open of bar `at`, rank by move since the open (or prev close) in ATR units;
    follow=True buys winners/shorts losers, False fades them.  Exit 15:15."""
    rv = relvol(arr, at)
    out = []
    for d in range(arr["open"].shape[0]):
        base = arr["open"][d, :, 0] if since == "open" else pclose[d]
        px = arr["close"][d, :, at - 1]
        mv = (px - base) / atr[d]
        ok = (rv[d] >= rv_min) & ~np.isnan(mv) & (arr["open"][d, :, 0] >= 50)
        idx = np.where(ok)[0]
        if len(idx) < 2 * n:
            continue
        order = idx[np.argsort(mv[idx])]
        legs = [(s, -1) for s in order[:n]] + [(s, 1) for s in order[-n:]]
        for s, sd in legs:
            sd = sd if follow else -sd
            entry = arr["open"][d, s, at]
            if np.isnan(entry):
                continue
            r, why = sim(arr, d, s, at, sd, entry, entry - sd * stop_atr * atr[d, s])
            out.append((d, s, sd, at, r, why))
    return out


def evaluate(trades, days):
    if not trades:
        return None
    t = pd.DataFrame(trades, columns=["d", "s", "side", "bar", "gross", "why"])
    t["net"] = t.gross - COST_BPS
    t["day"] = [days[i] for i in t.d]
    daily = t.groupby("day").net.mean()
    cut = days[int(len(days) * 0.6)]
    res = {}
    for name, m in (("IS", daily.index < cut), ("OOS", daily.index >= cut)):
        x = daily[m]
        tt = t[t.day.isin(x.index)]
        res[name] = {"days": int(len(x)), "trades": int(len(tt)), "net": round(float(tt.net.mean()), 2),
                     "gross": round(float(tt.gross.mean()), 2),
                     "t": round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2) if len(x) > 2 else 0.0,
                     "long": round(float(tt[tt.side > 0].net.mean()), 2) if (tt.side > 0).any() else None,
                     "short": round(float(tt[tt.side < 0].net.mean()), 2) if (tt.side < 0).any() else None}
    return res


def main():
    days, symbols, arr, atr, pclose = load()
    print(f"hourly panel: {len(days)} sessions {days[0]}..{days[-1]}, {len(symbols)} symbols; OOS from {days[int(len(days)*0.6)]}", flush=True)
    grid = {}
    for top in (10, 20):
        for rv in (1.5, 2.5):
            grid[f"orb60_top{top}_rv{rv}_rangestop"] = lambda top=top, rv=rv: orb60(arr, atr, top=top, rv_min=rv)
            grid[f"orb60_top{top}_rv{rv}_0.5atr"] = lambda top=top, rv=rv: orb60(arr, atr, top=top, rv_min=rv, stop=0.5)
            grid[f"orb60fade_top{top}_rv{rv}"] = lambda top=top, rv=rv: orb60(arr, atr, top=top, rv_min=rv, stop=0.5, contra=True)
    for at in (1, 3, 5):
        for follow in (True, False):
            for since in ("open", "pclose"):
                for rv in (1.0, 2.0):
                    k = f"{'mom' if follow else 'rev'}_at{at}_{since}_rv{rv}"
                    grid[k] = lambda at=at, follow=follow, since=since, rv=rv: rank_at(
                        arr, atr, at=at, follow=follow, since=since, rv_min=rv, pclose=pclose)
    res = {}
    for k, fn in grid.items():
        r = evaluate(fn(), days)
        res[k] = r
        if r:
            print(f"{k:<32} " + "  ".join(f"{p}: net {v['net']:+6.1f} (gross {v['gross']:+6.1f}) t={v['t']:+5.2f} n={v['trades']}"
                                          for p, v in r.items()), flush=True)
    OUT.write_text(json.dumps({"days": [days[0], days[-1]], "results": res}, indent=1))


if __name__ == "__main__":
    main()
