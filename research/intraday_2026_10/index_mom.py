"""Index intraday momentum (Gao, Han, Li & Zhou 2018 JFE) on NIFTY / BANKNIFTY, hourly ~2y.

Signal: return from the previous close to the end of the first hour (10:15), and
optionally the return of the penultimate hour.  Trade the index in the signal's
direction from the 14:15 open to the 15:15 open (last-hour), or from 10:15 to 15:15.
Futures cost ~4.5 bps round trip (STT 0.02% sell, exchange, GST, stamp, 1 bp slip/leg).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
COST = 4.5


def bars(sym):
    d = yf.download(sym, period="730d", interval="1h", progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index = d.index.tz_convert("Asia/Kolkata")
    d = d.rename(columns=str.lower)
    d["day"] = d.index.date
    d["h"] = d.index.hour
    return d


def test(sym):
    d = bars(sym)
    daily = d.groupby("day").agg(close=("close", "last"))
    pc = daily.close.shift(1)
    rows = []
    for day, g in d.groupby("day"):
        g = g.set_index("h")
        if not {9, 10, 13, 14, 15}.issubset(g.index) or pd.isna(pc.get(day)):
            continue
        r1 = g.at[9, "close"] / pc[day] - 1                  # prev close -> 10:15
        r_pen = g.at[13, "close"] / g.at[13, "open"] - 1     # 13:15 -> 14:15
        last = g.at[15, "open"] / g.at[14, "open"] - 1       # 14:15 -> 15:15 (trade)
        rest = g.at[15, "open"] / g.at[10, "open"] - 1       # 10:15 -> 15:15 (trade)
        rows.append((day, r1, r_pen, last, rest))
    t = pd.DataFrame(rows, columns=["day", "r1", "rpen", "last", "rest"]).set_index("day")
    cut = t.index[int(len(t) * 0.6)]
    print(f"\n{sym}: {len(t)} days {t.index[0]}..{t.index[-1]}, OOS from {cut}")
    for name, sig, ret in (("first-hour -> last hour", np.sign(t.r1), t["last"]),
                           ("first+penultimate -> last hour", np.sign(t.r1) * (np.sign(t.r1) == np.sign(t.rpen)), t["last"]),
                           ("first-hour -> rest of day", np.sign(t.r1), t["rest"]),
                           ("big first hour (|r1|>0.5%) -> rest", np.sign(t.r1) * (t.r1.abs() > 0.005), t["rest"]),
                           ("big first hour (|r1|>0.5%) -> last", np.sign(t.r1) * (t.r1.abs() > 0.005), t["last"])):
        pnl = (sig * ret * 1e4 - COST * (sig != 0))[sig != 0]
        for p, m in (("IS", pnl.index < cut), ("OOS", pnl.index >= cut)):
            x = pnl[m]
            tt = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else 0
            print(f"  {name:<36} {p}: n={len(x):4d} net {x.mean():+6.2f} bps/trade  t={tt:+5.2f}  hit {(x > 0).mean():.2f}")


for s in ("^NSEI", "^NSEBANK"):
    test(s)
