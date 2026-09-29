"""Point-in-time features for the gap-reversal book — one builder for research, training and live.

The rule that makes a backtest honest is enforced here once: a row for session
t may only use information that existed before t's open.

  * NSE equity and F&O bhavcopies and participant OI are published after the
    close, so session t sees values dated t-1 and earlier.
  * US and European closes, and the Indian ADRs in New York, finish before the
    NSE opens: their latest close dated before t is known.
  * Asian closes (Nikkei, Hang Seng...) come after the NSE opens, so only their
    t-1 close is usable — the same "strictly before t" rule covers it.

The universe is point-in-time: the most liquid EQ-series stocks by 20-session
traded value *as of t-1*, whatever they are called today — delisted and
demoted names included.  That removes the survivorship bias that a universe of
"today's NIFTY 500" builds into every cross-sectional model.

Research builds the whole history with targets; the live path appends an empty
row for the session being ranked and builds exactly the same columns for it.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nse_intraday_ai import nse_bhav, nse_extra, nse_fo

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "candles.sqlite3"
FLOWS = ROOT / "data" / "nse_flows"
SECTORS_CSV = ROOT / "data" / "nifty500_symbols.csv"
STOP_ATR = 0.75

# Yahoo daily series -> feature prefix.  Market-level: one value per session.
MARKET_SERIES = {
    "^NSEI": "nifty", "^NSEBANK": "bank", "^INDIAVIX": "ivix", "^NSMIDCP": "midcap",
    "^GSPC": "spx", "^IXIC": "ndx", "^RUT": "rut", "^VIX": "usvix", "^N225": "n225",
    "^HSI": "hsi", "^KS11": "kospi", "^GDAXI": "dax", "^FTSE": "ftse", "000001.SS": "shcomp",
    "EEM": "eem", "INDA": "inda", "DX-Y.NYB": "dxy", "USDINR=X": "inr", "CL=F": "crude",
    "GC=F": "gold", "HG=F": "copper", "^TNX": "us10y",
}
# New York ADR -> NSE symbol: the ADR trades after the NSE close.
ADRS = {"INFY": "INFY", "WIT": "WIPRO", "HDB": "HDFCBANK", "IBN": "ICICIBANK", "RDY": "DRREDDY"}
META_COLUMNS = ["turn_rank", "spread_bps", "val20_cr", "sector"]
TARGETS = ["short_bps", "long_bps", "intra_bps"]


@dataclass
class Inputs:
    bhav: pd.DataFrame                              # long: session, symbol, series, OHLC...
    fo: pd.DataFrame                                # long: session, symbol, kind, fut_*...
    series: dict[str, pd.DataFrame]                 # Yahoo daily OHLC by symbol
    participant: pd.DataFrame                       # date-indexed positioning ratios
    sectors: dict[str, str] = field(default_factory=dict)
    extra: pd.DataFrame | None = None               # MWPL crowding, bans, short sales


# ── loading ─────────────────────────────────────────────────────────────────

def load_series(symbols: list[str], db_path: Path | str = DB_PATH) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        q = ",".join("?" * len(symbols))
        raw = pd.read_sql_query(
            f"SELECT symbol, ts, open, high, low, close, volume FROM candles "
            f"WHERE interval='1d' AND symbol IN ({q})", con, params=symbols)
    finally:
        con.close()
    raw["d"] = pd.to_datetime(raw["ts"], utc=True, format="ISO8601").dt.tz_convert("Asia/Kolkata").dt.date
    raw = raw.drop_duplicates(["symbol", "d"], keep="last")
    out = {}
    for sym, g in raw.groupby("symbol"):
        g = g.set_index("d").sort_index()
        out[sym] = g.loc[g["close"] > 0, ["open", "high", "low", "close", "volume"]]
    return out


def load_participant(flows: Path = FLOWS) -> pd.DataFrame:
    """FII/Client/Pro/DII net positioning ratios in index and stock futures."""
    recs = []
    for p in sorted(flows.glob("oi_*.parquet")):
        try:
            d = pd.read_parquet(p)
        except Exception:                                   # noqa: BLE001
            continue
        rows = d.set_index(d["Client Type"].astype(str).str.strip())
        rec = {"date": pd.Timestamp(d["date"].iloc[0]).date()}
        for who in ("FII", "Client", "Pro", "DII"):
            if who not in rows.index:
                continue
            for kind, col in (("idx", "Future Index"), ("stk", "Future Stock")):
                long_, short = float(rows.loc[who, f"{col} Long"]), float(rows.loc[who, f"{col} Short"])
                rec[f"{who.lower()}_{kind}_ratio"] = (long_ - short) / (long_ + short) if long_ + short else np.nan
        recs.append(rec)
    if not recs:
        return pd.DataFrame()
    frame = pd.DataFrame(recs).set_index("date").sort_index()
    return frame[~frame.index.duplicated()]


def load_inputs(start: date, end: date) -> Inputs:
    sectors = {}
    if SECTORS_CSV.exists():
        meta = pd.read_csv(SECTORS_CSV)
        sectors = dict(zip(meta["Symbol"].astype(str), meta["Industry"].astype(str)))
    return Inputs(
        bhav=nse_bhav.load(start, end, series=("EQ", "BE")),
        fo=nse_fo.load(start, end),
        series=load_series(list(MARKET_SERIES) + list(ADRS)),
        participant=load_participant(),
        sectors=sectors,
        extra=nse_extra.load(start, end),
    )


def prior_value(series: pd.Series, sessions: pd.Index, max_age_days: int = 7) -> pd.Series:
    """For each session t, the latest value dated strictly before t (<= 7 days old)."""
    s = series.dropna().sort_index()
    if s.empty:
        return pd.Series(np.nan, index=sessions)
    idx = pd.to_datetime(pd.Index(s.index))
    sess = pd.to_datetime(sessions)
    pos = idx.searchsorted(sess, side="left") - 1
    safe = np.clip(pos, 0, None)
    age = (sess - idx[safe]).days
    vals = np.where((pos >= 0) & (age <= max_age_days), s.to_numpy()[safe], np.nan)
    return pd.Series(vals, index=sessions)


# ── building ────────────────────────────────────────────────────────────────

def _wide(frame: pd.DataFrame, col: str, sessions: pd.Index, symbols: pd.Index) -> pd.DataFrame:
    w = frame.pivot(index="session", columns="symbol", values=col)
    return w.reindex(index=sessions, columns=symbols)


def build(inputs: Inputs, *, universe: int = 400, live_session: date | None = None,
          min_price: float = 50.0, stop_atr: float = STOP_ATR, prefilter: int = 600) -> pd.DataFrame:
    """Feature rows (session, symbol) for the point-in-time liquid universe.

    Research (live_session=None): every session, with targets.  Live: only
    `live_session`, which must be after the last session in the data; it gets
    the same features, computed from the rows before it, and no targets.
    """
    raw = inputs.bhav.drop_duplicates(["session", "symbol", "series"])
    eq_long = raw.assign(eq=(raw["series"] == "EQ").astype(float))
    raw = raw.sort_values("series").drop_duplicates(["session", "symbol"], keep="last")  # EQ over BE
    sessions = pd.Index(sorted(raw["session"].unique()))
    if live_session is not None:
        if live_session <= sessions[-1]:
            raise ValueError(f"live session {live_session} is not after the data ({sessions[-1]})")
        sessions = sessions.append(pd.Index([live_session]))
    # Names never within the top `prefilter` by traded value cannot enter a
    # top-`universe` book; dropping them keeps the wide frames small.
    v = raw.pivot(index="session", columns="symbol", values="value").reindex(sessions)
    v20 = v.rolling(20, min_periods=10).mean().shift(1)
    symbols = v20.columns[(v20.rank(axis=1, ascending=False) <= prefilter).any()]
    raw = raw[raw["symbol"].isin(set(symbols))]
    W = lambda col: _wide(raw, col, sessions, symbols)  # noqa: E731
    o, h, l, c, pc = W("open"), W("high"), W("low"), W("close"), W("prev_close")
    val, trades, dq, dp = W("value"), W("trades"), W("deliv_qty"), W("deliv_pct")
    shares = W("volume")
    eq = eq_long.pivot_table(index="session", columns="symbol", values="eq", aggfunc="max") \
        .reindex(index=sessions, columns=symbols)
    bad = (o <= 0) | (c <= 0) | (pc <= 0) | (h < l)
    o, h, l, c, pc = (x.mask(bad) for x in (o, h, l, c, pc))

    # Corporate actions: NSE's PREV_CLOSE is not always adjusted on ex-dates
    # (BAJFINANCE 2025-06-17 opens at 0.10x "prev close": a 1:10 split).  No
    # stock can open 30% down or 35% up on a normal day, so such a day's gap,
    # return and range are unknown rather than signal.
    ratio = o / pc
    pc = pc.mask((ratio < 0.70) | (ratio > 1.35))
    r = c / pc - 1
    gap = o / pc - 1
    intra = c / o - 1
    trp = np.maximum(np.maximum(h - l, (h - pc).abs()), (l - pc).abs()) / pc
    atrp = trp.rolling(14, min_periods=10).mean()
    adj = (1 + r.fillna(0)).cumprod()
    scale = adj / c
    hi_adj, lo_adj = h * scale, l * scale

    lag = lambda x, k=1: x.shift(k)  # noqa: E731
    f: dict[str, pd.DataFrame] = {}
    f["gap1"], f["gap2"], f["gap3"] = lag(gap), lag(gap, 2), lag(gap, 3)
    f["gap_sum5"] = lag(gap.rolling(5, min_periods=3).sum())
    f["gap_mean20"] = lag(gap.rolling(20, min_periods=10).mean())
    f["intra1"] = lag(intra)
    f["intra_sum5"] = lag(intra.rolling(5, min_periods=3).sum())
    f["intra_mean20"] = lag(intra.rolling(20, min_periods=10).mean())
    f["tug20"] = f["gap_mean20"] - f["intra_mean20"]
    f["ret1"] = lag(r)
    for k in (5, 20, 60, 250):
        f[f"ret{k}"] = lag(adj / adj.shift(k) - 1)
    f["atr_pct"] = lag(atrp)
    f["vol20"] = lag(r.rolling(20, min_periods=10).std())
    f["gap1_atr"] = f["gap1"] / f["atr_pct"]
    f["range1"] = lag((h - l) / c)
    f["clv1"] = lag(((c - l) - (h - c)) / (h - l).replace(0, np.nan))
    f["upwick1"] = lag((h - np.maximum(o, c)) / (h - l).replace(0, np.nan))
    val20 = lag(val.rolling(20, min_periods=10).mean())
    f["relvol1"] = lag(val / val.rolling(20, min_periods=10).mean().shift(1))
    f["relvol5"] = lag(val.rolling(5).mean() / val.rolling(60, min_periods=20).mean())
    f["log_turn"] = np.log(val20)
    f["dist_hi250"] = lag(adj / hi_adj.rolling(250, min_periods=60).max() - 1)
    f["dist_lo250"] = lag(adj / lo_adj.rolling(250, min_periods=60).min() - 1)
    delta = adj.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    f["rsi14"] = lag(100 - 100 / (1 + up / dn.replace(0, np.nan)))
    f["log_price"] = np.log(lag(c))
    f["gap1_x_relvol"] = f["gap1"] * np.log1p(f["relvol1"].clip(0, 20))
    # NSE-only: delivery, trade size, series
    f["deliv1"] = lag(dp)
    f["deliv_z"] = lag((dp - dp.rolling(60, min_periods=20).mean())
                       / (dp.rolling(60, min_periods=20).std() + 1e-9))
    f["deliv_mean20"] = lag(dp.rolling(20, min_periods=10).mean())
    f["deliv_val_rel"] = lag((dq * c) / (dq * c).rolling(20, min_periods=10).mean().shift(1))
    ats = np.log(val / trades.replace(0, np.nan))
    f["avgtrade_z"] = lag((ats - ats.rolling(60, min_periods=20).mean())
                          / (ats.rolling(60, min_periods=20).std() + 1e-9))
    f["log_avgtrade"] = lag(ats)
    f["trades_rel"] = lag(trades / trades.rolling(20, min_periods=10).mean().shift(1))
    f["gap1_x_lowdeliv"] = f["gap1"] * (1 - f["deliv1"] / 100)
    f["was_be_20"] = lag((1 - eq.fillna(0)).rolling(20, min_periods=1).max())

    # Effective spread (Abdi & Ranaldo 2017) from close/high/low, causal: the
    # evaluator's liquidity cost, never a model input.
    lc, eta = np.log(c), (np.log(h) + np.log(l)) / 2
    spread_bps = np.sqrt((4 * (lc - eta) * (lc - eta.shift(-1)))
                         .rolling(60, min_periods=30).mean().shift(2).clip(lower=0)) * 1e4

    rank_val = val20.rank(axis=1, ascending=False)
    base = (rank_val <= universe) & (lag(c) >= min_price) & (lag(eq) == 1) & f["atr_pct"].gt(0)
    if live_session is None:
        # Research: the label must exist and the stock must still be EQ that
        # day (a move to trade-for-trade is announced in advance).
        in_univ = base & (eq == 1) & o.notna() & c.notna()
    else:
        in_univ = base.loc[[live_session]].reindex(sessions, fill_value=False)

    sec_of = pd.Series({s: inputs.sectors.get(s, "Other") for s in symbols})
    for name in ("gap1", "ret1", "ret5", "intra1"):
        med = f[name].where(in_univ).T.groupby(sec_of).transform("median").T
        f[f"{name}_sec"] = med
        f[f"{name}_vs_sec"] = f[name] - med
    g1 = f["gap1"].where(in_univ)
    f["gap1_xrank"] = g1.rank(axis=1, pct=True)

    mkt = pd.DataFrame(index=sessions)
    mkt["x_gap1_med"] = g1.median(axis=1)
    mkt["x_gap1_breadth"] = (g1 > 0).sum(axis=1) / g1.notna().sum(axis=1)
    mkt["x_gap1_disp"] = g1.std(axis=1)
    mkt["x_ret1_med"] = f["ret1"].where(in_univ).median(axis=1)
    mkt["x_intra1_med"] = f["intra1"].where(in_univ).median(axis=1)
    mkt["x_deliv1_med"] = f["deliv1"].where(in_univ).median(axis=1)

    ser = inputs.series
    for sym, key in MARKET_SERIES.items():
        if sym not in ser:
            continue
        s, cl = ser[sym], ser[sym]["close"]
        mkt[f"{key}_r1"] = prior_value(cl / cl.shift(1) - 1, sessions)
        mkt[f"{key}_r5"] = prior_value(cl / cl.shift(5) - 1, sessions)
        if key in ("nifty", "bank", "midcap"):
            mkt[f"{key}_gap1"] = prior_value(s["open"] / cl.shift(1) - 1, sessions)
            mkt[f"{key}_intra1"] = prior_value(s["close"] / s["open"] - 1, sessions)
            mkt[f"{key}_r20"] = prior_value(cl / cl.shift(20) - 1, sessions)
        if key in ("ivix", "usvix"):
            mkt[f"{key}_lvl"] = prior_value(cl, sessions)
    adr = pd.DataFrame(np.nan, index=sessions, columns=symbols)
    for a, nse in ADRS.items():
        if a in ser and nse in adr.columns and "spx_r1" in mkt:
            ra = ser[a]["close"] / ser[a]["close"].shift(1) - 1
            adr[nse] = prior_value(ra, sessions) - mkt["spx_r1"]
    f["adr_excess"] = adr

    fo = inputs.fo
    if fo is not None and not fo.empty:
        fo = fo.drop_duplicates(["session", "symbol"])
        stk = fo[fo["kind"] == "STOCK"]
        FW = lambda col: _wide(stk, col, sessions, symbols)  # noqa: E731
        fut_close, oi_all, chg_all = FW("fut_close"), FW("fut_oi_all"), FW("fut_chg_oi_all")
        call_oi, put_oi, call_chg, put_chg = FW("call_oi"), FW("put_oi"), FW("call_chg_oi"), FW("put_chg_oi")
        fut_val, opt_val = FW("fut_value"), FW("opt_value")
        fo_day = pd.Series(sessions.isin(set(stk["session"])), index=sessions)
        f["fno1"] = lag(FW("fno").fillna(0.0).where(fo_day, np.nan, axis=0))
        basis = fut_close / c - 1
        f["basis_rel1"] = lag(basis.sub(basis.median(axis=1), axis=0))
        f["oi_chg1"] = lag(chg_all / (oi_all - chg_all).replace(0, np.nan))
        f["oi_chg5"] = lag(oi_all / oi_all.shift(5) - 1)
        f["oi_x_ret1"] = f["oi_chg1"] * np.sign(f["ret1"])     # + build-up, - covering / unwinding
        f["oi_x_gap1"] = f["oi_chg1"] * np.sign(f["gap1"])
        lpcr = np.log(put_oi / call_oi.replace(0, np.nan))
        f["pcr1"] = lag(lpcr)
        f["pcr_chg1"] = lag(lpcr - lpcr.shift(1))
        f["call_chg_pct1"] = lag(call_chg / (call_oi - call_chg).replace(0, np.nan))
        f["put_chg_pct1"] = lag(put_chg / (put_oi - put_chg).replace(0, np.nan))
        lov = np.log(opt_val.replace(0, np.nan)) - np.log(fut_val.replace(0, np.nan))
        f["opt_fut_z"] = lag((lov - lov.rolling(60, min_periods=20).mean())
                             / (lov.rolling(60, min_periods=20).std() + 1e-9))
        f["fut_val_rel1"] = lag(fut_val / fut_val.rolling(20, min_periods=10).mean().shift(1))
        idx = fo[fo["kind"] == "INDEX"]
        spot = {"NIFTY": ser.get("^NSEI"), "BANKNIFTY": ser.get("^NSEBANK")}
        for u in ("NIFTY", "BANKNIFTY"):
            g = idx[idx["symbol"] == u].set_index("session").sort_index()
            if g.empty:
                continue
            key = u.lower()
            ipcr = np.log(g["put_oi"] / g["call_oi"])
            mkt[f"fo_{key}_pcr"] = prior_value(ipcr, sessions)
            mkt[f"fo_{key}_pcr_chg"] = prior_value(ipcr - ipcr.shift(1), sessions)
            mkt[f"fo_{key}_oi_chg"] = prior_value(
                g["fut_chg_oi_all"] / (g["fut_oi_all"] - g["fut_chg_oi_all"]), sessions)
            if spot[u] is not None:
                b = g["fut_close"] / spot[u]["close"].reindex(g.index) - 1
                mkt[f"fo_{key}_basis"] = prior_value(b, sessions)

    # Crowding and shorting (NSE combineoi / short-selling files).  "No Fresh
    # Positions" at the end of t-1 *is* session t's F&O ban, known the evening
    # before.  A short-selling file dated D reports trades of the session before
    # D; it is used one session later still, so the feature sees trades <= t-2.
    ex = inputs.extra
    if ex is not None and not ex.empty:
        ex = ex.drop_duplicates(["session", "symbol"])
        EW = lambda col: _wide(ex, col, sessions, symbols)  # noqa: E731
        ex_day = pd.Series(sessions.isin(set(ex["session"])), index=sessions)
        f["mwpl_util1"] = lag(EW("mwpl_util"))
        f["ban_today"] = lag(EW("no_fresh").fillna(0.0).where(ex_day, np.nan, axis=0))
        short = EW("short_qty").fillna(0.0).where(ex_day, np.nan, axis=0)
        f["short_frac"] = lag(short / shares.shift(1).replace(0, np.nan))
        mkt["x_banned"] = lag(EW("no_fresh").sum(axis=1).where(ex_day))
        mkt["x_mwpl_med"] = lag(EW("mwpl_util").median(axis=1))

    part = inputs.participant
    if part is not None and not part.empty:
        for col in part.columns:
            mkt[f"oi_{col}"] = prior_value(part[col], sessions)
            mkt[f"oi_{col}_d5"] = prior_value(part[col] - part[col].shift(5), sessions)

    dts = pd.to_datetime(sessions)
    mkt["dow"] = dts.dayofweek
    mkt["month"] = dts.month
    mkt["dom"] = dts.day
    expiry_wd = np.where(dts >= pd.Timestamp("2025-09-02"), 1, 3)   # NSE weekly expiry: Tue since
    mkt["is_expiry_wd"] = (dts.dayofweek == expiry_wd).astype(float)

    mask = in_univ.stack()
    rows = mask[mask].index
    out = pd.DataFrame({k: v.stack().reindex(rows).astype("float32").to_numpy() for k, v in f.items()},
                       index=rows)
    out.index.names = ["session", "symbol"]
    out["turn_rank"] = rank_val.stack().reindex(rows).astype("float32").to_numpy()
    out["spread_bps"] = spread_bps.stack().reindex(rows).astype("float32").to_numpy()
    out["val20_cr"] = (val20 / 1e7).stack().reindex(rows).astype("float32").to_numpy()
    out["sector"] = [sec_of[s] for s in out.index.get_level_values("symbol")]
    m = mkt.reindex(out.index.get_level_values("session")).astype("float32")
    for col in m.columns:
        out[col] = m[col].to_numpy()
    if live_session is None:
        stop = stop_atr * atrp.shift(1)
        short = ((o - c) / o * 1e4).where(~(h >= o * (1 + stop)), -stop * 1e4)
        long_ = ((c - o) / o * 1e4).where(~(l <= o * (1 - stop)), -stop * 1e4)
        for name, frame in (("short_bps", short), ("long_bps", long_), ("intra_bps", intra * 1e4)):
            out[name] = frame.stack().reindex(rows).astype("float32").to_numpy()
        out = out[out["short_bps"].notna() & out["gap1"].notna()]
    else:
        out = out[out["gap1"].notna()]
    return out


def history(start: date = date(2016, 1, 1), end: date = date(2099, 1, 1), **kw) -> pd.DataFrame:
    """The full research/training panel."""
    return build(load_inputs(start, end), **kw)


def live(session: date, *, lookback_days: int = 420, **kw) -> pd.DataFrame:
    """Feature rows for `session` from the ~13 months of data before it."""
    inputs = load_inputs(session - timedelta(days=lookback_days), session - timedelta(days=1))
    return build(inputs, live_session=session, **kw)
