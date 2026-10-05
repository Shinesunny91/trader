"""Live NSE-500 intraday monitor and the paper "shadow board".

What it is
----------
* **Monitor** (information, not advice): every 5 minutes during the session the
  5-minute bars of the NIFTY-500 are refreshed into the candle cache and turned
  into a snapshot: stocks in play (volume vs the same time of day over the last
  20 sessions), gainers/losers, new session highs/lows, names near their price
  band (circuit), and today's NSE corporate announcements with the price
  reaction since.  It also watches the gap-reversal book's open shorts and
  pushes a phone alert when one nears or hits its stop.
* **Shadow board** (paper only): the three intraday strategies that came
  closest in the 2026-10-05 research (docs/research-log.md) run forward on live
  data.  None is validated — every one was within noise of zero after costs —
  so their signals are graded on paper and never pushed as trades.  A strategy
  becomes *eligible for review* only after MIN_SESSIONS live sessions with a
  positive mean and Holm-adjusted p < 0.05 over the board.

Point-in-time: signals use completed bars only (a bar is complete once its
5 minutes have elapsed), so re-running a scan reproduces the same signals.
"""
from __future__ import annotations

import math
import sqlite3
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nse_intraday_ai.atomic_io import atomic_read_json, atomic_write_json

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "intraday"
LIVE_PATH = OUT / "live.json"
BOOK_PATH = OUT / "shadow_book.csv"
DB_PATH = ROOT / "data" / "candles.sqlite3"
UNIVERSE_CSV = ROOT / "data" / "nifty500_symbols.csv"

NBARS = 75                     # 09:15 .. 15:25
SQ_BAR = 72                    # 15:15 -> exit at that bar's open
FIRST_HOUR = 12                # bars 0..11 = 09:15..10:15
COST_BPS = 13.3                # Groww MIS round trip incl. 3 bps slippage per leg
BASELINE_SESSIONS = 20
MIN_SESSIONS = 60              # live sessions before a strategy can be reviewed
NEAR_STOP_PCT = 1.0            # gap-book alert threshold
NOISE_DESC = {
    "ESOP/ESOS/ESPS", "Trading Window", "Copy of Newspaper Publication", "Loss of Share Certificates",
    "Duplicate Share Certificate", "Certificate under SEBI (Depositories and Participants) Regulations, 2018",
    "Shareholders meeting", "Record Date", "Book Closure", "General Updates", "Updates",
}

# Research reference (2026-10-05, 5-min Jul-Oct 2026, out-of-sample half): net bps/trade, t.
RESEARCH = {
    "orb60": ("Opening-range (first hour) breakout on stocks in play, stop at the far side",
              "1-hour 2023-26 OOS +0.2 bps (t -0.03)"),
    "gap_fade": ("Fade today's opening gap (>= 1 ATR) back to yesterday's close, stop 0.5 ATR",
                 "5-min OOS +12.3 bps (t 1.1, n 87)"),
    "vol_breakout": ("New session high/low after 10:15 on >= 1.5x normal volume, stop 0.4 ATR",
                     "5-min OOS +4.9 bps (t 0.5)"),
}


@contextmanager
def _quiet():
    """Silence numpy's all-NaN / empty-slice warnings (missing bars are normal)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


# ── data ────────────────────────────────────────────────────────────────────
@dataclass
class DayPanel:
    session: date
    symbols: list[str]                 # NSE symbols without .NS
    o: np.ndarray                      # (S, NBARS) float, NaN where missing
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    v: np.ndarray
    nbar: int                          # completed bars available (bars 0..nbar-1)

    def idx(self, symbol: str) -> int:
        return self.symbols.index(symbol)


@dataclass
class Context:
    prev_close: np.ndarray             # (S,)
    atr: np.ndarray                    # (S,) 14-session ATR, rupees, known before the session
    cum_base: np.ndarray               # (S, NBARS) mean cumulative volume through each bar
    band: np.ndarray                   # (S,) price band % (NaN = none/unknown)


def universe() -> list[str]:
    return sorted(pd.read_csv(UNIVERSE_CSV)["Symbol"].astype(str).str.strip())


def completed_bars(now: datetime) -> int:
    """Number of 5-min bars of the session that have fully elapsed at `now`."""
    minutes = now.hour * 60 + now.minute - 555
    return int(min(max(minutes // 5, 0), NBARS))


def _read_5m(symbols: list[str], start: date, end: date, db_path: Path) -> pd.DataFrame:
    a = pd.Timestamp(start, tz=IST).tz_convert("UTC").isoformat()
    b = (pd.Timestamp(end, tz=IST) + pd.Timedelta(days=1)).tz_convert("UTC").isoformat()
    tickers = [f"{s}.NS" for s in symbols]
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        frame = pd.read_sql_query(
            f"SELECT symbol, ts, open, high, low, close, volume FROM candles WHERE interval='5m' "
            f"AND ts>=? AND ts<? AND symbol IN ({','.join('?' * len(tickers))})", con, params=[a, b, *tickers])
    finally:
        con.close()
    if frame.empty:
        return frame
    ts = pd.to_datetime(frame["ts"], utc=True, format="ISO8601").dt.tz_convert(IST)
    on_grid = (ts.dt.second == 0) & (ts.dt.minute % 5 == 0)
    frame, ts = frame[on_grid.to_numpy()].copy(), ts[on_grid]
    frame["session"] = ts.dt.date.to_numpy()
    frame["bar"] = ((ts.dt.hour * 60 + ts.dt.minute - 555) // 5).to_numpy()
    frame["symbol"] = frame["symbol"].str.removesuffix(".NS")
    return frame[(frame["bar"] >= 0) & (frame["bar"] < NBARS)]


def to_arrays(frame: pd.DataFrame, symbols: list[str]) -> dict[str, np.ndarray]:
    si = {s: i for i, s in enumerate(symbols)}
    out = {k: np.full((len(symbols), NBARS), np.nan) for k in ("open", "high", "low", "close", "volume")}
    if frame.empty:
        return out
    f = frame[frame["symbol"].isin(si)].drop_duplicates(["symbol", "bar"], keep="last")
    r, b = f["symbol"].map(si).to_numpy(), f["bar"].to_numpy().astype(int)
    for k in out:
        out[k][r, b] = f[k].to_numpy(dtype=float)
    return out


def load_day(session: date, symbols: list[str], nbar: int, db_path: Path = DB_PATH) -> DayPanel:
    a = to_arrays(_read_5m(symbols, session, session, db_path), symbols)
    for k in a:                                   # never look at an incomplete bar
        a[k][:, nbar:] = np.nan
    return DayPanel(session, symbols, a["open"], a["high"], a["low"], a["close"], a["volume"], nbar)


def volume_baseline(session: date, symbols: list[str], db_path: Path = DB_PATH,
                    sessions: int = BASELINE_SESSIONS) -> np.ndarray:
    """Mean cumulative volume through each bar over the previous `sessions` sessions."""
    frame = _read_5m(symbols, session - timedelta(days=int(sessions * 1.7) + 10), session - timedelta(days=1), db_path)
    base = np.full((len(symbols), NBARS), np.nan)
    if frame.empty:
        return base
    days = sorted(frame["session"].unique())[-sessions:]
    stack = []
    for d in days:
        vol = to_arrays(frame[frame["session"] == d], symbols)["volume"]
        full = np.isfinite(vol).sum(axis=1) >= NBARS - 3          # skip partial sessions
        cum = np.nancumsum(vol, axis=1)
        cum[~full] = np.nan
        stack.append(cum)
    if not stack:
        return base
    arr = np.stack(stack)
    enough = np.isfinite(arr).sum(axis=0) >= min(10, len(stack))
    with _quiet():
        mean = np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=0) if len(stack) else base
    return np.where(enough, mean, np.nan)


def daily_context(session: date, symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(prev close, ATR14) from the official bhavcopy, sessions strictly before `session`."""
    from nse_intraday_ai import nse_bhav

    bhav = nse_bhav.load(session - timedelta(days=45), session - timedelta(days=1),
                         columns=["symbol", "series", "high", "low", "close", "prev_close"])
    pc, atr = np.full(len(symbols), np.nan), np.full(len(symbols), np.nan)
    if bhav.empty:
        return pc, atr
    bhav = bhav.sort_values("session")
    tr = np.maximum(bhav["high"] - bhav["low"],
                    np.maximum((bhav["high"] - bhav["prev_close"]).abs(), (bhav["low"] - bhav["prev_close"]).abs()))
    bhav = bhav.assign(tr=tr)
    g = bhav.groupby("symbol")
    last = g["close"].last()
    a14 = g["tr"].apply(lambda s: s.tail(14).mean() if len(s) >= 10 else np.nan)
    si = pd.Index(symbols)
    return last.reindex(si).to_numpy(dtype=float), a14.reindex(si).to_numpy(dtype=float)


def bands_for(session: date, symbols: list[str], *, fetch: bool) -> np.ndarray:
    from nse_intraday_ai import nse_bands
    try:
        t = nse_bands.load(session, fetch=fetch, log=lambda *_: None)
    except Exception:                                   # noqa: BLE001 — optional input
        return np.full(len(symbols), np.nan)
    return t["band"].reindex(symbols).to_numpy(dtype=float) if not t.empty else np.full(len(symbols), np.nan)


# ── snapshot ────────────────────────────────────────────────────────────────
def snapshot(p: DayPanel, ctx: Context) -> pd.DataFrame:
    """One row per symbol with data: price, move, relative volume, highs/lows, band distance."""
    n = p.nbar
    if n == 0:
        return pd.DataFrame()
    with _quiet():
        last = p.c[:, n - 1]
        o0 = p.o[:, 0]
        cumv = np.nancumsum(p.v, axis=1)[:, n - 1]
        rv = cumv / ctx.cum_base[:, n - 1]
        hi_prev = np.nanmax(p.h[:, : n - 1], axis=1) if n > 1 else np.full(len(last), np.nan)
        lo_prev = np.nanmin(p.lo[:, : n - 1], axis=1) if n > 1 else np.full(len(last), np.nan)
        upper = ctx.prev_close * (1 + ctx.band / 100)
        lower = ctx.prev_close * (1 - ctx.band / 100)
        typical = (p.h + p.lo + p.c) / 3
        vwap = np.nansum(typical * np.nan_to_num(p.v), axis=1) / np.where(cumv > 0, cumv, np.nan)
    df = pd.DataFrame({
        "symbol": p.symbols, "last": last, "chg_pct": (last / ctx.prev_close - 1) * 100,
        "from_open_pct": (last / o0 - 1) * 100, "gap_pct": (o0 / ctx.prev_close - 1) * 100,
        "rel_volume": rv, "turnover_cr": cumv * np.where(np.isfinite(vwap), vwap, last) / 1e7,
        "new_high": p.h[:, n - 1] > hi_prev, "new_low": p.lo[:, n - 1] < lo_prev,
        "vs_vwap_pct": (last / vwap - 1) * 100, "band": ctx.band,
        "to_upper_pct": (upper / last - 1) * 100, "to_lower_pct": (1 - lower / last) * 100,
        "atr_pct": ctx.atr / ctx.prev_close * 100,
    })
    return df[np.isfinite(df["last"])].reset_index(drop=True)


# ── shadow strategies (pure; completed bars only) ───────────────────────────
@dataclass
class Signal:
    strategy: str
    symbol: str
    side: int                          # +1 long, -1 short
    bar: int                           # entry bar index
    entry: float
    stop: float
    target: float | None = None
    reason: str = ""

    @property
    def time(self) -> str:
        m = 555 + 5 * self.bar
        return f"{m // 60:02d}:{m % 60:02d}"


def _rv_at(p: DayPanel, ctx: Context, bar: int) -> np.ndarray:
    with _quiet():
        return np.nancumsum(p.v[:, : bar + 1], axis=1)[:, -1] / ctx.cum_base[:, bar]


def orb60(p: DayPanel, ctx: Context, *, top: int = 10, rv_min: float = 2.5) -> list[Signal]:
    if p.nbar <= FIRST_HOUR:
        return []
    rv = _rv_at(p, ctx, FIRST_HOUR - 1)
    with _quiet():
        hi = np.nanmax(p.h[:, :FIRST_HOUR], axis=1)
        lo = np.nanmin(p.lo[:, :FIRST_HOUR], axis=1)
    ok = (rv >= rv_min) & (p.o[:, 0] >= 50) & np.isfinite(hi) & np.isfinite(p.c[:, FIRST_HOUR - 1])
    cand = np.where(ok)[0]
    cand = cand[np.argsort(-rv[cand])][:top]
    out = []
    for s in cand:
        side = int(np.sign(p.c[s, FIRST_HOUR - 1] - p.o[s, 0]))
        if side == 0:
            continue
        level = hi[s] if side > 0 else lo[s]
        for b in range(FIRST_HOUR, min(p.nbar, SQ_BAR)):
            if not np.isfinite(p.h[s, b]):
                continue
            if (side > 0 and p.h[s, b] > level) or (side < 0 and p.lo[s, b] < level):
                entry = max(level, p.o[s, b]) if side > 0 else min(level, p.o[s, b])
                out.append(Signal("orb60", p.symbols[s], side, b, round(float(entry), 2),
                                  round(float(lo[s] if side > 0 else hi[s]), 2),
                                  reason=f"first-hour volume {rv[s]:.1f}x normal"))
                break
    return out


def gap_fade(p: DayPanel, ctx: Context, *, min_gap_atr: float = 1.0, top: int = 10,
             stop_atr: float = 0.5) -> list[Signal]:
    if p.nbar < 2:
        return []
    with _quiet():
        gap = (p.o[:, 0] - ctx.prev_close) / ctx.atr
    ok = np.isfinite(gap) & (np.abs(gap) >= min_gap_atr) & (p.o[:, 0] >= 50) & np.isfinite(p.o[:, 1])
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-np.abs(gap[idx]))][:top]
    out = []
    for s in idx:
        side = -1 if gap[s] > 0 else 1
        entry = float(p.o[s, 1])
        out.append(Signal("gap_fade", p.symbols[s], side, 1, round(entry, 2),
                          round(entry - side * stop_atr * ctx.atr[s], 2), round(float(ctx.prev_close[s]), 2),
                          reason=f"gap {gap[s]:+.1f} ATR"))
    return out


def vol_breakout(p: DayPanel, ctx: Context, *, rv_min: float = 1.5, stop_atr: float = 0.4,
                 n_max: int = 10) -> list[Signal]:
    out: list[Signal] = []
    done: set[int] = set()
    if p.nbar <= FIRST_HOUR:
        return out
    with _quiet():
        cum = np.nancumsum(p.v, axis=1) / ctx.cum_base
        runhi = np.fmax.accumulate(np.nan_to_num(p.h, nan=-np.inf), axis=1)
        runlo = np.fmin.accumulate(np.nan_to_num(p.lo, nan=np.inf), axis=1)
    for b in range(FIRST_HOUR, min(p.nbar, SQ_BAR - 3)):
        for side, brk in ((1, p.h[:, b] > runhi[:, b - 1]), (-1, p.lo[:, b] < runlo[:, b - 1])):
            for s in np.where(brk & (cum[:, b - 1] >= rv_min) & np.isfinite(ctx.atr))[0]:
                if s in done or len(out) >= n_max:
                    continue
                level = runhi[s, b - 1] if side > 0 else runlo[s, b - 1]
                entry = max(level, p.o[s, b]) if side > 0 else min(level, p.o[s, b])
                out.append(Signal("vol_breakout", p.symbols[s], side, b, round(float(entry), 2),
                                  round(float(entry - side * stop_atr * ctx.atr[s]), 2),
                                  reason=f"new session {'high' if side > 0 else 'low'} on {cum[s, b - 1]:.1f}x volume"))
                done.add(s)
    return out


STRATEGIES = {"orb60": orb60, "gap_fade": gap_fade, "vol_breakout": vol_breakout}


def grade(p: DayPanel, sig: Signal) -> dict:
    """Paper result of a signal up to the last completed bar (or the 15:15 cover)."""
    s = p.idx(sig.symbol)
    o, h, lo, c = p.o[s], p.h[s], p.lo[s], p.c[s]
    status, exit_px, exit_bar = "OPEN", float("nan"), None
    for b in range(sig.bar, min(p.nbar, SQ_BAR)):
        if not np.isfinite(h[b]):
            continue
        if (sig.side > 0 and lo[b] <= sig.stop) or (sig.side < 0 and h[b] >= sig.stop):
            exit_px = sig.stop if b == sig.bar else (min(sig.stop, o[b]) if sig.side > 0 else max(sig.stop, o[b]))
            status, exit_bar = "STOP", b
            break
        if sig.target is not None and b > sig.bar and (
                (sig.side > 0 and h[b] >= sig.target) or (sig.side < 0 and lo[b] <= sig.target)):
            exit_px = max(sig.target, o[b]) if sig.side > 0 else min(sig.target, o[b])
            status, exit_bar = "TARGET", b
            break
    if status == "OPEN":
        if p.nbar > SQ_BAR and np.isfinite(o[SQ_BAR]):
            status, exit_px, exit_bar = "CLOSED", float(o[SQ_BAR]), SQ_BAR
        else:
            valid = np.where(np.isfinite(c[: p.nbar]))[0]
            exit_px = float(c[valid[-1]]) if valid.size else sig.entry
    gross = sig.side * (exit_px - sig.entry) / sig.entry * 1e4
    return {**asdict(sig), "time": sig.time, "status": status, "exit": round(float(exit_px), 2),
            "exit_bar": exit_bar, "gross_bps": round(float(gross), 1), "net_bps": round(float(gross - COST_BPS), 1)}


def run_strategies(p: DayPanel, ctx: Context) -> list[dict]:
    rows = []
    for fn in STRATEGIES.values():
        try:
            rows += [grade(p, s) for s in fn(p, ctx)]
        except Exception:                                # noqa: BLE001 — one strategy never breaks the monitor
            continue
    return rows


# ── shadow book & promotion ─────────────────────────────────────────────────
def record_day(session: date, rows: list[dict], path: Path = BOOK_PATH) -> int:
    """Replace `session`'s rows in the shadow book with the final grades."""
    final = [r for r in rows if r["status"] != "OPEN"]
    new = pd.DataFrame(final)
    if not new.empty:
        new.insert(0, "session", session.isoformat())
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["session"] != session.isoformat()]
    book = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    book.to_csv(tmp, index=False)
    tmp.replace(path)
    return len(new)


def board_stats(book: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: live sessions, trades, net bps/trade, t-stat of daily means, Holm p, status."""
    from math import erf

    rows = []
    for name in STRATEGIES:
        t = book[book["strategy"] == name] if not book.empty else pd.DataFrame()
        if t.empty:
            rows.append({"strategy": name, "sessions": 0, "trades": 0, "net_bps": None, "t": None, "p": 1.0})
            continue
        daily = t.groupby("session")["net_bps"].mean()
        tstat = float(daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily)))) if len(daily) > 2 and daily.std() > 0 else 0.0
        p = 0.5 * (1 - erf(tstat / math.sqrt(2)))                       # one-sided, normal approx.
        rows.append({"strategy": name, "sessions": int(len(daily)), "trades": int(len(t)),
                     "net_bps": round(float(t["net_bps"].mean()), 1), "t": round(tstat, 2), "p": p,
                     "hit_rate": round(float((t["net_bps"] > 0).mean()), 3)})
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(df)
    df["holm_p"] = [min(1.0, max((m - j) * df["p"].iloc[j] for j in range(i + 1))) for i in range(m)]
    df["status"] = [
        "eligible for review" if (r.sessions >= MIN_SESSIONS and r.holm_p < 0.05 and (r.net_bps or 0) > 0)
        else f"paper ({r.sessions}/{MIN_SESSIONS} sessions)" for r in df.itertuples()]
    return df


# ── announcements ───────────────────────────────────────────────────────────
def todays_announcements(session: date, symbols: set[str]) -> pd.DataFrame:
    from nse_intraday_ai import nse_corp
    try:
        df = nse_corp.announcements(session, session, nse_corp.Client(min_interval=0.5, log=lambda *_: None))
    except Exception:                                    # noqa: BLE001 — NSE may block; monitor keeps going
        return pd.DataFrame(columns=["symbol", "ts", "desc", "text"])
    if df.empty:
        return df
    df = df[df["symbol"].isin(symbols) & ~df["desc"].isin(NOISE_DESC)].copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts", ascending=False)


def announcement_rows(ann: pd.DataFrame, p: DayPanel) -> list[dict]:
    rows = []
    for r in ann.head(60).itertuples():
        reaction = None
        if r.symbol in p.symbols and p.nbar:
            s = p.idx(r.symbol)
            m = r.ts.hour * 60 + r.ts.minute - 555
            b = m // 5
            last = p.c[s, p.nbar - 1]
            ref = p.o[s, 0] if b <= 0 else (p.c[s, min(b, p.nbar) - 1] if b <= p.nbar else np.nan)
            if np.isfinite(ref) and np.isfinite(last) and b < p.nbar:
                reaction = round(float((last / ref - 1) * 100), 2)
        rows.append({"time": r.ts.strftime("%H:%M"), "symbol": r.symbol, "category": r.desc,
                     "text": str(getattr(r, "text", "") or "")[:220], "move_since_pct": reaction})
    return rows


# ── gap-book watch ──────────────────────────────────────────────────────────
def gap_book_watch(p: DayPanel, picks_payload: dict | None) -> list[dict]:
    if not picks_payload or picks_payload.get("session") != p.session.isoformat() or not p.nbar:
        return []
    rows = []
    for pk in picks_payload.get("picks", []):
        sym = str(pk.get("symbol", "")).removesuffix(".NS")
        if pk.get("reserve") or sym not in p.symbols or not pk.get("stop_price"):
            continue
        s = p.idx(sym)
        stop, entry = float(pk["stop_price"]), float(pk.get("entry") or p.o[s, 0])
        with _quiet():
            hit = bool(np.nanmax(p.h[s, : min(p.nbar, SQ_BAR)]) >= stop)
        last = float(p.c[s, p.nbar - 1])
        rows.append({"symbol": sym, "entry": entry, "stop": stop, "last": last, "stop_hit": hit,
                     "to_stop_pct": round((stop / last - 1) * 100, 2),
                     "pnl_pct": round((entry - last) / entry * 100, 2)})
    return rows


def gap_alerts(rows: list[dict], sent: set[str]) -> list[tuple[str, str]]:
    """(key, message) for alerts not yet sent: stop hit, or within NEAR_STOP_PCT of it."""
    out = []
    for r in rows:
        if r["stop_hit"] and f"hit:{r['symbol']}" not in sent:
            out.append((f"hit:{r['symbol']}", f"🔴 {r['symbol']} reached its stop ₹{r['stop']:,.2f} — "
                                              "check your SL order filled."))
        elif not r["stop_hit"] and 0 <= r["to_stop_pct"] <= NEAR_STOP_PCT and f"near:{r['symbol']}" not in sent:
            out.append((f"near:{r['symbol']}", f"⚠ {r['symbol']} is {r['to_stop_pct']:.1f}% from its stop "
                                               f"₹{r['stop']:,.2f} (last ₹{r['last']:,.2f}). Keep the stop — don't move it."))
    return out


# ── one scan ────────────────────────────────────────────────────────────────
def _clean(records: list[dict]) -> list[dict]:
    def fix(v):
        if isinstance(v, (np.floating, float)):
            return None if not math.isfinite(float(v)) else round(float(v), 3)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    return [{k: fix(v) for k, v in r.items()} for r in records]


def build_live(p: DayPanel, ctx: Context, ann: pd.DataFrame, picks_payload: dict | None,
               now: datetime) -> dict:
    snap = snapshot(p, ctx)
    uni = set(universe()) if UNIVERSE_CSV.exists() else set(p.symbols)
    if not snap.empty:
        snap = snap[snap["symbol"].isin(uni)].reset_index(drop=True)
    signals = [s for s in run_strategies(p, ctx) if s["symbol"] in uni]
    payload = {"session": p.session.isoformat(), "generated_at": now.isoformat(timespec="seconds"),
               "nbar": p.nbar, "last_bar": None, "breadth": {}, "in_play": [], "gainers": [], "losers": [],
               "highs": [], "lows": [], "circuit_watch": [], "announcements": announcement_rows(ann, p),
               "signals": _clean(signals), "gap_book": _clean(gap_book_watch(p, picks_payload)),
               "research": {k: list(v) for k, v in RESEARCH.items()}}
    if p.nbar:
        m = 555 + 5 * (p.nbar - 1)
        payload["last_bar"] = f"{m // 60:02d}:{m % 60:02d}"
    if not snap.empty:
        payload["breadth"] = {"advancing": int((snap.chg_pct > 0).sum()), "declining": int((snap.chg_pct < 0).sum()),
                              "unchanged": int((snap.chg_pct == 0).sum()), "symbols": int(len(snap)),
                              "median_chg_pct": round(float(snap.chg_pct.median()), 2)}
        cols = ["symbol", "last", "chg_pct", "from_open_pct", "gap_pct", "rel_volume", "turnover_cr",
                "vs_vwap_pct", "to_upper_pct", "to_lower_pct", "band", "new_high", "new_low"]
        live = snap[snap["turnover_cr"] >= 1]                   # ignore names with < ₹1 cr traded so far
        payload["in_play"] = _clean(live.sort_values("rel_volume", ascending=False).head(40)[cols].to_dict("records"))
        payload["gainers"] = _clean(live.sort_values("chg_pct", ascending=False).head(20)[cols].to_dict("records"))
        payload["losers"] = _clean(live.sort_values("chg_pct").head(20)[cols].to_dict("records"))
        payload["highs"] = _clean(live[live.new_high].sort_values("rel_volume", ascending=False).head(20)[cols].to_dict("records"))
        payload["lows"] = _clean(live[live.new_low].sort_values("rel_volume", ascending=False).head(20)[cols].to_dict("records"))
        near = snap[(snap.to_upper_pct <= 1.5) | (snap.to_lower_pct <= 1.5)]
        payload["circuit_watch"] = _clean(near.sort_values("chg_pct", ascending=False)[cols].to_dict("records"))
    return payload


def read_live(path: Path = LIVE_PATH) -> dict | None:
    data = atomic_read_json(path, default=None)
    return data if isinstance(data, dict) else None


def write_live(payload: dict, path: Path = LIVE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
