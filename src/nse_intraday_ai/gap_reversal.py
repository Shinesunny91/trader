"""Overnight-gap reversal: short yesterday's biggest gap-ups for one session.

The one intraday effect in this workspace that survives every test it has been
put through.  Indian equities earn their return overnight and give part of it
back during the session, and the give-back concentrates in names that gapped
up the day before (the "tug of war" between overnight and intraday traders —
Lou, Polk & Skouras 2019 document the same effect in US stocks).

The trade, decided entirely before the open:

  universe   the 300 most liquid NIFTY 500 names by 20-session rupee turnover
  rank       by YESTERDAY's overnight gap, open(t-1) / close(t-2) - 1
  action     SHORT the top 8, equal rupee weight, MIS (intraday)
  entry      at the official open — a market order in the 09:00-09:07
             pre-open call auction, which fills at the opening price
  stop       buy-stop at entry + 0.75 x the 14-day daily ATR
  exit       buy to cover at 15:15 (the broker auto-squares MIS at ~15:20)

Evidence (costs = Groww MIS schedule at the actual position size plus 3 bps of
slippage per leg; regenerate with `scripts/gap_reversal.py backtest`).  The
configuration was fixed on the first two rows before the third was run once:

  10 years, daily bars   +24.7 bps/trade net, t = 9.2, positive in every
  (2,486 sessions)       calendar year 2016-2026, max drawdown 14.4% (1x)
  49 dev sessions, 5m    +11.6% of capital, profit factor 1.34, +5.9% and
  (2026-07-01..09-07)    +5.6% in the two halves, max drawdown 3.2%
  14 held-out sessions   +10.6%, profit factor 2.63, 10 of 14 days up, ~4 sd
  (2026-09-08..09-28)    above random shorts from the same universe; the
                         model-ranked book it replaces lost 0.3% over them

The held-out window was unusually kind (NIFTY fell 17.5 bps/day intraday
across it).  The decade says to expect ~0.2% of capital a day on average, with
losing weeks — not 10% a fortnight.

Three things that matter more than they look:

* **The entry time is most of the edge.**  Entering at 09:20 instead of at the
  open roughly halves the gross edge over the dev window and leaves the book
  near break-even after costs.  The signal is known the evening before, so
  there is no reason to be late: place the orders in the pre-open auction.
* **Selection is only half the return.**  In a falling market any short book
  makes money; the strategy's claim is the part above a random short.  That
  gap was ~4 sd in both the dev and the held-out windows.
* **Not every name can be shorted every day.**  Brokers block MIS shorts on
  names under ASM/GSM surveillance or in the trade-for-trade segment, and a
  big gapper is exactly what lands there.  The pick list is ranked; skip a
  blocked name and take the next one.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nse_intraday_ai.costs import round_trip_cost

IST = "Asia/Kolkata"
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "candles.sqlite3"
UNIVERSE_CSV = ROOT / "data" / "nifty500_symbols.csv"

# 5-minute bar index within an NSE session: bar k starts at 09:15 + 5k min.
BARS_PER_SESSION = 75
SQUARE_OFF_BAR = 72          # the 15:15 bar; exit at its open


@dataclass(frozen=True)
class GapReversalConfig:
    universe_size: int = 300
    picks: int = 8
    stop_atr: float = 0.75
    min_price: float = 50.0
    turnover_window: int = 20
    atr_window: int = 14
    capital: float = 10_00_000.0
    slippage_bps_per_leg: float = 3.0
    square_off: str = "15:15"
    # Ranked names published beyond `picks`, so a blocked short can be
    # replaced by the next one without re-running anything.
    reserves: int = 4


@dataclass
class Pick:
    rank: int
    symbol: str
    side: str
    gap_prev_pct: float          # yesterday's overnight gap, the ranking score
    prev_close: float
    atr: float                   # 14-day daily ATR, price units
    stop_distance: float         # rupees above the fill
    quantity: int
    position_value: float        # at the previous close — the fill is unknown
    turnover_cr: float           # 20-session average daily turnover, ₹ crore
    reserve: bool = False
    entry: float | None = None   # filled in once the session has opened
    stop_price: float | None = None

    @property
    def stop_pct(self) -> float:
        return self.stop_distance / self.prev_close * 100 if self.prev_close else 0.0

    def ticket(self, square_off: str = "15:15") -> str:
        head = f"SELL SHORT {self.quantity} {self.symbol.removesuffix('.NS')} (MIS)"
        if self.entry is not None and self.stop_price is not None:
            return (f"{head} filled ~₹{self.entry:,.2f} | stop-loss BUY SL-M "
                    f"₹{self.stop_price:,.2f} | cover at {square_off}")
        return (f"{head} at the open (pre-open 09:00-09:07) | stop-loss BUY SL-M "
                f"at fill + ₹{self.stop_distance:,.2f} (~{self.stop_pct:.1f}%) | "
                f"cover at {square_off}")

    def to_dict(self) -> dict:
        out = asdict(self)
        out["stop_pct"] = round(self.stop_pct, 3)
        return out


@dataclass
class Trade:
    session: date
    symbol: str
    side: str
    entry: float
    exit: float
    quantity: int
    exit_reason: str             # STOP | SQUARE_OFF | LAST_BAR
    exit_time: str
    gross: float
    costs: float

    @property
    def net(self) -> float:
        return self.gross - self.costs

    @property
    def gross_bps(self) -> float:
        return (self.entry - self.exit) / self.entry * 1e4 if self.side == "SHORT" else \
               (self.exit - self.entry) / self.entry * 1e4


@dataclass
class BacktestResult:
    config: GapReversalConfig
    trades: list[Trade] = field(default_factory=list)
    sessions: list[date] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=["session", "symbol", "side", "entry", "exit",
                                         "quantity", "exit_reason", "exit_time",
                                         "gross", "costs", "net", "gross_bps"])
        rows = [{**asdict(t), "net": t.net, "gross_bps": t.gross_bps} for t in self.trades]
        return pd.DataFrame(rows)

    def daily(self) -> pd.DataFrame:
        frame = self.frame()
        per = frame.groupby("session")["net"].sum() if not frame.empty else pd.Series(dtype=float)
        per = per.reindex(self.sessions, fill_value=0.0)
        out = pd.DataFrame({"session": per.index, "net": per.to_numpy()})
        out["cum_net"] = out["net"].cumsum()
        out["cum_pct"] = out["cum_net"] / self.config.capital * 100
        return out

    def summary(self) -> dict:
        frame = self.frame()
        daily = self.daily()
        net = frame["net"] if not frame.empty else pd.Series(dtype=float)
        wins, losses = net[net > 0].sum(), -net[net <= 0].sum()
        eq = daily["net"].cumsum()
        dd = float((eq.cummax().clip(lower=0) - eq).max()) if len(eq) else 0.0
        sd = daily["net"].std()
        return {
            "sessions": len(self.sessions),
            "first": str(self.sessions[0]) if self.sessions else None,
            "last": str(self.sessions[-1]) if self.sessions else None,
            "trades": int(len(frame)),
            "net_rupees": round(float(net.sum()), 2),
            "net_pct": round(float(net.sum()) / self.config.capital * 100, 3),
            "win_rate_pct": round(float((net > 0).mean() * 100), 1) if len(net) else 0.0,
            "profit_factor": round(float(wins / losses), 3) if losses > 0 else None,
            "avg_gross_bps": round(float(frame["gross_bps"].mean()), 2) if len(frame) else 0.0,
            "avg_net_bps": round(float((frame["net"] / (frame["entry"] * frame["quantity"])).mean() * 1e4), 2)
            if len(frame) else 0.0,
            "costs_rupees": round(float(frame["costs"].sum()), 2) if len(frame) else 0.0,
            "up_days": int((daily["net"] > 0).sum()),
            "down_days": int((daily["net"] < 0).sum()),
            "max_drawdown_pct": round(dd / self.config.capital * 100, 3),
            "stops_hit": int((frame["exit_reason"] == "STOP").sum()) if len(frame) else 0,
            "daily_sharpe_annualised": round(float(daily["net"].mean() / sd * math.sqrt(250)), 2)
            if len(daily) > 1 and sd > 0 else None,
        }


# ── data access ─────────────────────────────────────────────────────────────

def nifty500_symbols(path: Path | str = UNIVERSE_CSV) -> list[str]:
    frame = pd.read_csv(path)
    col = "Symbol" if "Symbol" in frame.columns else frame.columns[0]
    return sorted({f"{str(s).strip()}.NS" for s in frame[col] if str(s).strip()})


def load_daily(symbols: list[str] | None = None, since: str = "2025-06-01",
               db_path: Path | str = DB_PATH) -> dict[str, pd.DataFrame]:
    """Wide daily OHLCV frames (index = session date, columns = symbol)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        raw = pd.read_sql_query(
            "SELECT symbol, ts, open, high, low, close, volume FROM candles "
            "WHERE interval='1d' AND symbol LIKE '%.NS' AND ts>=?", con, params=(since,))
    finally:
        con.close()
    if symbols is not None:
        raw = raw[raw["symbol"].isin(set(symbols))]
    raw["session"] = pd.to_datetime(raw["ts"], utc=True, format="ISO8601") \
        .dt.tz_convert(IST).dt.date
    raw = raw.drop_duplicates(["symbol", "session"], keep="last")
    wide = {f: raw.pivot(index="session", columns="symbol", values=f).sort_index()
            for f in ("open", "high", "low", "close", "volume")}
    # Yahoo writes placeholder bars on exchange holidays: volume 0, O=H=L=C at
    # the last close (480 of them on the 2026-09-14 Ganesh Chaturthi holiday).
    # Left in, the holiday becomes "yesterday" and its zero gap replaces the
    # real one, so a whole session is ranked on nothing.
    bad = ((wide["open"] <= 0) | (wide["close"] <= 0) | (wide["high"] < wide["low"])
           | ~(wide["volume"] > 0))
    wide = {k: v.mask(bad) for k, v in wide.items()}
    # A date on which almost nothing traded is a holiday, not a session.
    real = wide["close"].notna().sum(axis=1) >= 0.5 * wide["close"].shape[1]
    return {k: v[real] for k, v in wide.items()}


def load_session_bars(symbol: str, session: date,
                      db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """One session of 5m bars indexed 0..74 (09:15 .. 15:25 IST)."""
    start = pd.Timestamp(session, tz=IST)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        frame = pd.read_sql_query(
            "SELECT ts, open, high, low, close FROM candles WHERE symbol=? AND interval='5m' "
            "AND ts>=? AND ts<?", con,
            params=(symbol, start.tz_convert("UTC").isoformat(),
                    (start + pd.Timedelta(days=1)).tz_convert("UTC").isoformat()))
    finally:
        con.close()
    if frame.empty:
        return frame
    ts = pd.to_datetime(frame["ts"], utc=True, format="ISO8601").dt.tz_convert(IST)
    on_grid = (ts.dt.second == 0) & (ts.dt.minute % 5 == 0)      # drop snapshot quotes
    frame = frame[on_grid.to_numpy()].copy()
    ts = ts[on_grid]
    minute = ts.dt.hour * 60 + ts.dt.minute
    frame["bar"] = ((minute - 555) // 5).to_numpy()
    frame = frame[(frame["bar"] >= 0) & (frame["bar"] < BARS_PER_SESSION)]
    frame = frame.drop_duplicates("bar", keep="first").set_index("bar").sort_index()
    return frame[["open", "high", "low", "close"]]


# ── the signal ──────────────────────────────────────────────────────────────

def features(daily: dict[str, pd.DataFrame], config: GapReversalConfig = GapReversalConfig()
             ) -> dict[str, pd.DataFrame]:
    """Per-(session, symbol) inputs, each known before that session's open."""
    o, h, l, c, v = (daily[k] for k in ("open", "high", "low", "close", "volume"))
    prev_close = c.shift(1)
    true_range = np.maximum(np.maximum(h - l, (h - prev_close).abs()), (l - prev_close).abs())
    return {
        "prev_close": prev_close,
        "gap_prev": (o / c.shift(1) - 1).shift(1),               # yesterday's gap
        "atr": true_range.rolling(config.atr_window, min_periods=10).mean().shift(1),
        "turnover": (c * v).rolling(config.turnover_window, min_periods=10).mean().shift(1),
    }


def select(daily: dict[str, pd.DataFrame], session: date,
           config: GapReversalConfig = GapReversalConfig()) -> list[Pick]:
    """The ranked short list for `session`, from bars strictly before it."""
    history = {k: v[v.index < session] for k, v in daily.items()}
    if history["close"].empty:
        return []
    # Live, today's daily bar does not exist before the open.  Appending an
    # empty row for the session lets the shift-by-one in `features` land the
    # last completed session's values on it — the same arithmetic a replay
    # does on a session that is already in the data.
    index = history["close"].index.append(pd.Index([session]))
    feats = features({k: v.reindex(index) for k, v in history.items()}, config)
    table = pd.DataFrame({name: feats[key].loc[session] for name, key in (
        ("prev_close", "prev_close"), ("gap", "gap_prev"), ("atr", "atr"), ("turnover", "turnover"),
    )}).dropna(subset=["prev_close", "turnover"])
    table = table[table["prev_close"] >= config.min_price]
    table = table.sort_values("turnover", ascending=False).head(config.universe_size)
    table = table.dropna(subset=["gap", "atr"])
    table = table[table["atr"] > 0].sort_values("gap", ascending=False)

    per_position = config.capital / config.picks
    picks: list[Pick] = []
    for rank, (symbol, row) in enumerate(table.head(config.picks + config.reserves).iterrows(), 1):
        qty = int(per_position // row["prev_close"])
        if qty <= 0:
            continue
        picks.append(Pick(
            rank=rank, symbol=str(symbol), side="SHORT",
            gap_prev_pct=round(float(row["gap"]) * 100, 3),
            prev_close=round(float(row["prev_close"]), 2),
            atr=round(float(row["atr"]), 4),
            stop_distance=round(float(config.stop_atr * row["atr"]), 2),
            quantity=qty,
            position_value=round(qty * float(row["prev_close"]), 2),
            turnover_cr=round(float(row["turnover"]) / 1e7, 2),
            reserve=rank > config.picks,
        ))
    return picks


# ── execution simulation ────────────────────────────────────────────────────

def simulate_pick(pick: Pick, bars: pd.DataFrame, session: date,
                  config: GapReversalConfig = GapReversalConfig()) -> Trade | None:
    """Short at the 09:15 open, buy-stop at entry + stop_atr*ATR, cover at 15:15.

    Stop is checked on every bar including the first; if a later bar opens
    through the stop the fill is that open (a gap through a stop fills worse,
    never better).  Exit at the open of the 15:15 bar, or — if that bar is
    missing — at the last close before it.
    """
    if bars.empty or 0 not in bars.index:
        return None
    entry = float(bars.at[0, "open"])
    if not math.isfinite(entry) or entry <= 0:
        return None
    qty = int((config.capital / config.picks) // entry)
    if qty <= 0:
        return None
    stop = entry + config.stop_atr * pick.atr
    exit_price, reason, exit_bar = None, "SQUARE_OFF", SQUARE_OFF_BAR
    for bar in range(0, SQUARE_OFF_BAR):
        if bar not in bars.index:
            continue
        hi = float(bars.at[bar, "high"])
        if math.isfinite(hi) and hi >= stop:
            opened = float(bars.at[bar, "open"])
            exit_price = stop if bar == 0 else max(stop, opened)
            reason, exit_bar = "STOP", bar
            break
    if exit_price is None:
        if SQUARE_OFF_BAR in bars.index and math.isfinite(bars.at[SQUARE_OFF_BAR, "open"]):
            exit_price = float(bars.at[SQUARE_OFF_BAR, "open"])
        else:
            # Missing square-off bar: the last close before it, never after.
            closes = bars.loc[bars.index < SQUARE_OFF_BAR, "close"].dropna()
            if closes.empty:
                return None
            exit_price, reason, exit_bar = float(closes.iloc[-1]), "LAST_BAR", int(closes.index[-1])
    minutes = 555 + 5 * exit_bar
    gross = (entry - exit_price) * qty
    costs = round_trip_cost(entry, exit_price, qty,
                            slippage_bps_per_leg=config.slippage_bps_per_leg).total
    return Trade(session=session, symbol=pick.symbol, side="SHORT", entry=round(entry, 4),
                 exit=round(exit_price, 4), quantity=qty, exit_reason=reason,
                 exit_time=f"{minutes // 60:02d}:{minutes % 60:02d}",
                 gross=round(gross, 2), costs=round(costs, 2))


def backtest(sessions: list[date], config: GapReversalConfig = GapReversalConfig(),
             *, daily: dict[str, pd.DataFrame] | None = None,
             bar_loader=load_session_bars) -> BacktestResult:
    """Replay the book over `sessions` on cached 5m bars.

    A pick with no bars that session (suspended, not yet listed) is skipped and
    the next-ranked reserve takes its slot, as a live trader would do.
    """
    if daily is None:
        since = (pd.Timestamp(min(sessions)) - pd.Timedelta(days=90)).date().isoformat()
        daily = load_daily(nifty500_symbols(), since=since)
    result = BacktestResult(config=config, sessions=list(sessions))
    for session in sessions:
        taken = 0
        for pick in select(daily, session, config):
            if taken >= config.picks:
                break
            trade = simulate_pick(pick, bar_loader(pick.symbol, session), session, config)
            if trade is not None:
                result.trades.append(trade)
                taken += 1
    return result


def cached_sessions(since: str, until: str | None = None, *, min_symbols: int = 300,
                    db_path: Path | str = DB_PATH) -> list[date]:
    """Sessions with 5m bars for most of the universe — the replayable ones."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = con.execute(
            "SELECT substr(ts, 1, 10) AS d, COUNT(DISTINCT symbol) FROM candles "
            "WHERE interval='5m' AND symbol LIKE '%.NS' AND ts>=? AND ts LIKE '%T03:45:00%' "
            "GROUP BY d", (since,)).fetchall()
    finally:
        con.close()
    # 03:45 UTC is the 09:15 IST opening bar: one row per symbol per session.
    days = sorted(date.fromisoformat(d) for d, n in rows if n >= min_symbols)
    if until:
        days = [d for d in days if d <= date.fromisoformat(until)]
    return days


def backtest_daily(daily: dict[str, pd.DataFrame],
                   config: GapReversalConfig = GapReversalConfig(),
                   cost_bps: float = 13.0) -> pd.DataFrame:
    """Decade-scale check on daily bars: open -> close, stop via the day's high.

    Exact for this trade on daily data — a short entered AT the open is
    stopped out if and only if the day's high reaches entry + stop, because a
    gap through the stop is impossible when the entry is the open itself.
    What daily bars cannot see is the 15:15 exit (the close is used), so this
    is the long-sample *sanity check*; `backtest` on 5m bars is the execution
    model.  Returns one row per session: mean net bps per trade across picks.
    """
    o, h, c = daily["open"], daily["high"], daily["close"]
    feats = features(daily, config)
    ok = feats["prev_close"].notna() & (feats["prev_close"] >= config.min_price) \
        & feats["turnover"].notna()
    liquid = feats["turnover"].where(ok).rank(axis=1, ascending=False) <= config.universe_size
    score = feats["gap_prev"].where(liquid & feats["atr"].gt(0) & o.notna() & c.notna())
    chosen = score.rank(axis=1, ascending=False, method="first") <= config.picks
    short = (o - c) / o * 1e4
    stop = config.stop_atr * feats["atr"]
    short = short.where(~(h >= o + stop), -stop / o * 1e4)
    per_trade = short.where(chosen) - cost_bps
    out = pd.DataFrame({"net_bps": per_trade.mean(axis=1), "picks": chosen.sum(axis=1)})
    return out[out["picks"] > 0]


# ── keeping the inputs fresh ────────────────────────────────────────────────

def _normalise_download(raw: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """One symbol's OHLCV out of a yfinance batch download, IST-indexed."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [frame.columns.get_level_values(i) for i in range(frame.columns.nlevels)]
        level = next((i for i, lv in enumerate(levels) if symbol in set(lv)), None)
        if level is None:
            return pd.DataFrame()
        frame = frame.xs(symbol, axis=1, level=level)
    frame = frame.rename(columns=lambda c: str(c).lower())
    need = ["open", "high", "low", "close"]
    if not all(c in frame.columns for c in need):
        return pd.DataFrame()
    out = frame[need + (["volume"] if "volume" in frame.columns else [])].dropna(subset=need).copy()
    if "volume" not in out:
        out["volume"] = 0.0
    idx = pd.to_datetime(out.index)
    idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    if interval == "1d":
        # Daily bars are dates; stamp them at the close so the cache's UTC
        # canonicalisation round-trips to the same calendar day (matches
        # scripts/fetch_daily.py, which wrote the decade of history).
        idx = idx.normalize() + pd.Timedelta(hours=15, minutes=30)
    else:
        keep = (idx.second == 0) & (idx.minute % 5 == 0)     # drop snapshot quotes
        out, idx = out[keep], idx[keep]
    out.index = idx
    return out[~out.index.duplicated(keep="last")].astype(float)


def refresh(symbols: list[str], *, interval: str, period: str, batch: int = 60,
            db_path: Path | str = DB_PATH, log=print) -> set[str]:
    """Download `period` of `interval` bars into the candle cache.

    Returns the symbols that actually arrived.  Yahoo reports a failed request
    as "possibly delisted", so a network or clock problem looks like 500
    delistings — callers must check the count, not trust the call.
    """
    import warnings

    import yfinance as yf

    from nse_intraday_ai.candle_cache import CandleCache

    cache = CandleCache(db_path)
    arrived: set[str] = set()
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(chunk, period=period, interval=interval, group_by="ticker",
                                  auto_adjust=False, progress=False, threads=True)
        except Exception as exc:                                    # noqa: BLE001
            log(f"  download failed for {len(chunk)} symbols: {type(exc).__name__}: {exc}")
            continue
        for symbol in chunk:
            frame = _normalise_download(raw, symbol, interval)
            if not frame.empty and cache.save(symbol, interval, frame):
                arrived.add(symbol)
    return arrived


def missing_intraday(symbols: list[str], session: date, *, bar: int = 0,
                     db_path: Path | str = DB_PATH) -> list[str]:
    """Symbols with no cached 5m bar number `bar` (0 = 09:15) for `session`.

    Check the bar you actually need: after the close that is the square-off
    bar, because a morning run caches a partial opening bar that would
    otherwise pass for a complete session.
    """
    stamp = pd.Timestamp(session, tz=IST) + pd.Timedelta(minutes=555 + 5 * bar)
    key = stamp.tz_convert("UTC").isoformat()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        have = {r[0] for r in con.execute(
            f"SELECT symbol FROM candles WHERE interval='5m' AND ts=? AND symbol IN "
            f"({','.join('?' * len(symbols))})", (key, *symbols))} if symbols else set()
    finally:
        con.close()
    return [s for s in symbols if s not in have]


# ── the published pick list (read by the app, the phone push and `record`) ──

OUT_DIR = ROOT / "data" / "gap_reversal"
PICKS_PATH = OUT_DIR / "picks.json"


def next_session(now: pd.Timestamp | None = None) -> date:
    """Today if the session has not closed yet, else the next weekday.

    Exchange holidays are not known here; the `levels` step finds out (no
    opening bar) and says so.
    """
    now = pd.Timestamp.now(tz=IST) if now is None else now
    day = now.date()
    if now.weekday() < 5 and (now.hour, now.minute) < (15, 30):
        return day
    day = (pd.Timestamp(day) + pd.offsets.BDay(1)).date()
    return day


def read_picks(path: Path | None = None) -> dict | None:
    """The saved pick-list payload as written, whatever session it is for."""
    import json

    path = PICKS_PATH if path is None else path
    return json.loads(path.read_text()) if path.exists() else None


def save_picks(session: date, picks: list[Pick], config: GapReversalConfig = GapReversalConfig(),
               path: Path | None = None, **extra) -> None:
    import json

    path = PICKS_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session": session.isoformat(),
        "generated_at": pd.Timestamp.now(tz=IST).isoformat(timespec="seconds"),
        "config": asdict(config),
        "picks": [p.to_dict() for p in picks],
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_picks(session: date, path: Path | None = None) -> list[Pick] | None:
    """The saved list for `session`, or None if the file is for another day."""
    payload = read_picks(path)
    if payload is None or payload.get("session") != session.isoformat():
        return None
    fields = set(Pick.__dataclass_fields__)
    return [Pick(**{k: v for k, v in p.items() if k in fields}) for p in payload["picks"]]


class StaleDataError(RuntimeError):
    """The daily bars needed to rank a session are missing or incomplete."""


def session_coverage(symbols: list[str], since: date,
                     db_path: Path | str = DB_PATH) -> dict[date, int]:
    """Symbols with a *real* daily bar (volume > 0) on each date since `since`."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = con.execute(
            "SELECT symbol, ts FROM candles WHERE interval='1d' AND volume > 0 AND ts >= ?",
            (pd.Timestamp(since).isoformat(),)).fetchall()
    finally:
        con.close()
    wanted = set(symbols)
    out: dict[date, set[str]] = {}
    for symbol, ts in rows:
        if symbol in wanted:
            day = pd.Timestamp(ts).tz_convert(IST).date()
            out.setdefault(day, set()).add(symbol)
    return {d: len(v) for d, v in sorted(out.items())}


def check_fresh(symbols: list[str], session: date, *, min_coverage: float = 0.9,
                db_path: Path | str = DB_PATH) -> date:
    """The last completed session before `session`, or raise StaleDataError.

    Every weekday between the newest well-covered session and `session` must
    be a holiday — i.e. carry no real bars at all.  A weekday with *some* real
    bars is a session whose download is incomplete (2026-09-29: the clock was
    wrong at boot, 194 of 500 names arrived, and the ranking silently used the
    day before).  A weekday with none after a successful refresh is a holiday.
    """
    since = (pd.Timestamp(session) - pd.Timedelta(days=15)).date()
    coverage = session_coverage(symbols, since, db_path=db_path)
    need = min_coverage * len(symbols)
    complete = [d for d, n in coverage.items() if d < session and n >= need]
    if not complete:
        raise StaleDataError(f"no complete daily session in the 15 days before {session}")
    last = complete[-1]
    for day in pd.bdate_range(last + pd.Timedelta(days=1), session - pd.Timedelta(days=1)):
        n = coverage.get(day.date(), 0)
        if n > 0:
            raise StaleDataError(
                f"daily bars for {day.date()} are incomplete ({n}/{len(symbols)} names) — "
                f"ranking {session} on {last} would use the wrong 'yesterday'")
    return last


def publish_picks(session: date | None = None, config: GapReversalConfig = GapReversalConfig(),
                  *, fetch: bool = True, retries: int = 3, log=print) -> tuple[date, date, list[Pick]]:
    """Rank the universe for `session` and write the pick list.

    Returns (session, the last completed session the ranking used, picks).
    Raises StaleDataError rather than publish a list built on missing data.
    """
    import time

    session = session or next_session()
    symbols = nifty500_symbols()
    if fetch:
        log(f"refreshing daily bars for {len(symbols)} names...")
        todo = symbols
        for attempt in range(retries):
            arrived = refresh(todo, interval="1d", period="3mo", log=log)
            todo = [s for s in todo if s not in arrived]
            if len(todo) <= 0.05 * len(symbols):
                break
            log(f"  {len(todo)} names did not arrive; retrying ({attempt + 1}/{retries})")
            time.sleep(20 * (attempt + 1))
        if len(todo) > 0.1 * len(symbols):
            raise StaleDataError(f"{len(todo)}/{len(symbols)} names failed to download "
                                 "(network, clock or Yahoo outage)")
    last = check_fresh(symbols, session)
    daily = load_daily(symbols, since=(pd.Timestamp(session) - pd.Timedelta(days=120)).date().isoformat())
    picks = select(daily, session, config)
    save_picks(session, picks, config, based_on=last.isoformat())
    return session, last, picks
