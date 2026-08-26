from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

_INTERVAL_MINUTES = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60}


def drop_synthetic_bars(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Remove rows that are not real candles for this interval.

    Yahoo's intraday endpoints append a *snapshot* row carrying the current
    quote: it sits off the interval grid (e.g. 09:15:17 on a 5m series), has
    zero volume, and O=H=L=C at the last traded price.  It is not a bar.

    Left in, it does real damage: a backtest that fills "at the next bar's
    open" fills at that stale snapshot price instead of the next genuine open,
    which on 2026-08-12 handed the simulated book a 0.8% head start on a DIXON
    short and turned a losing session into a winning one.  Live, the same row
    makes the newest "closed bar" a flat doji that distorts ATR and volume-z.
    """
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    minutes = _INTERVAL_MINUTES.get(interval)
    if not minutes:
        return frame
    aligned = (frame.index.second == 0) & (frame.index.minute % minutes == 0)
    return frame[aligned]


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    symbol   TEXT    NOT NULL,
    interval TEXT    NOT NULL,
    ts       TEXT    NOT NULL,
    open     REAL    NOT NULL,
    high     REAL    NOT NULL,
    low      REAL    NOT NULL,
    close    REAL    NOT NULL,
    volume   REAL    NOT NULL,
    PRIMARY KEY (symbol, interval, ts)
);
CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles (symbol, interval, ts DESC);
"""


class CandleCache:
    """SQLite-backed OHLCV store.

    YFinanceProvider saves every successful fetch here and reads from here
    when the network is unavailable.  Data accumulates indefinitely across
    sessions — the longer the system runs, the richer the historical dataset
    available for backtesting and ML features.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CREATE_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ── Write ──────────────────────────────────────────────────────────────

    def save(self, symbol: str, interval: str, frame: pd.DataFrame) -> int:
        """Upsert candles from a normalised OHLCV DataFrame. Returns rows saved."""
        if frame.empty:
            return 0
        rows = []
        for ts, row in frame.iterrows():
            # Canonicalize to UTC isoformat: ts is stored as TEXT and is part
            # of the PRIMARY KEY, so "+05:30" and "+00:00" spellings of the
            # same instant would otherwise coexist as duplicate bars
            # (2026-07-07 incident: mixed forms produced duplicate-index
            # frames that crashed strategy evaluation).
            if hasattr(ts, "tz_convert") and getattr(ts, "tzinfo", None) is not None:
                ts_str = ts.tz_convert("UTC").isoformat()
            elif hasattr(ts, "isoformat"):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
            rows.append((
                symbol, interval, ts_str,
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]),
                float(row.get("volume", 0)),
            ))
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO candles
                   (symbol, interval, ts, open, high, low, close, volume)
                   VALUES (?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    # ── Read ───────────────────────────────────────────────────────────────

    def load(
        self,
        symbol: str,
        interval: str,
        since: datetime | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Return cached candles as a DatetimeIndex DataFrame, newest-first limited."""
        since_str = since.isoformat() if since else "1970-01-01"
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT ts, open, high, low, close, volume
                   FROM candles
                   WHERE symbol=? AND interval=? AND ts>=?
                   ORDER BY ts DESC LIMIT ?""",
                (symbol, interval, since_str, limit),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
        df = df.set_index("ts").sort_index()
        # Defensive: legacy rows may still carry both timestamp spellings of
        # one instant — a duplicate index breaks indicator/strategy code.
        df = df[~df.index.duplicated(keep="last")]
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        return drop_synthetic_bars(df, interval)

    def load_period(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """Load candles for the yfinance-style period string (e.g. '1d', '5d', '60d')."""
        unit = period[-1]
        n = int(period[:-1])
        delta = {"d": timedelta(days=n), "w": timedelta(weeks=n),
                 "m": timedelta(days=n * 30)}.get(unit, timedelta(days=n))
        since = datetime.now(IST) - delta
        return self.load(symbol, interval, since=since)

    def latest_ts(self, symbol: str, interval: str) -> datetime | None:
        """Timestamp of the most recent cached candle for symbol/interval."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(ts) FROM candles WHERE symbol=? AND interval=?",
                (symbol, interval),
            ).fetchone()
        if not row or not row[0]:
            return None
        return datetime.fromisoformat(row[0])

    def symbol_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(DISTINCT symbol) FROM candles").fetchone()[0]

    def candle_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]

    def stats(self) -> dict:
        with self._connect() as conn:
            symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM candles").fetchone()[0]
            total   = conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
            oldest  = conn.execute("SELECT MIN(ts) FROM candles").fetchone()[0]
            newest  = conn.execute("SELECT MAX(ts) FROM candles").fetchone()[0]
        return {"symbols": symbols, "candles": total, "oldest": oldest, "newest": newest}
