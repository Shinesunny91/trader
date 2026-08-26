"""Two India-specific daily feeds the engine was not using, both free.

Everything the engine currently scores is price and volume derived from a US
quote vendor. That misses the two readings Indian desks actually watch, and both
are published by NSE itself as plain CSV with no key and years of archive:

**Delivery percentage** (`sec_bhavdata_full`). NSE settles intraday churn
against actual delivery, and reports the split per symbol per day. A stock that
rose on 70% delivery was bought by people willing to hold it overnight; the same
move on 20% delivery is speculative flow that unwinds by the close. Nothing in
the current feature set can distinguish those, because Yahoo's volume field
counts both identically.

**Participant open interest** (`fao_participant_oi`). Daily FII / DII / Pro /
Client positioning in index and stock futures. FII net index-future positioning
is the closest thing Indian equities have to a published sentiment gauge.

Both are *daily and lagged* — published after the close — so they can only be
used as prior-day context for the next session. That is a real constraint, not a
detail: using same-day delivery data to score a same-day signal is look-ahead,
and the loader deliberately makes the lag explicit.
"""
from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BHAV_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
OI_URL = "https://archives.nseindia.com/content/nsccl/fao_participant_oi_{ddmmyyyy}.csv"

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA, "Accept": "text/csv,*/*"}


def _get(url: str, timeout: float = 30.0) -> bytes | None:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
    except requests.RequestException:
        return None
    # A market holiday is a 404, which is information rather than an error.
    if r.status_code != 200 or not r.content or len(r.content) < 200:
        return None
    return r.content


# ── Delivery percentage ────────────────────────────────────────────────────

def fetch_delivery(day: date) -> pd.DataFrame:
    """Per-symbol delivery split for one session. Empty on a holiday."""
    raw = _get(BHAV_URL.format(ddmmyyyy=day.strftime("%d%m%Y")))
    if raw is None:
        return pd.DataFrame()
    frame = pd.read_csv(io.BytesIO(raw))
    frame.columns = [c.strip().upper() for c in frame.columns]
    needed = {"SYMBOL", "SERIES", "DELIV_PER", "DELIV_QTY", "TTL_TRD_QNTY"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame[frame["SERIES"].astype(str).str.strip() == "EQ"].copy()
    for col in ("DELIV_PER", "DELIV_QTY", "TTL_TRD_QNTY", "NO_OF_TRADES", "CLOSE_PRICE"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["symbol"] = frame["SYMBOL"].astype(str).str.strip() + ".NS"
    frame["date"] = pd.Timestamp(day)
    # Average trade size is a crude but useful participation read: the same
    # turnover split over few large trades looks different from many small ones.
    frame["avg_trade_qty"] = frame["TTL_TRD_QNTY"] / frame["NO_OF_TRADES"].replace(0, pd.NA)
    return frame[["date", "symbol", "DELIV_PER", "DELIV_QTY", "TTL_TRD_QNTY",
                  "avg_trade_qty"]].rename(columns={
        "DELIV_PER": "deliv_pct", "DELIV_QTY": "deliv_qty", "TTL_TRD_QNTY": "traded_qty",
    }).dropna(subset=["deliv_pct"])


# ── Participant open interest ──────────────────────────────────────────────

def fetch_participant_oi(day: date) -> pd.DataFrame:
    """FII / DII / Pro / Client futures + options OI for one session."""
    raw = _get(OI_URL.format(ddmmyyyy=day.strftime("%d%m%Y")))
    if raw is None:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(io.BytesIO(raw), skiprows=1)
    except Exception:                                   # noqa: BLE001
        return pd.DataFrame()
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Client Type" not in frame.columns:
        return pd.DataFrame()
    frame["Client Type"] = frame["Client Type"].astype(str).str.strip()
    for col in frame.columns:
        if col != "Client Type":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.Timestamp(day)
    return frame


def participant_features(oi: pd.DataFrame) -> dict:
    """Collapse one day's OI table into a few readings worth scoring.

    Net index-future positioning is the headline: FIIs run large directional
    index books and the *change* in that net is the sentiment signal, so this
    returns levels and the caller differences them across days.
    """
    if oi.empty or "Client Type" not in oi.columns:
        return {}
    rows = oi.set_index("Client Type")

    def val(who: str, col: str) -> float:
        try:
            return float(rows.loc[who, col])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    out: dict[str, float] = {}
    for who in ("FII", "DII", "Client", "Pro"):
        long_ = val(who, "Future Index Long")
        short = val(who, "Future Index Short")
        total = long_ + short
        out[f"{who.lower()}_idx_fut_net"] = long_ - short
        # Ratio in [-1, 1] — comparable across days regardless of total OI.
        out[f"{who.lower()}_idx_fut_ratio"] = (long_ - short) / total if total else float("nan")
        s_long = val(who, "Future Stock Long")
        s_short = val(who, "Future Stock Short")
        s_total = s_long + s_short
        out[f"{who.lower()}_stk_fut_ratio"] = (
            (s_long - s_short) / s_total if s_total else float("nan")
        )
    return out


# ── Bulk loading with an on-disk cache ─────────────────────────────────────

def load_history(
    start: date,
    end: date,
    cache_dir: Path,
    *,
    what: str = "delivery",
    verbose: bool = False,
    pause: float = 0.4,
) -> pd.DataFrame:
    """Fetch a date range once and reuse it. Holidays cache as empty markers."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fetch = fetch_delivery if what == "delivery" else fetch_participant_oi

    parts: list[pd.DataFrame] = []
    day = start
    while day <= end:
        if day.weekday() >= 5:                       # NSE is shut at weekends
            day += timedelta(days=1)
            continue
        path = cache_dir / f"{what}_{day:%Y%m%d}.parquet"
        marker = cache_dir / f"{what}_{day:%Y%m%d}.holiday"
        if path.exists():
            parts.append(pd.read_parquet(path))
        elif marker.exists():
            pass
        else:
            import time
            time.sleep(pause)          # be a polite client of a free archive
            frame = fetch(day)
            if frame.empty:
                marker.touch()
                if verbose:
                    print(f"  {day} no data (holiday or missing)")
            else:
                frame.to_parquet(path)
                parts.append(frame)
                if verbose:
                    print(f"  {day} {len(frame)} rows")
        day += timedelta(days=1)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def delivery_features(history: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol delivery features, lagged one session.

    The lag is the whole point: delivery data for session T is published after
    T closes, so the earliest a signal may use it is T+1. Everything here is
    shifted accordingly, and the column names say so.
    """
    if history.empty:
        return pd.DataFrame()
    h = history.sort_values(["symbol", "date"]).copy()
    g = h.groupby("symbol", sort=False)

    h["deliv_pct_20d"] = g["deliv_pct"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    h["deliv_pct_z"] = g["deliv_pct"].transform(
        lambda s: (s - s.rolling(60, min_periods=20).mean())
        / (s.rolling(60, min_periods=20).std() + 1e-9)
    )
    h["deliv_qty_ratio"] = h["deliv_qty"] / g["deliv_qty"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    ).replace(0, pd.NA)
    h["avg_trade_qty_z"] = g["avg_trade_qty"].transform(
        lambda s: (s - s.rolling(60, min_periods=20).mean())
        / (s.rolling(60, min_periods=20).std() + 1e-9)
    )

    cols = ["deliv_pct", "deliv_pct_20d", "deliv_pct_z", "deliv_qty_ratio", "avg_trade_qty_z"]
    out = h[["date", "symbol"] + cols].copy()
    # Shift so row `date` carries the PREVIOUS session's published values.
    for col in cols:
        out[col] = out.groupby("symbol", sort=False)[col].shift(1)
    return out.rename(columns={c: f"prev_{c}" for c in cols})
