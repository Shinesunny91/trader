"""NSE participant-wise open interest (`fao_participant_oi`), free and daily.

FII / DII / Pro / Client positioning in index and stock futures and options.
FII net index-future positioning is the closest thing Indian equities have to
a published sentiment gauge. The file is published after the close, so it is
prior-day context only; `features.py` applies the one-session lag.

(Delivery % used to be fetched here too; it now comes from the bhavcopy in
`nse_bhav`, so the duplicate fetcher was removed.)
"""
from __future__ import annotations

import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

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


def load_history(
    start: date,
    end: date,
    cache_dir: Path,
    *,
    what: str = "oi",
    verbose: bool = False,
    pause: float = 0.4,
) -> pd.DataFrame:
    """Fetch a date range once and reuse it. Holidays cache as empty markers.

    Files land as `<cache_dir>/oi_YYYYMMDD.parquet` (read by features.py).
    """
    if what != "oi":
        raise ValueError(f"unsupported feed {what!r} (only 'oi' remains)")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

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
        elif not marker.exists():
            time.sleep(pause)          # be a polite client of a free archive
            frame = fetch_participant_oi(day)
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
