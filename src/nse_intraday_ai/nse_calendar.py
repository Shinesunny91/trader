"""NSE trading calendar: exchange holidays, cached locally.

`gap_reversal.next_session()` used to treat every weekday as a session, so on
exchange holidays (e.g. Gandhi Jayanti, 2026-10-02) the 08:45 job still
published — and pushed to the phone — a short list for a day with no market.

Sources, in order:
1. NSE's holiday master (``/api/holiday-master?type=trading``), cached per
   year in ``data/nse_holidays.json`` and refreshed every 7 days.
2. Holiday markers written by the bhavcopy downloader
   (``data/nse_bhav/<yyyy>/bhav_<yyyymmdd>.holiday``) — authoritative after
   the fact, so past holidays are known even with no network.

If neither knows a date, it is assumed to be a trading day (weekdays only):
failing open matches the old behaviour rather than silently skipping a day.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "nse_holidays.json"
BHAV = ROOT / "data" / "nse_bhav"
_TTL_SECONDS = 7 * 24 * 3600
_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}
_memo: set[date] | None = None


def _fetch() -> list[str]:
    import requests

    s = requests.Session()
    s.get("https://www.nseindia.com/", headers=_HEADERS, timeout=10)
    r = s.get(_URL, headers=_HEADERS, timeout=10)
    r.raise_for_status()
    rows = r.json().get("CM", [])          # CM = capital-market (equity) segment
    return [datetime.strptime(x["tradingDate"], "%d-%b-%Y").date().isoformat() for x in rows]


def _load() -> set[date]:
    from nse_intraday_ai.atomic_io import atomic_read_json, atomic_write_json

    cached = atomic_read_json(CACHE, default={}) or {}
    days = set(cached.get("holidays", []))
    if time.time() - cached.get("fetched_at", 0) > _TTL_SECONDS:
        try:
            days |= set(_fetch())
            atomic_write_json(CACHE, {"fetched_at": time.time(), "holidays": sorted(days)})
        except Exception as exc:                                 # noqa: BLE001 — network is optional
            print(f"nse_calendar: holiday refresh failed ({type(exc).__name__}); using cache/markers")
    out = {date.fromisoformat(d) for d in days}
    for marker in BHAV.glob("*/bhav_*.holiday"):
        out.add(datetime.strptime(marker.stem.split("_")[1], "%Y%m%d").date())
    return out


def holidays(refresh: bool = False) -> set[date]:
    global _memo
    if _memo is None or refresh:
        _memo = _load()
    return _memo


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in holidays()


def next_trading_day(day: date) -> date:
    """The first trading day strictly after `day`."""
    d = day + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d
