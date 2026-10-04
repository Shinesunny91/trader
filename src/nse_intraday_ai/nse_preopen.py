"""NSE's pre-open call auction: today's opening prices, known by ~09:12.

Orders are collected from 09:00 (market orders only until 09:05 since
2026-09-07; entry closes at a random moment in 09:08-09:10), the auction clears
at one equilibrium price per stock (which *becomes* the official open), and the
continuous market starts at 09:15.  So between ~09:12 and 09:15 every stock's
opening gap is public — the freshest cross-sectional information of the day.
The open re-rank (timer nse-gap-final) uses it.

Historically the auction price equals the bhavcopy's official open, so features
built from it are backtestable exactly.  The feed also carries what NSE does not
archive — the unmatched buy/sell quantities and the auction volume — so every
snapshot is saved: after a few hundred sessions the weekly learner can test
whether auction imbalance adds information (it cannot be tested before).
"""
from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "data" / "nse_preopen"
URL = "https://www.nseindia.com/api/market-data-pre-open?key={key}"
IST = ZoneInfo("Asia/Kolkata")
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market",
}


class PreOpenNotReady(RuntimeError):
    """The feed is not (yet) today's final auction result."""


def parse(payload: dict) -> pd.DataFrame:
    rows = []
    for item in payload.get("data", []):
        m, d = item.get("metadata", {}), item.get("detail", {}).get("preOpenMarket", {})
        rows.append({
            "symbol": str(m.get("symbol", "")).strip(), "series": str(m.get("series", "")).strip(),
            "iep": d.get("IEP"), "prev_close": m.get("previousClose"),
            "final_qty": d.get("finalQuantity"), "buy_qty": d.get("totalBuyQuantity"),
            "sell_qty": d.get("totalSellQuantity"), "ato_buy": d.get("atoBuyQty"),
            "ato_sell": d.get("atoSellQty"), "updated": d.get("lastUpdateTime"),
        })
    frame = pd.DataFrame(rows)
    for col in ("iep", "prev_close", "final_qty", "buy_qty", "sell_qty", "ato_buy", "ato_sell"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["updated"] = pd.to_datetime(frame["updated"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    return frame


def fetch(key: str = "ALL", *, session: requests.Session | None = None, retries: int = 3) -> pd.DataFrame:
    s = session or requests.Session()
    last = None
    for attempt in range(retries):
        try:
            r = s.get(URL.format(key=key), headers=_HEADERS, timeout=20)
            if r.status_code == 200 and r.content[:1] == b"{":
                return parse(r.json())
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(3 * (attempt + 1))
    raise PreOpenNotReady(f"pre-open feed unavailable ({last})")


# NSE restructured the pre-open on 2026-09-07: market orders only 09:00-09:05,
# limit-only 09:05-09:10, order entry closes at a RANDOM time in 09:08-09:10,
# then matching to 09:12.  Before that, entry closed at exactly 09:08.
NEW_AUCTION_FROM = date(2026, 9, 7)


def _final_cutoff(day: date) -> str:
    return "09:10" if day >= NEW_AUCTION_FROM else "09:08"


def final_for(day: date, frame: pd.DataFrame) -> pd.DataFrame:
    """Keep EQ rows whose auction for `day` has finished (updated after order entry closed)."""
    eq = frame[(frame["series"] == "EQ") & (frame["iep"] > 0) & (frame["prev_close"] > 0)]
    stamp = eq["updated"].dropna()
    if stamp.empty or stamp.max().date() != day:
        raise PreOpenNotReady(f"pre-open feed is for {stamp.max().date() if not stamp.empty else '?'}, "
                              f"not {day}")
    cutoff = pd.Timestamp.combine(day, datetime.strptime(_final_cutoff(day), "%H:%M").time())
    done = eq[eq["updated"] >= cutoff]
    if len(done) < 0.8 * len(eq):
        raise PreOpenNotReady(f"only {len(done)}/{len(eq)} auctions final — too early")
    return done


def opens(frame: pd.DataFrame) -> pd.DataFrame:
    """symbol -> (open, prev_close) for features.build(opens=...)."""
    return frame.set_index("symbol")[["iep", "prev_close"]].rename(columns={"iep": "open"})


def archive(day: date, frame: pd.DataFrame) -> Path:
    path = ARCHIVE_DIR / f"{day:%Y}" / f"preopen_{day:%Y%m%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def wait_and_fetch(day: date, *, deadline: str = "09:14", poll_seconds: int = 10, log=print) -> pd.DataFrame:
    """Poll until the auction for `day` is final (or the deadline passes)."""
    stop = pd.Timestamp.combine(day, datetime.strptime(deadline, "%H:%M").time()).tz_localize(IST)
    while True:
        try:
            return final_for(day, fetch())
        except PreOpenNotReady as exc:
            if pd.Timestamp.now(tz=IST) >= stop:
                raise
            log(f"  pre-open not final yet ({exc}); retrying")
            time.sleep(poll_seconds)


def load_archive(start: date, end: date) -> pd.DataFrame:
    """Archived auction snapshots in [start, end] as (session, symbol, ...) rows."""
    parts = []
    for year in range(start.year, end.year + 1):
        for p in sorted((ARCHIVE_DIR / f"{year}").glob("preopen_*.parquet")):
            d = date(int(p.stem[8:12]), int(p.stem[12:14]), int(p.stem[14:16]))
            if start <= d <= end:
                f = pd.read_parquet(p)
                f.insert(0, "session", d)
                parts.append(f)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def archived_sessions() -> int:
    return len(list(ARCHIVE_DIR.glob("*/preopen_*.parquet")))
