"""NSE's own end-of-day files: every listed stock, split-adjusted prev close, delivery.

Why this exists rather than more Yahoo:

* **Survivorship.** Yahoo is queried for today's NIFTY 500, so a decade of
  history contains only names that were winners enough to be in the index
  now.  A cross-sectional model trained on that learns "don't short future
  winners", which backtests beautifully and cannot be traded.  The bhavcopy
  lists every stock that traded that day, delisted and demoted ones included,
  so the universe can be formed point-in-time.
* **Corporate actions.** ``PREV_CLOSE`` is NSE's *adjusted* previous close on
  ex-dates, so overnight gaps and ATR are correct through splits and bonuses
  without an adjustment table.
* **Information Yahoo does not have:** delivery quantity (how much of the day's
  volume was bought to hold), trade count (average trade size), and the
  series (EQ = intraday-shortable; BE = trade-for-trade, not shortable).
* **Reliability.** One official file per day, published ~18:00 IST, instead of
  500 requests that fail together when the network or clock hiccups.

Sources (all public, no key):
  2019-10 ..  sec_bhavdata_full_DDMMYYYY.csv       OHLC + prev close + trades + delivery
  .. 2024-07  cmDDMONYYYYbhav.csv.zip              OHLC + prev close + trades
  .. now      MTO_DDMMYYYY.DAT                     delivery (joined to the above)
"""
from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ARCHIVE = "https://archives.nseindia.com"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "nse_bhav"
FULL_FROM = date(2019, 10, 1)         # sec_bhavdata_full exists from here
SERIES = ("EQ", "BE")                 # EQ: normal (MIS-shortable); BE: trade-for-trade
COLUMNS = ["symbol", "series", "open", "high", "low", "close", "prev_close", "volume",
           "value", "trades", "deliv_qty", "deliv_pct"]
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def _get(session: requests.Session, url: str, retries: int = 3, min_size: int = 500) -> bytes | None:
    """Body of `url`, None on 404 (a holiday or a file that does not exist).

    `min_size` guards against error pages served with a 200; lists that can
    legitimately be a few bytes long (the F&O ban list) pass a smaller one.
    """
    for attempt in range(retries):
        try:
            r = session.get(url, headers=_HEADERS, timeout=30)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        if (r.status_code == 200 and len(r.content) >= min_size
                and not r.content.lstrip().startswith(b"<")):
            return r.content
        time.sleep(2 * (attempt + 1))
    raise ConnectionError(f"could not fetch {url}")


def _num(frame: pd.DataFrame, cols) -> None:
    for c in cols:
        frame[c] = pd.to_numeric(frame[c].astype(str).str.strip().replace({"-": None, "": None}),
                                 errors="coerce")


def _file_date(values: pd.Series) -> date | None:
    stamps = pd.to_datetime(values.astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
    stamps = stamps.dropna()
    return stamps.mode().iloc[0].date() if len(stamps) else None


def _parse_full(raw: bytes) -> pd.DataFrame:
    f = pd.read_csv(io.BytesIO(raw), skipinitialspace=True, encoding="latin-1")
    f.columns = [c.strip().upper() for c in f.columns]
    file_date = _file_date(f["DATE1"])
    f["SERIES"] = f["SERIES"].astype(str).str.strip()
    f = f[f["SERIES"].isin(SERIES)].copy()
    out = pd.DataFrame({
        "symbol": f["SYMBOL"].astype(str).str.strip(), "series": f["SERIES"],
        "open": f["OPEN_PRICE"], "high": f["HIGH_PRICE"], "low": f["LOW_PRICE"],
        "close": f["CLOSE_PRICE"], "prev_close": f["PREV_CLOSE"], "volume": f["TTL_TRD_QNTY"],
        "value": f["TURNOVER_LACS"], "trades": f["NO_OF_TRADES"],
        "deliv_qty": f["DELIV_QTY"], "deliv_pct": f["DELIV_PER"],
    })
    _num(out, COLUMNS[2:])
    out["value"] = out["value"] * 1e5                       # lakh -> rupees
    out.attrs["file_date"] = file_date
    return out


def _parse_cm(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        f = pd.read_csv(z.open(z.namelist()[0]), encoding="latin-1")
    f.columns = [c.strip().upper() for c in f.columns]
    file_date = _file_date(f["TIMESTAMP"])
    f["SERIES"] = f["SERIES"].astype(str).str.strip()
    f = f[f["SERIES"].isin(SERIES)].copy()
    out = pd.DataFrame({
        "symbol": f["SYMBOL"].astype(str).str.strip(), "series": f["SERIES"],
        "open": f["OPEN"], "high": f["HIGH"], "low": f["LOW"], "close": f["CLOSE"],
        "prev_close": f["PREVCLOSE"], "volume": f["TOTTRDQTY"], "value": f["TOTTRDVAL"],
        "trades": f.get("TOTALTRADES"),
    })
    _num(out, ["open", "high", "low", "close", "prev_close", "volume", "value", "trades"])
    out.attrs["file_date"] = file_date
    return out


def _parse_mto(raw: bytes) -> pd.DataFrame:
    rows = []
    for line in raw.decode("latin-1").splitlines():
        parts = line.split(",")
        if len(parts) >= 7 and parts[0].strip() == "20":
            rows.append((parts[2].strip(), parts[3].strip(), parts[5].strip(), parts[6].strip()))
    f = pd.DataFrame(rows, columns=["symbol", "series", "deliv_qty", "deliv_pct"])
    _num(f, ["deliv_qty", "deliv_pct"])
    return f


def fetch_day(day: date, session: requests.Session | None = None) -> pd.DataFrame | None:
    """One session's EQ/BE rows, or None if NSE has no file (holiday).

    NSE's archive answers some holiday dates with a copy of the previous
    session's file (2026-09-14 returns 2026-09-11's rows verbatim), so the date
    printed *inside* the file decides, not the file name.
    """
    s = session or requests.Session()
    if day >= FULL_FROM:
        raw = _get(s, f"{ARCHIVE}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv")
        frame = None
        if raw is not None:
            try:
                frame = _parse_full(raw)
            except Exception:                                  # noqa: BLE001
                # 2022-08-08 is an .xlsx served under the .csv name; the older
                # bhavcopy + MTO pair below covers that day.
                frame = None
        if frame is not None:
            if frame.attrs.pop("file_date", None) != day:
                return None
            return frame
    mon = day.strftime("%b").upper()
    raw = _get(s, f"{ARCHIVE}/content/historical/EQUITIES/{day:%Y}/{mon}/cm{day:%d}{mon}{day:%Y}bhav.csv.zip")
    if raw is None:
        return None
    frame = _parse_cm(raw)
    if frame.attrs.pop("file_date", None) != day:
        return None
    mto = _get(s, f"{ARCHIVE}/archives/equities/mto/MTO_{day:%d%m%Y}.DAT")
    if mto is not None:
        frame = frame.merge(_parse_mto(mto), on=["symbol", "series"], how="left")
    else:
        frame["deliv_qty"] = float("nan")
        frame["deliv_pct"] = float("nan")
    return frame[COLUMNS]


def _path(day: date, cache: Path) -> Path:
    return cache / f"{day:%Y}" / f"bhav_{day:%Y%m%d}.parquet"


def _holiday_marker(day: date, cache: Path) -> Path:
    return cache / f"{day:%Y}" / f"bhav_{day:%Y%m%d}.holiday"


def backfill(start: date, end: date, *, cache: Path = CACHE, workers: int = 3,
             pause: float = 0.2, log=print) -> tuple[int, int]:
    """Download every missing weekday in [start, end]. Returns (fetched, holidays)."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if d.weekday() < 5 and not _path(d, cache).exists()
            and not _holiday_marker(d, cache).exists()]
    counts = {"ok": 0, "holiday": 0}

    def work(chunk: list[date]) -> None:
        s = requests.Session()
        for d in chunk:
            try:
                frame = fetch_day(d, s)
            except Exception as exc:                          # noqa: BLE001
                log(f"  {d}: {type(exc).__name__}: {exc}")    # retried on the next run
                continue
            _path(d, cache).parent.mkdir(parents=True, exist_ok=True)
            if frame is None or frame.empty:
                # A weekday with no file is a holiday — but only once it is in
                # the past; today's file may simply not be published yet.
                if d < date.today():
                    _holiday_marker(d, cache).touch()
                    counts["holiday"] += 1
            else:
                frame.to_parquet(_path(d, cache), index=False)
                counts["ok"] += 1
            time.sleep(pause)

    chunks = [todo[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, chunks))
    return counts["ok"], counts["holiday"]


def load(start: date, end: date, *, cache: Path = CACHE, series: tuple[str, ...] = ("EQ",),
         columns: list[str] | None = None) -> pd.DataFrame:
    """Long frame (session, symbol, ...) of cached sessions in [start, end]."""
    parts = []
    for year in range(start.year, end.year + 1):
        for p in sorted((cache / f"{year}").glob("bhav_*.parquet")):
            d = date(int(p.stem[5:9]), int(p.stem[9:11]), int(p.stem[11:13]))
            if start <= d <= end:
                f = pd.read_parquet(p, columns=columns)
                f.insert(0, "session", d)
                parts.append(f)
    if not parts:
        return pd.DataFrame(columns=["session", *COLUMNS])
    frame = pd.concat(parts, ignore_index=True)
    if series and "series" in frame:
        frame = frame[frame["series"].isin(series)]
    return frame
