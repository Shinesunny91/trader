"""Three small NSE files about crowding and shorting, one row per symbol per file date.

  combineoi_DDMMYYYY.zip     futures-equivalent OI against the market-wide position
                             limit (MWPL), end of day D.  A stock near its MWPL is
                             crowded; at 95% it enters the F&O ban.
  fo_secban_DDMMYYYY.csv     securities in the F&O ban *for trade date D* — published
                             the evening before, so known before D's open.
  shortselling_DDMMYYYY.csv  institutional short sales reported in file D (they
                             describe the previous trade date).

Stored per file date D as data/nse_extra/YYYY/extra_YYYYMMDD.parquet with
columns symbol, mwpl_util, no_fresh, in_ban (for trade date D), short_qty.
Timing is applied by the feature builder: in_ban is usable for session D
itself; the other two only from the next session on (short sales: two).
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

from nse_intraday_ai.nse_bhav import ARCHIVE, _get

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "nse_extra"


def _combineoi(raw: bytes, day: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        f = pd.read_csv(z.open(name), skipinitialspace=True, encoding="latin-1")
    f.columns = [c.strip() for c in f.columns]
    f = f[f["NSE Symbol"].notna()]                      # some old files carry blank rows
    stamp = pd.to_datetime(f["Date"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
    if stamp.notna().any() and stamp.dropna().mode().iloc[0].date() != day:
        return pd.DataFrame()                            # stale copy served on a holiday
    mwpl = pd.to_numeric(f["MWPL"], errors="coerce")
    # Older files have plain OI; the delta-based "future equivalent" column came later.
    oi_col = "Future Equivalent Open Interest" if "Future Equivalent Open Interest" in f else "Open Interest"
    fut_eq = pd.to_numeric(f[oi_col], errors="coerce")
    return pd.DataFrame({
        "symbol": f["NSE Symbol"].astype(str).str.strip(),
        "mwpl_util": (fut_eq / mwpl).where(mwpl > 0),
        "no_fresh": f["Limit for Next Day"].astype(str).str.contains("No Fresh", case=False).astype(float),
    })


def _secban(raw: bytes) -> set[str]:
    lines = raw.decode("latin-1").splitlines()
    return {ln.split(",")[1].strip() for ln in lines[1:] if "," in ln and ln.split(",")[1].strip()}


def _shortselling(raw: bytes) -> pd.DataFrame:
    f = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    if f.empty or "Symbol Name" not in f.columns:
        return pd.DataFrame(columns=["symbol", "short_qty"])
    f = f[f["Symbol Name"].notna()]
    f["symbol"] = f["Symbol Name"].astype(str).str.strip()
    f["short_qty"] = pd.to_numeric(f["Quantity"], errors="coerce")
    return f.groupby("symbol", as_index=False)["short_qty"].sum()


def fetch_day(day: date, session: requests.Session | None = None) -> pd.DataFrame | None:
    s = session or requests.Session()
    ddmmyyyy = day.strftime("%d%m%Y")
    oi_raw = _get(s, f"{ARCHIVE}/archives/nsccl/mwpl/combineoi_{ddmmyyyy}.zip")
    if oi_raw is None:
        return None                                      # no F&O day -> holiday
    frame = _combineoi(oi_raw, day)
    if frame.empty:
        return None
    ban_raw = _get(s, f"{ARCHIVE}/archives/fo/sec_ban/fo_secban_{ddmmyyyy}.csv", min_size=10)
    banned = _secban(ban_raw) if ban_raw else set()
    ss_raw = _get(s, f"{ARCHIVE}/archives/equities/shortSelling/shortselling_{ddmmyyyy}.csv",
                  min_size=10)
    shorts = _shortselling(ss_raw) if ss_raw else pd.DataFrame(columns=["symbol", "short_qty"])
    symbols = sorted({x for x in set(frame["symbol"]) | banned | set(shorts["symbol"])
                      if isinstance(x, str) and x})
    out = pd.DataFrame({"symbol": symbols}).merge(frame, on="symbol", how="left")
    out["in_ban"] = out["symbol"].isin(banned).astype(float)
    out = out.merge(shorts, on="symbol", how="left")
    return out


def _path(day: date) -> Path:
    return CACHE / f"{day:%Y}" / f"extra_{day:%Y%m%d}.parquet"


def _marker(day: date) -> Path:
    return CACHE / f"{day:%Y}" / f"extra_{day:%Y%m%d}.holiday"


def backfill(start: date, end: date, *, workers: int = 3, pause: float = 0.15, log=print) -> tuple[int, int]:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if d.weekday() < 5 and not _path(d).exists() and not _marker(d).exists()]
    counts = {"ok": 0, "holiday": 0}

    def work(chunk):
        s = requests.Session()
        for d in chunk:
            try:
                frame = fetch_day(d, s)
            except Exception as exc:                          # noqa: BLE001
                log(f"  {d}: {type(exc).__name__}: {exc}")
                continue
            _path(d).parent.mkdir(parents=True, exist_ok=True)
            if frame is None or frame.empty:
                if d < date.today():
                    _marker(d).touch()
                    counts["holiday"] += 1
            else:
                frame.to_parquet(_path(d), index=False)
                counts["ok"] += 1
            time.sleep(pause)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, [todo[i::workers] for i in range(workers)]))
    return counts["ok"], counts["holiday"]


def load(start: date, end: date) -> pd.DataFrame:
    parts = []
    for year in range(start.year, end.year + 1):
        for p in sorted((CACHE / f"{year}").glob("extra_*.parquet")):
            d = date(int(p.stem[6:10]), int(p.stem[10:12]), int(p.stem[12:14]))
            if start <= d <= end:
                f = pd.read_parquet(p)
                f.insert(0, "session", d)
                parts.append(f)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["session", "symbol", "mwpl_util", "no_fresh", "in_ban", "short_qty"])
