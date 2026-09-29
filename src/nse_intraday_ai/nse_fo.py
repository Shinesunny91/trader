"""NSE's F&O bhavcopy, reduced to one row per underlying per day.

The equity file says what a stock did; the derivatives file says how traders
are *positioned* in it.  For an overnight-gap trade that is the natural
companion: a gap made with fresh futures longs (price up, OI up) is conviction,
one made by shorts covering (price up, OI down) is fuel that is already spent,
and a stock whose futures trade at an unusual premium to spot is one the
leveraged crowd is leaning on.

Per underlying per session (all published after the close, so usable only
from the next session on):

  fno            1 if the stock had futures listed (point-in-time F&O membership:
                 always intraday-shortable, and arbitrage-linked to its future)
  fut_close      near-month futures close          fut_oi / fut_chg_oi   near month
  fut_oi_all     futures OI across expiries        fut_chg_oi_all
  fut_value      futures traded value (rupees)
  call_oi, put_oi, call_chg_oi, put_chg_oi         options OI across strikes/expiries
  opt_value      options premium traded (rupees)

Index underlyings (NIFTY, BANKNIFTY) are kept too; they describe the market.

Sources:  ..2024-07-05  content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip
          2024-07-08..  content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip (UDiFF)
"""
from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nse_intraday_ai.nse_bhav import ARCHIVE, _get

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "nse_fo"
UDIFF_FROM = date(2024, 7, 8)
COLUMNS = ["symbol", "kind", "fno", "fut_close", "fut_oi", "fut_chg_oi", "fut_oi_all",
           "fut_chg_oi_all", "fut_value", "call_oi", "put_oi", "call_chg_oi", "put_chg_oi",
           "opt_value"]


def _standardise_old(f: pd.DataFrame) -> tuple[pd.DataFrame, date | None]:
    f.columns = [c.strip().upper() for c in f.columns]
    stamp = pd.to_datetime(f["TIMESTAMP"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
    file_date = stamp.dropna().mode().iloc[0].date() if stamp.notna().any() else None
    inst = f["INSTRUMENT"].astype(str).str.strip()
    out = pd.DataFrame({
        "symbol": f["SYMBOL"].astype(str).str.strip(),
        "kind": np.select([inst.isin(["FUTSTK", "FUTIDX"]), inst.isin(["OPTSTK", "OPTIDX"])],
                          ["FUT", "OPT"], "OTHER"),
        "index": inst.isin(["FUTIDX", "OPTIDX"]),
        "expiry": pd.to_datetime(f["EXPIRY_DT"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce"),
        "opt": f["OPTION_TYP"].astype(str).str.strip(),
        "close": pd.to_numeric(f["CLOSE"], errors="coerce"),
        "oi": pd.to_numeric(f["OPEN_INT"], errors="coerce"),
        "chg_oi": pd.to_numeric(f["CHG_IN_OI"], errors="coerce"),
        "value": pd.to_numeric(f["VAL_INLAKH"], errors="coerce") * 1e5,
    })
    return out, file_date


def _standardise_udiff(f: pd.DataFrame) -> tuple[pd.DataFrame, date | None]:
    f.columns = [c.strip() for c in f.columns]
    stamp = pd.to_datetime(f["TradDt"].astype(str).str.strip(), errors="coerce")
    file_date = stamp.dropna().mode().iloc[0].date() if stamp.notna().any() else None
    tp = f["FinInstrmTp"].astype(str).str.strip()
    out = pd.DataFrame({
        "symbol": f["TckrSymb"].astype(str).str.strip(),
        "kind": np.select([tp.isin(["STF", "IDF"]), tp.isin(["STO", "IDO"])], ["FUT", "OPT"], "OTHER"),
        "index": tp.isin(["IDF", "IDO"]),
        "expiry": pd.to_datetime(f["XpryDt"].astype(str).str.strip(), errors="coerce"),
        "opt": f["OptnTp"].astype(str).str.strip(),
        "close": pd.to_numeric(f["ClsPric"], errors="coerce"),
        "oi": pd.to_numeric(f["OpnIntrst"], errors="coerce"),
        "chg_oi": pd.to_numeric(f["ChngInOpnIntrst"], errors="coerce"),
        "value": pd.to_numeric(f["TtlTrfVal"], errors="coerce"),
    })
    return out, file_date


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Contract-level rows -> one row per underlying."""
    fut = rows[rows["kind"] == "FUT"]
    opt = rows[rows["kind"] == "OPT"]
    near = fut.sort_values("expiry").drop_duplicates("symbol", keep="first").set_index("symbol")
    fa = fut.groupby("symbol").agg(fut_oi_all=("oi", "sum"), fut_chg_oi_all=("chg_oi", "sum"),
                                   fut_value=("value", "sum"))
    calls = opt[opt["opt"].isin(["CE"])].groupby("symbol").agg(call_oi=("oi", "sum"),
                                                                  call_chg_oi=("chg_oi", "sum"))
    puts = opt[opt["opt"].isin(["PE"])].groupby("symbol").agg(put_oi=("oi", "sum"),
                                                                put_chg_oi=("chg_oi", "sum"))
    ov = opt.groupby("symbol").agg(opt_value=("value", "sum"))
    out = pd.DataFrame(index=sorted(set(rows["symbol"])))
    out = out.join(near[["close", "oi", "chg_oi"]].rename(
        columns={"close": "fut_close", "oi": "fut_oi", "chg_oi": "fut_chg_oi"}))
    out = out.join(fa).join(calls).join(puts).join(ov)
    idx = rows.groupby("symbol")["index"].any()
    out["kind"] = np.where(idx.reindex(out.index).fillna(False), "INDEX", "STOCK")
    out["fno"] = out["fut_close"].notna().astype(float)
    out = out.reset_index().rename(columns={"index": "symbol"})
    return out[COLUMNS]


def fetch_day(day: date, session: requests.Session | None = None) -> pd.DataFrame | None:
    s = session or requests.Session()
    if day >= UDIFF_FROM:
        url = f"{ARCHIVE}/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
        parse = _standardise_udiff
    else:
        mon = day.strftime("%b").upper()
        url = f"{ARCHIVE}/content/historical/DERIVATIVES/{day:%Y}/{mon}/fo{day:%d}{mon}{day:%Y}bhav.csv.zip"
        parse = _standardise_old
    raw = _get(s, url)
    if raw is None:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        frame = pd.read_csv(z.open(z.namelist()[0]), encoding="latin-1", low_memory=False)
    rows, file_date = parse(frame)
    if file_date != day:            # the archive serves stale copies on some holidays
        return None
    return aggregate(rows)


def _path(day: date, cache: Path) -> Path:
    return cache / f"{day:%Y}" / f"fo_{day:%Y%m%d}.parquet"


def _marker(day: date, cache: Path) -> Path:
    return cache / f"{day:%Y}" / f"fo_{day:%Y%m%d}.holiday"


def backfill(start: date, end: date, *, cache: Path = CACHE, workers: int = 3,
             pause: float = 0.2, log=print) -> tuple[int, int]:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if d.weekday() < 5 and not _path(d, cache).exists()
            and not _marker(d, cache).exists()]
    counts = {"ok": 0, "holiday": 0}

    def work(chunk: list[date]) -> None:
        s = requests.Session()
        for d in chunk:
            try:
                frame = fetch_day(d, s)
            except Exception as exc:                          # noqa: BLE001
                log(f"  {d}: {type(exc).__name__}: {exc}")
                continue
            _path(d, cache).parent.mkdir(parents=True, exist_ok=True)
            if frame is None or frame.empty:
                if d < date.today():
                    _marker(d, cache).touch()
                    counts["holiday"] += 1
            else:
                frame.to_parquet(_path(d, cache), index=False)
                counts["ok"] += 1
            time.sleep(pause)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, [todo[i::workers] for i in range(workers)]))
    return counts["ok"], counts["holiday"]


def load(start: date, end: date, *, cache: Path = CACHE) -> pd.DataFrame:
    parts = []
    for year in range(start.year, end.year + 1):
        for p in sorted((cache / f"{year}").glob("fo_*.parquet")):
            d = date(int(p.stem[3:7]), int(p.stem[7:9]), int(p.stem[9:11]))
            if start <= d <= end:
                f = pd.read_parquet(p)
                f.insert(0, "session", d)
                parts.append(f)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["session", *COLUMNS])
