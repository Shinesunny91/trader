"""NSE price bands (circuit limits) and series per security, from NSE's daily `sec_list.csv`.

Why it matters for a short book: a stock with a 5% band cannot trade above
prev_close × 1.05.  If a short's stop sits at or beyond that circuit, the stop
can never trigger — and a stock locked at its upper circuit cannot be bought
back at all (a short then goes to the exchange's short-delivery auction).
F&O stocks have no fixed band ("No Band"); they are returned as NaN.

The file also gives TODAY's series: small caps are often moved overnight from
EQ to BE (trade-for-trade, delivery only) or worse, and such a name cannot be
shorted intraday even though yesterday's bhavcopy listed it as EQ.

The file is fetched once per day and archived (data/nse_bands/sec_list_YYYYMMDD.csv),
which also builds a point-in-time history for future research.
"""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "nse_bands"
URL = "https://nsearchives.nseindia.com/content/equities/sec_list.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0 Safari/537.36", "Accept": "text/csv,*/*"}
# A stop must sit at least this far (percentage points) inside the circuit.
CIRCUIT_MARGIN_PCT = 0.5
SHORTABLE_SERIES = "EQ"


def parse(raw: bytes | str) -> pd.DataFrame:
    """Frame indexed by symbol with `band` (%, NaN = no band) and `series`.

    A symbol listed in several series keeps its EQ row if it has one.
    """
    text = raw.decode() if isinstance(raw, bytes) else raw
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "symbol": df["Symbol"].str.strip(),
        "series": df["Series"].str.strip(),
        "band": pd.to_numeric(df["Band"].str.strip(), errors="coerce"),
    })
    out = out.sort_values("series", key=lambda s: s.ne(SHORTABLE_SERIES), kind="stable")
    return out.drop_duplicates("symbol").set_index("symbol")


def load(day: date | None = None, *, fetch: bool = True, cache: Path = CACHE, log=print) -> pd.DataFrame:
    """Bands and series in force for session `day` (default today).

    Files are archived by DOWNLOAD date.  For today or a future session the
    file is (re)downloaded once per calendar day; the answer is the newest
    archived file downloaded on or before `day` (point-in-time), or an empty
    frame if there is none.
    """
    today = date.today()
    day = day or today
    path = cache / f"sec_list_{today:%Y%m%d}.csv"
    if fetch and day >= today and not path.exists():
        try:
            import requests
            r = requests.get(URL, headers=_HEADERS, timeout=20)
            if r.status_code == 200 and b"Band" in r.content[:200]:
                cache.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
            else:
                log(f"  price bands: HTTP {r.status_code}")
        except Exception as exc:                                  # noqa: BLE001 — optional input
            log(f"  price bands: download failed ({type(exc).__name__})")
    files = [p for p in sorted(cache.glob("sec_list_*.csv")) if p.stem[-8:] <= f"{day:%Y%m%d}"]
    if not files:
        return pd.DataFrame(columns=["series", "band"])
    if files[-1].stem[-8:] != f"{today:%Y%m%d}":
        log(f"  price bands: using {files[-1].name}")
    return parse(files[-1].read_bytes())


def circuit_risk(symbol: str, stop_pct: float, table: pd.DataFrame,
                 margin: float = CIRCUIT_MARGIN_PCT) -> bool:
    """True if a short's stop (in % above entry) is not safely inside the band."""
    sym = symbol.removesuffix(".NS")
    if sym not in table.index:
        return False
    band = table.at[sym, "band"]
    return bool(pd.notna(band) and stop_pct >= band - margin)


def untradeable_reason(symbol: str, stop_pct: float, table: pd.DataFrame) -> str | None:
    """Why a short in `symbol` should not be placed today, or None if it can be."""
    if table.empty:
        return None                                   # no information: never block
    sym = symbol.removesuffix(".NS")
    if sym not in table.index:
        return "not listed today"
    if table.at[sym, "series"] != SHORTABLE_SERIES:
        return f"{table.at[sym, 'series']} series (not MIS-shortable)"
    if circuit_risk(sym, stop_pct, table):
        return f"stop {stop_pct:.1f}% at/near the {table.at[sym, 'band']:g}% circuit"
    return None
