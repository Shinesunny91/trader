"""NSE corporate events — board meetings, results, announcements — cached, point in time.

Why: a gap with a catalyst (results, an order win, a buyback) tends to keep
running; a gap the exchange itself queried as unexplained ("Spurt in Volume",
"Price movement") tends to reverse (Chan 2003; Savor 2012).  The book shorts
yesterday's biggest gappers, so knowing which of them had news matters.

Sources (www.nseindia.com/api, a browser session with a Referer header):

  corporate-board-meetings      by MEETING date; the schedule is public days in
                                advance — `bm_timestamp` is when it became known
  corporates-financial-results  period=Quarterly, by broadcast date.  Goes
                                almost silent from 2025-03, when results moved
                                to the SEBI "Integrated Filing" ...
  integrated-filing-results     ... which has them from 2025-01 (paginated:
                                `size=` must cover `totalCount`)
  corporate-announcements       every filing with a category (`desc`); also
                                the earliest public time of most results (the
                                PDF "Outcome of Board Meeting" precedes XBRL)

Cache: data/nse_corp/<kind>/YYYY-MM.parquet plus <kind>/_manifest.json, which
records how far each month has been fetched and when.  A month is complete
once it was fetched after it ended; anything else is fetched again.

Symbols: NSE's APIs report the CURRENT symbol for old events (a 2017 Century
Textiles meeting comes back as ABREL).  `symbolchange.csv` maps them back to
the symbol in use on the event date, which is what the bhavcopies carry.

Point in time: every event carries the time it became public (IST, naive).
Feature rows for session t only see events before 08:45 on t (the morning
list); board meetings count from `bm_timestamp`.  Session windows:

    gap1 window of t   after 15:30 on t-2, before 09:15 on t-1
    last k sessions    after 15:30 on t-(k+1), before 08:45 on t

A feature whose window the cache does not cover is NaN, never 0.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "nse_corp"
BASE = "https://www.nseindia.com/api/"
URLS = {
    "board_meetings": BASE + "corporate-board-meetings?index=equities&from_date={a}&to_date={b}",
    "results": BASE + "corporates-financial-results?index=equities&from_date={a}&to_date={b}&period=Quarterly",
    "integrated": BASE + "integrated-filing-results?index=equities&from_date={a}&to_date={b}&size={size}",
    "announcements": BASE + "corporate-announcements?index=equities&from_date={a}&to_date={b}",
}
SYMBOL_CHANGE_URL = "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"
KINDS = ("board_meetings", "results", "integrated", "announcements")
HISTORY_START = date(2016, 1, 1)
INTEGRATED_START = date(2025, 1, 1)        # SEBI Integrated Filing: results move there
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
# Exchange-initiated queries about an unexplained move: documented NO-news gaps.
CLARIFICATION_RX = re.compile(r"price movement|spurt in (?:volume|price)|clarification sought|"
                              r"reply to clarification|movement in price", re.I)
# The exchange asking about a specific news item: there IS a story.
NEWS_QUERY_RX = re.compile(r"news verification|news clarification", re.I)
_RESULT_DESC_RX = re.compile(r"financial results?\b|integrated filing- ?financ", re.I)
_RESULT_TEXT_RX = re.compile(r"financial results?\b", re.I)
_NOT_RESULT_RX = re.compile(r"delay|non-submission|non submission|trading window", re.I)
_BM_CANCEL_RX = re.compile(r"adjourn|cancel|postpone|defer", re.I)
RESULT_CLUSTER_DAYS = 3        # filings within 3 days of a results event are the same news
SESSION_OPEN = pd.Timedelta(hours=9, minutes=15)
SESSION_CLOSE = pd.Timedelta(hours=15, minutes=30)
LIST_CUTOFF = pd.Timedelta(hours=8, minutes=45)
DAYS_TO_BM_CAP = 30
DAYS_SINCE_RESULTS_CAP = 120


# ── parsing (pure; tested on recorded JSON) ─────────────────────────────────

def _ts(values: Iterable, fmt: str = "%d-%b-%Y %H:%M:%S") -> pd.Series:
    s = pd.Series(list(values), dtype=object).replace({"-": None, "": None})
    out = pd.to_datetime(s, format=fmt, errors="coerce")
    miss = out.isna() & s.notna()
    if miss.any():                                 # e.g. "28-Feb-2017 19:08" without seconds
        out[miss] = pd.to_datetime(s[miss], format="%d-%b-%Y %H:%M", errors="coerce")
    return out


def _records(payload) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("data", [])
    return [r for r in (payload or []) if isinstance(r, dict)]


def is_results_meeting(purpose: str, desc: str) -> bool:
    """A board meeting that will consider financial results."""
    purpose, desc = str(purpose or ""), str(desc or "")
    if _BM_CANCEL_RX.search(purpose):
        return False
    return bool(re.search(r"result", purpose, re.I) or _RESULT_TEXT_RX.search(desc))


def parse_board_meetings(payload) -> pd.DataFrame:
    rec = _records(payload)
    cols = ["symbol", "bm_date", "known_at", "purpose", "desc", "is_results"]
    if not rec:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({
        "symbol": [str(r.get("bm_symbol") or "").strip().upper() for r in rec],
        "bm_date": pd.to_datetime(pd.Series([r.get("bm_date") for r in rec]), format="%d-%b-%Y", errors="coerce"),
        "known_at": _ts(r.get("bm_timestamp") for r in rec),
        "purpose": [str(r.get("bm_purpose") or "") for r in rec],
        "desc": [str(r.get("bm_desc") or "")[:400] for r in rec],
    })
    df["is_results"] = [is_results_meeting(p, d) for p, d in zip(df["purpose"], df["desc"])]
    df = df[(df["symbol"] != "") & df["bm_date"].notna()]
    return df.drop_duplicates(["symbol", "bm_date", "purpose", "known_at"]).reset_index(drop=True)[cols]


def parse_results(payload) -> pd.DataFrame:
    """corporates-financial-results rows -> symbol, ts (public time), relating_to."""
    rec = _records(payload)
    cols = ["symbol", "ts", "relating_to", "consolidated"]
    if not rec:
        return pd.DataFrame(columns=cols)
    bc = _ts(r.get("broadCastDate") for r in rec)
    ex = _ts(r.get("exchdisstime") for r in rec)
    df = pd.DataFrame({
        "symbol": [str(r.get("symbol") or "").strip().upper() for r in rec],
        "ts": bc.fillna(ex),
        "relating_to": [str(r.get("relatingTo") or "") for r in rec],
        "consolidated": [str(r.get("consolidated") or "") for r in rec],
    })
    df = df[(df["symbol"] != "") & df["ts"].notna()]
    return df.drop_duplicates().reset_index(drop=True)[cols]


def parse_integrated(payload) -> pd.DataFrame:
    """integrated-filing-results rows (financials only) -> symbol, ts, relating_to."""
    rec = [r for r in _records(payload) if "financ" in str(r.get("type") or "Financ").lower()
           and "revis" not in str(r.get("type_Sub") or "").lower()]
    cols = ["symbol", "ts", "relating_to", "consolidated"]
    if not rec:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({
        "symbol": [str(r.get("symbol") or "").strip().upper() for r in rec],
        "ts": _ts(r.get("broadcast_Date") or r.get("creation_Date") for r in rec),
        "relating_to": [str(r.get("qe_Date") or "") for r in rec],
        "consolidated": [str(r.get("consolidated") or "") for r in rec],
    })
    df = df[(df["symbol"] != "") & df["ts"].notna()]
    return df.drop_duplicates().reset_index(drop=True)[cols]


def parse_announcements(payload) -> pd.DataFrame:
    rec = _records(payload)
    cols = ["symbol", "ts", "desc", "text", "industry", "seq_id"]
    if not rec:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({
        "symbol": [str(r.get("symbol") or "").strip().upper() for r in rec],
        "ts": _ts(r.get("an_dt") for r in rec),
        "desc": [str(r.get("desc") or "") for r in rec],
        "text": [str(r.get("attchmntText") or "")[:300] for r in rec],
        "industry": [str(r.get("smIndustry") or "") for r in rec],
        "seq_id": [str(r.get("seq_id") or "") for r in rec],
    })
    df = df[(df["symbol"] != "") & df["ts"].notna()]
    return df.drop_duplicates(["seq_id", "symbol", "ts"]).reset_index(drop=True)[cols]


PARSERS: dict[str, Callable] = {"board_meetings": parse_board_meetings, "results": parse_results,
                                "integrated": parse_integrated, "announcements": parse_announcements}


def parse_symbol_changes(raw: bytes | str) -> pd.DataFrame:
    """symbolchange.csv (company, old, new, DD-MON-YYYY; no header) -> old, new, date."""
    text = raw.decode("latin-1") if isinstance(raw, bytes) else raw
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.rsplit(",", 3)]
        if len(parts) != 4:
            continue
        try:
            d = datetime.strptime(parts[3].title(), "%d-%b-%Y")
        except ValueError:
            continue
        if parts[1] and parts[2] and parts[1] != parts[2]:
            rows.append((parts[1].upper(), parts[2].upper(), pd.Timestamp(d)))
    return pd.DataFrame(rows, columns=["old", "new", "date"]).sort_values("date").reset_index(drop=True)


def symbol_as_of(symbols: pd.Series, when: pd.Series, changes: pd.DataFrame | None) -> pd.Series:
    """Map each (current symbol, event time) to the symbol in use at that time."""
    out = symbols.astype(str).copy()
    if changes is None or changes.empty or out.empty:
        return out
    by_new: dict[str, list] = {}
    for old, new, d in changes.itertuples(index=False):
        by_new.setdefault(new, []).append((np.datetime64(d, "ns"), old))
    when_d = pd.to_datetime(when).dt.normalize()
    vals, ws = out.to_numpy(dtype=object), when_d.to_numpy(dtype="datetime64[ns]")
    for i in np.flatnonzero(out.isin(set(by_new)).to_numpy()):
        sym, w = vals[i], ws[i]
        for _ in range(8):                         # follow chains of renames back in time
            hits = [(d, old) for d, old in by_new.get(sym, ()) if d > w]
            if not hits:
                break
            sym = min(hits)[1]                     # the first rename after the event
        vals[i] = sym
    return pd.Series(vals, index=out.index)


# ── fetching ────────────────────────────────────────────────────────────────

class Client:
    """A polite NSE session: cookie warm-up, ~1 request/second, retry with backoff."""

    def __init__(self, min_interval: float = 1.0, log=print):
        import requests

        self.s = requests.Session()
        self.s.headers.update(_HEADERS)
        self.min_interval, self.log, self._last, self._warm_at = min_interval, log, 0.0, 0.0

    def _wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()

    def warm(self) -> None:
        try:
            self._wait()
            self.s.get("https://www.nseindia.com/", timeout=15)   # 403 is common; cookies still help
        except Exception:                                         # noqa: BLE001
            pass
        self._warm_at = time.monotonic()

    def get(self, url: str, retries: int = 6, timeout: float = 60) -> bytes:
        if time.monotonic() - self._warm_at > 600:
            self.warm()
        err = None
        for k in range(retries):
            self._wait()
            try:
                r = self.s.get(url, timeout=timeout)
                if r.status_code == 200 and r.content:
                    return r.content
                err = f"HTTP {r.status_code}"
                if r.status_code in (401, 403):
                    self.warm()
            except Exception as exc:                              # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(60.0, 2.0 * 2 ** k))
        raise RuntimeError(f"{url}: {err}")

    def get_json(self, url: str, **kw):
        raw = self.get(url, **kw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{url}: not JSON ({raw[:80]!r})") from exc


def _fmt(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def fetch_window(kind: str, a: date, b: date, client: Client) -> pd.DataFrame:
    """One API call (or a few, if the window must be split) for [a, b]."""
    if kind == "integrated":
        size = 5000
        while True:
            payload = client.get_json(URLS[kind].format(a=_fmt(a), b=_fmt(b), size=size))
            total = int(payload.get("totalCount") or 0) if isinstance(payload, dict) else 0
            if len(_records(payload)) >= total or size >= 40000:
                break
            size = max(size * 2, total + 100)
        return parse_integrated(payload)
    try:
        payload = client.get_json(URLS[kind].format(a=_fmt(a), b=_fmt(b)))
    except RuntimeError:
        if b <= a:
            raise
        mid = a + (b - a) // 2                     # time-outs on heavy windows: split
        return pd.concat([fetch_window(kind, a, mid, client),
                          fetch_window(kind, mid + timedelta(days=1), b, client)], ignore_index=True)
    return PARSERS[kind](payload)


def board_meetings(a: date, b: date, client: Client | None = None) -> pd.DataFrame:
    return fetch_window("board_meetings", a, b, client or Client())


def results(a: date, b: date, client: Client | None = None) -> pd.DataFrame:
    client = client or Client()
    parts = [fetch_window("results", a, b, client)]
    if b >= INTEGRATED_START:
        parts.append(fetch_window("integrated", max(a, INTEGRATED_START), b, client))
    return pd.concat(parts, ignore_index=True)


def announcements(a: date, b: date, client: Client | None = None) -> pd.DataFrame:
    return fetch_window("announcements", a, b, client or Client())


# ── cache ───────────────────────────────────────────────────────────────────

def _month_start(d: date) -> date:
    return d.replace(day=1)


def _month_end(d: date) -> date:
    nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return nxt - timedelta(days=1)


def months(a: date, b: date) -> list[date]:
    out, m = [], _month_start(a)
    while m <= b:
        out.append(m)
        m = _month_end(m) + timedelta(days=1)
    return out


def _dir(kind: str, cache: Path) -> Path:
    return cache / kind


def _manifest_path(kind: str, cache: Path) -> Path:
    return _dir(kind, cache) / "_manifest.json"


def read_manifest(kind: str, cache: Path = CACHE) -> dict:
    from nse_intraday_ai.atomic_io import atomic_read_json

    return atomic_read_json(_manifest_path(kind, cache), default={}) or {}


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f".{path.stem}_")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _store(kind: str, month: date, df: pd.DataFrame, through: date, cache: Path, *, append: bool) -> int:
    from nse_intraday_ai.atomic_io import atomic_write_json

    path = _dir(kind, cache) / f"{month:%Y-%m}.parquet"
    if append and path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    key = ["seq_id", "symbol", "ts"] if kind == "announcements" else list(df.columns)
    df = df.drop_duplicates(key).reset_index(drop=True)
    _write_parquet(df, path)
    man = read_manifest(kind, cache)
    man[f"{month:%Y-%m}"] = {"through": through.isoformat(), "rows": int(len(df)),
                             "fetched_at": datetime.now().isoformat(timespec="seconds")}
    atomic_write_json(_manifest_path(kind, cache), dict(sorted(man.items())))
    return len(df)


def month_complete(kind: str, month: date, cache: Path = CACHE) -> bool:
    ent = read_manifest(kind, cache).get(f"{month:%Y-%m}")
    if not ent:
        return False
    end = _month_end(month)
    return (ent["through"] >= end.isoformat()
            and datetime.fromisoformat(ent["fetched_at"]).date() > end)


def _chunks(a: date, b: date, days: int) -> list[tuple[date, date]]:
    out = []
    while a <= b:
        e = min(b, a + timedelta(days=days - 1))
        out.append((a, e))
        a = e + timedelta(days=1)
    return out


def fetch_month(kind: str, month: date, client: Client, *, cache: Path = CACHE,
                today: date | None = None, log=print) -> int:
    """Fetch (or finish) one month of `kind` into the cache; returns rows stored."""
    today = today or date.today()
    end = _month_end(month)
    if kind == "announcements":
        # Resumable inside the month: weekly calls, the parquet rewritten after each.
        ent = read_manifest(kind, cache).get(f"{month:%Y-%m}")
        fetched = datetime.fromisoformat(ent["fetched_at"]).date() if ent else None
        start = month
        if ent and fetched is not None:
            # Days before the fetch day were complete when fetched; restart there.
            done = min(date.fromisoformat(ent["through"]), fetched - timedelta(days=1))
            start = max(month, done + timedelta(days=1))
        stop = min(end, today)
        n = 0
        for a, b in _chunks(start, stop, 7):
            n = _store(kind, month, fetch_window(kind, a, b, client), b, cache, append=True)
        return n
    stop = end if kind == "board_meetings" else min(end, today)   # meetings: future schedule too
    if kind == "integrated" and end < INTEGRATED_START:
        return _store(kind, month, parse_integrated([]), stop, cache, append=False)
    return _store(kind, month, fetch_window(kind, month, stop, client), stop, cache, append=False)


def backfill(start: date, end: date, kinds: Iterable[str] = KINDS, *, cache: Path = CACHE,
             client: Client | None = None, log=print) -> dict:
    """Fetch every month in [start, end] not yet complete; resumable and idempotent."""
    client = client or Client(log=log)
    stats = {}
    for kind in kinds:
        last = end + timedelta(days=45) if kind == "board_meetings" else end   # the schedule ahead
        todo = [m for m in months(start, last)
                if not (kind == "integrated" and _month_end(m) < INTEGRATED_START)
                and not month_complete(kind, m, cache)]
        if kind == "announcements":
            todo = todo[::-1]          # newest first: 2025+ results timing depends on them
        t0, done = time.monotonic(), 0
        log(f"[{kind}] {len(todo)} months to fetch")
        for m in todo:
            try:
                n = fetch_month(kind, m, client, cache=cache, log=log)
            except Exception as exc:                              # noqa: BLE001
                log(f"[{kind}] {m:%Y-%m} FAILED: {exc}")
                continue
            done += 1
            rate = (time.monotonic() - t0) / done
            log(f"[{kind}] {m:%Y-%m}: {n} rows ({done}/{len(todo)}, "
                f"ETA {rate * (len(todo) - done) / 60:.0f} min)")
        stats[kind] = done
    return stats


def refresh_symbol_changes(cache: Path = CACHE, max_age_days: int = 7, client: Client | None = None) -> None:
    path = cache / "symbolchange.csv"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return
    raw = (client or Client()).get(SYMBOL_CHANGE_URL)
    if len(parse_symbol_changes(raw)) > 100:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, path)


def refresh(start: date, end: date, *, cache: Path = CACHE, log=print, bm_ahead_days: int = 45) -> dict:
    """Incremental daily update: recent results/announcements, and the meeting schedule ahead."""
    client = Client(log=log)
    try:
        refresh_symbol_changes(cache, client=client)
    except Exception as exc:                                      # noqa: BLE001
        log(f"  symbol changes refresh failed: {exc}")
    stats = {}
    for kind, a, b in (("board_meetings", start, end + timedelta(days=bm_ahead_days)),
                       ("results", start, end), ("integrated", max(start, INTEGRATED_START), end),
                       ("announcements", start, end)):
        n = 0
        for m in months(a, b):
            if month_complete(kind, m, cache):
                continue
            try:
                n += fetch_month(kind, m, client, cache=cache, today=max(end, date.today()), log=log)
            except Exception as exc:                              # noqa: BLE001
                log(f"  corp {kind} {m:%Y-%m} failed: {exc}")
        stats[kind] = n
    return stats


# ── loading ─────────────────────────────────────────────────────────────────

@dataclass
class CorpEvents:
    """Tidy point-in-time events; timestamps are naive IST."""
    board_meetings: pd.DataFrame      # symbol, bm_date, known_at, purpose, desc, is_results
    results: pd.DataFrame             # symbol, ts, source   (one row per results event)
    announcements: pd.DataFrame       # symbol, ts, desc, text
    coverage: dict[str, np.ndarray] = field(default_factory=dict)   # kind -> sorted datetime64[D]

    def covered(self, kind: str, days) -> np.ndarray:
        cov = self.coverage.get(kind)
        d = np.asarray(pd.to_datetime(pd.Series(days)).values.astype("datetime64[D]"))
        if cov is None or len(cov) == 0:
            return np.zeros(len(d), dtype=bool)
        pos = np.clip(np.searchsorted(cov, d), 0, len(cov) - 1)
        return cov[pos] == d


def _coverage(kind: str, cache: Path, a: date, b: date) -> np.ndarray:
    days = []
    for key, ent in read_manifest(kind, cache).items():
        m = datetime.strptime(key, "%Y-%m").date()
        if _month_end(m) < a or m > b:
            continue
        through = date.fromisoformat(ent["through"])
        if kind != "board_meetings":   # a day is complete only if fetched after it ended
            through = min(through, datetime.fromisoformat(ent["fetched_at"]).date() - timedelta(days=1))
        if through >= m:
            days.append(np.arange(np.datetime64(m), np.datetime64(through) + np.timedelta64(1, "D"), dtype="datetime64[D]"))
    return np.unique(np.concatenate(days)) if days else np.array([], dtype="datetime64[D]")


def _read(kind: str, cache: Path, a: date, b: date) -> pd.DataFrame:
    frames = []
    for m in months(a, b):
        p = _dir(kind, cache) / f"{m:%Y-%m}.parquet"
        if p.exists():
            try:
                frames.append(pd.read_parquet(p))
            except Exception:                                     # noqa: BLE001
                continue
    frames = [f for f in frames if not f.empty]
    if not frames:
        return PARSERS[kind]([])
    return pd.concat(frames, ignore_index=True)


def _results_from_announcements(ann: pd.DataFrame, bm: pd.DataFrame) -> pd.DataFrame:
    """Results visible in announcements: results filings, and an 'Outcome of Board
    Meeting' on the day a results meeting was scheduled."""
    if ann.empty:
        return pd.DataFrame(columns=["symbol", "ts"])
    hit = ann["desc"].str.contains(_RESULT_DESC_RX) & ~ann["desc"].str.contains(_NOT_RESULT_RX)
    outcome = ann["desc"].str.contains("outcome of board meeting", case=False)
    hit |= outcome & ann["text"].str.contains(_RESULT_TEXT_RX)
    if not bm.empty:
        sched = set(zip(bm.loc[bm["is_results"], "symbol"], bm.loc[bm["is_results"], "bm_date"].dt.normalize()))
        on_day = pd.Series([(s, d) in sched for s, d in zip(ann["symbol"], ann["ts"].dt.normalize())],
                           index=ann.index)
        hit |= outcome & on_day
    return ann.loc[hit, ["symbol", "ts"]]


def cluster_results(df: pd.DataFrame, days: int = RESULT_CLUSTER_DAYS) -> pd.DataFrame:
    """Keep the first public time of each results event (later filings of the
    same results — consolidated, XBRL, the integrated filing — are not news)."""
    if df.empty:
        return df.reset_index(drop=True)
    df = df.dropna(subset=["ts"]).sort_values(["symbol", "ts"])
    gap = df.groupby("symbol")["ts"].diff()
    return df[gap.isna() | (gap > pd.Timedelta(days=days))].reset_index(drop=True)


_memo: dict = {}


def load(start: date, end: date, *, cache: Path = CACHE, map_symbols: bool = True) -> CorpEvents:
    """Events relevant to sessions in [start, end] (with look-back/ahead margins)."""
    a, b = start - timedelta(days=DAYS_SINCE_RESULTS_CAP + 15), end + timedelta(days=DAYS_TO_BM_CAP + 15)
    stamp = tuple((k, _manifest_path(k, cache).stat().st_mtime if _manifest_path(k, cache).exists() else 0)
                  for k in KINDS)
    key = (str(cache), a, b, map_symbols, stamp)
    if key in _memo:
        return _memo[key]
    bm = _read("board_meetings", cache, a, b)
    res = pd.concat([_read("results", cache, a, b).assign(source="results"),
                     _read("integrated", cache, a, b).assign(source="integrated")], ignore_index=True)
    ann = _read("announcements", cache, a, b)
    changes = None
    sc = cache / "symbolchange.csv"
    if map_symbols and sc.exists():
        try:
            changes = parse_symbol_changes(sc.read_bytes())
        except Exception:                                         # noqa: BLE001
            changes = None
    if changes is not None:
        if not bm.empty:
            bm["symbol"] = symbol_as_of(bm["symbol"], bm["bm_date"], changes).to_numpy()
        if not res.empty:
            res["symbol"] = symbol_as_of(res["symbol"], res["ts"], changes).to_numpy()
        if not ann.empty:
            ann["symbol"] = symbol_as_of(ann["symbol"], ann["ts"], changes).to_numpy()
    from_ann = _results_from_announcements(ann, bm).assign(source="announcements")
    allres = pd.concat([res[["symbol", "ts", "source"]], from_ann], ignore_index=True)
    allres["ts"] = pd.to_datetime(allres["ts"])
    cov = {k: _coverage(k, cache, a, b) for k in KINDS}
    # Results are complete where the legacy endpoint is cached and, from 2025
    # (when they migrate to the Integrated Filing, whose XBRL often lags the
    # PDF announcement by hours), the integrated and announcement caches too.
    legacy = cov["results"]
    modern = np.intersect1d(cov["integrated"], cov["announcements"])
    cov["results_any"] = legacy[(legacy < np.datetime64(INTEGRATED_START)) | np.isin(legacy, modern)]
    out = CorpEvents(board_meetings=bm.reset_index(drop=True), results=cluster_results(allres),
                     announcements=ann[["symbol", "ts", "desc"]].reset_index(drop=True), coverage=cov)
    _memo.clear()
    _memo[key] = out
    return out


def empty() -> CorpEvents:
    return CorpEvents(PARSERS["board_meetings"]([]), pd.DataFrame(columns=["symbol", "ts", "source"]),
                      PARSERS["announcements"]([]), {})


# ── point-in-time features ──────────────────────────────────────────────────

FEATURES = ("res_in_gap1", "res_night0", "res_recent3", "bm_results_today", "days_to_bm",
            "days_since_results", "ann_n_gap1", "ann_n_5d", "clar_gap1", "newsq_2d")


def _shift(arr: np.ndarray, k: int) -> np.ndarray:
    """arr[i - k] with NaT before the start."""
    out = np.full(arr.shape, np.datetime64("NaT", "ns")).astype(arr.dtype)
    if k < len(arr):
        out[k:] = arr[:len(arr) - k]
    return out


def _window_counts(times: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Events with lo < t < hi (strict); NaN where a bound is NaT."""
    n = np.searchsorted(times, hi, side="left") - np.searchsorted(times, lo, side="right")
    out = np.maximum(n, 0).astype("float64")
    out[np.isnat(lo) | np.isnat(hi)] = np.nan
    return out


def _counts_by_symbol(ev: pd.DataFrame, symbols: pd.Index, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    out = np.zeros((len(lo), len(symbols)))
    bad = np.isnat(lo) | np.isnat(hi)
    out[bad, :] = np.nan
    if ev.empty:
        return out
    col = {s: j for j, s in enumerate(symbols)}
    for sym, g in ev.groupby("symbol"):
        j = col.get(sym)
        if j is None:
            continue
        t = np.sort(g["ts"].to_numpy(dtype="datetime64[ns]"))
        out[:, j] = _window_counts(t, lo, hi)
    return out


def corp_features(ev: CorpEvents | None, sessions: Iterable, symbols: Iterable,
                  cutoff: pd.Timedelta = LIST_CUTOFF) -> dict[str, pd.DataFrame]:
    """Session x symbol frames for FEATURES; NaN wherever the cache cannot vouch."""
    sessions, symbols = pd.Index(sessions), pd.Index(symbols)
    nan = lambda: pd.DataFrame(np.nan, index=sessions, columns=symbols)  # noqa: E731
    if ev is None or len(sessions) == 0:
        return {k: nan() for k in FEATURES}
    day = pd.to_datetime(pd.Series(list(sessions))).to_numpy(dtype="datetime64[ns]")
    O, C, K = day + SESSION_OPEN.to_timedelta64(), day + SESSION_CLOSE.to_timedelta64(), \
        day + cutoff.to_timedelta64()
    O1, C2 = _shift(O, 1), _shift(C, 2)
    win = {"gap1": (C2, O1), "night0": (_shift(C, 1), K), "3d": (_shift(C, 4), K),
           "5d": (_shift(C, 6), K), "2d": (_shift(C, 3), K)}

    def valid(kind: str, lo: np.ndarray) -> np.ndarray:
        """The cache covers every day from the window start through t-1."""
        ok = ~np.isnat(lo)
        lo_d = np.where(ok, lo, day).astype("datetime64[D]")
        prev = _shift(day, 1).astype("datetime64[D]")
        prev = np.where(np.isnat(prev), lo_d, prev)
        return ok & ev.covered(kind, lo_d) & ev.covered(kind, prev)

    frame = lambda arr, ok: pd.DataFrame(np.where(ok[:, None], arr, np.nan),  # noqa: E731
                                         index=sessions, columns=symbols)
    keep = set(symbols)
    res = ev.results[ev.results["symbol"].isin(keep)] if not ev.results.empty else ev.results
    ann = ev.announcements[ev.announcements["symbol"].isin(keep)] if not ev.announcements.empty \
        else ev.announcements
    out: dict[str, pd.DataFrame] = {}
    for name, w in (("res_in_gap1", "gap1"), ("res_night0", "night0"), ("res_recent3", "3d")):
        lo, hi = win[w]
        out[name] = frame(np.minimum(_counts_by_symbol(res, symbols, lo, hi), 1), valid("results_any", lo))
    for name, w in (("ann_n_gap1", "gap1"), ("ann_n_5d", "5d")):
        lo, hi = win[w]
        out[name] = frame(_counts_by_symbol(ann, symbols, lo, hi), valid("announcements", lo))
    lo, hi = win["2d"]
    if not ann.empty:
        clar = ann[ann["desc"].str.contains(CLARIFICATION_RX)]
        newsq = ann[ann["desc"].str.contains(NEWS_QUERY_RX)]
    else:
        clar = newsq = ann
    ok2 = valid("announcements", lo)
    out["clar_gap1"] = frame(np.minimum(_counts_by_symbol(clar, symbols, lo, hi), 1), ok2)
    out["newsq_2d"] = frame(np.minimum(_counts_by_symbol(newsq, symbols, lo, hi), 1), ok2)

    # Days since the last results event public before the cutoff.
    since = np.full((len(day), len(symbols)), np.nan)
    col = {s: j for j, s in enumerate(symbols)}
    day_d = day.astype("datetime64[D]")
    if not res.empty:
        for sym, g in res.groupby("symbol"):
            j = col.get(sym)
            if j is None:
                continue
            t = np.sort(g["ts"].to_numpy(dtype="datetime64[ns]"))
            idx = np.searchsorted(t, K, side="left") - 1
            has = idx >= 0
            last = t[np.clip(idx, 0, None)].astype("datetime64[D]")
            since[has, j] = (day_d[has] - last[has]).astype(int)
    since = np.minimum(since, DAYS_SINCE_RESULTS_CAP)
    # "None in 120 days" is only a fact if the cache covers those 120 days.
    lookback_ok = valid("results_any", (day_d - np.timedelta64(DAYS_SINCE_RESULTS_CAP, "D")).astype("datetime64[ns]"))
    since = np.where(np.isnan(since) & lookback_ok[:, None], DAYS_SINCE_RESULTS_CAP, since)
    out["days_since_results"] = frame(since, valid("results_any", _shift(day, 1)))

    # The board-meeting schedule: known from bm_timestamp (or, when missing,
    # two days before the meeting — SEBI's minimum notice for results).
    today = np.zeros((len(day), len(symbols)))
    to_bm = np.full((len(day), len(symbols)), float(DAYS_TO_BM_CAP))
    bm = ev.board_meetings
    if not bm.empty:
        bm = bm[bm["is_results"]]
        for sym, g in bm.groupby("symbol"):
            j = col.get(sym)
            if j is None:
                continue
            md = g["bm_date"].to_numpy(dtype="datetime64[ns]").astype("datetime64[D]")
            known = g["known_at"].to_numpy(dtype="datetime64[ns]")
            known = np.where(np.isnat(known), (md - np.timedelta64(2, "D")).astype("datetime64[ns]"), known)
            vis = known[None, :] < K[:, None]                                  # sessions x meetings
            dd = (md[None, :] - day_d[:, None]).astype(int).astype(float)
            dd = np.where(vis & (dd >= 0), dd, np.inf)
            nearest = dd.min(axis=1)
            today[:, j] = (nearest == 0).astype(float)
            to_bm[:, j] = np.minimum(nearest, DAYS_TO_BM_CAP)
    bm_ok = ev.covered("board_meetings", day_d)
    out["bm_results_today"] = frame(today, bm_ok)
    out["days_to_bm"] = frame(to_bm, bm_ok)
    return {k: out[k].astype("float32") for k in FEATURES}


def results_day_names(symbols: Iterable[str], session: date, *, cache: Path = CACHE) -> dict[str, list[str]]:
    """For logging: which of `symbols` have a results meeting today / results news
    in the gap1 window / results overnight.  Cache only (no network), so it is
    safe inside replays; weekday arithmetic stands in for the NSE calendar.
    """
    if not any(_manifest_path(k, cache).exists() for k in KINDS):
        return {}
    syms = [str(s).removesuffix(".NS") for s in symbols]
    try:
        # Year-quantised window: one load serves every session of a replay year.
        ev = load(date(session.year, 1, 1), date(session.year, 12, 31), cache=cache)
        days = pd.bdate_range(session - timedelta(days=10), session).date
        feats = corp_features(ev, days, syms)
    except Exception:                                             # noqa: BLE001
        return {}
    row = {k: feats[k].iloc[-1] for k in ("bm_results_today", "res_in_gap1", "res_night0")}
    return {k: sorted(v.index[v.fillna(0) > 0]) for k, v in row.items()}
