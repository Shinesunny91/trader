"""Multi-source market-news ingestion and an intraday news risk gate.

Why this exists, and what it is *not*
-------------------------------------
The README used to say news/sentiment was skipped because there is no free,
reliable intraday NSE feed.  That is half right: there is no free feed fast
enough to *predict* a move.  By the time a headline reaches an RSS endpoint,
the move is over — which is the same "signal arrives after the move" failure
the price engine already suffers from.  So this module deliberately does not
try to trade the news.

It uses news for the one thing a slow feed is genuinely good at: **knowing
when the price model's assumptions have broken**.  An ATR-derived stop assumes
the last hour of volatility describes the next hour.  A results announcement,
a block deal, a regulatory action or an index-inclusion headline invalidates
that assumption outright, and those are exactly the situations where a
mean-reversion or breakout signal fires into a one-way tape.  So the gate is:

* **fresh high-impact stock news  →  abstain** (the setup is untradeable, not
  bullish or bearish — we do not know which side the repricing lands on);
* **macro risk-off headlines      →  size down / require more confidence**;
* **direction                     →  recorded, never acted on** until the
  audit trail below proves it out.

Every fetched item is written to `data/news.sqlite3` with its fetch time, so
the direction question becomes answerable from our own archive after a few
weeks of running — the same evidence-gate discipline the meta-model uses.

Sources (all free, no API key):
  * Google News RSS          — per-symbol and macro queries, India edition
  * Yahoo Finance RSS        — per-ticker headlines
  * Moneycontrol / Economic Times / Business Standard — market-wide RSS
  * NSE corporate announcements API — the authoritative disclosure feed
"""
from __future__ import annotations

import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import requests

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[2]
NEWS_DB = ROOT / "data" / "news.sqlite3"

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)
_TIMEOUT = 8

# ── Market-wide RSS sources ───────────────────────────────────────────────────
MACRO_FEEDS: dict[str, str] = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "bs_markets": "https://www.business-standard.com/rss/markets-106.rss",
    "google_macro": (
        "https://news.google.com/rss/search?q="
        + quote_plus("Nifty OR Sensex OR RBI OR \"Indian markets\" when:1d")
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    ),
}

GOOGLE_SYMBOL_RSS = (
    "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
)
YAHOO_SYMBOL_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
NSE_ANNOUNCEMENTS = "https://www.nseindia.com/api/corporate-announcements?index=equities"

# ── Lexicons ──────────────────────────────────────────────────────────────────
# Deliberately small and finance-specific.  A general-purpose sentiment model
# scores "profit falls less than expected" as negative; these lists are tuned
# for Indian market copy and are easy to audit and extend.
_POSITIVE = {
    "beats", "beat", "surges", "surge", "jumps", "jump", "rallies", "rally", "gains",
    "upgrade", "upgraded", "outperform", "buy", "record high", "order win", "bags order",
    "wins order", "approval", "approved", "dividend", "bonus issue", "buyback",
    "stake buy", "expansion", "profit rises", "profit jumps", "revenue growth",
    "margin expansion", "raises guidance", "inclusion", "index inclusion",
}
_NEGATIVE = {
    "misses", "miss", "plunges", "plunge", "slumps", "slump", "falls", "drops", "tanks",
    "downgrade", "downgraded", "underperform", "sell", "cut to", "loss widens",
    "net loss", "profit falls", "profit declines", "resigns", "resignation",
    "probe", "raid", "searches", "investigation", "fraud", "default", "insolvency",
    "nclt", "sebi bars", "ban", "penalty", "fine", "recall", "downgrade outlook",
    "guidance cut", "stake sale", "block deal", "pledge", "exclusion",
}
# Headlines that make the *volatility model* invalid regardless of direction.
_HIGH_IMPACT = {
    "results", "q1 results", "q2 results", "q3 results", "q4 results", "earnings",
    "block deal", "bulk deal", "stake sale", "open offer", "merger", "demerger",
    "acquisition", "acquires", "takeover", "delisting", "rights issue", "qip",
    "sebi", "cbi", "ed ", "income tax", "raid", "probe", "fraud", "insolvency",
    "credit rating", "downgrade", "upgrade", "index inclusion", "index exclusion",
    "circuit", "trading halt", "suspended", "board meeting", "fund raise",
}
_MACRO_RISK_OFF = {
    "crash", "selloff", "sell-off", "plunge", "tumble", "rout", "panic", "war",
    "attack", "tariff", "sanction", "rate hike", "inflation spike", "recession",
    "downgrade india", "fii selling", "foreign outflows", "circuit breaker",
    "emergency meeting", "geopolitical",
}


@dataclass(frozen=True)
class NewsItem:
    source: str
    title: str
    link: str
    published: datetime | None
    symbol: str | None = None        # tagged NSE symbol, if any
    sentiment: float = 0.0           # [-1, +1], lexicon-derived
    impact: float = 0.0              # [0, 1], "does this invalidate the model"
    macro_risk: float = 0.0          # [0, 1], broad-market stress

    @property
    def age_minutes(self) -> float | None:
        if self.published is None:
            return None
        now = datetime.now(tz=IST)
        return max(0.0, (now - self.published).total_seconds() / 60)


def _score(title: str) -> tuple[float, float, float]:
    text = " " + title.lower() + " "
    pos = sum(1 for word in _POSITIVE if word in text)
    neg = sum(1 for word in _NEGATIVE if word in text)
    impact = sum(1 for word in _HIGH_IMPACT if word in text)
    macro = sum(1 for word in _MACRO_RISK_OFF if word in text)
    total = pos + neg
    sentiment = 0.0 if total == 0 else (pos - neg) / total
    return sentiment, min(1.0, impact / 2.0), min(1.0, macro / 2.0)


def _nse_datetime(raw: str) -> datetime:
    """Parse NSE's '12-Aug-2026 09:14:32' announcement stamps."""
    return datetime.strptime(raw, "%d-%b-%Y %H:%M:%S")


def _parse_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for parser in (parsedate_to_datetime, datetime.fromisoformat, _nse_datetime):
        try:
            value = parser(raw)
        except (TypeError, ValueError):
            continue
        if value.tzinfo is None:
            value = value.replace(tzinfo=IST)
        return value.astimezone(IST)
    return None


# ── Symbol tagging ────────────────────────────────────────────────────────────
# Headline text names *companies*, not tickers, and matching on the ticker root
# is badly wrong: "Reliance Chemotex re-appoints..." and "Reliance (NYSE:RS)
# reaches new high" both contain "Reliance" but neither is RELIANCE.NS.  A
# false-positive high-impact tag would veto a perfectly good signal, so the
# match requires the *full* registered company name (minus corporate suffixes).
_SUFFIXES = re.compile(
    r"\b(ltd|limited|ltd\.|corporation|corp|inc|plc|company|co|the)\b\.?", re.IGNORECASE
)
_NON_WORD = re.compile(r"[^a-z0-9 ]+")


def _normalise_name(name: str) -> str:
    cleaned = _SUFFIXES.sub(" ", name.lower())
    cleaned = _NON_WORD.sub(" ", cleaned)
    return " ".join(cleaned.split())


def company_name_map(cache_path: Path | str | None = None) -> dict[str, str]:
    """``normalised company name -> NSE symbol`` for the NIFTY 500 universe."""
    import csv

    path = Path(cache_path or ROOT / "data" / "nifty500_symbols.csv")
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = _normalise_name(row.get("Company Name", ""))
            symbol = (row.get("Symbol") or "").strip().upper()
            # Single-word names ("Infosys") are safe; drop anything shorter
            # than 4 characters, which would match inside unrelated words.
            if symbol and len(name) >= 4:
                mapping[name] = symbol
    return mapping


def tag_symbols(title: str, names: dict[str, str]) -> str | None:
    """Longest full-company-name match in `title`, or None."""
    text = " " + _normalise_name(title) + " "
    best: tuple[int, str] | None = None
    for name, symbol in names.items():
        if f" {name} " in text and (best is None or len(name) > best[0]):
            best = (len(name), symbol)
    return best[1] if best else None


def _parse_rss(text: str, source: str) -> list[NewsItem]:
    items: list[NewsItem] = []
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return items
    # RSS 2.0 <item> and Atom <entry> both appear across these sources.
    nodes = root.iter("item")
    for node in nodes:
        title = (node.findtext("title") or "").strip()
        if not title:
            continue
        link = (node.findtext("link") or "").strip()
        published = _parse_time(node.findtext("pubDate") or node.findtext("published"))
        sentiment, impact, macro = _score(title)
        items.append(
            NewsItem(source, title, link, published, None, sentiment, impact, macro)
        )
    return items


def _fetch(url: str, session: requests.Session) -> str | None:
    try:
        response = session.get(url, timeout=_TIMEOUT)
        if response.status_code != 200:
            return None
        return response.text
    except requests.RequestException:
        return None


def fetch_macro_news(
    *, feeds: dict[str, str] | None = None, workers: int = 6, names: dict[str, str] | None = None
) -> list[NewsItem]:
    """Market-wide headlines from every configured macro feed, concurrently.

    Items naming a NIFTY 500 company are tagged with its symbol, so a single
    market-wide fetch also yields stock-level news for free.
    """
    feeds = feeds or MACRO_FEEDS
    names = company_name_map() if names is None else names
    session = requests.Session()
    session.headers.update({"User-Agent": _UA, "Accept": "application/rss+xml, */*"})
    items: list[NewsItem] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, url, session): name for name, url in feeds.items()}
        for future in as_completed(futures):
            text = future.result()
            if not text:
                continue
            for item in _parse_rss(text, futures[future]):
                symbol = tag_symbols(item.title, names) if names else None
                items.append(item if symbol is None else replace(item, symbol=symbol))
    return items


def fetch_symbol_news(
    symbol: str, *, company_name: str | None = None, names: dict[str, str] | None = None
) -> list[NewsItem]:
    """Headlines for one NSE symbol, from Google News and Yahoo Finance.

    Both endpoints return loose keyword matches, so every item is re-checked
    against the company-name map and anything that resolves to a *different*
    company (or to none at all) is dropped.
    """
    from nse_intraday_ai.data import to_nse_symbol, to_yahoo_symbol

    base = to_nse_symbol(symbol)
    ticker = to_yahoo_symbol(symbol)
    names = company_name_map() if names is None else names
    if company_name is None:
        company_name = next((n for n, s in names.items() if s == base), base)
    query = quote_plus(f'"{company_name}" (NSE OR shares OR stock) when:1d')
    session = requests.Session()
    session.headers.update({"User-Agent": _UA, "Accept": "application/rss+xml, */*"})

    items: list[NewsItem] = []
    for url in (GOOGLE_SYMBOL_RSS.format(query=query), YAHOO_SYMBOL_RSS.format(ticker=ticker)):
        text = _fetch(url, session)
        if not text:
            continue
        for item in _parse_rss(text, "google" if "google" in url else "yahoo"):
            if tag_symbols(item.title, names) != base:
                continue
            items.append(replace(item, symbol=base))
    return items


def fetch_nse_announcements(*, session: requests.Session | None = None) -> list[NewsItem]:
    """Corporate announcements straight from NSE (the authoritative source).

    NSE requires a cookie handshake against the site root before its JSON API
    responds, hence the warm-up GET.
    """
    session = session or requests.Session()
    session.headers.update(
        {"User-Agent": _UA, "Accept": "application/json", "Referer": "https://www.nseindia.com/"}
    )
    try:
        session.get("https://www.nseindia.com/", timeout=_TIMEOUT)
        response = session.get(NSE_ANNOUNCEMENTS, timeout=_TIMEOUT)
        if response.status_code != 200:
            return []
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    items: list[NewsItem] = []
    for row in payload if isinstance(payload, list) else payload.get("data", []):
        if not isinstance(row, dict):
            continue
        # `desc` is a bare category ("Press Release"); `attchmntText` carries
        # the actual disclosure text, which is what the lexicon needs to see.
        category = str(row.get("desc") or row.get("subject") or "").strip()
        detail = _strip_html(str(row.get("attchmntText") or ""))
        subject = f"{category}: {detail}" if detail else category
        symbol = str(row.get("symbol") or "").strip().upper()
        if not subject or not symbol:
            continue
        published = _parse_time(row.get("sort_date") or row.get("an_dt"))
        sentiment, impact, macro = _score(subject)
        items.append(
            NewsItem(
                "nse_announcement", subject[:400], str(row.get("attchmntFile") or ""),
                published, symbol,
                sentiment,
                # An official disclosure is by definition model-invalidating:
                # whatever it says, the last hour's ATR no longer describes the
                # next hour.  Routine filings are capped *below* the veto
                # threshold instead — an investor presentation is not an event.
                min(impact, 0.2) if category.lower() in _ROUTINE_FILINGS else max(impact, 0.6),
                macro,
            )
        )
    return items


# Disclosure categories that fire constantly and move nothing.
_ROUTINE_FILINGS = {
    "analysts/institutional investor meet/con. call updates",
    "investor presentation",
    "trading window",
    "newspaper publication",
    "certificate under reg. 74 (5) of sebi (dp) regulations, 2018",
    "change in company secretary/compliance officer",
    "loss of share certificates",
}

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    from html import unescape

    return " ".join(unescape(_TAG_RE.sub(" ", text)).split())


# ── Persistence: our own archive, so direction becomes testable later ─────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    fetched_at TEXT NOT NULL,
    published  TEXT,
    source     TEXT NOT NULL,
    symbol     TEXT,
    title      TEXT NOT NULL,
    link       TEXT,
    sentiment  REAL NOT NULL,
    impact     REAL NOT NULL,
    macro_risk REAL NOT NULL,
    PRIMARY KEY (source, title, published)
);
CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news (symbol, published DESC);
CREATE INDEX IF NOT EXISTS idx_news_time ON news (published DESC);
"""


def archive(items: list[NewsItem], db_path: Path | str = NEWS_DB) -> int:
    """Persist items; returns the number of genuinely new rows."""
    if not items:
        return 0
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.executescript(_SCHEMA)
        now = datetime.now(tz=IST).isoformat(timespec="seconds")
        before = con.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        con.executemany(
            "INSERT OR IGNORE INTO news"
            " (fetched_at, published, source, symbol, title, link, sentiment, impact, macro_risk)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    now,
                    item.published.isoformat() if item.published else None,
                    item.source, item.symbol, item.title, item.link,
                    item.sentiment, item.impact, item.macro_risk,
                )
                for item in items
            ],
        )
        con.commit()
        return con.execute("SELECT COUNT(*) FROM news").fetchone()[0] - before
    finally:
        con.close()


# ── The gate the scanner actually consults ───────────────────────────────────


@dataclass
class NewsContext:
    """A refreshable snapshot of the news tape, consulted per candidate signal."""

    fetched_at: datetime | None = None
    macro_items: list[NewsItem] = field(default_factory=list)
    symbol_items: dict[str, list[NewsItem]] = field(default_factory=dict)
    # How stale the snapshot may get before a refresh is forced.
    ttl_seconds: int = 300
    # A stock headline younger than this makes the volatility model unreliable.
    fresh_minutes: int = 45
    impact_threshold: float = 0.5
    fetch_seconds: float = 0.0

    @property
    def macro_risk(self) -> float:
        """Broad-market stress in [0, 1] from the last hour of macro headlines."""
        recent = [
            item for item in self.macro_items
            if item.age_minutes is not None and item.age_minutes <= 90
        ]
        if not recent:
            return 0.0
        return min(1.0, sum(item.macro_risk for item in recent) / 3.0)

    @property
    def is_stale(self) -> bool:
        if self.fetched_at is None:
            return True
        return (datetime.now(tz=IST) - self.fetched_at).total_seconds() > self.ttl_seconds

    def fresh_symbol_news(self, symbol: str) -> list[NewsItem]:
        from nse_intraday_ai.data import to_nse_symbol

        base = to_nse_symbol(symbol)
        return [
            item for item in self.symbol_items.get(base, [])
            if item.age_minutes is not None and item.age_minutes <= self.fresh_minutes
        ]

    def entry_gate(self, symbol: str) -> tuple[bool, float, str | None]:
        """Should this signal be taken, and at what size?

        Returns ``(allow, size_multiplier, reason)``.  Direction is
        intentionally ignored — see the module docstring.
        """
        hot = [
            item for item in self.fresh_symbol_news(symbol)
            if item.impact >= self.impact_threshold
        ]
        if hot:
            return False, 0.0, f"fresh high-impact news: {hot[0].title[:90]}"
        risk = self.macro_risk
        if risk >= 0.6:
            return True, 0.5, f"macro risk-off tape (score {risk:.2f}) — half size"
        if risk >= 0.3:
            return True, 0.75, f"elevated macro headlines (score {risk:.2f}) — reduced size"
        return True, 1.0, None


def refresh_news_context(
    symbols: list[str] | None = None,
    *,
    existing: NewsContext | None = None,
    archive_to: Path | str | None = NEWS_DB,
    include_announcements: bool = True,
) -> NewsContext:
    """Fetch macro feeds + NSE announcements (+ optional per-symbol queries).

    Per-symbol Google/Yahoo queries are one HTTP request each, so pass only the
    handful of symbols actually being considered — never the whole universe.
    """
    context = existing or NewsContext()
    if existing is not None and not existing.is_stale and not symbols:
        return existing

    started = time.monotonic()
    items = fetch_macro_news()
    if include_announcements:
        items += fetch_nse_announcements()

    by_symbol: dict[str, list[NewsItem]] = {}
    for item in items:
        if item.symbol:
            by_symbol.setdefault(item.symbol, []).append(item)

    if symbols:
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
            futures = {pool.submit(fetch_symbol_news, s): s for s in symbols}
            for future in as_completed(futures):
                try:
                    found = future.result()
                except Exception:
                    continue
                items.extend(found)
                for item in found:
                    if item.symbol:
                        by_symbol.setdefault(item.symbol, []).append(item)

    if archive_to is not None:
        try:
            archive(items, archive_to)
        except sqlite3.Error:
            pass

    context.fetched_at = datetime.now(tz=IST)
    context.macro_items = [item for item in items if item.symbol is None]
    context.symbol_items = by_symbol
    context.fetch_seconds = time.monotonic() - started
    return context


def recent_archive(
    *, symbol: str | None = None, hours: int = 24, db_path: Path | str = NEWS_DB
) -> list[dict]:
    """Read back the local archive — used by the app and by future studies."""
    path = Path(db_path)
    if not path.exists():
        return []
    since = (datetime.now(tz=IST) - timedelta(hours=hours)).isoformat()
    con = sqlite3.connect(str(path))
    try:
        con.row_factory = sqlite3.Row
        if symbol:
            rows = con.execute(
                "SELECT * FROM news WHERE symbol=? AND published>=? ORDER BY published DESC",
                (symbol, since),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM news WHERE published>=? ORDER BY published DESC LIMIT 200",
                (since,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()
