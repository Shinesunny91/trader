"""Offline tests for the news layer — parsing, tagging, scoring and the gate.

Nothing here touches the network; the fetchers are exercised by passing them
canned payloads, because a test that depends on today's headlines is a test
that fails for reasons unrelated to the code.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai.news_feed import (
    NewsContext,
    NewsItem,
    _parse_rss,
    _parse_time,
    _score,
    archive,
    recent_archive,
    tag_symbols,
)

IST = ZoneInfo("Asia/Kolkata")

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Reliance Industries Q1 results beat estimates, profit rises</title>
    <link>https://example.com/1</link>
    <pubDate>Tue, 11 Aug 2026 10:30:00 +0530</pubDate>
  </item>
  <item>
    <title>Sensex plunges 900 points in a broad selloff as FII selling accelerates</title>
    <link>https://example.com/2</link>
    <pubDate>Tue, 11 Aug 2026 11:00:00 +0530</pubDate>
  </item>
</channel></rss>"""

NAMES = {
    "reliance industries": "RELIANCE",
    "infosys": "INFY",
    "tata consultancy services": "TCS",
    "state bank of india": "SBIN",
}


# ── parsing ───────────────────────────────────────────────────────────────────

def test_rss_parsing_extracts_titles_and_times():
    items = _parse_rss(RSS, "test")
    assert len(items) == 2
    assert items[0].published.tzinfo is not None
    assert items[0].published.hour == 10


def test_malformed_feed_yields_nothing_rather_than_raising():
    assert _parse_rss("not xml at all", "test") == []
    assert _parse_rss("", "test") == []


def test_nse_announcement_timestamps_parse():
    """NSE stamps look like '12-Aug-2026 09:14:32' — neither RFC822 nor ISO."""
    parsed = _parse_time("12-Aug-2026 09:14:32")
    assert parsed is not None and parsed.month == 8 and parsed.hour == 9


def test_unparseable_time_is_none():
    assert _parse_time("sometime last Tuesday") is None
    assert _parse_time(None) is None


# ── symbol tagging ────────────────────────────────────────────────────────────

def test_full_company_name_is_required_not_the_ticker_root():
    """The failure this guards: 'Reliance Chemotex' is not RELIANCE."""
    assert tag_symbols("Reliance Industries Q1 profit rises", NAMES) == "RELIANCE"
    assert tag_symbols("Reliance Chemotex re-appoints MD", NAMES) is None
    assert tag_symbols("Reliance (NYSE:RS) reaches 12-month high", NAMES) is None


def test_tagging_survives_punctuation_and_suffixes():
    assert tag_symbols("Infosys Ltd. wins a large deal", NAMES) == "INFY"
    assert tag_symbols("State Bank of India to raise dollar bonds", NAMES) == "SBIN"


def test_longest_match_wins():
    names = {**NAMES, "tata consultancy": "WRONG"}
    assert tag_symbols("Tata Consultancy Services posts results", names) == "TCS"


def test_untagged_headline_returns_none():
    assert tag_symbols("Gold prices climb on Fed bets", NAMES) is None


# ── scoring ───────────────────────────────────────────────────────────────────

def test_sentiment_direction():
    assert _score("Infosys profit rises, upgrade to buy")[0] > 0
    assert _score("Company reports net loss, downgrade")[0] < 0
    assert _score("Board meeting scheduled")[0] == 0


def test_high_impact_is_independent_of_sentiment():
    _, impact, _ = _score("SEBI bars promoter; block deal reported")
    assert impact > 0.5


def test_macro_risk_detects_broad_stress():
    _, _, macro = _score("Sensex crash: panic selling as war fears mount")
    assert macro > 0.5


# ── the gate ──────────────────────────────────────────────────────────────────

def _item(minutes_ago, *, symbol=None, impact=0.0, macro=0.0):
    return NewsItem(
        "test", "headline", "", datetime.now(tz=IST) - timedelta(minutes=minutes_ago),
        symbol, 0.0, impact, macro,
    )


def test_fresh_high_impact_news_blocks_the_trade():
    context = NewsContext(symbol_items={"RELIANCE": [_item(5, symbol="RELIANCE", impact=0.8)]})
    allow, size, reason = context.entry_gate("RELIANCE")
    assert not allow and size == 0.0 and "high-impact" in reason


def test_stale_news_does_not_block():
    context = NewsContext(symbol_items={"RELIANCE": [_item(600, symbol="RELIANCE", impact=0.8)]})
    assert context.entry_gate("RELIANCE")[0]


def test_routine_low_impact_news_does_not_block():
    context = NewsContext(symbol_items={"RELIANCE": [_item(5, symbol="RELIANCE", impact=0.2)]})
    assert context.entry_gate("RELIANCE")[0]


def test_macro_risk_off_reduces_size_without_blocking():
    context = NewsContext(macro_items=[_item(10, macro=1.0) for _ in range(3)])
    allow, size, reason = context.entry_gate("RELIANCE")
    assert allow and size < 1.0 and reason


def test_quiet_tape_allows_full_size():
    allow, size, reason = NewsContext().entry_gate("RELIANCE")
    assert allow and size == 1.0 and reason is None


def test_context_staleness():
    assert NewsContext().is_stale
    assert not NewsContext(fetched_at=datetime.now(tz=IST)).is_stale


# ── archive ───────────────────────────────────────────────────────────────────

def test_archive_round_trip_and_dedup(tmp_path):
    db = tmp_path / "news.sqlite3"
    items = [
        NewsItem("src", "Infosys wins order", "u1", datetime.now(tz=IST), "INFY", 1.0, 0.6, 0.0),
        NewsItem("src", "Market falls", "u2", datetime.now(tz=IST), None, -1.0, 0.0, 0.5),
    ]
    assert archive(items, db) == 2
    assert archive(items, db) == 0, "re-archiving the same items must not duplicate"
    rows = recent_archive(symbol="INFY", db_path=db)
    assert len(rows) == 1 and rows[0]["title"] == "Infosys wins order"


def test_archive_of_nothing_is_a_noop(tmp_path):
    assert archive([], tmp_path / "news.sqlite3") == 0


def test_recent_archive_on_missing_db_is_empty(tmp_path):
    assert recent_archive(db_path=tmp_path / "absent.sqlite3") == []
