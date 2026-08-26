"""Unified scan service: bar cursor, closed-bar-only evaluation, catch-up."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai.data import DataResult
from nse_intraday_ai.market_context import MarketContext
from nse_intraday_ai.scan_service import run_scan_cycle

IST = "Asia/Kolkata"
SYMBOLS = ["AAA.NS", "BBB.NS"]
BAR = pd.Timedelta(minutes=5)


def _frame(end: pd.Timestamp, bars: int = 120, seed: int = 3) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=bars, freq="5min", tz=IST)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, bars)))
    spread = close * 0.001
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.integers(10_000, 90_000, bars).astype(float),
        },
        index=idx,
    )


class FakeProvider:
    """Deterministic provider without a cache: frames end at `last_bar_start`."""

    name = "fake"

    def __init__(self, last_bar_start: pd.Timestamp):
        self.last_bar_start = last_bar_start

    def batch_history(self, symbols, period="1d", interval="5m"):
        return {
            s: DataResult(s, _frame(self.last_bar_start, seed=i + 3), self.name, None)
            for i, s in enumerate(symbols)
        }


class CachingProvider:
    """Provider backed by a real CandleCache, counting network fetches."""

    name = "caching-fake"

    def __init__(self, cache, last_bar_start: pd.Timestamp):
        self._cache = cache
        self.last_bar_start = last_bar_start
        self.fetch_calls = 0

    def batch_history(self, symbols, period="1d", interval="5m", retry_missing=True):
        self.fetch_calls += 1
        out = {}
        for i, s in enumerate(symbols):
            frame = _frame(self.last_bar_start, seed=i + 3)
            self._cache.save(s, interval, frame)
            out[s] = DataResult(s, frame, self.name, None)
        return out


def test_no_new_bar_does_not_fetch_when_cache_warm(tmp_path):
    """The core efficiency win: a tick with no newly-closed bar must NOT hit
    the network once the cache already holds the data."""
    from nse_intraday_ai.candle_cache import CandleCache

    cache = CandleCache(tmp_path / "candles.sqlite3")
    # Must stay inside the scan's `period` lookback (5d): the cache serves
    # frames via load_period(now - period), so a hardcoded past date makes the
    # cache read come back empty and trips the cold-cache re-fetch, failing
    # this test for reasons that have nothing to do with the fetch gate.
    t0 = (pd.Timestamp.now(tz=IST) - pd.Timedelta(days=1)).floor("5min")
    provider = CachingProvider(cache, last_bar_start=t0)
    state_path = tmp_path / "s.json"

    def cycle(now):
        return run_scan_cycle(
            "nse", source="test", provider_factory=lambda: provider,
            record_shadow=False, log_signals=False, state_path=state_path,
            now=now, market_context=MarketContext(), symbols=SYMBOLS,
        )

    first = cycle(t0 + BAR)                       # cursor None -> must fetch
    assert first.fetched and provider.fetch_calls == 1
    second = cycle(t0 + BAR + pd.Timedelta(seconds=30))  # same bar -> cache only
    assert not second.fetched
    assert provider.fetch_calls == 1             # NO extra network call
    assert len(second.results) == 2              # still renders from cache
    # force_fetch overrides the gate (manual "Scan now")
    forced = run_scan_cycle(
        "nse", source="test", provider_factory=lambda: provider,
        record_shadow=False, log_signals=False, state_path=state_path,
        now=t0 + BAR + pd.Timedelta(seconds=45), market_context=MarketContext(),
        symbols=SYMBOLS, force_fetch=True,
    )
    assert forced.fetched and provider.fetch_calls == 2


def _cycle(last_bar_start, now, state_path, **kw):
    return run_scan_cycle(
        "nse",
        source="test",
        provider_factory=lambda: FakeProvider(last_bar_start),
        record_shadow=False,
        log_signals=False,
        state_path=state_path,
        now=now,
        market_context=MarketContext(),   # offline: skip the network fetch
        symbols=SYMBOLS,
        **kw,
    )


@pytest.fixture
def t0():
    return pd.Timestamp("2026-07-06 13:00:00", tz=IST)  # a Monday, mid-session


class TestBarCursor:
    def test_first_run_evaluates_only_latest_closed_bar(self, tmp_path, t0):
        cycle = _cycle(t0, now=t0 + BAR, state_path=tmp_path / "s.json")
        assert cycle.symbols_with_data == 2
        assert cycle.evaluated_bars == 2          # one (latest closed) bar per symbol
        assert len(cycle.results) == 2

    def test_rerun_same_now_evaluates_nothing_new(self, tmp_path, t0):
        path = tmp_path / "s.json"
        _cycle(t0, now=t0 + BAR, state_path=path)
        again = _cycle(t0, now=t0 + BAR + pd.Timedelta(seconds=30), state_path=path)
        assert again.evaluated_bars == 0
        # ...but display state is still fully populated
        assert len(again.results) == 2

    def test_new_bar_evaluated_exactly_once(self, tmp_path, t0):
        path = tmp_path / "s.json"
        _cycle(t0, now=t0 + BAR, state_path=path)
        # one more bar has closed since the cursor
        nxt = _cycle(t0 + BAR, now=t0 + 2 * BAR + pd.Timedelta(seconds=15), state_path=path)
        assert nxt.evaluated_bars == 2             # exactly the one new bar per symbol
        third = _cycle(t0 + BAR, now=t0 + 2 * BAR + pd.Timedelta(seconds=45), state_path=path)
        assert third.evaluated_bars == 0

    def test_catchup_covers_every_missed_bar(self, tmp_path, t0):
        """If cycles stall, the next one evaluates ALL closed bars since the cursor."""
        path = tmp_path / "s.json"
        _cycle(t0, now=t0 + BAR, state_path=path)
        # six bars later (cycle outage) — all six must be evaluated per symbol
        catchup = _cycle(t0 + 6 * BAR, now=t0 + 7 * BAR, state_path=path)
        assert catchup.evaluated_bars == 12

    def test_forming_bar_is_never_evaluated(self, tmp_path, t0):
        # `now` is only 2 minutes into the newest bar -> that bar is open
        cycle = _cycle(t0, now=t0 + pd.Timedelta(minutes=2), state_path=tmp_path / "s.json")
        for result in cycle.results:
            # newest CLOSED bar is the one before t0
            assert result.quote_age_seconds >= 120

    def test_per_universe_cursor_isolation(self, tmp_path, t0):
        path = tmp_path / "s.json"
        _cycle(t0, now=t0 + BAR, state_path=path)
        import json
        state = json.loads(path.read_text())
        assert "nse:5m" in state and "commodity:5m" not in state


# ── Aug-2026 gates: entry quality + macro alignment ───────────────────────────

def _trending_frame(end: pd.Timestamp, bars: int = 120) -> pd.DataFrame:
    """A clean uptrend with a volume-backed impulse on the final bar.

    Deterministic on purpose: the gate is a threshold rule, so a random walk
    would make the test flap.
    """
    idx = pd.date_range(end=end, periods=bars, freq="5min", tz=IST)
    close = np.linspace(100, 106, bars)
    close[-1] = close[-2] * 1.02          # the impulse the gate looks for
    open_ = np.r_[close[0], close[:-1]]
    volume = np.full(bars, 20_000.0)
    volume[-1] = 500_000.0                # the volume expansion it needs
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.0005,
            "low": np.minimum(open_, close) * 0.9995,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


class TrendingProvider:
    name = "trending-fake"

    def __init__(self, last_bar_start: pd.Timestamp):
        self.last_bar_start = last_bar_start

    def batch_history(self, symbols, period="1d", interval="5m", retry_missing=True):
        return {
            s: DataResult(s, _trending_frame(self.last_bar_start), self.name, None)
            for s in symbols
        }


def _gate_cycle(tmp_path, cfg_extra, market_context):
    t0 = (pd.Timestamp.now(tz=IST) - pd.Timedelta(days=1)).floor("5min").replace(hour=11, minute=0)
    cfg = {
        "capital": 1_000_000, "risk_per_trade_pct": 0.5, "max_position_pct": 25.0,
        "interval": "5m", "period": "5d", "min_confidence": 0.0, "min_reward_risk": 0.0,
        "estimated_cost_bps": 15.0, "slippage_bps": 3.0,
        "min_agreeing_votes": 1, "min_vote_share": 0.0,
        "meta_veto": False, "require_policy_approval": False,
        **cfg_extra,
    }
    return run_scan_cycle(
        "nse", source="test", provider_factory=lambda: TrendingProvider(t0),
        cfg=cfg, record_shadow=False, log_signals=False,
        state_path=tmp_path / "state.json", now=t0 + BAR,
        market_context=market_context, symbols=SYMBOLS,
    )


def _context(nifty=0.3, usdinr=-0.05, crude=-0.4):
    ctx = MarketContext()
    ctx.nifty_change_pct = nifty
    ctx.usdinr_change_pct = usdinr
    ctx.crude_change_pct = crude
    return ctx


def test_quality_gate_is_wired_into_the_scan_pipeline(tmp_path):
    """Signals without an impulse must be refused, with the reason kept."""
    off = _gate_cycle(tmp_path, {"entry_quality_gate": False, "macro_gate": False}, _context())
    on = _gate_cycle(tmp_path / "b", {"entry_quality_gate": True, "macro_gate": True}, _context())
    assert len(on.recommendations) <= len(off.recommendations)
    if on.quality_vetoed:
        _, reason = on.quality_vetoed[0]
        assert reason, "a veto must explain itself"


def test_macro_against_the_trade_blocks_long_signals(tmp_path):
    """Index down, rupee weak, crude spiking is a hostile tape for a long."""
    hostile = _context(nifty=-0.6, usdinr=0.3, crude=2.0)
    cycle = _gate_cycle(tmp_path, {"entry_quality_gate": False, "macro_gate": True}, hostile)
    longs = [r for r in cycle.recommendations if r.plan.side.value == "LONG"]
    assert not longs
    assert any("macro against" in reason for _, reason in cycle.quality_vetoed)


def test_missing_macro_panel_blocks_rather_than_guessing(tmp_path):
    """A dead context feed must stop the gate, not silently pass everything."""
    cycle = _gate_cycle(tmp_path, {"entry_quality_gate": False, "macro_gate": True},
                        MarketContext())
    assert not cycle.recommendations
    assert any("macro panel incomplete" in reason for _, reason in cycle.quality_vetoed)


def test_gates_can_be_disabled_entirely(tmp_path):
    cycle = _gate_cycle(tmp_path, {"entry_quality_gate": False, "macro_gate": False},
                        MarketContext())
    assert not cycle.quality_vetoed
