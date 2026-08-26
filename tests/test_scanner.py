import pandas as pd

from nse_intraday_ai.data import (
    DataResult,
    _parse_nifty50_csv,
    _parse_nifty500_csv,
    load_commodity_symbols,
    to_yahoo_symbol,
)
from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.scanner import iter_scan_universe, rank_actionable
from nse_intraday_ai.strategies import EnsembleConfig


class FakeProvider:
    name = "fake"

    def history(self, symbol, period="1d", interval="1m"):
        idx = pd.date_range("2026-01-01 09:15", periods=90, freq="min")
        close = [100 + i * 0.08 for i in range(90)]
        frame = pd.DataFrame(
            {
                "open": [value - 0.03 for value in close],
                "high": [value + 0.25 for value in close],
                "low": [value - 0.25 for value in close],
                "close": close,
                "volume": [100_000 + i * 3000 for i in range(90)],
            },
            index=idx,
        )
        return DataResult(symbol, frame, self.name)


def test_parse_nifty500_rejects_short_universe():
    raw = "Company Name,Industry,Symbol,Series,ISIN Code\nA Ltd,Test,AAA,EQ,INE0\n"

    try:
        _parse_nifty500_csv(raw)
    except ValueError as exc:
        assert "returned only" in str(exc)
    else:
        raise AssertionError("Expected short universe to be rejected")


def test_parse_nifty50_accepts_index_sized_universe():
    rows = ["Company Name,Industry,Symbol,Series,ISIN Code"]
    rows.extend(f"Company {i},Test,SYM{i},EQ,INE{i:05d}" for i in range(50))

    symbols = _parse_nifty50_csv("\n".join(rows))

    assert len(symbols) == 50
    assert symbols[0] == "SYM0.NS"


def test_commodity_symbols_are_not_rewritten_as_nse_equities():
    universe = load_commodity_symbols()

    assert "GC=F" in universe.symbols
    assert to_yahoo_symbol("GC=F") == "GC=F"
    assert to_yahoo_symbol("cl=f") == "CL=F"


def test_scanner_runs_symbols_in_parallel_shape():
    results = list(
        iter_scan_universe(
            symbols=["AAA.NS", "BBB.NS", "CCC.NS"],
            provider_factory=FakeProvider,
            period="1d",
            interval="1m",
            risk_config=RiskConfig(min_confidence=95),
            ensemble_config=EnsembleConfig(min_weighted_confidence=95),
            max_workers=2,
        )
    )

    assert {result.symbol for result in results} == {"AAA.NS", "BBB.NS", "CCC.NS"}
    assert all(result.rows == 90 for result in results)
    assert rank_actionable(results) == []


# ── synthetic-bar filtering ───────────────────────────────────────────────────

def test_drop_synthetic_bars_removes_off_grid_quote_rows():
    """Yahoo appends a live-quote row that is not a candle; filling against it
    fabricates a price that never traded."""
    import pandas as pd

    from nse_intraday_ai.candle_cache import drop_synthetic_bars

    index = pd.DatetimeIndex([
        "2026-08-12 09:15:00", "2026-08-12 09:15:17",   # <- the snapshot row
        "2026-08-12 09:20:00", "2026-08-12 09:25:00",
    ], tz="Asia/Kolkata")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}, index=index
    )
    cleaned = drop_synthetic_bars(frame, "5m")
    assert len(cleaned) == 3
    assert pd.Timestamp("2026-08-12 09:15:17", tz="Asia/Kolkata") not in cleaned.index


def test_drop_synthetic_bars_respects_the_interval_grid():
    import pandas as pd

    from nse_intraday_ai.candle_cache import drop_synthetic_bars

    index = pd.date_range("2026-08-12 09:15", periods=6, freq="1min", tz="Asia/Kolkata")
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "volume": 1.0}, index=index)
    assert len(drop_synthetic_bars(frame, "1m")) == 6   # all valid at 1m
    assert len(drop_synthetic_bars(frame, "5m")) == 2   # only 09:15 and 09:20


def test_drop_synthetic_bars_passes_through_unknown_intervals():
    import pandas as pd

    from nse_intraday_ai.candle_cache import drop_synthetic_bars

    index = pd.date_range("2026-08-12", periods=3, freq="1D", tz="Asia/Kolkata")
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "volume": 1.0}, index=index)
    assert len(drop_synthetic_bars(frame, "1d")) == 3
