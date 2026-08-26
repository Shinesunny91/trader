import pandas as pd

from nse_intraday_ai.context_series import MarketContextSeries, build_context_series
from nse_intraday_ai.market_context import MarketContext, is_commodity_symbol


def _frame(start, periods, step=0.1, base=100.0, freq="5min"):
    idx = pd.date_range(start, periods=periods, freq=freq, tz="Asia/Kolkata")
    closes = [base + i * step for i in range(periods)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [10_000] * periods,
        },
        index=idx,
    )


def test_is_commodity_symbol():
    assert is_commodity_symbol("GC=F")
    assert is_commodity_symbol("USDINR=X")
    assert not is_commodity_symbol("RELIANCE.NS")
    assert not is_commodity_symbol("^NSEI")


def test_extra_symbol_adj_metals_inverse_dxy():
    ctx = MarketContext(dxy_change_pct=-0.15)
    assert ctx.extra_symbol_adj("GC=F", "LONG") > 0    # dollar down -> gold long boosted
    assert ctx.extra_symbol_adj("GC=F", "SHORT") < 0
    ctx_up = MarketContext(dxy_change_pct=+0.15)
    assert ctx_up.extra_symbol_adj("GC=F", "LONG") < 0


def test_extra_symbol_adj_energy_follows_risk():
    risk_on = MarketContext(global_risk=+0.8)
    assert risk_on.extra_symbol_adj("CL=F", "LONG") > 0
    assert risk_on.extra_symbol_adj("CL=F", "SHORT") < 0
    # safe havens get the inverse (mild)
    assert risk_on.extra_symbol_adj("GC=F", "LONG") < 0


def test_extra_symbol_adj_stock_sector_alignment():
    ctx = MarketContext(sector_index_regimes={"Banking": "TRENDING_UP"})
    assert ctx.extra_symbol_adj("HDFCBANK.NS", "LONG") > 0
    assert ctx.extra_symbol_adj("HDFCBANK.NS", "SHORT") < 0


def test_context_series_is_causal():
    frames = {"^INDIAVIX": _frame("2026-06-24 09:15", 60, step=0.0, base=14.0)}
    series = build_context_series(frames)
    before = pd.Timestamp("2026-06-24 09:00", tz="Asia/Kolkata")
    assert series.at(before).vix_value is None  # nothing known before first bar
    during = pd.Timestamp("2026-06-24 10:00", tz="Asia/Kolkata")
    assert series.at(during).vix_value == 14.0


def test_context_series_staleness_guard():
    frames = {"^INDIAVIX": _frame("2026-06-24 09:15", 12, step=0.0, base=14.0)}
    series = build_context_series(frames)
    much_later = pd.Timestamp("2026-06-24 18:00", tz="Asia/Kolkata")
    assert series.at(much_later).vix_value is None  # data too stale to use


def test_intraday_yield_momentum_not_scored():
    # Deliberately unused in scoring (Jul-2026 ablation: pure noise at 1h
    # horizon) — the field is display-only.
    ctx = MarketContext(us10y_change_pct=+1.0)
    assert ctx.extra_symbol_adj("GC=F", "LONG") == 0.0


def test_extra_symbol_adj_inr_conversion_tailwind():
    # Weakening INR lifts MCX (INR-denominated) commodity longs across groups.
    ctx = MarketContext(usdinr_change_pct=+0.15)
    assert ctx.extra_symbol_adj("GC=F", "LONG") > 0
    assert ctx.extra_symbol_adj("CL=F", "LONG") > 0
    assert ctx.extra_symbol_adj("ZW=F", "LONG") > 0
    assert ctx.extra_symbol_adj("GC=F", "SHORT") < 0


def test_extra_symbol_adj_copper_follows_china():
    ctx = MarketContext(hsi_change_pct=+0.8)
    assert ctx.extra_symbol_adj("HG=F", "LONG") > 0
    assert ctx.extra_symbol_adj("GC=F", "LONG") == 0.0  # china proxy is industrial-only


def test_us_vix_uncertainty_penalises_equities_both_sides():
    ctx = MarketContext(us_vix_value=30.0)
    assert ctx.extra_symbol_adj("RELIANCE.NS", "LONG") < 0
    assert ctx.extra_symbol_adj("RELIANCE.NS", "SHORT") < 0
    calm = MarketContext(us_vix_value=15.0)
    assert calm.extra_symbol_adj("RELIANCE.NS", "LONG") == 0.0


def test_us_vix_level_survives_longer_staleness():
    frames = {"^VIX": _frame("2026-06-24 19:00", 12, step=0.0, base=28.0)}
    series = build_context_series(frames)
    next_morning = pd.Timestamp("2026-06-25 03:00", tz="Asia/Kolkata")
    assert series.at(next_morning).us_vix_value == 28.0  # level valid for hours
    much_later = pd.Timestamp("2026-06-26 12:00", tz="Asia/Kolkata")
    assert series.at(much_later).us_vix_value is None


def test_vwap_breadth_penalises_fighting_the_tape_only():
    bull_tape = MarketContext(vwap_breadth_pct=0.72)
    assert bull_tape.extra_symbol_adj("RELIANCE.NS", "SHORT") < 0
    assert bull_tape.extra_symbol_adj("RELIANCE.NS", "LONG") == 0.0  # no boost
    bear_tape = MarketContext(vwap_breadth_pct=0.25)
    assert bear_tape.extra_symbol_adj("RELIANCE.NS", "LONG") < 0
    assert bear_tape.extra_symbol_adj("RELIANCE.NS", "SHORT") == 0.0
    neutral = MarketContext(vwap_breadth_pct=0.50)
    assert neutral.extra_symbol_adj("RELIANCE.NS", "LONG") == 0.0


def test_vwap_breadth_series_causal_lookup():
    from nse_intraday_ai.context_series import build_vwap_breadth_series
    frames = {f"S{i}.NS": _frame("2026-06-24 09:15", 60, step=0.1 if i % 2 else -0.1)
              for i in range(12)}
    breadth = build_vwap_breadth_series(frames, min_symbols=10)
    assert breadth is not None and not breadth.empty
    assert 0.0 <= float(breadth.iloc[-1]) <= 1.0
    series = MarketContextSeries(vwap_breadth=breadth)
    ctx = series.at(pd.Timestamp("2026-06-24 12:00", tz="Asia/Kolkata"))
    assert ctx.vwap_breadth_pct is not None
    early = series.at(pd.Timestamp("2026-06-24 09:00", tz="Asia/Kolkata"))
    assert early.vwap_breadth_pct is None


def test_empty_context_series_returns_neutral_context():
    series = MarketContextSeries()
    ctx = series.at(pd.Timestamp("2026-06-24 10:00", tz="Asia/Kolkata"))
    assert ctx.index_regime == "UNKNOWN"
    assert ctx.global_risk == 0.0
    assert ctx.extra_symbol_adj("RELIANCE.NS", "LONG") == 0.0
