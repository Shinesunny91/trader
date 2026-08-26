import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai.entry_quality import (
    EntryQuality,
    GateConfig,
    compute_entry_quality,
    macro_alignment,
    passes_entry_gate,
)
from nse_intraday_ai.indicators import add_indicators

IST = "Asia/Kolkata"


def _frame(closes, volumes=None):
    index = pd.date_range("2026-08-11 09:15", periods=len(closes), freq="5min", tz=IST)
    closes = np.asarray(closes, dtype=float)
    volumes = np.full(len(closes), 1000.0) if volumes is None else np.asarray(volumes, float)
    return add_indicators(pd.DataFrame({
        "open": closes, "high": closes * 1.002, "low": closes * 0.998,
        "close": closes, "volume": volumes,
    }, index=index))


# ── feature extraction ────────────────────────────────────────────────────────

def test_impulse_is_measured_in_atr_and_signed_by_side():
    closes = list(np.linspace(100, 101, 30)) + [103.0]     # sharp final push up
    frame = _frame(closes)
    long_q = compute_entry_quality(frame, "LONG")
    short_q = compute_entry_quality(frame, "SHORT")
    assert long_q is not None and short_q is not None
    assert long_q.impulse_atr > 0
    assert short_q.impulse_atr == pytest.approx(-long_q.impulse_atr)


def test_returns_none_without_enough_history():
    assert compute_entry_quality(_frame([100.0, 101.0]), "LONG") is None
    assert compute_entry_quality(None, "LONG") is None


def test_bars_since_extreme_counts_back_from_the_high():
    closes = list(np.linspace(100, 105, 20)) + list(np.linspace(105, 103, 6))
    quality = compute_entry_quality(_frame(closes), "LONG")
    assert quality.bars_since_extreme >= 5


# ── conviction ────────────────────────────────────────────────────────────────

def test_conviction_needs_both_volume_and_impulse():
    assert EntryQuality(3.0, 2.0, 0.5, 55, 1).has_conviction
    assert not EntryQuality(3.0, 0.5, 0.5, 55, 1).has_conviction   # no impulse
    assert not EntryQuality(0.5, 2.0, 0.5, 55, 1).has_conviction   # no volume


def test_strong_is_stricter_than_conviction():
    modest = EntryQuality(2.5, 2.0, 0.5, 55, 1)
    strong = EntryQuality(3.5, 3.0, 0.5, 55, 1)
    assert modest.has_conviction and not modest.is_strong
    assert strong.has_conviction and strong.is_strong


# ── macro alignment ───────────────────────────────────────────────────────────

def test_macro_signs_flip_with_side():
    kwargs = dict(nifty_change_pct=0.2, usdinr_change_pct=-0.05, crude_change_pct=-0.4)
    long_m = macro_alignment("LONG", **kwargs)
    short_m = macro_alignment("SHORT", **kwargs)
    assert long_m.score == pytest.approx(-short_m.score)
    assert long_m.score > 0, "index up, rupee strong, crude down should favour a long"


def test_crude_rally_is_a_headwind_for_indian_equity_longs():
    calm = macro_alignment("LONG", nifty_change_pct=0.0, usdinr_change_pct=0.0,
                           crude_change_pct=0.0)
    spike = macro_alignment("LONG", nifty_change_pct=0.0, usdinr_change_pct=0.0,
                            crude_change_pct=2.0)
    assert spike.score < calm.score


def test_weak_rupee_is_a_headwind_for_equity_longs():
    strong = macro_alignment("LONG", nifty_change_pct=0.0, usdinr_change_pct=-0.2,
                             crude_change_pct=0.0)
    weak = macro_alignment("LONG", nifty_change_pct=0.0, usdinr_change_pct=0.2,
                           crude_change_pct=0.0)
    assert weak.score < strong.score


def test_coverage_tracks_missing_feeds():
    partial = macro_alignment("LONG", nifty_change_pct=0.1, usdinr_change_pct=None,
                              crude_change_pct=None)
    assert partial.coverage == 1 and not partial.is_complete


def test_components_are_clipped_so_one_outlier_cannot_dominate():
    extreme = macro_alignment("LONG", nifty_change_pct=500.0, usdinr_change_pct=-500.0,
                              crude_change_pct=-500.0)
    assert extreme.score == pytest.approx(9.0)   # 3 components x clip of 3


# ── the gate ──────────────────────────────────────────────────────────────────

_GOOD_MACRO = macro_alignment("LONG", nifty_change_pct=0.2, usdinr_change_pct=-0.05,
                              crude_change_pct=-0.3)
_BAD_MACRO = macro_alignment("LONG", nifty_change_pct=-0.3, usdinr_change_pct=0.2,
                             crude_change_pct=1.0)


def test_gate_rejects_signals_without_an_impulse():
    quiet = EntryQuality(0.1, 0.2, 0.3, 50, 10)
    result = passes_entry_gate(quiet, _GOOD_MACRO)
    assert not result.allow and "no conviction" in result.reason


def test_gate_rejects_macro_pointing_the_other_way():
    conviction = EntryQuality(3.0, 2.5, 0.5, 60, 1)
    result = passes_entry_gate(conviction, _BAD_MACRO)
    assert not result.allow and "macro against" in result.reason


def test_gate_accepts_conviction_with_supportive_macro():
    conviction = EntryQuality(3.0, 2.5, 0.5, 60, 1)
    result = passes_entry_gate(conviction, _GOOD_MACRO)
    assert result.allow and result.is_strong


def test_gate_requires_full_macro_coverage_by_default():
    conviction = EntryQuality(3.0, 2.5, 0.5, 60, 1)
    partial = macro_alignment("LONG", nifty_change_pct=0.5, usdinr_change_pct=None,
                              crude_change_pct=None)
    assert not passes_entry_gate(conviction, partial).allow
    relaxed = GateConfig(require_macro_coverage=False)
    assert passes_entry_gate(conviction, partial, config=relaxed).allow


def test_gate_can_be_run_on_quality_alone():
    conviction = EntryQuality(3.0, 2.5, 0.5, 60, 1)
    config = GateConfig(require_macro_coverage=False)
    assert passes_entry_gate(conviction, _BAD_MACRO, config=config).allow is False
    # ...but with the macro floor removed as well, quality alone is enough
    config = GateConfig(require_macro_coverage=False, min_macro_score=-99)
    assert passes_entry_gate(conviction, _BAD_MACRO, config=config).allow


def test_gate_handles_missing_features_safely():
    assert not passes_entry_gate(None, _GOOD_MACRO).allow
    assert not passes_entry_gate(EntryQuality(3.0, 2.5, 0.5, 60, 1), None).allow
