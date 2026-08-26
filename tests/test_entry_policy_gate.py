"""The study's vectorised gate must be the gate that ships.

`scripts/entry_policy_study.py` decides whether filtering with
`entry_quality.passes_entry_gate` before ranking helps or hurts the live book.
It evaluates the gate as vectorised numpy over 67K rows rather than by calling
the real function 67K times, which is a re-implementation — and a study that
measures a slightly different gate from the one that ships would answer the
wrong question. This pins them together.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from entry_policy_study import gate_verdicts, macro_score  # noqa: E402
from nse_intraday_ai.entry_quality import (  # noqa: E402
    EntryQuality,
    GateConfig,
    MacroAlignment,
    macro_alignment,
    passes_entry_gate,
)


def _sample(n=400, seed=11):
    rng = np.random.default_rng(seed)
    side = rng.choice(["LONG", "SHORT"], n)
    sign = np.where(side == "LONG", 1.0, -1.0)
    # Raw instrument moves, then signed exactly as build_dataset.add_macro does.
    nifty, usdinr, crude = (rng.normal(0, s, n) for s in (0.25, 0.12, 0.7))
    return pd.DataFrame({
        "side": side,
        "vol_z": rng.normal(1.0, 2.0, n),
        "run6": rng.normal(1.0, 1.8, n),
        "ext_vwap": rng.normal(0, 1.0, n),
        "rsi": rng.uniform(20, 80, n),
        "age_extreme": rng.integers(0, 20, n),
        "nifty": nifty, "usdinr": usdinr, "crude": crude,
        "m_nifty": nifty * sign,
        "m_inr": -usdinr * sign,
        "m_crude": -crude * sign,
    })


def test_vectorised_gate_matches_the_shipped_gate():
    frame = _sample()
    vector = gate_verdicts(frame).to_numpy()
    loop = []
    for row in frame.itertuples():
        macro = macro_alignment(
            row.side,
            nifty_change_pct=row.nifty,
            usdinr_change_pct=row.usdinr,
            crude_change_pct=row.crude,
        )
        quality = EntryQuality(row.vol_z, row.run6, row.ext_vwap, row.rsi, row.age_extreme)
        loop.append(passes_entry_gate(quality, macro, config=GateConfig()).allow)
    assert vector.tolist() == loop
    # The fixture must actually exercise both verdicts, or the test is vacuous.
    assert 0 < vector.sum() < len(vector)


def test_vectorised_macro_score_matches():
    frame = _sample(120, seed=3)
    vector = macro_score(frame).to_numpy()
    loop = np.array([
        macro_alignment(r.side, nifty_change_pct=r.nifty, usdinr_change_pct=r.usdinr,
                        crude_change_pct=r.crude).score
        for r in frame.itertuples()
    ])
    np.testing.assert_allclose(vector, loop, atol=1e-9)


def test_incomplete_macro_panel_is_refused_by_both():
    frame = _sample(50, seed=7)
    frame.loc[frame.index[:10], ["m_crude", "crude"]] = np.nan
    assert not gate_verdicts(frame).to_numpy()[:10].any()
    assert not passes_entry_gate(
        EntryQuality(5.0, 5.0, 0.0, 50.0, 1),
        MacroAlignment(nifty=1.0, inr=1.0, crude=None, score=2.0, coverage=2),
        config=GateConfig(),
    ).allow
