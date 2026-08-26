"""The signal-ranking model's contracts.

The failure this file mostly guards against is silent degradation: a model that
loads but scores a feature set it was not trained on, or a live path that
quietly falls back to a different ranking without anyone noticing. That
happened once already — `sim_today.py` used a lighter feature builder than
`build_dataset.py`, so the trained model could not score live signals at all
and fell back to the composite rank without failing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai.signal_model import (
    ALL_FEATURES,
    FEATURE_NAMES,
    REGIMES,
    SignalModel,
    expectancy_note,
    feature_matrix,
    load_if_available,
    train,
)

BARRIER = "1.5_3.0_12"


def synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES})
    frame["side"] = rng.choice(["LONG", "SHORT"], n)
    frame["regime"] = rng.choice(REGIMES, n)
    # A learnable signal so the forest has something to fit.
    frame[f"net_bps_{BARRIER}"] = frame["run6"] * 10 + rng.normal(scale=2, size=n)
    return frame


# ── feature contract ──────────────────────────────────────────────────────────

def test_feature_matrix_has_the_declared_shape_and_order():
    X = feature_matrix(synthetic(50))
    assert X.shape == (50, len(ALL_FEATURES))


def test_missing_features_raise_rather_than_defaulting_to_zero():
    """Silently feeding zeros for a whole family produces confident nonsense."""
    frame = synthetic(50).drop(columns=["clv_flow", "xs_with_crowd"])
    with pytest.raises(ValueError) as excinfo:
        feature_matrix(frame)
    assert "clv_flow" in str(excinfo.value)


def test_side_and_regime_become_numeric_columns():
    frame = synthetic(20)
    frame["side"] = "LONG"
    frame["regime"] = "RANGING"
    X = feature_matrix(frame)
    assert X[:, ALL_FEATURES.index("is_long")].min() == 1.0
    assert X[:, ALL_FEATURES.index("regime_RANGING")].min() == 1.0
    assert X[:, ALL_FEATURES.index("regime_HIGH_VOL")].max() == 0.0


def test_infinities_and_nans_do_not_reach_the_model():
    frame = synthetic(30)
    frame.loc[0, "run6"] = np.inf
    frame.loc[1, "vol_z"] = np.nan
    assert np.isfinite(feature_matrix(frame)).all()


# ── train / persist / score ───────────────────────────────────────────────────

def test_round_trip_preserves_predictions(tmp_path):
    frame = synthetic(500)
    model = train(frame, barrier=BARRIER, validation={"net_pct": 1.0})
    path = tmp_path / "signal_model.json"
    model.save(path)

    reloaded = SignalModel.load(path)
    assert np.allclose(model.score(frame), reloaded.score(frame))
    assert reloaded.validation["net_pct"] == 1.0
    assert reloaded.n_events == 500


def test_load_refuses_a_model_trained_on_different_features(tmp_path):
    import json

    frame = synthetic(200)
    path = tmp_path / "signal_model.json"
    train(frame, barrier=BARRIER).save(path)
    payload = json.loads(path.read_text())
    payload["feature_names"] = payload["feature_names"][:-3]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError) as excinfo:
        SignalModel.load(path)
    assert "retrain" in str(excinfo.value)


def test_load_if_available_is_none_when_nothing_is_installed(tmp_path):
    assert load_if_available(tmp_path) is None


def test_load_if_available_is_none_when_the_forest_is_missing(tmp_path):
    frame = synthetic(200)
    path = tmp_path / "signal_model.json"
    train(frame, barrier=BARRIER).save(path)
    path.with_suffix(".forest.pkl").unlink()
    assert load_if_available(tmp_path) is None


def test_model_learns_the_planted_signal():
    """Sanity: if the target is a feature, ranking must recover it."""
    frame = synthetic(800)
    model = train(frame, barrier=BARRIER)
    scores = model.score(frame)
    top = frame.loc[np.argsort(-scores)[:80], f"net_bps_{BARRIER}"].mean()
    bottom = frame.loc[np.argsort(-scores)[-80:], f"net_bps_{BARRIER}"].mean()
    assert top > bottom


def test_scoring_without_a_loaded_forest_raises():
    model = SignalModel(mu=[0.0], sd=[1.0], feature_names=list(ALL_FEATURES),
                        trained_at="", n_events=0, barrier=BARRIER)
    with pytest.raises(RuntimeError):
        model.score(synthetic(5))


# ── the note ──────────────────────────────────────────────────────────────────

def test_expectancy_note_without_a_model_names_the_fallback():
    note = expectancy_note(None)
    assert "No ranking model" in note and "0.55%" in note


def test_expectancy_note_carries_the_caveat_with_the_number():
    frame = synthetic(100)
    model = train(frame, barrier=BARRIER, validation={
        "net_pct": 7.36, "sessions": 34, "profit_factor": 1.93,
    })
    note = expectancy_note(model)
    assert "+7.36%" in note and "34 held-out sessions" in note
    assert "spans zero" in note, "the interval caveat must travel with the number"
    assert "not a profit forecast" in note


# ── the live-path contract that actually broke ────────────────────────────────

def test_dataset_builder_and_model_agree_on_the_feature_list():
    """One feature contract, or the live path silently falls back."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_dataset

    source = Path(build_dataset.__file__).read_text()
    # Every declared feature must be produced somewhere in the builder.
    missing = [name for name in FEATURE_NAMES if f'"{name}"' not in source]
    assert not missing, f"build_dataset.py does not emit: {missing}"
