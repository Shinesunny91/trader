"""Purging, embargo and uniqueness — the things that decide whether a
long-horizon backtest is measurement or flattery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai.ml import (
    concurrency,
    conformal_threshold,
    purged_walk_forward,
    uniqueness_weights,
    walk_forward_predict,
)


def overlapping(n=300, hold=21):
    """One label per bar, each spanning `hold` bars — the realistic case."""
    start = pd.Series(np.arange(n))
    end = start + hold
    return start, end


def test_concurrency_counts_live_labels():
    start, end = overlapping(100, hold=10)
    live = concurrency(start, end)
    # Deep in the series every bar has ~11 labels open across it.
    assert live[50] == pytest.approx(11, abs=1)
    assert live[0] == 1                       # nothing open before the first


def test_uniqueness_weights_downweight_crowded_periods():
    start, end = overlapping(200, hold=20)
    w = uniqueness_weights(start, end)
    assert w.mean() == pytest.approx(1.0, rel=1e-6)
    # The first samples have fewer concurrent labels, so they weigh more.
    assert w[0] > w[100]


def test_non_overlapping_labels_are_all_equally_unique():
    start = pd.Series([0, 10, 20, 30])
    end = pd.Series([9, 19, 29, 39])
    w = uniqueness_weights(start, end)
    assert np.allclose(w, 1.0)


def test_purging_removes_every_training_label_that_overlaps_the_test():
    """The whole point: no training label may end inside a test window."""
    start, end = overlapping(400, hold=21)
    splits = purged_walk_forward(start, end, n_splits=4, embargo=0, min_train=10)
    assert splits, "expected at least one usable split"
    s, e = start.to_numpy(), end.to_numpy()
    for sp in splits:
        assert (e[sp.train] < sp.test_start).all(), (
            "a training label resolves inside the test window — that is leakage"
        )


def test_embargo_extends_the_exclusion_after_the_test_window():
    start, end = overlapping(400, hold=5)
    lax = purged_walk_forward(start, end, n_splits=4, embargo=0, min_train=10)
    strict = purged_walk_forward(start, end, n_splits=4, embargo=50, min_train=10)
    # An embargo can only ever remove training samples, never add them.
    for a, b in zip(lax, strict):
        assert len(b.train) <= len(a.train)


def test_train_and_test_never_intersect():
    start, end = overlapping(300, hold=10)
    for sp in purged_walk_forward(start, end, n_splits=5, embargo=3, min_train=10):
        assert not set(sp.train.tolist()) & set(sp.test.tolist())


def test_walk_forward_learns_a_real_signal():
    """Sanity: with an actual relationship, OOS predictions must correlate."""
    rng = np.random.default_rng(0)
    n = 1200
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(2.0 * x["a"] - 1.0 * x["b"] + rng.normal(scale=0.5, size=n))
    start = pd.Series(np.arange(n))
    end = start + 3
    pred = walk_forward_predict(x, y, start, end, n_splits=4, embargo=2)
    ok = pred.notna()
    assert ok.sum() > 200
    assert np.corrcoef(pred[ok], y[ok])[0, 1] > 0.7


def test_walk_forward_finds_nothing_in_pure_noise():
    """The more important direction — it must NOT manufacture an edge."""
    rng = np.random.default_rng(1)
    n = 1000
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.normal(size=n))          # unrelated to x
    start = pd.Series(np.arange(n))
    end = start + 21                            # heavily overlapping
    pred = walk_forward_predict(x, y, start, end, n_splits=4, embargo=5)
    ok = pred.notna()
    assert ok.sum() > 100
    corr = np.corrcoef(pred[ok], y[ok])[0, 1]
    assert abs(corr) < 0.15, f"leakage: found corr {corr:.3f} in pure noise"


def test_conformal_threshold_is_a_quantile_of_winning_scores():
    scores = np.linspace(0, 1, 200)
    outcomes = np.where(scores > 0.5, 1.0, -1.0)     # only high scores win
    keep_half = conformal_threshold(scores, outcomes, target_rate=0.5)
    assert 0.7 < keep_half < 0.8                     # median of the winners
    keep_all = conformal_threshold(scores, outcomes, target_rate=1.0)
    assert keep_all <= 0.51


def test_conformal_threshold_abstains_when_evidence_is_thin():
    assert conformal_threshold(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 0.5) == float("-inf")
