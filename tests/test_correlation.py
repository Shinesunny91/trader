"""Intra-stock dependency — the risk half, which is the half that works."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai.correlation import (
    breadth, cluster_labels, effective_positions, most_correlated,
    pick_diversified, return_matrix, rolling_corr,
)


def frames_from(returns: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=len(next(iter(returns.values()))), freq="D")
    out = {}
    for s, r in returns.items():
        close = 100 * np.exp(np.cumsum(r))
        out[s] = pd.DataFrame({"close": close}, index=idx)
    return out


def test_identical_names_are_one_bet_not_three():
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 0.01, 400)
    frames = frames_from({f"C{i}.NS": shared.copy() for i in range(3)})
    corr = rolling_corr(return_matrix(frames), window=300)
    assert effective_positions(corr, list(frames)) == pytest.approx(1.0, abs=0.05)


def test_independent_names_are_three_bets():
    rng = np.random.default_rng(1)
    frames = frames_from({f"I{i}.NS": rng.normal(0, 0.01, 600) for i in range(3)})
    corr = rolling_corr(return_matrix(frames), window=500)
    assert effective_positions(corr, list(frames)) > 2.5


def test_effective_positions_sits_between_the_extremes():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 0.01, 600)
    frames = frames_from({
        f"H{i}.NS": 0.7 * common + 0.3 * rng.normal(0, 0.01, 600) for i in range(3)
    })
    corr = rolling_corr(return_matrix(frames), window=500)
    eff = effective_positions(corr, list(frames))
    assert 1.0 < eff < 3.0


def test_diversified_pick_skips_the_correlated_runner_up():
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.01, 600)
    frames = frames_from({
        "A.NS": common, "B.NS": common * 0.99 + rng.normal(0, 0.0005, 600),
        "C.NS": rng.normal(0, 0.01, 600),
    })
    corr = rolling_corr(return_matrix(frames), window=500)
    ranked = pd.DataFrame({"symbol": ["A.NS", "B.NS", "C.NS"], "_score": [3.0, 2.0, 1.0]})
    picks = pick_diversified(ranked, corr, n=2, max_corr=0.7)
    got = [p.symbol for p in picks]
    assert got[0] == "A.NS"
    assert "B.NS" not in got, "B is the same bet as A and must be skipped"
    assert "C.NS" in got


def test_diversification_can_be_switched_off_by_a_loose_threshold():
    """A scaled copy correlates at exactly 1.0, so only max_corr=1.0 admits it."""
    rng = np.random.default_rng(4)
    common = rng.normal(0, 0.01, 400)
    frames = frames_from({"A.NS": common, "B.NS": common * 0.99})
    corr = rolling_corr(return_matrix(frames), window=300)
    ranked = pd.DataFrame({"symbol": ["A.NS", "B.NS"], "_score": [2.0, 1.0]})
    assert corr.at["A.NS", "B.NS"] == pytest.approx(1.0)
    assert len(pick_diversified(ranked, corr, n=2, max_corr=0.999)) == 1
    assert len(pick_diversified(ranked, corr, n=2, max_corr=1.0)) == 2


def test_most_correlated_excludes_the_symbol_itself():
    rng = np.random.default_rng(5)
    common = rng.normal(0, 0.01, 400)
    frames = frames_from({"A.NS": common, "B.NS": common * 0.9 + rng.normal(0, 0.003, 400),
                          "C.NS": rng.normal(0, 0.01, 400)})
    corr = rolling_corr(return_matrix(frames), window=300)
    mc = most_correlated(corr, "A.NS", top=2)
    assert "A.NS" not in mc.index
    assert mc.index[0] == "B.NS"


def test_clusters_group_names_that_move_together():
    rng = np.random.default_rng(6)
    g1 = rng.normal(0, 0.01, 500)
    g2 = rng.normal(0, 0.01, 500)
    frames = frames_from({
        "X1.NS": g1, "X2.NS": g1 * 0.98 + rng.normal(0, 0.001, 500),
        "Y1.NS": g2, "Y2.NS": g2 * 0.98 + rng.normal(0, 0.001, 500),
    })
    corr = rolling_corr(return_matrix(frames), window=400)
    labels = cluster_labels(corr, threshold=0.7)
    assert labels["X1.NS"] == labels["X2.NS"]
    assert labels["Y1.NS"] == labels["Y2.NS"]
    assert labels["X1.NS"] != labels["Y1.NS"]


def test_breadth_detects_a_market_moving_as_one():
    rng = np.random.default_rng(7)
    common = rng.normal(0, 0.01, 300)
    together = frames_from({f"T{i}.NS": common * 0.98 + rng.normal(0, 0.001, 300)
                            for i in range(5)})
    apart = frames_from({f"S{i}.NS": rng.normal(0, 0.01, 300) for i in range(5)})
    b_together = breadth(return_matrix(together), window=60)
    b_apart = breadth(return_matrix(apart), window=60)
    assert b_together["avg_pairwise_corr"] > 0.9
    assert b_apart["avg_pairwise_corr"] < 0.3


def test_correlation_uses_returns_not_price_levels():
    """Two unrelated names both drifting up must not look correlated."""
    rng = np.random.default_rng(8)
    a = rng.normal(0.002, 0.01, 600)
    b = rng.normal(0.002, 0.01, 600)
    corr = rolling_corr(return_matrix(frames_from({"A.NS": a, "B.NS": b})), window=500)
    assert abs(corr.at["A.NS", "B.NS"]) < 0.2
