from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai import meta as M
from nse_intraday_ai import ranker as R

FAST = R.RankerConfig(max_iter=60, learning_rate=0.1, max_leaf_nodes=7, min_samples_leaf=20)


def _panel(n_days: int = 800, n_syms: int = 40, seed: int = 0, leak_from: str | None = None):
    """Synthetic panel: short return = 40 * signal + noise.  `leak` equals the
    label only on/after `leak_from` (and is noise before), to catch look-ahead."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-01", periods=n_days).date
    idx = pd.MultiIndex.from_product([days, [f"S{i}" for i in range(n_syms)]],
                                     names=["session", "symbol"])
    n = len(idx)
    signal = rng.normal(size=n).astype("float32")
    y = 40 * signal + rng.normal(0, 60, size=n)
    leak = rng.normal(size=n)
    if leak_from:
        late = pd.to_datetime(idx.get_level_values("session")) >= leak_from
        leak[late] = y[late] / 60
    return pd.DataFrame({
        "signal": signal, "noise": rng.normal(size=n).astype("float32"),
        "leak": leak.astype("float32"),
        "short_bps": y.astype("float32"), "long_bps": (-y).astype("float32"),
        "intra_bps": (-y).astype("float32"),
        "turn_rank": np.tile(np.arange(1, n_syms + 1), n_days).astype("float32"),
        "spread_bps": np.full(n, 5.0, dtype="float32"), "val20_cr": np.ones(n, dtype="float32"),
        "sector": "X"}, index=idx)


def test_ranker_learns_a_planted_signal_out_of_sample():
    p = _panel()
    oos = R.walk_forward(p, FAST, start="2018-01-01", refit="YS", min_train_years=1.0)
    scored = oos.dropna()
    assert len(scored) > 0
    corr = np.corrcoef(scored, p.loc[scored.index, "signal"])[0, 1]
    assert corr > 0.8
    model_book = R.book(p, oos, k=5, universe=40, cost_bps=0.0)
    random_book = R.book(p, p["noise"], k=5, universe=40, cost_bps=0.0)
    assert model_book.mean() > random_book.mean() + 20


def test_walk_forward_cannot_use_information_from_the_block_it_scores():
    # `leak` is the answer itself, but only from 2018-07 on.  A model that ever
    # trained on the block it scores would exploit it there; a walk-forward
    # model has only seen noise in that column before each block.
    p = _panel(leak_from="2018-07-02")[["leak", "noise", "short_bps", "long_bps", "intra_bps",
                                         "turn_rank", "spread_bps", "val20_cr", "sector"]]
    oos = R.walk_forward(p, FAST, start="2018-07-02", refit="YS", min_train_years=1.0)
    first_block = oos.dropna()
    first_block = first_block[pd.to_datetime(first_block.index.get_level_values("session")) < "2019-01-01"]
    corr = np.corrcoef(first_block, p.loc[first_block.index, "leak"])[0, 1]
    assert abs(corr) < 0.1
    # ...whereas an in-sample fit on the same block finds it immediately.
    block = p.loc[first_block.index]
    insample = R.Ranker.fit(block, FAST, features=["leak", "noise"]).score(block)
    assert np.corrcoef(insample, block["leak"])[0, 1] > 0.8


def test_save_and_load_reproduce_scores(tmp_path):
    p = _panel(n_days=300)
    model = R.Ranker.fit(p, FAST)
    loaded = R.Ranker.load(model.save(tmp_path / "m"))
    np.testing.assert_allclose(model.score(p), loaded.score(p))
    assert loaded.features == model.features and loaded.trained_through == model.trained_through


def test_score_refuses_a_frame_missing_features():
    p = _panel(n_days=200)
    model = R.Ranker.fit(p, FAST)
    with pytest.raises(KeyError):
        model.score(p.drop(columns=["signal"]))


def test_book_takes_top_k_per_session_and_charges_costs():
    p = _panel(n_days=5, n_syms=10)
    score = pd.Series(np.tile(np.arange(10, 0, -1), 5).astype(float), index=p.index)  # S0 best
    b = R.book(p, score, k=2, universe=10, cost_bps=13.0)
    expected = p["short_bps"].groupby(level="session").apply(lambda s: s.iloc[:2].mean()) - 13.0
    np.testing.assert_allclose(b.to_numpy(), expected.to_numpy(), rtol=1e-5)
    liq = R.book(p, score, k=2, universe=10, cost_bps=None)            # fees 7 + spread 5
    np.testing.assert_allclose(liq.to_numpy(), (expected + 13.0 - 12.0).to_numpy(), rtol=1e-5)


def test_hedge_moves_weight_to_the_better_expert_and_keeps_a_floor():
    st = M.MetaState(weights={"rule": 0.5, "ranker": 0.5}, eta=0.5, floor=0.05)
    for i in range(60):
        st.update(f"2026-01-{i:02d}" if i < 10 else f"2026-02-{i:02d}", {"rule": -40.0, "ranker": 60.0})
    assert st.weights["ranker"] > 0.9
    assert st.weights["rule"] >= 0.05 / 2 - 1e-12


def test_hedge_update_is_idempotent_per_session():
    st = M.MetaState()
    st.update("2026-03-02", {"rule": 50.0, "ranker": -50.0})
    w = dict(st.weights)
    st.update("2026-03-02", {"rule": -500.0, "ranker": 500.0})
    assert st.weights == w


def test_cusum_alarm_needs_a_persistent_shortfall():
    st = M.MetaState(expected_bps=20, slack_bps=10, threshold_bps=300)
    for i, r in enumerate([-100, 200, -100, 200] * 5):
        st.update(f"2026-04-{i + 1:02d}", {}, book_return_bps=r)
    assert not st.alarm
    for i in range(20):
        st.update(f"2026-05-{i + 1:02d}", {}, book_return_bps=-10.0)
    assert st.alarm


def test_hedge_replay_is_causal():
    p = _panel(n_days=120, n_syms=20)
    scores = {"good": p["signal"], "bad": -p["signal"]}
    daily, weights = M.hedge_replay(scores, p["short_bps"], k=3, cost_bps=0.0)
    assert weights.iloc[0]["good"] == pytest.approx(0.5)            # day 1: nothing learned yet
    assert weights.iloc[-1]["good"] > 0.9
    assert daily.iloc[-60:].mean() > 0


def test_guard_keeps_the_ranker_until_it_is_significantly_worse():
    st = M.MetaState(guard_window=60, guard_t=-2.0)
    assert st.guard() == ("ranker", None)                         # no record yet
    rng = np.random.default_rng(0)
    for i in range(60):                                           # ranker better on average
        st.track.append({"session": f"a{i:03d}", "ranker": 60 + rng.normal(0, 80),
                         "rule": 30 + rng.normal(0, 80)})
    assert st.guard()[0] == "ranker"
    for i in range(60):                                           # ranker clearly worse lately
        st.track.append({"session": f"b{i:03d}", "ranker": -40 + rng.normal(0, 30),
                         "rule": 30 + rng.normal(0, 30)})
    expert, t = st.guard()
    assert expert == "rule" and t < -2
