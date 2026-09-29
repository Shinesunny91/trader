from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai import meta as M
from nse_intraday_ai import nse_bhav
from nse_intraday_ai import pipeline as P


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(nse_bhav, "CACHE", tmp_path)

    def put(day: date, holiday: bool = False):
        path = nse_bhav._holiday_marker(day, tmp_path) if holiday else nse_bhav._path(day, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return put


def test_previous_session_skips_weekends_and_verified_holidays(cache):
    cache(date(2026, 9, 11))                       # Friday: a session
    cache(date(2026, 9, 14), holiday=True)         # Monday: holiday marker
    assert P.previous_session(date(2026, 9, 15)) == date(2026, 9, 11)
    cache(date(2026, 9, 15))
    assert P.previous_session(date(2026, 9, 16)) == date(2026, 9, 15)


def test_previous_session_refuses_a_day_without_file_or_marker(cache):
    # 2026-09-29 morning: the previous day's file had not arrived.  The morning
    # must stop rather than silently rank on the day before it.
    cache(date(2026, 9, 28))
    with pytest.raises(P.DataNotReady):
        P.previous_session(date(2026, 9, 30))


def _rows(n=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([[date(2026, 9, 29)], [f"S{i}" for i in range(n)]],
                                     names=["session", "symbol"])
    return pd.DataFrame({"gap1": rng.normal(0, 0.02, n), "short_bps": rng.normal(0, 100, n),
                         "turn_rank": np.arange(1, n + 1, dtype=float)}, index=idx)


def test_top_book_takes_the_k_best_scores_net_of_cost():
    rows = _rows()
    score = pd.Series(np.arange(30, 0, -1, dtype=float), index=rows.index)   # S0 best
    mask = rows["turn_rank"] <= 30
    got = P.top_book(rows["short_bps"], score, mask, k=4, cost=10.0)
    assert got == pytest.approx(rows["short_bps"].iloc[:4].mean() - 10.0)
    assert P.top_book(rows["short_bps"], score, rows["turn_rank"] <= 2, k=4) is None


def test_blend_follows_the_weights():
    rows = _rows()
    mask = pd.Series(True, index=rows.index)
    a = pd.Series(np.arange(30, dtype=float), index=rows.index)
    scores = {"rule": a, "ranker": -a}
    only_rule = P.blended(scores, M.MetaState(weights={"rule": 1.0, "ranker": 1e-9}), mask)
    assert only_rule.idxmax() == a.idxmax()
    only_ranker = P.blended(scores, M.MetaState(weights={"rule": 1e-9, "ranker": 1.0}), mask)
    assert only_ranker.idxmax() == a.idxmin()


def test_traded_expert_prefers_the_most_refined_list_its_record_allows():
    st = M.MetaState(guard_window=5, guard_t=-2.0)
    assert P.traded_expert(st, {"rule"}) == "rule"
    assert P.traded_expert(st, {"rule", "ranker"}) == "ranker"                 # no record yet
    assert P.traded_expert(st, {"rule", "ranker", "ranker_open"}) == "ranker_open"
    for i in range(5):     # open model clearly worse than the morning model lately
        st.track.append({"session": f"x{i}", "rule": 10.0 + i, "ranker": 60.0 + i % 2,
                         "ranker_open": -50.0 + i % 3})
    assert P.traded_expert(st, {"rule", "ranker", "ranker_open"}) == "ranker"


def test_open_rerank_keeps_the_morning_list_without_an_open_model(monkeypatch, tmp_path):
    from nse_intraday_ai import nse_preopen as PO

    monkeypatch.setattr(M, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(P.R, "load_champion", lambda name="ranker": None)
    monkeypatch.setattr(PO, "ARCHIVE_DIR", tmp_path / "preopen")

    def unavailable(*a, **k):
        raise PO.PreOpenNotReady("feed down")
    monkeypatch.setattr(PO, "wait_and_fetch", unavailable)
    assert P.open_rerank(date(2026, 9, 30), log=lambda *a: None) is None      # quietly

    snap = pd.DataFrame({"symbol": ["A"], "series": ["EQ"], "iep": [101.0], "prev_close": [100.0]})
    assert P.open_rerank(date(2026, 9, 30), preopen=snap, log=lambda *a: None) is None
    assert (tmp_path / "preopen" / "2026" / "preopen_20260930.parquet").exists()   # archived anyway
