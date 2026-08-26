"""Meta-label model: feature parity, persistence, veto behavior, training."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai.backtest import CandidateEvent, _candidate_vote_stats
from nse_intraday_ai.meta_model import (
    FEATURE_NAMES,
    MetaModel,
    extract_features,
    features_from_event,
    features_from_plan,
    load_if_available,
    model_path,
    plan_vote_stats,
    train_meta_model,
)
from nse_intraday_ai.models import Side, StrategySignal, TradePlan


def _vote(strategy, side, confidence=70.0, weight=1.0, entry=100.0):
    return StrategySignal(
        strategy=strategy, side=side, confidence=confidence,
        entry=entry, stop_loss=99.0, target=102.0,
        reason="test", weight=weight,
    )


def _plan(side=Side.LONG, confidence=72.0, ts="2026-07-06T13:35:00+05:30"):
    votes = (
        _vote("trend_continuation", side, 70.0, 1.2),
        _vote("ema_scalp", side, 60.0, 0.8),
        _vote("vwap_mean_reversion", Side.SHORT if side == Side.LONG else Side.LONG, 65.0, 1.0),
        _vote("supertrend", side, 40.0, 1.0),          # below 55-conf cut: not counted
        StrategySignal("rsi_divergence", Side.WAIT, 80.0, None, None, None, "wait"),
    )
    return TradePlan(
        symbol="TEST.NS", side=side, confidence=confidence,
        entry=100.0, stop_loss=99.2, target=101.6,
        quantity=10, risk_amount=8.0, reward_amount=16.0, reward_risk=2.0,
        decision="ACTIONABLE",
        reasons=("2 strategies agree on LONG; weighted vote share 0.55.",
                 "Market regime: TRENDING_UP | Session multiplier: 1.06x."),
        strategy_votes=votes,
        timestamp=ts,
    )


def _event_from_plan(plan, ts):
    count, share, names = _candidate_vote_stats(plan)
    return CandidateEvent(
        timestamp=ts, symbol=plan.symbol, side=plan.side,
        entry=float(plan.entry), stop_loss=float(plan.stop_loss), target=float(plan.target),
        confidence=plan.confidence, reward_risk=plan.reward_risk,
        agreeing_count=count, vote_share=share,
        exit_time=ts, exit_price=101.0, exit_reason="TARGET",
        per_share_pnl=1.0, reasons=plan.reasons, agreeing_strategies=names,
    )


class TestFeatureParity:
    def test_plan_vote_stats_matches_backtest(self):
        plan = _plan()
        assert plan_vote_stats(plan) == _candidate_vote_stats(plan)

    def test_live_and_event_features_identical(self):
        """The live scan path must score exactly what training scored."""
        plan = _plan()
        ts = pd.Timestamp(plan.timestamp)
        event = _event_from_plan(plan, ts)
        assert features_from_plan(plan) == pytest.approx(features_from_event(event))

    def test_feature_vector_length_matches_names(self):
        assert len(features_from_plan(_plan())) == len(FEATURE_NAMES)

    def test_plan_without_levels_returns_none(self):
        plan = _plan()
        plan = TradePlan(**{**plan.__dict__, "entry": None})
        assert features_from_plan(plan) is None

    def test_regime_and_session_extracted_from_reasons(self):
        features = extract_features(
            confidence=70, reward_risk=1.5, agreeing_count=2, vote_share=0.6,
            side=Side.LONG, timestamp=datetime(2026, 7, 6, 10, 0),
            entry=100.0, stop_loss=99.0, target=102.0,
            reasons=("Market regime: HIGH_VOL | Session multiplier: 0.85x.",),
            agreeing_strategies=("ema_scalp",),
        )
        named = dict(zip(FEATURE_NAMES, features))
        assert named["HIGH_VOL"] == 1.0 and named["TRENDING_UP"] == 0.0
        assert named["sess_mult"] == pytest.approx(0.85)
        assert named["ema_scalp"] == 1.0 and named["supertrend"] == 0.0


def _synthetic_events(n=600, seed=7):
    """Signals where LONG morning trades win and SHORT afternoon trades lose."""
    rng = np.random.default_rng(seed)
    events = []
    base = pd.Timestamp("2026-05-04 09:30:00+05:30")
    for i in range(n):
        day, minute = divmod(i, 12)
        ts = base + pd.Timedelta(days=day % 40, minutes=25 * minute)
        long_side = rng.random() < 0.5
        good = long_side and ts.hour < 12
        gross = (rng.normal(40, 20) if good else rng.normal(-15, 20)) / 1e4 * 100
        events.append(CandidateEvent(
            timestamp=ts, symbol="SYN.NS", side=Side.LONG if long_side else Side.SHORT,
            entry=100.0, stop_loss=99.3, target=101.4,
            confidence=float(rng.uniform(60, 95)), reward_risk=2.0,
            agreeing_count=2, vote_share=0.7,
            exit_time=ts + pd.Timedelta(minutes=30), exit_price=100.0 + gross,
            exit_reason="TIME_EXIT", per_share_pnl=gross,
            reasons=("Market regime: RANGING | Session multiplier: 1.00x.",),
            agreeing_strategies=("trend_continuation", "ema_scalp"),
        ))
    return events


class TestTrainingAndPersistence:
    def test_train_learns_planted_structure(self):
        events = _synthetic_events()
        model = train_meta_model(events, universe="test", cost_bps=6.0, gates=(60, 1.0, 1))
        good = [e for e in events if e.side == Side.LONG and e.timestamp.hour < 12]
        bad = [e for e in events if e.side == Side.SHORT and e.timestamp.hour >= 12]
        good_scores = [model.score(features_from_event(e)) for e in good[:50]]
        bad_scores = [model.score(features_from_event(e)) for e in bad[:50]]
        assert np.mean(good_scores) > np.mean(bad_scores) + 0.1

    def test_save_load_roundtrip_scores_identically(self, tmp_path):
        model = train_meta_model(_synthetic_events(), universe="test",
                                 cost_bps=6.0, gates=(60, 1.0, 1))
        path = tmp_path / "meta_model_test.json"
        model.save(path)
        loaded = MetaModel.load(path)
        features = features_from_plan(_plan())
        assert loaded.score(features) == pytest.approx(model.score(features))
        assert loaded.taus == model.taus

    def test_load_rejects_stale_feature_schema(self, tmp_path):
        model = train_meta_model(_synthetic_events(), universe="test",
                                 cost_bps=6.0, gates=(60, 1.0, 1))
        model.feature_names = model.feature_names[:-1]
        path = tmp_path / "meta_model_test.json"
        model.save(path)
        with pytest.raises(ValueError, match="different features"):
            MetaModel.load(path)

    def test_load_if_available_missing_returns_none(self, tmp_path):
        assert load_if_available("nope", tmp_path) is None
        assert model_path("nope", tmp_path).name == "meta_model_nope.json"

    def test_veto_thresholds_are_ordered_and_score_in_range(self):
        model = train_meta_model(_synthetic_events(), universe="test",
                                 cost_bps=6.0, gates=(60, 1.0, 1))
        assert model.taus["0.3"] <= model.taus["0.5"] <= model.taus["0.7"]
        score = model.score(features_from_plan(_plan()))
        assert 0.0 < score < 1.0
        assert model.passes(features_from_plan(_plan()), 0.5) == (score >= model.taus["0.5"])
        with pytest.raises(KeyError):
            model.tau(0.9)
