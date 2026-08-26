import pandas as pd

from nse_intraday_ai.models import Side, StrategySignal
from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.strategies import BaseStrategy, EnsembleConfig, VotingSignalEngine


class FixedStrategy(BaseStrategy):
    def __init__(self, name, signal):
        self.name = name
        self._signal = signal

    def evaluate(self, df):
        return self._signal


def sample_frame():
    idx = pd.date_range("2026-01-01 09:15", periods=90, freq="min")
    return pd.DataFrame(
        {
            "open": [100 + i * 0.05 for i in range(90)],
            "high": [101 + i * 0.05 for i in range(90)],
            "low": [99 + i * 0.05 for i in range(90)],
            "close": [100.4 + i * 0.05 for i in range(90)],
            "volume": [100_000 + i * 1000 for i in range(90)],
        },
        index=idx,
    )


def test_ensemble_withholds_when_only_one_strategy_agrees():
    strategies = (
        FixedStrategy("one", StrategySignal("one", Side.LONG, 92, 100, 99, 103, "test")),
        FixedStrategy("two", StrategySignal("two", Side.WAIT, 0, None, None, None, "test")),
    )
    engine = VotingSignalEngine(strategies, EnsembleConfig(min_agreeing_votes=2))

    _, plan = engine.analyze("TEST.NS", sample_frame(), RiskConfig(min_confidence=70))

    assert plan.decision == "WAIT"
    assert not plan.is_actionable


def test_ensemble_allows_high_confidence_agreement_with_valid_risk():
    strategies = (
        FixedStrategy("one", StrategySignal("one", Side.LONG, 88, 100, 99, 103, "test")),
        FixedStrategy("two", StrategySignal("two", Side.LONG, 90, 100.1, 99.2, 103.2, "test")),
        FixedStrategy("three", StrategySignal("three", Side.WAIT, 0, None, None, None, "test")),
    )
    engine = VotingSignalEngine(
        strategies,
        EnsembleConfig(min_agreeing_votes=2, min_vote_share=0.60, min_weighted_confidence=75),
    )

    _, plan = engine.analyze("TEST.NS", sample_frame(), RiskConfig(min_confidence=75))

    assert plan.decision == "ACTIONABLE"
    assert plan.side == Side.LONG
    assert plan.quantity > 0
    assert plan.reward_risk >= 1.5
