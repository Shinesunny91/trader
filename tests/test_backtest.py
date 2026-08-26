import pandas as pd

from nse_intraday_ai.backtest import BacktestConfig, CandidateEvent, run_backtest, run_backtest_from_events
from nse_intraday_ai.data import DataResult
from nse_intraday_ai.models import Side
from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.strategies import EnsembleConfig


def trending_frame(start=100.0, minutes=180):
    idx = pd.date_range("2026-01-01 09:15", periods=minutes, freq="min", tz="Asia/Kolkata")
    close = [start + i * 0.08 for i in range(minutes)]
    return pd.DataFrame(
        {
            "open": [value - 0.02 for value in close],
            "high": [value + 0.25 for value in close],
            "low": [value - 0.25 for value in close],
            "close": close,
            "volume": [100_000 + i * 2000 for i in range(minutes)],
        },
        index=idx,
    )


def test_backtest_returns_summary_and_frames():
    data = {"TEST.NS": DataResult("TEST.NS", trending_frame(), "test")}

    summary, trades, equity = run_backtest(
        data=data,
        risk_config=RiskConfig(min_confidence=55, min_reward_risk=0.8),
        ensemble_config=EnsembleConfig(min_agreeing_votes=1, min_vote_share=0.5, min_weighted_confidence=55),
        config=BacktestConfig(
            starting_capital=100_000,
            cooldown_minutes=60,
            warmup_candles=80,
            max_hold_minutes=30,
        ),
    )

    assert summary.starting_capital == 100_000
    assert summary.ending_capital >= 0
    assert not equity.empty
    assert len(trades) <= 2


def event_at(timestamp: str, per_share_pnl: float) -> CandidateEvent:
    ts = pd.Timestamp(timestamp, tz="Asia/Kolkata")
    return CandidateEvent(
        timestamp=ts,
        symbol="TEST.NS",
        side=Side.LONG,
        entry=100,
        stop_loss=99,
        target=102,
        confidence=90,
        reward_risk=2,
        agreeing_count=1,
        vote_share=1,
        exit_time=ts + pd.Timedelta(minutes=5),
        exit_price=100 + per_share_pnl,
        exit_reason="STOP_LOSS" if per_share_pnl < 0 else "TARGET",
        per_share_pnl=per_share_pnl,
        reasons=("test event",),
        agreeing_strategies=("test_strategy",),
    )


def test_backtest_loss_controls_skip_late_and_after_daily_loss():
    late_summary, late_trades, _ = run_backtest_from_events(
        events=[event_at("2026-01-01 15:10", 2)],
        starting_capital=100_000,
        risk_per_trade_pct=1,
        max_position_pct=100,
        cooldown_minutes=0,
        min_confidence=80,
        min_reward_risk=1.5,
        min_agreeing_votes=1,
        min_vote_share=0.5,
        config=BacktestConfig(no_new_trade_after="15:00"),
    )
    assert late_summary.trades == 0
    assert late_trades.empty

    loss_limited_summary, loss_limited_trades, _ = run_backtest_from_events(
        events=[
            event_at("2026-01-01 10:00", -1),
            event_at("2026-01-01 10:10", 2),
        ],
        starting_capital=100_000,
        risk_per_trade_pct=1,
        max_position_pct=100,
        cooldown_minutes=0,
        min_confidence=80,
        min_reward_risk=1.5,
        min_agreeing_votes=1,
        min_vote_share=0.5,
        config=BacktestConfig(no_new_trade_after="", daily_loss_limit_pct=0.5),
    )
    assert loss_limited_summary.trades == 1
    assert loss_limited_summary.pnl < 0
    assert len(loss_limited_trades) == 1
