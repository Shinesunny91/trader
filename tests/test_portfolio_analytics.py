import pandas as pd
import numpy as np

from nse_intraday_ai.execution_plan import (
    build_execution_plan,
    kelly_position_size,
    STOP_ATR,
    TARGET_ATR,
    BREAKEVEN_ATR,
    TRAIL_LOCK_TRIGGER_ATR,
    TRAIL_LOCK_DISTANCE_ATR,
)
from nse_intraday_ai.portfolio_sim import (
    SimConfig,
    SimResult,
    IntradayPortfolioSimulator,
    monte_carlo_permutation,
)


def test_execution_plan_ratchets_and_kelly():
    plan = build_execution_plan(
        symbol="SBIN.NS",
        side="LONG",
        signal_price=800.0,
        atr=8.0,
        capital=1_000_000.0,
        risk_per_trade_pct=1.0,
        predicted_net_bps=28.5,
        model_rank=1,
    )
    assert plan.tradable is True
    assert plan.stop_distance == STOP_ATR * 8.0
    assert plan.target_distance == TARGET_ATR * 8.0
    assert "lock stop" in plan.trail_lock_at
    assert plan.predicted_net_bps == 28.5
    assert plan.model_rank == 1

    ticket_text = plan.order_ticket()
    assert "profit lock" in ticket_text
    assert "model edge" in ticket_text

    k_size = kelly_position_size(1_000_000.0, win_rate=0.50, win_loss_payoff=2.0, kelly_multiplier=0.5)
    assert 5_000.0 <= k_size <= 50_000.0


def test_sim_result_advanced_metrics():
    dates = pd.date_range("2026-08-01", periods=10, freq="D")
    trades_data = {
        "entry_time": dates,
        "exit_time": dates,
        "symbol": ["TCS.NS"] * 10,
        "side": ["LONG"] * 10,
        "entry": [3500.0] * 10,
        "exit": [3550.0] * 10,
        "quantity": [50] * 10,
        "gross_pnl": [2500.0 if i % 3 != 0 else -1500.0 for i in range(10)],
        "costs": [100.0] * 10,
        "net_pnl": [2400.0 if i % 3 != 0 else -1600.0 for i in range(10)],
        "turnover": [350000.0] * 10,
        "bars_held": [5] * 10,
        "atr": [20.0] * 10,
        "exit_reason": ["TARGET"] * 10,
        "note": [""] * 10,
    }
    trades_df = pd.DataFrame(trades_data)
    equity_df = pd.DataFrame({
        "timestamp": dates,
        "equity": 1_000_000.0 + np.cumsum(trades_df["net_pnl"]),
    })
    daily_df = pd.DataFrame({
        "pnl": trades_df["net_pnl"].to_numpy(),
        "trades": [1] * 10,
    }, index=dates.date)

    cfg = SimConfig()
    result = SimResult(
        trades=trades_df,
        equity=equity_df,
        daily=daily_df,
        config=cfg,
        starting_capital=1_000_000.0,
        ending_capital=float(equity_df["equity"].iloc[-1]),
    )

    assert result.win_rate == 60.0
    assert result.profit_factor > 1.0
    assert result.expectancy_rupees > 0
    assert result.expectancy_bps > 0
    assert result.sharpe_ratio != 0
    assert result.sortino_ratio != 0
    assert result.calmar_ratio > 0
    assert 0 <= result.kelly_fraction <= 1.0

    summary_str = result.summary()
    assert "daily Sharpe" in summary_str
    assert "Sortino" in summary_str
    assert "Calmar" in summary_str
    assert "Kelly frac" in summary_str


def test_monte_carlo_permutation():
    trades = pd.DataFrame({
        "net_pnl": [1500.0, -800.0, 2200.0, -900.0, 1800.0, -750.0, 3100.0, -1100.0, 1400.0, -600.0],
    })
    mc = monte_carlo_permutation(trades, starting_capital=1_000_000.0, num_paths=200, random_seed=42)
    assert mc["num_paths"] == 200
    assert mc["p50_equity"] > 1_000_000.0
    assert mc["p5_equity"] <= mc["p50_equity"] <= mc["p95_equity"]
    assert 0.0 <= mc["max_dd_p95_pct"] <= 100.0
    assert 0.0 <= mc["risk_of_ruin_pct"] <= 100.0
    assert len(mc["percentile_curves"]["p50"]) == 11
