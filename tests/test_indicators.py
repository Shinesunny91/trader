import pandas as pd

from nse_intraday_ai.indicators import add_indicators


def test_add_indicators_returns_expected_columns():
    idx = pd.date_range("2026-01-01 09:15", periods=80, freq="min")
    df = pd.DataFrame(
        {
            "open": [100 + i * 0.1 for i in range(80)],
            "high": [101 + i * 0.1 for i in range(80)],
            "low": [99 + i * 0.1 for i in range(80)],
            "close": [100.5 + i * 0.1 for i in range(80)],
            "volume": [100_000 + i * 100 for i in range(80)],
        },
        index=idx,
    )

    out = add_indicators(df)

    expected_cols = [
        "ema_9", "ema_21", "ema_50", "rsi_14", "atr_14", "vwap", "volume_z",
        "clv", "clv_flow", "streak", "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "poc", "vah", "val", "cpr_p", "cpr_bc", "cpr_tc", "regime",
    ]
    for column in expected_cols:
        assert column in out.columns
    assert out["atr_14"].iloc[-1] > 0
    assert 0 <= out["rsi_14"].iloc[-1] <= 100
    assert -1.0 <= out["clv"].iloc[-1] <= 1.0
    assert out["poc"].iloc[-1] > 0
    assert out["vah"].iloc[-1] >= out["val"].iloc[-1]
