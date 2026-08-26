import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from nse_intraday_ai.indicators import add_indicators
from nse_intraday_ai.models import Side, StrategySignal, TradePlan
from nse_intraday_ai.scanner import ScanResult
from nse_intraday_ai.signal_model import extract_live_features_row, score_and_rank_scan_results

IST = ZoneInfo("Asia/Kolkata")


def _build_test_scan_result(symbol: str, side: str = "LONG", conf: float = 75.0) -> ScanResult:
    idx = pd.date_range("2026-08-25 09:15", periods=30, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(30)],
            "high": [102.0 + i for i in range(30)],
            "low": [99.0 + i for i in range(30)],
            "close": [101.0 + i for i in range(30)],
            "volume": [50_000 + i * 500 for i in range(30)],
        },
        index=idx,
    )
    df = add_indicators(df)
    plan_side = Side.LONG if side == "LONG" else Side.SHORT
    plan = TradePlan(
        symbol=symbol,
        side=plan_side,
        confidence=conf,
        entry=float(df["close"].iloc[-1]),
        stop_loss=float(df["close"].iloc[-1] - 2.0),
        target=float(df["close"].iloc[-1] + 5.0),
        quantity=50,
        risk_amount=100.0,
        reward_amount=250.0,
        reward_risk=2.5,
        decision="ACTIONABLE",
        reasons=("test reason",),
        strategy_votes=(StrategySignal(strategy="trend_momentum", side=plan_side, confidence=conf, entry=float(df["close"].iloc[-1]), stop_loss=float(df["close"].iloc[-1] - 2.0), target=float(df["close"].iloc[-1] + 5.0), reason="test", weight=1.0),),
        timestamp="2026-08-25 10:00:00",
    )
    return ScanResult(symbol=symbol, plan=plan, last_close=float(df["close"].iloc[-1]), frame=df)


def test_extract_live_features_row():
    res = _build_test_scan_result("RELIANCE.NS", "LONG", 80.0)
    feats = extract_live_features_row(res)
    assert feats is not None
    assert feats["symbol"] == "RELIANCE.NS"
    assert feats["side"] == "LONG"
    assert feats["rsi"] > 0
    assert "clv" in feats
    assert "clv_flow" in feats
    assert "vol_z" in feats
    assert "ext_vwap" in feats


def test_score_and_rank_scan_results():
    r1 = _build_test_scan_result("TCS.NS", "LONG", 70.0)
    r2 = _build_test_scan_result("INFY.NS", "LONG", 85.0)
    r3 = _build_test_scan_result("HDFCBANK.NS", "SHORT", 78.0)

    scored = score_and_rank_scan_results([r1, r2, r3])
    assert len(scored) == 3
    for s in scored:
        assert s.rank_score is not None
        assert isinstance(s.rank_score, float)
