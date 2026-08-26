from nse_intraday_ai.learning import ShadowLearner
from nse_intraday_ai.models import Side, StrategySignal, TradePlan
from nse_intraday_ai.scanner import ScanResult


def make_result():
    vote = StrategySignal(
        strategy="trend",
        side=Side.LONG,
        confidence=80,
        entry=100,
        stop_loss=99,
        target=102,
        reason="test",
    )
    plan = TradePlan(
        symbol="TEST.NS",
        side=Side.WAIT,
        confidence=80,
        entry=None,
        stop_loss=None,
        target=None,
        quantity=0,
        risk_amount=0,
        reward_amount=0,
        reward_risk=0,
        decision="WAIT",
        reasons=("blocked by strict threshold",),
        strategy_votes=(vote,),
        timestamp="2026-01-01T09:15:00+05:30",
    )
    return ScanResult(
        symbol="TEST.NS",
        plan=plan,
        rows=100,
        source="test",
        last_close=100,
        public_ltp=100,
    )


def test_shadow_learner_records_evaluates_and_estimates(tmp_path):
    learner = ShadowLearner(tmp_path / "learner.sqlite3")
    result = make_result()

    inserted = learner.record_scan_results([result], min_confidence=55, enforce_session=False)
    evaluated = learner.evaluate_pending({"TEST.NS": 101}, horizon_minutes=0, enforce_session=False)
    learner.refresh_policy()
    estimate = learner.estimate_for_result(result, min_samples=1)

    assert inserted == 1
    assert evaluated == 1
    assert estimate.is_trained
    assert estimate.samples == 1
    assert estimate.avg_reward_bps > 0
    assert estimate.confidence_bonus > 0


# ── Per-universe policy (2026-08-17) ────────────────────────────────────────
# Before this, the state key had no universe component and both scanners wrote
# to one table: 121,023 NSE observations averaging -1.8 bps set the verdict for
# 3,365 commodity observations averaging +4.7 bps, vetoing 94.6% of futures
# signals whose blocked subset had actually returned +4.98 bps.

from datetime import datetime
from zoneinfo import ZoneInfo

from nse_intraday_ai.learning import (
    ASSUMED_COST_BPS, COMMODITY_COST_BPS, cost_bps_for, is_commodity,
)

IST = ZoneInfo("Asia/Kolkata")


def test_commodity_and_equity_never_share_a_policy_arm(tmp_path):
    learner = ShadowLearner(tmp_path / "s.sqlite3", candle_cache=None)
    ts = datetime(2026, 8, 17, 10, 30, tzinfo=IST)
    common = dict(side=Side.LONG, confidence=72.0, reward_risk=1.8, vote_count=2, ts=ts)
    equity = learner._state_key_from_features(**common, symbol="RELIANCE.NS")
    future = learner._state_key_from_features(**common, symbol="GC=F")
    assert equity != future
    assert future.startswith("C|")
    # the equity key keeps its historical shape so the existing table stays valid
    assert equity == "LONG|conf70|rr1.5|votes2|mid"


def test_futures_are_charged_futures_costs():
    assert cost_bps_for("GC=F") == COMMODITY_COST_BPS == 8.0
    assert cost_bps_for("RELIANCE.NS") == ASSUMED_COST_BPS == 18.0
    assert is_commodity("CL=F") and not is_commodity("INFY.NS")


def test_futures_use_a_24h_clock_not_the_nse_session(tmp_path):
    """The NSE buckets filed 14:00 IST -> 09:59 next day, 20 of 24 hours and the
    whole US overlap, into one 'close' bucket shared with Indian equities."""
    learner = ShadowLearner(tmp_path / "s.sqlite3", candle_cache=None)
    at = lambda h, m=0: datetime(2026, 8, 17, h, m, tzinfo=IST)
    buckets = {learner._tod_bucket(at(h), symbol="CL=F") for h in (2, 7, 11, 16, 21)}
    assert len(buckets) == 5, "each futures session phase needs its own arm"
    # equities keep the NSE windows
    assert learner._tod_bucket(at(9, 30), symbol="INFY.NS") == "open"
    assert learner._tod_bucket(at(21), symbol="INFY.NS") == "close"


def test_purge_keeps_futures_observations_outside_nse_hours(tmp_path):
    """`purge_and_migrate` DISCARDED 9,385 commodity rows in 30 days by applying
    the equity clock to 24-hour futures."""
    learner = ShadowLearner(tmp_path / "s.sqlite3", candle_cache=None)
    evening = datetime(2026, 8, 17, 21, 0, tzinfo=IST).isoformat()
    with learner._connect() as conn:
        for symbol in ("CL=F", "RELIANCE.NS"):
            conn.execute(
                "INSERT INTO shadow_observations (observed_at, symbol, side, confidence,"
                " reward_risk, entry_price, vote_count, state_key, status, reward_bps,"
                " outcome, reasons_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (evening, symbol, "LONG", 75.0, 1.8, 100.0, 2, "k", "EVALUATED", 25.0, "WIN", "[]"),
            )
    learner.purge_and_migrate()
    with learner._connect() as conn:
        status = dict(conn.execute("SELECT symbol, status FROM shadow_observations").fetchall())
    assert status["CL=F"] == "EVALUATED", "a 21:00 IST crude bar is a real observation"
    assert status["RELIANCE.NS"] == "DISCARDED", "NSE is shut at 21:00"
