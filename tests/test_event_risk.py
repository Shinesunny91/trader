import pandas as pd

from nse_intraday_ai.event_risk import event_risk_penalty, upcoming_event


def _ist(value):
    return pd.Timestamp(value, tz="Asia/Kolkata")


def test_eia_window_hits_energy_hardest():
    # Wednesday 2026-07-01 20:10 IST — inside the EIA window
    penalty, label = event_risk_penalty(_ist("2026-07-01 20:10"), "CL=F")
    assert penalty == 12.0 and "EIA" in label
    gold_penalty, _ = event_risk_penalty(_ist("2026-07-01 20:10"), "GC=F")
    assert gold_penalty == 4.0
    # outside the window
    assert event_risk_penalty(_ist("2026-07-01 21:00"), "CL=F")[0] == 0.0


def test_nfp_first_friday_only():
    # 2026-07-03 is the first Friday of July
    penalty, label = event_risk_penalty(_ist("2026-07-03 18:05"), "GC=F")
    assert penalty == 12.0 and "payrolls" in label
    # 2026-07-10 is the second Friday — no NFP window
    assert event_risk_penalty(_ist("2026-07-10 18:05"), "GC=F")[0] == 0.0


def test_jobless_claims_thursday():
    penalty, label = event_risk_penalty(_ist("2026-07-02 18:00"), "NG=F")
    assert penalty == 6.0 and "claims" in label


def test_equities_expiry_afternoon_only():
    # NSE expiry moved Thursday -> Tuesday on 2025-09-02.
    # 2026-06-30 is the last Tuesday of June -> monthly expiry.
    assert event_risk_penalty(_ist("2026-06-30 14:00"), "RELIANCE.NS")[0] == 6.0
    assert event_risk_penalty(_ist("2026-06-30 10:00"), "RELIANCE.NS")[0] == 0.0
    # 2026-06-23 is an ordinary Tuesday -> weekly expiry, smaller de-rate
    assert event_risk_penalty(_ist("2026-06-23 14:00"), "RELIANCE.NS")[0] == 4.0
    # Thursday is no longer an NSE expiry day
    assert event_risk_penalty(_ist("2026-06-25 14:00"), "RELIANCE.NS")[0] == 0.0
    # US release windows are after NSE close — equities unaffected
    assert event_risk_penalty(_ist("2026-07-01 20:10"), "RELIANCE.NS")[0] == 0.0


def test_expiry_weekday_history_respects_the_rule_change():
    from nse_intraday_ai.macro_context import nse_expiry_weekday
    from datetime import date

    assert nse_expiry_weekday(date(2025, 6, 1)) == 3    # Thursday, pre-change
    assert nse_expiry_weekday(date(2026, 6, 1)) == 1    # Tuesday, post-change


def test_upcoming_event_note():
    assert "EIA" in upcoming_event(_ist("2026-07-01 20:10"))
    assert upcoming_event(_ist("2026-07-01 12:00")) is None
