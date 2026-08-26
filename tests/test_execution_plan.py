import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai.execution_plan import (
    MAX_CONCURRENT,
    MAX_TRADES_PER_DAY,
    STOP_ATR,
    TARGET_ATR,
    build_execution_plan,
    expectancy_note,
)


def plan(**overrides):
    kwargs = dict(
        symbol="RELIANCE.NS", side="LONG", signal_price=1900.0, atr=6.0,
        capital=10_00_000.0,
    )
    kwargs.update(overrides)
    return build_execution_plan(**kwargs)


# ── sizing ────────────────────────────────────────────────────────────────────

def test_position_respects_the_max_position_cap():
    ticket = plan()
    assert ticket.tradable
    assert ticket.position_value <= 10_00_000 * 0.33 + 1900


def test_risk_matches_the_stop_distance():
    ticket = plan()
    assert ticket.stop_distance == pytest.approx(STOP_ATR * 6.0)
    assert ticket.target_distance == pytest.approx(TARGET_ATR * 6.0)
    assert ticket.risk_rupees == pytest.approx(ticket.quantity * ticket.stop_distance)


def test_reward_to_risk_matches_the_configured_multiple():
    ticket = plan()
    expected = TARGET_ATR / STOP_ATR
    assert ticket.reward_rupees / ticket.risk_rupees == pytest.approx(expected)
    assert expected > 1.0, "the target must sit further away than the stop"


def test_small_account_is_refused_rather_than_sized_down():
    """Below the cost floor the flat ₹20 legs eat the edge; refuse instead."""
    ticket = plan(capital=50_000.0)
    assert not ticket.tradable and "cost floor" in ticket.note
    assert ticket.quantity == 0


def test_bigger_positions_cost_fewer_bps():
    small = plan(capital=5_00_000.0)
    large = plan(capital=50_00_000.0)
    assert small.tradable and large.tradable
    assert large.est_cost_bps < small.est_cost_bps


# ── guards ────────────────────────────────────────────────────────────────────

def test_daily_cap_is_enforced_when_one_is_set(monkeypatch):
    import nse_intraday_ai.execution_plan as module

    monkeypatch.setattr(module, "MAX_TRADES_PER_DAY", 3)
    assert not module.build_execution_plan(
        symbol="RELIANCE.NS", side="LONG", signal_price=1900.0, atr=6.0,
        capital=10_00_000.0, taken_today=3,
    ).tradable


def test_zero_cap_means_unlimited_not_refuse_everything(monkeypatch):
    """0 disables the cap; it must not disable ticketing."""
    import nse_intraday_ai.execution_plan as module

    monkeypatch.setattr(module, "MAX_TRADES_PER_DAY", 0)
    ticket = module.build_execution_plan(
        symbol="RELIANCE.NS", side="LONG", signal_price=1900.0, atr=6.0,
        capital=10_00_000.0, taken_today=25,
    )
    assert ticket.tradable, "an uncapped book must still issue tickets"


def test_concurrency_cap_is_enforced():
    ticket = plan(already_open=MAX_CONCURRENT)
    assert not ticket.tradable and "already open" in ticket.note


def test_missing_atr_is_refused():
    assert not plan(atr=0.0).tradable
    assert not plan(atr=-1.0).tradable


def test_wait_side_is_refused():
    assert not plan(side="WAIT").tradable


# ── the ticket text ───────────────────────────────────────────────────────────

def test_ticket_says_next_bar_open_not_a_fixed_entry_price():
    """The signal bar's close is gone; quoting it as an entry is the bug."""
    text = plan().order_ticket()
    assert "NEXT 5-minute bar's open" in text
    assert "1900.00" not in text, "must not present the stale close as the entry"


def test_ticket_quotes_stop_relative_to_the_fill():
    text = plan().order_ticket()
    assert f"fill − ₹{STOP_ATR * 6.0:.2f}" in text
    assert f"fill + ₹{TARGET_ATR * 6.0:.2f}" in text


def test_short_ticket_uses_the_right_verbs_and_signs():
    text = plan(side="SHORT").order_ticket()
    assert text.startswith("SELL SHORT")
    assert "BUY TO COVER" in text
    assert f"fill + ₹{STOP_ATR * 6.0:.2f}" in text   # stop sits above the fill on a short
    assert f"fill − ₹{TARGET_ATR * 6.0:.2f}" in text


def test_ticket_carries_the_square_off_and_time_stop():
    text = plan().order_ticket()
    assert "15:15" in text and "60 min" in text


def test_refused_ticket_explains_itself():
    text = plan(capital=50_000.0).order_ticket()
    assert text.startswith("NO TRADE —") and len(text) > 20


def test_expectancy_note_reports_every_gate_not_just_the_best():
    note = expectancy_note()
    for value in ("-0.17%", "-0.55%", "-2.82%"):
        assert value in note, f"{value} missing — the note must not cherry-pick"
    assert "not a profit forecast" in note


def test_expectancy_note_defers_to_a_model_note_when_one_is_installed():
    note = expectancy_note("MODEL SAYS SOMETHING")
    assert "MODEL SAYS SOMETHING" in note
    assert "None of the rule-based gates was profitable" in note
