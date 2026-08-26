import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai.costs import (
    min_position_for_cost_target,
    round_trip_bps,
    round_trip_cost,
)


def test_flat_brokerage_makes_small_positions_expensive():
    """The whole point of the model: cost per rupee falls as size rises."""
    small = round_trip_bps(1000, 25)     # ₹25,000
    large = round_trip_bps(1000, 500)    # ₹5,00,000
    assert small > large
    assert small > 14 and large < 11


def test_brokerage_caps_at_twenty_per_leg():
    breakdown = round_trip_cost(1000, 1000, 5000)   # ₹50L turnover per leg
    assert breakdown.brokerage == 40.0              # 2 legs x ₹20 cap


def test_brokerage_is_percentage_below_the_cap():
    # ₹30,000 per leg -> 0.03% = ₹9, below the ₹20 cap
    breakdown = round_trip_cost(1000, 1000, 30)
    assert breakdown.brokerage == 18.0


def test_stt_is_charged_on_the_sell_side_only():
    breakdown = round_trip_cost(100, 200, 1000)   # sell turnover 2x buy
    assert breakdown.stt == 200 * 1000 * 0.00025
    assert breakdown.stamp == 100 * 1000 * 0.00003


def test_extra_legs_cost_more():
    two = round_trip_cost(1000, 1010, 200, legs=2).total
    three = round_trip_cost(1000, 1010, 200, legs=3).total
    assert three > two, "a partial exit is an extra order and must cost more"


def test_min_position_for_cost_target_is_monotone():
    tight = min_position_for_cost_target(1000, 8.0)
    loose = min_position_for_cost_target(1000, 15.0)
    assert tight > loose
    # A target below the pure-percentage floor is unreachable at any size.
    assert min_position_for_cost_target(1000, 1.0) == float("inf")


def test_zero_quantity_is_free():
    assert round_trip_cost(1000, 1010, 0).total == 0.0
    assert round_trip_bps(1000, 0) == 0.0
