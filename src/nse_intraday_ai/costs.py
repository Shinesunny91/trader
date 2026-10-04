"""Actual NSE intraday equity transaction costs, not a flat bps guess.

The engine has been charging a flat 15 bps commission + 3 bps slippage on
every round trip.  That is roughly right for a ₹50,000 position and roughly
double the truth for a ₹2,00,000 one, because the dominant term — discount
broker brokerage — is a **flat ₹20 per order**, not a percentage.  Since the
measured gross edge on the best signal subset is only a few bps, getting this
wrong is the difference between a strategy that is hopeless and one that is
merely marginal, and it changes position sizing: bigger positions amortise the
flat fee, so the cost curve argues for fewer, larger trades.

Charge structure (NSE equity intraday / MIS, 2026):

  brokerage      min(₹20, 0.03% of turnover) per executed order
  STT            0.025% of turnover, SELL side only
  exchange txn   0.00297% of turnover, both sides
  SEBI turnover  0.0001% of turnover, both sides
  stamp duty     0.003% of turnover, BUY side only
  GST            18% on (brokerage + exchange txn + SEBI)

Slippage is modelled separately and is not a tax: it scales with how much of
the bar's liquidity the order consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

# Rates as fractions of turnover.
BROKERAGE_PCT = 0.0003          # 0.03%
BROKERAGE_CAP = 20.0            # ₹ per executed order
STT_SELL_PCT = 0.00025          # 0.025%, sell side only
EXCHANGE_TXN_PCT = 0.0000297    # 0.00297%, both sides
SEBI_PCT = 0.000001             # 0.0001%, both sides
STAMP_BUY_PCT = 0.00003         # 0.003%, buy side only
GST_PCT = 0.18


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage + self.stt + self.exchange
            + self.sebi + self.stamp + self.gst + self.slippage
        )

    def bps_on(self, entry_turnover: float) -> float:
        return self.total / entry_turnover * 1e4 if entry_turnover else 0.0


def _brokerage(turnover: float) -> float:
    return min(BROKERAGE_CAP, turnover * BROKERAGE_PCT)


def round_trip_cost(
    entry_price: float,
    exit_price: float,
    quantity: int,
    *,
    slippage_bps_per_leg: float = 2.5,
    legs: int = 2,
) -> CostBreakdown:
    """Full statutory + brokerage + slippage cost of one intraday round trip.

    `legs` counts *executed orders*, so scaling out of a position in two
    tranches is 3 legs, not 2 — each tranche pays its own ₹20 floor.  That is
    exactly why partial exits are not free, and why the simulator has to model
    them rather than assume a single fill.
    """
    if quantity <= 0 or entry_price <= 0:
        return CostBreakdown(0, 0, 0, 0, 0, 0, 0)

    buy_turnover = entry_price * quantity
    sell_turnover = exit_price * quantity
    total_turnover = buy_turnover + sell_turnover

    # Flat ₹20 applies per order; with more legs the same turnover is split
    # across more orders, so charge the per-leg minimum on each slice.
    per_leg_turnover = total_turnover / max(legs, 2)
    brokerage = _brokerage(per_leg_turnover) * max(legs, 2)

    stt = sell_turnover * STT_SELL_PCT
    exchange = total_turnover * EXCHANGE_TXN_PCT
    sebi = total_turnover * SEBI_PCT
    stamp = buy_turnover * STAMP_BUY_PCT
    gst = (brokerage + exchange + sebi) * GST_PCT
    slippage = total_turnover * slippage_bps_per_leg / 1e4

    return CostBreakdown(brokerage, stt, exchange, sebi, stamp, gst, slippage)


def round_trip_bps(
    price: float, quantity: int, *, slippage_bps_per_leg: float = 2.5, legs: int = 2
) -> float:
    """Round-trip cost in bps of entry turnover, assuming a flat exit price.

    This is what the signal engine needs *before* it knows the exit: the
    hurdle a trade has to clear to be worth taking at this position size.
    """
    if price <= 0 or quantity <= 0:
        return 0.0
    breakdown = round_trip_cost(
        price, price, quantity, slippage_bps_per_leg=slippage_bps_per_leg, legs=legs
    )
    return breakdown.bps_on(price * quantity)


def min_position_for_cost_target(
    price: float, target_bps: float, *, slippage_bps_per_leg: float = 2.5
) -> float:
    """Smallest position value whose round-trip cost is within `target_bps`.

    Below this size the flat ₹20 legs dominate and the trade cannot clear its
    own costs no matter how good the signal is — the sizing rule should refuse
    the trade rather than take it small.
    """
    # Percentage-only component (everything except the flat brokerage floor).
    pct_component = (
        STT_SELL_PCT / 2 + EXCHANGE_TXN_PCT + SEBI_PCT + STAMP_BUY_PCT / 2
    ) * 1e4 + 2 * slippage_bps_per_leg
    pct_component *= 1 + 0.0  # GST on txn/SEBI is negligible at this precision
    headroom_bps = target_bps - pct_component
    if headroom_bps <= 0:
        return float("inf")
    # Two flat legs of ₹20 plus GST on them.
    flat = 2 * BROKERAGE_CAP * (1 + GST_PCT)
    return flat / (headroom_bps / 1e4)
