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
from enum import Enum

# ── Groww rate card, per segment (2026) ────────────────────────────────────
#
# Added 2026-08-18 for the intra-week (swing) book, which holds overnight and
# therefore settles as DELIVERY, not MIS.  The difference is not a detail:
#
#   STT, intraday : 0.025% on the SELL leg only
#   STT, delivery : 0.100% on BOTH legs
#
# — a 4x jump in the dominant statutory charge, plus a per-scrip DP fee on the
# sell day that a percentage model cannot express.  A ₹1,00,000 round trip
# costs ~13 bps as MIS and ~29 bps as CNC.  Intra-week is still the better
# ratio because it is chasing whole-percent moves rather than a few bps, but
# the hurdle has to be charged honestly or the backtest is fiction.
#
# Brokerage note: Groww charges min(₹20, 0.1%) with a ₹5 floor, where the
# legacy constants above assume 0.03%.  At the position sizes this book uses
# (₹1L+) the ₹20 cap binds under both, so intraday results are unchanged; the
# rates differ only on small orders.
#
# These are published rates.  Verify against a contract note before sizing real
# money — brokers change them, and agri commodities are CTT-exempt.


class Segment(str, Enum):
    EQUITY_INTRADAY = "equity_intraday"     # MIS, squared off same day
    EQUITY_DELIVERY = "equity_delivery"     # CNC, held overnight
    COMMODITY_FUTURES = "commodity_futures" # MCX


GROWW_BROKERAGE_PCT = 0.001     # 0.1% of order value
GROWW_BROKERAGE_CAP = 20.0      # ₹ per executed order
GROWW_BROKERAGE_MIN = 5.0       # ₹ floor per executed order
DP_CHARGE_PER_SELL = 20.0       # ₹ per scrip on the day of a delivery sell
IPFT_PCT = 0.000001             # NSE investor protection fund, 0.0001%

# (stt_buy, stt_sell, exchange_txn, stamp_buy)
_SEGMENT_RATES: dict[Segment, tuple[float, float, float, float]] = {
    Segment.EQUITY_INTRADAY:   (0.0,    0.00025, 0.0000297, 0.00003),
    Segment.EQUITY_DELIVERY:   (0.001,  0.001,   0.0000297, 0.00015),
    Segment.COMMODITY_FUTURES: (0.0,    0.0001,  0.000026,  0.00002),
}

# MCX commodity transaction tax applies to non-agricultural contracts only.
AGRI_ROOTS = {"ZC", "ZS", "ZW", "CT", "SB", "KC", "CC", "LE", "HE"}


def is_agri(symbol: str) -> bool:
    return symbol.split("=")[0].upper() in AGRI_ROOTS


def _groww_brokerage(turnover: float) -> float:
    if turnover <= 0:
        return 0.0
    return min(GROWW_BROKERAGE_CAP, max(GROWW_BROKERAGE_MIN, turnover * GROWW_BROKERAGE_PCT))


def segment_round_trip_cost(
    entry_price: float,
    exit_price: float,
    quantity: int,
    *,
    segment: Segment = Segment.EQUITY_DELIVERY,
    slippage_bps_per_leg: float = 2.5,
    symbol: str = "",
) -> CostBreakdown:
    """Groww round-trip cost for one position in the given segment.

    Unlike `round_trip_cost` this charges STT on the correct legs per segment
    and adds the flat DP fee that delivery sells attract, so an intra-week hold
    is not silently priced as an intraday one.
    """
    if quantity <= 0 or entry_price <= 0:
        return CostBreakdown(0, 0, 0, 0, 0, 0, 0)

    buy_turnover = entry_price * quantity
    sell_turnover = exit_price * quantity
    total_turnover = buy_turnover + sell_turnover

    stt_buy_pct, stt_sell_pct, exch_pct, stamp_pct = _SEGMENT_RATES[segment]
    if segment is Segment.COMMODITY_FUTURES and is_agri(symbol):
        stt_sell_pct = 0.0          # agri contracts are CTT-exempt

    brokerage = _groww_brokerage(buy_turnover) + _groww_brokerage(sell_turnover)
    stt = buy_turnover * stt_buy_pct + sell_turnover * stt_sell_pct
    exchange = total_turnover * exch_pct
    sebi = total_turnover * SEBI_PCT
    ipft = total_turnover * IPFT_PCT
    stamp = buy_turnover * stamp_pct
    dp = DP_CHARGE_PER_SELL if segment is Segment.EQUITY_DELIVERY else 0.0
    gst = (brokerage + exchange + sebi + ipft + dp) * GST_PCT
    slippage = total_turnover * slippage_bps_per_leg / 1e4

    # DP and IPFT ride in the existing fields so the dataclass stays stable:
    # DP with brokerage (both are broker-side flat fees), IPFT with sebi.
    return CostBreakdown(
        brokerage=brokerage + dp,
        stt=stt,
        exchange=exchange,
        sebi=sebi + ipft,
        stamp=stamp,
        gst=gst,
        slippage=slippage,
    )


def segment_round_trip_bps(
    price: float,
    quantity: int,
    *,
    segment: Segment = Segment.EQUITY_DELIVERY,
    slippage_bps_per_leg: float = 2.5,
    symbol: str = "",
) -> float:
    """The hurdle a position must clear, in bps of entry turnover."""
    if price <= 0 or quantity <= 0:
        return 0.0
    breakdown = segment_round_trip_cost(
        price, price, quantity, segment=segment,
        slippage_bps_per_leg=slippage_bps_per_leg, symbol=symbol,
    )
    return breakdown.bps_on(price * quantity)


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
