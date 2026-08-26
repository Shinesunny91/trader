"""Size-aware slippage: what it actually costs to push a large order through a bar.

Every cost number in this repo prices slippage as a **constant**
`slippage_bps_per_leg` — 1.5 bps in the live book.  That is a defensible figure
for the order the book actually sends today: ₹3,30,000 into a name whose median
5-minute bar turns over ₹2.4 crore, which is about 1.4% of the bar.  It is
fiction for the ₹16,50,000 order that 5x intraday margin makes possible, and
the difference is not a rounding error — it is the entire measured edge.

Why this module has to exist before any leverage question can be answered:

    A constant-slippage model makes leverage look free.  Multiply the position
    by five and a constant-bps cost model multiplies gross P&L and cost by the
    same five, so the net percentage return on equity multiplies by five too.
    Nothing in the arithmetic pushes back.  Reality pushes back through market
    impact, which grows with the *square root* of participation, so the fifth
    rupee of size earns less than the first.  A leverage study run on a
    constant-slippage simulator is not a conservative estimate of the truth; it
    is a different question with a better answer.

The model is the standard square-root impact law, expressed per bar rather than
per day because a 5-minute intraday book consumes a bar, not a session:

    slippage_bps = base_bps + coefficient x sigma_bps x sqrt(participation)

where `participation` is order value / bar turnover and `sigma_bps` is the
bar's own volatility (ATR over price, in bps).  `coefficient` is the one free
parameter; the literature puts the daily-horizon version near 0.5-1.0 and this
module defaults to 1.0, the pessimistic end, because the asymmetry of the
mistake is severe: under-charging impact on a levered book does not make the
backtest slightly optimistic, it inverts the sign of the conclusion.

Calibration status: **not calibrated against this account's fills.** No
contract notes have been fed back into it, so treat the absolute level as an
assumption and the *shape* — cost rising with size — as the finding.  The
existing constant 1.5 bps is not better-grounded; it is the same assumption
with the size term set to zero.
"""
from __future__ import annotations

# The default 5-minute participation ceiling.  At 1% of a bar's turnover the
# square-root term is roughly the size of the base spread cost, which is about
# where a market order stops being a price-taker and starts being the price.
MAX_PARTICIPATION_PCT = 1.0

# Pessimistic end of the published range for the square-root law.
DEFAULT_IMPACT_COEFFICIENT = 1.0


def participation(position_value: float, bar_turnover: float) -> float:
    """Order value as a fraction of the bar's own turnover.

    Returns 0.0 when turnover is unknown or non-positive, which makes the
    impact term vanish — the caller is then back to the constant model and
    should know it.  `bar_turnover` of zero is a real occurrence in the cache
    (a bar with no trades), not a data error to raise on.
    """
    if bar_turnover <= 0 or position_value <= 0:
        return 0.0
    return position_value / bar_turnover


def impact_bps(
    position_value: float,
    bar_turnover: float,
    sigma_bps: float,
    *,
    coefficient: float = DEFAULT_IMPACT_COEFFICIENT,
) -> float:
    """Market impact of one leg, in bps of the order's own value.

    `sigma_bps` is the bar's volatility in bps — ATR/price x 1e4 is the
    measure the rest of the codebase already carries (`atr_bps`).
    """
    if coefficient <= 0 or sigma_bps <= 0:
        return 0.0
    rate = participation(position_value, bar_turnover)
    if rate <= 0:
        return 0.0
    return coefficient * sigma_bps * rate**0.5


def slippage_bps_for(
    position_value: float,
    bar_turnover: float,
    sigma_bps: float,
    *,
    base_bps: float,
    coefficient: float = DEFAULT_IMPACT_COEFFICIENT,
) -> float:
    """Total per-leg slippage: the constant spread cost plus size impact.

    With `coefficient=0` this returns `base_bps` exactly, which is what every
    existing study assumed — so the old results are the special case of this
    one, not a different model.
    """
    return base_bps + impact_bps(
        position_value, bar_turnover, sigma_bps, coefficient=coefficient
    )


def max_position_for_participation(
    bar_turnover: float, max_participation_pct: float = MAX_PARTICIPATION_PCT
) -> float:
    """Largest order value that stays within `max_participation_pct` of a bar.

    This is the constraint that actually binds a large account, and it binds
    *per name*, not per portfolio: ₹50,00,000 of buying power is not a position
    size, it is a budget to spread across names whose bars can absorb it.
    Returns infinity when turnover is unknown, so a missing-liquidity path
    degrades to the previous unconstrained behaviour rather than silently
    refusing every trade.
    """
    if bar_turnover <= 0 or max_participation_pct <= 0:
        return float("inf")
    return bar_turnover * max_participation_pct / 100.0
