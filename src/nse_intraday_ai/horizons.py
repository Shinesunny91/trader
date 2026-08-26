"""Four trading horizons as one comparable object.

The workspace grew a separate engine per horizon — a 5-minute intraday scanner
and a daily swing book — with different costs, different labels and different
validation.  That made them impossible to compare, which matters because the
single most important fact about this system is *how the edge/cost ratio changes
with holding period*:

    hold        round trip      typical move        ratio
    minutes     ~10 bps (MIS)   ~5 bps              hopeless
    5 sessions  ~37 bps (CNC)   ~300 bps            plausible
    21 sessions ~37 bps (CNC)   ~700 bps            better
    250 sessions ~37 bps (CNC)  ~2500 bps           dominated by beta

Costs are roughly *fixed per round trip*; the move being chased grows with the
square root of time.  So the cost drag falls monotonically with horizon while
the thing you must predict gets harder to time.  Somewhere in the middle is the
best trade-off, and the only way to find it is to measure all four the same way.

Each horizon below therefore fixes: the bar interval, the hold in sessions, the
settlement segment (which decides STT), a stop width appropriate to the holding
period, and how many independent observations a decade of data actually
provides — the last is the honest sample-size ceiling and is printed with every
result.
"""
from __future__ import annotations

from dataclasses import dataclass

from nse_intraday_ai.costs import Segment


@dataclass(frozen=True)
class Horizon:
    key: str
    label: str
    interval: str            # bar size the decision is made on
    hold_sessions: int       # 1 = close the same day
    stop_atr: float
    target_atr: float        # 0 = no target, exit on time
    rebalance_every: int     # sessions between decisions
    segment_equity: Segment
    note: str

    @property
    def overnight(self) -> bool:
        return self.hold_sessions > 1

    def segment_for(self, universe: str) -> Segment:
        if universe == "commodity":
            return Segment.COMMODITY_FUTURES
        return self.segment_equity

    def independent_obs(self, sessions: int = 2500) -> int:
        """Roughly how many non-overlapping trades a decade of data supports.

        A 250-session hold over ten years is *ten* independent observations, not
        2,500.  Reporting a Sharpe or a t-statistic off overlapping windows is
        the most common way a long-horizon backtest lies, so the count is
        carried alongside every result rather than left implicit.
        """
        return max(1, sessions // max(self.hold_sessions, 1))


INTRADAY = Horizon(
    key="intraday",
    label="Intra-day",
    interval="5m",
    hold_sessions=1,
    stop_atr=2.0,
    target_atr=5.0,
    rebalance_every=1,
    segment_equity=Segment.EQUITY_INTRADAY,
    note="MIS, squared off at 15:15. Cheapest per trade, but the measured gross "
         "edge is a few bps against a ~10 bps round trip.",
)

INTRAWEEK = Horizon(
    key="intraweek",
    label="Intra-week",
    interval="1d",
    hold_sessions=5,
    stop_atr=2.5,
    target_atr=0.0,
    rebalance_every=5,
    segment_equity=Segment.EQUITY_DELIVERY,
    note="Enter Monday, exit Friday. Delivery settlement, so STT is 0.1% on "
         "both legs — the round trip roughly triples versus MIS.",
)

INTRAMONTH = Horizon(
    key="intramonth",
    label="Intra-month",
    interval="1d",
    hold_sessions=21,
    stop_atr=3.0,
    target_atr=0.0,
    rebalance_every=21,
    segment_equity=Segment.EQUITY_DELIVERY,
    note="~One calendar month. Same fixed cost spread over a move four times "
         "larger, which is where the arithmetic starts to work.",
)

INTRAYEAR = Horizon(
    key="intrayear",
    label="Intra-year",
    interval="1d",
    hold_sessions=250,
    stop_atr=4.0,
    target_atr=0.0,
    rebalance_every=63,      # review quarterly even though the hold is a year
    segment_equity=Segment.EQUITY_DELIVERY,
    note="A year-long position. Costs are negligible here, but so is the "
         "active decision — this horizon competes directly with buy and hold, "
         "and a decade supplies only ~10 independent observations.",
)

ALL: dict[str, Horizon] = {
    h.key: h for h in (INTRADAY, INTRAWEEK, INTRAMONTH, INTRAYEAR)
}

ORDER = ["intraday", "intraweek", "intramonth", "intrayear"]


def get(key: str) -> Horizon:
    if key not in ALL:
        raise ValueError(f"unknown horizon {key!r}; expected one of {ORDER}")
    return ALL[key]
