"""Scheduled macro event-risk windows — pure clock-based, no data feed.

Certain recurring releases reliably whipsaw intraday prices: EIA crude
inventories, US jobless claims, US non-farm payrolls, and (for NSE) the
monthly F&O expiry session.  Their timing is known in advance, so the engine
can de-rate new entries inside those windows without any external feed —
which also means backtests exercise the exact same logic as live scans.

Windows open 15 minutes before the release (positioning noise) and close 30
minutes after (initial whipsaw).  Penalties are confidence points.
"""
from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from nse_intraday_ai.market_context import COMMODITY_GROUP, is_commodity_symbol

IST = ZoneInfo("Asia/Kolkata")

_PRE_MINUTES = 15
_POST_MINUTES = 30


@dataclass(frozen=True)
class EventWindow:
    label: str
    # penalty in confidence points, keyed by commodity group ("*" = any group)
    penalties: dict[str, float]


# Weekly / monthly recurring releases, IST clock.
# NOTE: US releases shift one hour with US daylight saving; 18:00 IST is the
# summer time.  A wrongly-timed window costs at most a mild penalty for 45
# minutes, so the simple fixed-clock model is acceptable.
_EIA = EventWindow("EIA crude inventories", {"energy": 12.0, "*": 4.0})
_CLAIMS = EventWindow("US jobless claims", {"*": 6.0})
_NFP = EventWindow("US non-farm payrolls", {"precious": 12.0, "*": 8.0})


def _is_first_friday(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 4 and ts.day <= 7


def _in_window(minute_of_day: int, event_minute: int) -> bool:
    return event_minute - _PRE_MINUTES <= minute_of_day <= event_minute + _POST_MINUTES


def _active_commodity_event(ts: pd.Timestamp) -> EventWindow | None:
    minute = ts.hour * 60 + ts.minute
    if ts.weekday() == 2 and _in_window(minute, 20 * 60):        # Wed 20:00 IST
        return _EIA
    if _is_first_friday(ts) and _in_window(minute, 18 * 60):     # first Fri 18:00 IST
        return _NFP
    if ts.weekday() == 3 and _in_window(minute, 18 * 60):        # Thu 18:00 IST
        return _CLAIMS
    return None


def event_risk_penalty(ts, symbol: str) -> tuple[float, str | None]:
    """Confidence penalty for entering `symbol` at time `ts` (any tz)."""
    if ts is None:
        return 0.0, None
    try:
        ts = ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)
    except (TypeError, AttributeError):
        return 0.0, None

    if is_commodity_symbol(symbol):
        event = _active_commodity_event(ts)
        if event is None:
            return 0.0, None
        group = COMMODITY_GROUP.get(symbol, "")
        penalty = event.penalties.get(group, event.penalties.get("*", 0.0))
        return penalty, event.label

    # NSE equities: F&O expiry afternoon — unwind and pinning flows distort
    # intraday signals into the close.  The expiry weekday moved Thursday →
    # Tuesday on 2025-09-02 (SEBI circular Oct-2024); this code de-rated
    # Thursdays for the whole of 2026, penalising an ordinary session and
    # leaving the real expiry unguarded.  macro_context owns the date rule so
    # backtests before the change still use Thursday.
    from nse_intraday_ai.macro_context import is_nse_monthly_expiry, is_nse_weekly_expiry

    if ts.hour >= 13:
        if is_nse_monthly_expiry(ts):
            return 6.0, "NSE monthly F&O expiry afternoon"
        if is_nse_weekly_expiry(ts):
            return 4.0, "NSE weekly F&O expiry afternoon"
    return 0.0, None


def upcoming_event(now) -> str | None:
    """Human-readable note for the sidebar: active or imminent window."""
    try:
        now = now.tz_convert(IST) if now.tzinfo else now.tz_localize(IST)
    except (TypeError, AttributeError):
        return None
    event = _active_commodity_event(now)
    if event is not None:
        return f"{event.label} window active — commodity entries de-rated"
    from nse_intraday_ai.macro_context import is_nse_monthly_expiry, is_nse_weekly_expiry

    if now.hour >= 13 and is_nse_weekly_expiry(now):
        kind = "monthly" if is_nse_monthly_expiry(now) else "weekly"
        return f"NSE {kind} F&O expiry afternoon — equity entries de-rated"
    return None
