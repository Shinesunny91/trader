"""Session-phase, overnight and calendar context for NSE intraday signals.

`market_context.py` already fetches global instruments (S&P futures, Nikkei,
Hang Seng, DAX, DXY, USDINR, VIX).  What it did *not* model is **when those
instruments are actually informative**.  A Nikkei "1-hour change" read at
14:00 IST is three hours stale — Tokyo shut at 11:30 IST — while the DAX only
starts printing at 12:30 IST and is the dominant global driver of the NSE
afternoon.  Weighting all of them equally all day mixes live signal with
frozen numbers.

This module supplies, as pure functions of an IST timestamp:

* which global sessions are **live**, and per-region weights that decay once a
  region closes (so a closed market contributes its *level*, not a fake
  momentum reading);
* the **overnight block** — what happened between yesterday's NSE close and
  today's open (US cash session, Asia's morning) — which is what actually
  prices the gap;
* **calendar** state: NSE F&O expiry (Tuesday since 2025-09-02), month/quarter
  end, day of week, and the monsoon/results seasonality markers.

Everything is derived from a timestamp plus cached frames, so backtests and
live scans compute identical values.

Session map (IST), which is what the weights encode:

    05:30-08:00, 09:00-11:30   Tokyo        (JST 09:00-15:00)
    07:00-09:30, 10:30-13:30   Hong Kong    (HKT 09:30-16:00)
    09:15-15:30                NSE
    12:30-21:00                Frankfurt    (CET 09:00-17:30)
    19:00-01:30                US cash      (ET 09:30-16:00)
    ~24h                       ES=F, DXY, USDINR, crude
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

# NSE derivatives expiry moved Thursday -> Tuesday on 2025-09-02 (SEBI circular
# Oct-2024); BSE Sensex moved to Thursday.  Expiry-day flows distort intraday
# equity signals, so the engine needs the *current* day, not the historical one.
NSE_EXPIRY_WEEKDAY = 1          # Monday=0 ... Tuesday=1
NSE_EXPIRY_CHANGE_DATE = date(2025, 9, 2)
_LEGACY_EXPIRY_WEEKDAY = 3      # Thursday, for backtests before the change

# Region session windows in IST minutes-of-day, as (start, end) pairs.
_SESSIONS: dict[str, tuple[tuple[int, int], ...]] = {
    "japan":  ((5 * 60 + 30, 8 * 60), (9 * 60, 11 * 60 + 30)),
    "china":  ((7 * 60, 9 * 60 + 30), (10 * 60 + 30, 13 * 60 + 30)),
    "europe": ((12 * 60 + 30, 21 * 60),),
    "us":     ((19 * 60, 24 * 60), (0, 1 * 60 + 30)),
}

# How long after a region closes its last momentum reading stays usable.
_DECAY_MINUTES = 90


def _minute_of_day(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def _to_ist(ts) -> pd.Timestamp | None:
    if ts is None:
        return None
    try:
        ts = pd.Timestamp(ts)
    except Exception:
        return None
    try:
        return ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)
    except (TypeError, ValueError):
        return None


def is_session_live(region: str, ts) -> bool:
    ist = _to_ist(ts)
    if ist is None:
        return False
    minute = _minute_of_day(ist)
    return any(start <= minute < end for start, end in _SESSIONS.get(region, ()))


def minutes_since_close(region: str, ts) -> float | None:
    """Minutes since `region` last closed; 0.0 while it is open, None if unknown."""
    ist = _to_ist(ts)
    if ist is None:
        return None
    minute = _minute_of_day(ist)
    windows = _SESSIONS.get(region, ())
    if not windows:
        return None
    if any(start <= minute < end for start, end in windows):
        return 0.0
    past_closes = [end for _, end in windows if end <= minute]
    if past_closes:
        return float(minute - max(past_closes))
    # Region has not opened yet today — measure from yesterday's last close.
    return float(minute + 24 * 60 - max(end for _, end in windows))


def region_weight(region: str, ts) -> float:
    """Confidence a region's *momentum* reading deserves at this IST time.

    1.0 while its session is live, decaying linearly to 0 over the 90 minutes
    after it closes.  A closed market's last hour of trading is not news any
    more — that information is already in the NSE tape.
    """
    gap = minutes_since_close(region, ts)
    if gap is None:
        return 0.0
    if gap <= 0:
        return 1.0
    return max(0.0, 1.0 - gap / _DECAY_MINUTES)


@dataclass(frozen=True)
class SessionPhase:
    """Which external markets are steering the tape right now."""

    name: str
    japan: float
    china: float
    europe: float
    us: float

    @property
    def any_live(self) -> bool:
        return max(self.japan, self.china, self.europe, self.us) > 0


def session_phase(ts) -> SessionPhase:
    ist = _to_ist(ts)
    if ist is None:
        return SessionPhase("unknown", 0.0, 0.0, 0.0, 0.0)
    minute = _minute_of_day(ist)
    if minute < 9 * 60 + 15:
        name = "pre_open"
    elif minute < 11 * 60 + 30:
        name = "nse_asia"          # Tokyo/HK still trading alongside NSE
    elif minute < 12 * 60 + 30:
        name = "nse_gap"           # Asia gone, Europe not yet — thinnest window
    elif minute < 15 * 60 + 30:
        name = "nse_europe"        # Frankfurt open, US pre-market building
    else:
        name = "post_close"
    return SessionPhase(
        name,
        region_weight("japan", ist),
        region_weight("china", ist),
        region_weight("europe", ist),
        region_weight("us", ist),
    )


def nse_expiry_weekday(on: date | None = None) -> int:
    """The NSE F&O expiry weekday in force on `on` (Tuesday since 2025-09-02)."""
    if on is not None and on < NSE_EXPIRY_CHANGE_DATE:
        return _LEGACY_EXPIRY_WEEKDAY
    return NSE_EXPIRY_WEEKDAY


def is_nse_weekly_expiry(ts) -> bool:
    ist = _to_ist(ts)
    return ist is not None and ist.weekday() == nse_expiry_weekday(ist.date())


def is_nse_monthly_expiry(ts) -> bool:
    """Last expiry-weekday of the calendar month."""
    ist = _to_ist(ts)
    if ist is None or ist.weekday() != nse_expiry_weekday(ist.date()):
        return False
    return (ist + pd.Timedelta(days=7)).month != ist.month


@dataclass(frozen=True)
class CalendarContext:
    weekday: int                 # Monday=0
    is_weekly_expiry: bool
    is_monthly_expiry: bool
    is_expiry_eve: bool
    is_month_end: bool           # last 2 trading-ish days of the month
    is_quarter_end: bool
    month: int
    # Indian seasonality markers, useful mainly as model features rather than
    # hand-tuned rules: Q1/Q3 results season, and the monsoon window that
    # drives FMCG/agri/fertiliser flow.
    is_results_season: bool
    is_monsoon: bool

    def as_features(self) -> dict[str, float]:
        return {
            "cal_dow": self.weekday / 4.0,
            "cal_weekly_expiry": float(self.is_weekly_expiry),
            "cal_monthly_expiry": float(self.is_monthly_expiry),
            "cal_expiry_eve": float(self.is_expiry_eve),
            "cal_month_end": float(self.is_month_end),
            "cal_quarter_end": float(self.is_quarter_end),
            "cal_results_season": float(self.is_results_season),
            "cal_monsoon": float(self.is_monsoon),
        }


def calendar_context(ts) -> CalendarContext:
    ist = _to_ist(ts)
    if ist is None:
        return CalendarContext(0, False, False, False, False, False, 1, False, False)
    expiry_wd = nse_expiry_weekday(ist.date())
    days_in_month = ist.days_in_month
    return CalendarContext(
        weekday=ist.weekday(),
        is_weekly_expiry=ist.weekday() == expiry_wd,
        is_monthly_expiry=is_nse_monthly_expiry(ist),
        is_expiry_eve=ist.weekday() == (expiry_wd - 1) % 7,
        is_month_end=ist.day >= days_in_month - 2,
        is_quarter_end=ist.month in (3, 6, 9, 12) and ist.day >= days_in_month - 2,
        month=ist.month,
        # Indian listed companies report Q1 in Jul-Aug, Q2 Oct-Nov, Q3 Jan-Feb,
        # Q4 Apr-May; the first half of those months carries the density.
        is_results_season=(ist.month in (1, 4, 7, 10) and ist.day >= 10)
        or (ist.month in (2, 5, 8, 11) and ist.day <= 14),
        is_monsoon=ist.month in (6, 7, 8, 9),
    )


# ── Overnight block ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OvernightContext:
    """What happened between yesterday's NSE close and today's open."""

    prev_close: float | None = None
    today_open: float | None = None
    gap_pct: float | None = None          # index gap, % of previous close
    # Overnight moves of the instruments that trade while NSE is shut.
    us_overnight_pct: float | None = None  # S&P futures, NSE close -> now
    asia_morning_pct: float | None = None  # Nikkei/HSI since their open
    dxy_overnight_pct: float | None = None
    usdinr_overnight_pct: float | None = None
    crude_overnight_pct: float | None = None
    # Has the opening gap been filled yet today?
    gap_filled: bool | None = None

    def as_features(self) -> dict[str, float]:
        def clip(value, lo=-3.0, hi=3.0):
            return 0.0 if value is None else float(max(lo, min(hi, value)))

        return {
            "on_gap_pct": clip(self.gap_pct),
            "on_us_pct": clip(self.us_overnight_pct),
            "on_asia_pct": clip(self.asia_morning_pct),
            "on_dxy_pct": clip(self.dxy_overnight_pct, -1.5, 1.5),
            "on_usdinr_pct": clip(self.usdinr_overnight_pct, -1.5, 1.5),
            "on_crude_pct": clip(self.crude_overnight_pct, -5.0, 5.0),
            "on_gap_filled": -1.0 if self.gap_filled is None else float(self.gap_filled),
        }


def _session_bounds(frame: pd.DataFrame, day) -> pd.DataFrame:
    return frame[frame.index.normalize() == day]


def overnight_change_pct(frame: pd.DataFrame | None, ts, *, since_hour: int = 15) -> float | None:
    """Percent change of `frame` from ~yesterday's NSE close to `ts`.

    Used for instruments that keep trading overnight (ES=F, DXY, USDINR,
    crude): the move NSE has *not* priced in yet.
    """
    if frame is None or frame.empty:
        return None
    ist = _to_ist(ts)
    if ist is None:
        return None
    frame = frame[frame.index <= ist]
    if len(frame) < 2:
        return None
    anchor = ist.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=since_hour, minutes=30)
    prior = frame[frame.index <= anchor]
    if prior.empty:
        return None
    ref = float(prior["close"].iloc[-1])
    if ref == 0:
        return None
    return float((float(frame["close"].iloc[-1]) / ref - 1.0) * 100)


def build_overnight_context(
    *,
    index_frame: pd.DataFrame | None,
    ts,
    es_frame: pd.DataFrame | None = None,
    asia_frame: pd.DataFrame | None = None,
    dxy_frame: pd.DataFrame | None = None,
    usdinr_frame: pd.DataFrame | None = None,
    crude_frame: pd.DataFrame | None = None,
) -> OvernightContext:
    """Assemble the overnight block causally as of `ts`.

    All frames are OHLCV with tz-aware index and lowercase columns.  Anything
    missing simply stays None rather than defaulting to zero, so a missing feed
    is distinguishable from a flat market.
    """
    ist = _to_ist(ts)
    if ist is None:
        return OvernightContext()

    prev_close = today_open = gap_pct = None
    gap_filled = None
    if index_frame is not None and not index_frame.empty:
        hist = index_frame[index_frame.index <= ist]
        if not hist.empty:
            today = ist.normalize()
            session = _session_bounds(hist, today)
            earlier = hist[hist.index.normalize() < today]
            if not earlier.empty:
                prev_close = float(earlier["close"].iloc[-1])
            if not session.empty:
                today_open = float(session["open"].iloc[0])
            if prev_close and today_open:
                gap_pct = (today_open / prev_close - 1.0) * 100
                if gap_pct > 0:
                    gap_filled = bool(float(session["low"].min()) <= prev_close)
                elif gap_pct < 0:
                    gap_filled = bool(float(session["high"].max()) >= prev_close)
                else:
                    gap_filled = True

    return OvernightContext(
        prev_close=prev_close,
        today_open=today_open,
        gap_pct=gap_pct,
        us_overnight_pct=overnight_change_pct(es_frame, ist),
        asia_morning_pct=overnight_change_pct(asia_frame, ist),
        dxy_overnight_pct=overnight_change_pct(dxy_frame, ist),
        usdinr_overnight_pct=overnight_change_pct(usdinr_frame, ist),
        crude_overnight_pct=overnight_change_pct(crude_frame, ist),
        gap_filled=gap_filled,
    )


def phase_weighted_global_risk(
    ts,
    *,
    es_change_pct: float | None,
    japan_change_pct: float | None = None,
    china_change_pct: float | None = None,
    europe_change_pct: float | None = None,
) -> float:
    """Global risk appetite in [-1, +1], weighted by which sessions are live.

    The flat-weight version in `market_context.global_risk_score` treats a
    Nikkei reading at 14:00 IST (Tokyo closed 2.5h earlier) the same as at
    10:00 IST.  Here each regional momentum term is scaled by `region_weight`,
    and the surviving weights are renormalised, so the score always reflects
    markets that are actually trading.  S&P futures carry a floor weight
    because they trade nearly around the clock.
    """
    phase = session_phase(ts)
    terms: list[tuple[float, float]] = []   # (weight, saturated value)
    if es_change_pct is not None:
        # ES=F trades ~23h; weight it up while US cash is open.
        weight = 0.45 + 0.25 * phase.us
        terms.append((weight, max(-1.0, min(1.0, es_change_pct / 0.5))))
    if japan_change_pct is not None and phase.japan > 0:
        terms.append((0.20 * phase.japan, max(-1.0, min(1.0, japan_change_pct / 0.8))))
    if china_change_pct is not None and phase.china > 0:
        terms.append((0.20 * phase.china, max(-1.0, min(1.0, china_change_pct / 0.8))))
    if europe_change_pct is not None and phase.europe > 0:
        terms.append((0.25 * phase.europe, max(-1.0, min(1.0, europe_change_pct / 0.6))))
    if not terms:
        return 0.0
    total = sum(weight for weight, _ in terms)
    if total <= 0:
        return 0.0
    score = sum(weight * value for weight, value in terms) / total
    # Renormalising to the surviving weights would make a single stale-ish
    # region as loud as a full panel; damp by how much of the panel is present.
    coverage = min(1.0, total / 0.9)
    return max(-1.0, min(1.0, score * coverage))
