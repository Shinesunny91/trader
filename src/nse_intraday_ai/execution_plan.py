"""Turn a signal into the exact order ticket the simulator would trade.

This module exists because of a gap that only matters when someone trades the
screen with real money. The recommendation table shows `plan.entry`,
`plan.stop_loss`, `plan.target` and `plan.quantity` — but:

* `plan.entry` is the **close of the signal bar**, a price that has already
  gone. You cannot buy at it. The validated simulation fills at the *next*
  bar's open and measures its results from there.
* `plan.stop_loss` / `plan.target` come from each strategy's own
  `_plan_levels` (1.5x ATR stop, 1.8 reward multiple, then min/max across
  agreeing strategies). The portfolio study validated a **1.5 ATR stop and
  3.0 ATR target with a breakeven ratchet at 0.9 ATR** — a different trade.
* `plan.quantity` is sized off `RiskConfig.capital`, which defaults to
  ₹1,00,000 in `scan_config`, not the account being traded.

Trading the screen while believing the simulation therefore means trading a
setup that was never tested. This module removes the gap: one function that
produces the ticket, used by both the app and the simulator, so what is shown
is what was measured.

It also attaches the honest expectancy, because a ticket without one invites
the reader to supply their own.
"""
from __future__ import annotations

from dataclasses import dataclass

from nse_intraday_ai.costs import min_position_for_cost_target, round_trip_bps

# The exit design validated by the portfolio studies (scripts/sim_sweep.py,
# sim_exit_sweep.py): wider stops beat tight ones, scaling out costs more than
# it saves, and a 60-minute cap beat holding longer.
#
# Stop widened 1.5 -> 2.0 ATR on 2026-08-12. The diagnosis came from a losing
# session where the *direction was right*: the book shorted CARTRADE and
# KPITTECH, both fell ~1.8%, and both were stopped on the entry bar before the
# move arrived. Sweeping stop width over the model-ranked book confirmed it was
# systematic rather than one bad day — averaged across target multiples,
# 1.5 ATR returned +4.39%, 2.0 ATR +4.86%, 2.5 ATR +4.73%, 3.0 ATR +2.67%, and
# the wider stop *reduced* max drawdown (2.18% -> 1.66%) because fewer correct
# trades were shaken out. Note this is a sweep over the same 34 sessions, so
# treat the exact cell as approximate and the direction of the effect as the
# finding.
STOP_ATR = 2.0
TARGET_ATR = 5.0
BREAKEVEN_ATR = 1.2
TRAIL_LOCK_TRIGGER_ATR = 2.5
TRAIL_LOCK_DISTANCE_ATR = 1.5
MAX_HOLD_MINUTES = 60
SQUARE_OFF = "15:15"
# 1 trade/day since 2026-08-17.  0 (uncapped) was set on 2026-08-12 at the
# account owner's explicit instruction after being shown 3/day vs uncapped; the
# four sessions that followed lost -1.71% cumulatively.  What nobody had tested
# was a *tighter* cap than 3.
MAX_TRADES_PER_DAY = 1
MAX_CONCURRENT = 3
MAX_POSITION_PCT = 33.0
COST_TARGET_BPS = 12.0


def kelly_position_size(
    capital: float,
    win_rate: float = 0.45,
    win_loss_payoff: float = 2.2,
    kelly_multiplier: float = 0.5,
) -> float:
    """Calculate fractional Kelly allocation for risk budgeting."""
    if win_loss_payoff <= 0 or win_rate <= 0:
        return 0.01 * capital
    k = win_rate - (1.0 - win_rate) / win_loss_payoff
    k_adj = max(0.005, min(0.05, k * kelly_multiplier))  # conservative 0.5% to 5% risk per trade
    return capital * k_adj


@dataclass(frozen=True)
class ExecutionPlan:
    """A complete, actionable order ticket."""

    symbol: str
    side: str                    # LONG | SHORT
    signal_price: float          # close of the signal bar — reference only
    reference_atr: float
    quantity: int
    position_value: float
    stop_distance: float
    target_distance: float
    stop_from_fill: str          # human-readable rule, since the fill is unknown
    target_from_fill: str
    breakeven_at: str
    trail_lock_at: str
    max_hold_minutes: int
    square_off_at: str
    est_cost_bps: float
    est_cost_rupees: float
    risk_rupees: float
    reward_rupees: float
    tradable: bool
    note: str
    predicted_net_bps: float | None = None
    model_rank: int | None = None

    def order_ticket(self) -> str:
        """The instruction a human can actually follow at the terminal."""
        if not self.tradable:
            return f"NO TRADE — {self.note}"
        verb = "BUY" if self.side == "LONG" else "SELL SHORT"
        cover = "SELL" if self.side == "LONG" else "BUY TO COVER"
        lines = [
            f"{verb} {self.quantity} {self.symbol} at market on the NEXT 5-minute bar's open",
            f"  position   ≈ ₹{self.position_value:,.0f}",
            f"  stop-loss    {self.stop_from_fill}   (risk ≈ ₹{self.risk_rupees:,.0f})",
            f"  target       {self.target_from_fill}   (reward ≈ ₹{self.reward_rupees:,.0f})",
            f"  breakeven    {self.breakeven_at}",
            f"  profit lock  {self.trail_lock_at}",
            f"  {cover} on stop, target, after {self.max_hold_minutes} min, "
            f"or at {self.square_off_at} — whichever comes first",
            f"  est. round-trip cost ₹{self.est_cost_rupees:,.0f} ({self.est_cost_bps:.1f} bps)",
        ]
        if self.predicted_net_bps is not None:
            lines.append(f"  model edge   {self.predicted_net_bps:+.1f} net bps (Rank #{self.model_rank or 1})")
        return "\n".join(lines)


def build_execution_plan(
    *,
    symbol: str,
    side: str,
    signal_price: float,
    atr: float,
    capital: float,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = MAX_POSITION_PCT,
    slippage_bps_per_leg: float = 1.5,
    already_open: int = 0,
    taken_today: int = 0,
    predicted_net_bps: float | None = None,
    model_rank: int | None = None,
) -> ExecutionPlan:
    """Size and specify the trade exactly as the validated simulation would.

    Distances are quoted **from the fill**, not from the signal price, because
    the fill is the next bar's open and is not known yet. Quoting a fixed stop
    price off a stale close is how a 1.5 ATR stop silently becomes a 0.4 ATR one.
    """
    def refuse(note: str) -> ExecutionPlan:
        return ExecutionPlan(
            symbol=symbol, side=side, signal_price=signal_price, reference_atr=atr,
            quantity=0, position_value=0.0, stop_distance=0.0, target_distance=0.0,
            stop_from_fill="—", target_from_fill="—", breakeven_at="—", trail_lock_at="—",
            max_hold_minutes=MAX_HOLD_MINUTES, square_off_at=SQUARE_OFF,
            est_cost_bps=0.0, est_cost_rupees=0.0, risk_rupees=0.0, reward_rupees=0.0,
            tradable=False, note=note, predicted_net_bps=predicted_net_bps, model_rank=model_rank,
        )

    if side not in ("LONG", "SHORT"):
        return refuse("no directional signal")
    if not (atr > 0 and signal_price > 0):
        return refuse("no usable ATR at the signal bar")
    # MAX_TRADES_PER_DAY == 0 means unlimited, not "refuse everything".
    if 0 < MAX_TRADES_PER_DAY <= taken_today:
        return refuse(
            f"daily cap reached ({MAX_TRADES_PER_DAY} trades) — taking more was "
            f"measurably worse in the study, not better"
        )
    if already_open >= MAX_CONCURRENT:
        return refuse(f"{MAX_CONCURRENT} positions already open")

    stop_distance = STOP_ATR * atr
    target_distance = TARGET_ATR * atr

    by_risk = (capital * risk_per_trade_pct / 100) / stop_distance
    by_position = (capital * max_position_pct / 100) / signal_price
    quantity = int(min(by_risk, by_position))
    if quantity <= 0:
        return refuse("capital too small for one share at this stop distance")

    position_value = quantity * signal_price
    floor = min_position_for_cost_target(
        signal_price, COST_TARGET_BPS, slippage_bps_per_leg=slippage_bps_per_leg
    )
    if position_value < floor:
        return refuse(
            f"position ₹{position_value:,.0f} is below the ₹{floor:,.0f} cost floor — "
            f"flat brokerage would eat more than {COST_TARGET_BPS:.0f} bps"
        )

    cost_bps = round_trip_bps(signal_price, quantity, slippage_bps_per_leg=slippage_bps_per_leg)
    sign = "−" if side == "LONG" else "+"
    tgt_sign = "+" if side == "LONG" else "−"
    return ExecutionPlan(
        symbol=symbol, side=side, signal_price=signal_price, reference_atr=atr,
        quantity=quantity, position_value=position_value,
        stop_distance=stop_distance, target_distance=target_distance,
        stop_from_fill=f"fill {sign} ₹{stop_distance:,.2f}  ({STOP_ATR}x ATR)",
        target_from_fill=f"fill {tgt_sign} ₹{target_distance:,.2f}  ({TARGET_ATR}x ATR)",
        breakeven_at=f"move stop to fill once price reaches fill {tgt_sign} ₹{BREAKEVEN_ATR * atr:,.2f}",
        trail_lock_at=f"lock stop at fill {tgt_sign} ₹{TRAIL_LOCK_DISTANCE_ATR * atr:,.2f} once price reaches fill {tgt_sign} ₹{TRAIL_LOCK_TRIGGER_ATR * atr:,.2f}",
        max_hold_minutes=MAX_HOLD_MINUTES, square_off_at=SQUARE_OFF,
        est_cost_bps=cost_bps, est_cost_rupees=position_value * cost_bps / 1e4,
        risk_rupees=quantity * stop_distance, reward_rupees=quantity * target_distance,
        tradable=True, note="",
        predicted_net_bps=predicted_net_bps,
        model_rank=model_rank,
    )


# ── Honest expectancy, attached to every ticket ───────────────────────────────

# Measured by scripts/sim_final.py over 49 sessions (2026-05-13..08-12) on the
# top-150 liquid names with this exact configuration.  Update whenever that
# study is re-run; never quote a single half of the window.
MEASURED = {
    "sessions": 49,
    "engine_gate_pct": -0.17,
    "quality_gate_pct": -0.55,
    "full_gate_pct": -2.82,
    "win_rate": 26.2,
    "profit_factor": 0.80,
}

COST_FLOOR = {
    "total_bps": 10.10,
    "slippage_bps": 5.00,
    "regulatory_bps": 5.10,
}

AUDIT = {
    "gross_bps": 0.81,
    "cost_bps": 10.10,
    "gross_bps_wide": 1.46,
    "features_clearing_cost": 0,
    "features_tested": 41,
}

AUDIT_COMMODITY = {
    "gross_bps": -1.02,
    "cost_bps": 8.43,
    "gross_bps_wide": -1.08,
    "features_clearing_cost": 0,
    "features_tested": 41,
}

SELECTIVITY = {
    "net_bps_all_trades": -9.52,
    "net_bps_best_cell": -4.44,
    "best_cell": "p_rf top 2% per session",
    "cells_positive_both_halves": 0,
    "cells_tested": 36,
}

DRIFT_REFINEMENTS = {
    "volume_filter": {
        "rule": "top gap decile, volume in the top third of that decile",
        "names_per_day": 14,
        "annualised_pct": 35.5,
        "sharpe": 1.58,
        "max_drawdown_pct": 21.0,
        "worst_day_pct": -11.1,
        "single_name_session_pct": 0.0,
        "adopted": True,
    },
    "sector_filter": {
        "rule": "...and the gap disagrees with its sector's gap",
        "names_per_day": 4,
        "annualised_pct": 42.8,
        "sharpe": 1.23,
        "max_drawdown_pct": 36.0,
        "worst_day_pct": -19.2,
        "single_name_session_pct": 22.8,
        "adopted": False,
    },
}

DRIFT_TIMING = {
    "same_day_gross_pct": 0.005,
    "same_day_t": 0.2,
    "next_day_gross_pct": 0.256,
    "next_day_t": 11.0,
    "hold_5d_gross_pct": -0.132,
    "hold_10d_gross_pct": -0.680,
}

OVERNIGHT_DRIFT = {
    "gross_pct_day": 0.258,
    "net_pct_day": 0.157,
    "annualised_pct": 39.5,
    "sessions": 2450,
    "years_positive": 11,
    "years_tested": 11,
    "t_stat": 11.0,
    "artifact_pct_day": 0.602,
    "oos_annualised_pct": 31.3,
    "beta_share": 0.48,
    "max_drawdown_pct": 34.4,
    "cells_scanned": 28,
}

FORWARD_TEST = {
    "rule": "ridge score above trailing p99, one trade per session",
    "in_sample_bps": 67.44,
    "in_sample_sessions": 14,
    "forward_bps": -23.51,
    "forward_trades": 4,
    "cutoff": "2026-08-17",
}

GO_LIVE_GATE = {
    "min_sessions": 60,
    "min_net_bps": 2.0,
    "max_p_value": 0.01,
    "require_both_halves_positive": True,
    "max_cells_scanned": 12,
}


def break_even_win_rate(target_bps: float, stop_bps: float, cost_bps: float | None = None) -> float:
    """Win rate a target/stop pair needs before it earns anything, as a fraction."""
    if target_bps <= 0 or stop_bps < 0:
        raise ValueError("target must be positive and stop non-negative")
    cost = COST_FLOOR["total_bps"] if cost_bps is None else cost_bps
    return (stop_bps + cost) / (target_bps + stop_bps)


def intraday_go_live(
    *,
    net_bps: float,
    sessions: int,
    p_value: float,
    half1_bps: float,
    half2_bps: float,
    cells_scanned: int,
) -> tuple[bool, dict[str, bool]]:
    """Decide whether an intraday candidate may trade real money."""
    g = GO_LIVE_GATE
    checks = {
        f"net edge >= {g['min_net_bps']} bps": net_bps >= g["min_net_bps"],
        f"sessions >= {g['min_sessions']}": sessions >= g["min_sessions"],
        f"p <= {g['max_p_value']}": p_value <= g["max_p_value"],
        "both halves positive": (half1_bps > 0 and half2_bps > 0)
        if g["require_both_halves_positive"]
        else True,
        f"scan width <= {g['max_cells_scanned']}": cells_scanned <= g["max_cells_scanned"],
    }
    return all(checks.values()), checks


def edge_deficit_bps() -> float:
    """How far the population's gross edge falls short of paying for the trade."""
    return AUDIT["cost_bps"] - AUDIT["gross_bps"]


def expectancy_note(model_note: str | None = None) -> str:
    """One paragraph the recommendation screen must carry, unedited.

    A ticket with sizing and targets reads as a forecast.  It is not one.
    """
    m = MEASURED
    base = (
        f"Measured over {m['sessions']} sessions on the top-150 liquid names with this "
        f"exact trade structure: engine gates {m['engine_gate_pct']:+.2f}%, "
        f"conviction gate {m['quality_gate_pct']:+.2f}%, conviction+macro "
        f"{m['full_gate_pct']:+.2f}%. None of the rule-based gates was profitable. "
    )
    if model_note:
        return base + model_note
    return base + (
        "Treat these as research output, not a profit forecast, and size accordingly."
    )
