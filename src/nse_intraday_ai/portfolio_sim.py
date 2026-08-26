"""Intraday portfolio simulator: many concurrent positions, one trading day.

The existing backtester holds **one position at a time** and charges a flat bps
commission.  Neither matches how the account described in this project is
actually traded: ₹10,00,000 of capital, several names open at once, partial
exits, and — non-negotiably — everything squared off before the close, because
these are MIS/intraday positions that the broker will auto-square anyway
(usually at a worse price than a planned exit).

What this module models that the old one does not:

* **Concurrency and capital**: positions compete for a shared cash pool and a
  shared risk budget; a signal that does not fit is skipped, not queued.
* **Entry at the next bar's open.** A signal computed on the close of bar T
  cannot be filled at that close.  The old harness used the signal bar's close
  as the entry price, which quietly awards the last tick of the impulse to the
  strategy.  Here the fill is the open of bar T+1.
* **Partial exits**: scale out at the first target, trail the remainder.  Each
  tranche is a separate order and pays its own flat brokerage leg.
* **Real costs** (`costs.py`), which depend on position size rather than being
  a constant, and therefore actually interact with sizing.
* **Mandatory square-off** at a configurable time, with the closing slippage
  that forced exits really incur.
* **Daily loss limit and per-symbol cooldown**, evaluated on the live equity
  curve rather than after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd

from nse_intraday_ai.costs import min_position_for_cost_target, round_trip_cost
from nse_intraday_ai.liquidity import (
    max_position_for_participation,
    slippage_bps_for,
)


@dataclass(frozen=True)
class SimConfig:
    starting_capital: float = 10_00_000.0
    max_concurrent_positions: int = 4
    risk_per_trade_pct: float = 0.5         # % of equity risked to the stop
    max_position_pct: float = 25.0          # % of equity in one name
    max_gross_exposure_pct: float = 100.0   # % of equity deployed at once
    # Cost model
    slippage_bps_per_leg: float = 2.5
    # Size-aware slippage (`liquidity.py`).  Zero keeps the constant model that
    # every study before 2026-08-25 assumed, so those results are reproducible;
    # any book that leans on margin must set it, because a constant bps cost is
    # what makes leverage look free.
    impact_coefficient: float = 0.0
    # Refuse (or trim) a position that would be more than this share of the
    # signal bar's own turnover.  Zero disables the cap.  This is the constraint
    # that actually binds a large account: buying power is a budget to spread
    # across names, not a position size.
    max_participation_pct: float = 0.0
    # Refuse trades too small to clear their own flat-fee costs.
    cost_target_bps: float = 12.0
    # Exit design.  Distances are in ATR units measured at entry.
    stop_atr: float = 1.0
    target_atr: float = 1.6
    scale_out_atr: float = 0.9              # take partial profit here
    scale_out_fraction: float = 0.5
    trail_atr: float = 1.2                  # trail the remainder from its peak
    breakeven_after_atr: float = 0.6        # move stop to entry after this
    trail_lock_trigger_atr: float = 0.0     # lock profit once excursion reaches this
    trail_lock_distance_atr: float = 0.0    # locked stop level from entry
    max_hold_bars: int = 12                 # 12 x 5m = 60 minutes
    # Session controls (IST)
    no_new_entry_after: str = "14:45"
    square_off_at: str = "15:15"
    square_off_slippage_bps: float = 5.0    # forced exits cost more
    # Risk controls
    daily_loss_limit_pct: float = 2.0       # stop trading for the day
    symbol_cooldown_bars: int = 12
    max_trades_per_day: int = 12


@dataclass
class Position:
    symbol: str
    side: str                   # LONG | SHORT
    entry_time: pd.Timestamp
    entry_price: float
    quantity: int
    initial_quantity: int
    stop: float
    target: float
    scale_level: float
    atr: float
    bars_held: int = 0
    peak: float = 0.0
    scaled_out: bool = False
    legs: int = 1               # executed orders so far
    realised: float = 0.0       # gross P&L already banked from partial exits
    reason: str = ""
    # Per-leg slippage actually charged to this position.  Constant books leave
    # it at the config value; a size-aware book stamps the impact-inflated
    # figure here at entry so both legs are priced at the size that was traded.
    slip_bps: float = 0.0
    turnover: float = 0.0       # signal-bar turnover used to size and price it

    @property
    def sign(self) -> int:
        return 1 if self.side == "LONG" else -1

    def gross_at(self, price: float, quantity: int | None = None) -> float:
        qty = self.quantity if quantity is None else quantity
        return (price - self.entry_price) * self.sign * qty


@dataclass
class SimResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    daily: pd.DataFrame
    config: SimConfig
    starting_capital: float
    ending_capital: float

    @property
    def pnl(self) -> float:
        return self.ending_capital - self.starting_capital

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.starting_capital * 100

    @property
    def peak_exposure_x(self) -> float:
        """Highest gross exposure reached, as a multiple of starting capital.

        This is the margin the book would have had to be granted.  A study that
        reports a return without it is quoting a number the account may not
        have been allowed to earn.
        """
        if self.equity.empty or "exposure" not in self.equity:
            return 0.0
        return float(self.equity["exposure"].max()) / self.starting_capital

    @property
    def mean_exposure_x(self) -> float:
        """Average gross exposure while at least one position is open."""
        if self.equity.empty or "exposure" not in self.equity:
            return 0.0
        live = self.equity["exposure"][self.equity["exposure"] > 0]
        return float(live.mean()) / self.starting_capital if len(live) else 0.0

    def recency_split(self, tail_fraction: float = 0.5) -> str:
        """Is the result still working, or did it stop?

        A cumulative equity curve hides decay: a strategy that made everything
        in its first three weeks and has bled since reports the same headline
        number as one earning steadily.  The recent half is the part that
        predicts tomorrow, so it gets printed next to the headline every time.
        """
        if self.daily.empty or len(self.daily) < 4:
            return "  (too few sessions to split)"
        daily = self.daily.copy()
        cut = int(len(daily) * (1 - tail_fraction))
        lines = []
        for label, part in (("earlier", daily.iloc[:cut]), ("recent ", daily.iloc[cut:])):
            if part.empty:
                continue
            total = part["pnl"].sum()
            up = int((part["pnl"] > 0).sum())
            lines.append(
                f"  {label}  {len(part):3d} sessions  ₹{total:+9,.0f} "
                f"({total / self.starting_capital * 100:+6.2f}%)  "
                f"up {up}/{len(part)}  mean ₹{part['pnl'].mean():+,.0f}"
            )
        earlier = daily.iloc[:cut]["pnl"].sum()
        recent = daily.iloc[cut:]["pnl"].sum()
        if earlier > 0 and recent < 0:
            lines.append(
                "  ⚠ the entire gain is in the earlier period — the recent half, "
                "which is the part that predicts tomorrow, lost money"
            )
        return "\n".join(lines)

    @property
    def win_rate(self) -> float:
        if self.trades.empty:
            return 0.0
        return float((self.trades["net_pnl"] > 0).mean() * 100.0)

    @property
    def profit_factor(self) -> float:
        if self.trades.empty:
            return 0.0
        wins = self.trades[self.trades["net_pnl"] > 0]["net_pnl"].sum()
        losses = abs(self.trades[self.trades["net_pnl"] <= 0]["net_pnl"].sum())
        return float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)

    @property
    def max_drawdown(self) -> float:
        if self.equity.empty or "equity" not in self.equity:
            return 0.0
        eq = self.equity["equity"]
        return float((eq.cummax() - eq).max())

    @property
    def max_drawdown_pct(self) -> float:
        if self.starting_capital <= 0:
            return 0.0
        return float(self.max_drawdown / self.starting_capital * 100.0)

    @property
    def sharpe_ratio(self) -> float:
        if self.daily.empty or len(self.daily) < 2 or self.starting_capital <= 0:
            return 0.0
        returns = (self.daily["pnl"] / self.starting_capital).to_numpy(float)
        std = float(np.std(returns, ddof=1))
        return float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0.0

    @property
    def sortino_ratio(self) -> float:
        if self.daily.empty or len(self.daily) < 2 or self.starting_capital <= 0:
            return 0.0
        returns = (self.daily["pnl"] / self.starting_capital).to_numpy(float)
        downside = np.minimum(0.0, returns)
        downside_std = float(np.sqrt(np.mean(downside**2)))
        if downside_std <= 0:
            std = float(np.std(returns, ddof=1))
            downside_std = std if std > 0 else 1e-6
        return float(np.mean(returns) / downside_std * np.sqrt(252))

    @property
    def calmar_ratio(self) -> float:
        dd = self.max_drawdown_pct
        if dd <= 0:
            return 0.0
        n_days = max(1, len(self.daily))
        annualized_return = (self.pnl_pct / n_days) * 252.0
        return float(annualized_return / dd)

    @property
    def expectancy_rupees(self) -> float:
        if self.trades.empty:
            return 0.0
        return float(self.trades["net_pnl"].mean())

    @property
    def expectancy_bps(self) -> float:
        if self.trades.empty or "turnover" not in self.trades:
            return 0.0
        turnover = self.trades["turnover"].replace(0, np.nan)
        bps = self.trades["net_pnl"] / turnover * 1e4
        return float(bps.dropna().mean()) if len(bps.dropna()) else 0.0

    @property
    def kelly_fraction(self) -> float:
        """Full Kelly criterion: p - (1-p)/b where p=win rate, b=win/loss payoff ratio."""
        if self.trades.empty:
            return 0.0
        wins = self.trades[self.trades["net_pnl"] > 0]["net_pnl"]
        losses = abs(self.trades[self.trades["net_pnl"] <= 0]["net_pnl"])
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        p = len(wins) / len(self.trades)
        b = wins.mean() / max(losses.mean(), 1e-9)
        k = p - (1.0 - p) / b
        return float(max(0.0, min(1.0, k)))

    def summary(self) -> str:
        trades = self.trades
        if trades.empty:
            return f"no trades taken (capital unchanged at ₹{self.ending_capital:,.0f})"
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] <= 0]
        gross_win = wins["net_pnl"].sum()
        gross_loss = abs(losses["net_pnl"].sum())
        drawdown_val = self.max_drawdown
        return "\n".join([
            f"  capital      ₹{self.starting_capital:,.0f} → ₹{self.ending_capital:,.0f}"
            f"   ({self.pnl:+,.0f}, {self.pnl_pct:+.2f}%)",
            f"  trades       {len(trades)} over {self.daily.shape[0]} sessions"
            f"   ({len(trades) / max(self.daily.shape[0], 1):.1f}/day)",
            f"  win rate     {self.win_rate:.1f}%"
            f"   ({len(wins)}W / {len(losses)}L)",
            f"  avg win      ₹{wins['net_pnl'].mean():,.0f}" if len(wins) else "  avg win      —",
            f"  avg loss     ₹{losses['net_pnl'].mean():,.0f}" if len(losses) else "  avg loss     —",
            f"  profit factor {self.profit_factor:.2f}",
            f"  expectancy   ₹{self.expectancy_rupees:+,.0f} per trade",
            f"  total costs  ₹{trades['costs'].sum():,.0f}"
            f"   ({trades['costs'].sum() / max(trades['gross_pnl'].abs().sum(), 1) * 100:.1f}% of gross flow)",
            f"  gross P&L    ₹{trades['gross_pnl'].sum():+,.0f}"
            f"   → net ₹{trades['net_pnl'].sum():+,.0f}",
            f"  max drawdown ₹{drawdown_val:,.0f} ({self.max_drawdown_pct:.2f}%)",
            f"  daily Sharpe {self.sharpe_ratio:.2f} | Sortino {self.sortino_ratio:.2f} | Calmar {self.calmar_ratio:.2f}",
            f"  Kelly frac   {self.kelly_fraction * 100:.1f}% (Half-Kelly {self.kelly_fraction * 50:.1f}%)",
            f"  best day     ₹{self.daily['pnl'].max():+,.0f}   "
            f"worst day ₹{self.daily['pnl'].min():+,.0f}",
        ])


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


class IntradayPortfolioSimulator:
    """Replays pre-computed engine signals through a realistic intraday book."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()

    # ── sizing ───────────────────────────────────────────────────────────────

    def _size(self, equity: float, price: float, stop: float, deployed: float,
              turnover: float = 0.0) -> int:
        cfg = self.config
        risk_per_share = abs(price - stop)
        if risk_per_share <= 0 or price <= 0:
            return 0
        by_risk = (equity * cfg.risk_per_trade_pct / 100) / risk_per_share
        by_position = (equity * cfg.max_position_pct / 100) / price
        headroom = max(0.0, equity * cfg.max_gross_exposure_pct / 100 - deployed)
        by_exposure = headroom / price
        # The bar can only absorb so much.  Without this, raising
        # `max_gross_exposure_pct` to model 5x margin buys size the tape was
        # never going to give at the modelled price.
        by_liquidity = (
            max_position_for_participation(turnover, cfg.max_participation_pct) / price
            if cfg.max_participation_pct > 0 else float("inf")
        )
        quantity = int(min(by_risk, by_position, by_exposure, by_liquidity))
        if quantity <= 0:
            return 0
        # A position too small to amortise the flat ₹20 legs cannot clear its
        # own costs; take it properly sized or not at all.
        floor_value = min_position_for_cost_target(
            price, cfg.cost_target_bps, slippage_bps_per_leg=cfg.slippage_bps_per_leg
        )
        if quantity * price < floor_value:
            return 0
        return quantity

    @staticmethod
    def _bar_turnover(frame: pd.DataFrame, ts: pd.Timestamp, lookback: int = 12) -> float:
        """Rupee turnover the tape offered around the signal bar.

        A single bar's volume is noisy enough that sizing off it alone would
        swing the position by a factor of three between adjacent minutes, so
        this is the median of the last `lookback` closed bars up to and
        including the signal bar.  Only closed bars: at 09:15 there is exactly
        one, its own, which is the honest amount of information a book has at
        the moment the opening signal prints.
        """
        position_of = frame.index.searchsorted(ts, side="right")
        if position_of <= 0:
            return 0.0
        window = frame.iloc[max(0, position_of - lookback):position_of]
        if window.empty:
            return 0.0
        turnover = (window["close"] * window["volume"]).to_numpy(float)
        turnover = turnover[np.isfinite(turnover) & (turnover > 0)]
        return float(np.median(turnover)) if len(turnover) else 0.0

    # ── exits ────────────────────────────────────────────────────────────────

    def _close(
        self, position: Position, price: float, quantity: int, reason: str,
        *, slippage_bps: float | None = None,
    ) -> dict:
        cfg = self.config
        # `slippage_bps` overrides the *spread* term only (a forced square-off
        # crosses a wider spread).  Size impact is a property of the order, not
        # of why it was sent, so it is added on top rather than replaced —
        # otherwise the one exit a levered book most wants priced honestly, the
        # 15:15 forced unwind, would be the one charged least.
        base = cfg.slippage_bps_per_leg if slippage_bps is None else slippage_bps
        impact = max(0.0, position.slip_bps - cfg.slippage_bps_per_leg)
        slip = base + impact
        gross = position.gross_at(price, quantity)
        costs = round_trip_cost(
            position.entry_price, price, quantity,
            slippage_bps_per_leg=slip, legs=position.legs + 1,
        )
        return {
            "gross": gross, "costs": costs.total, "net": gross - costs.total,
            "price": price, "quantity": quantity, "reason": reason,
        }

    def _manage(self, position: Position, bar: pd.Series, ts: pd.Timestamp,
                force: bool) -> list[dict]:
        """Advance one position by one bar; returns the fills it produced.

        Ordering inside a bar is unknowable from OHLC alone, so the simulator
        always resolves the **stop before the target** — the pessimistic
        assumption.  Trailing and breakeven arm from the *next* bar, never
        retroactively within the bar that triggered them.
        """
        cfg = self.config
        fills: list[dict] = []
        high, low = float(bar["high"]), float(bar["low"])
        sign = position.sign
        position.bars_held += 1

        if force:
            price = float(bar["close"])
            fills.append(self._close(position, price, position.quantity, "SQUARE_OFF",
                                     slippage_bps=cfg.square_off_slippage_bps))
            position.quantity = 0
            return fills

        # Stop first (pessimistic).
        stop_hit = low <= position.stop if sign > 0 else high >= position.stop
        if stop_hit:
            # Distinguish the three ways a stop can fire, by where it sits now
            # relative to entry — a stop that has ratcheted to entry is a
            # scratch, not a loss, and lumping them together hid the fact that
            # most "stop-outs" in this book are breakeven exits.
            at_or_past_entry = (
                position.stop >= position.entry_price if sign > 0
                else position.stop <= position.entry_price
            )
            if not at_or_past_entry:
                reason = "STOP"
            elif position.stop == position.entry_price:
                reason = "BREAKEVEN"
            else:
                reason = "TRAIL_STOP"
            fills.append(self._close(position, position.stop, position.quantity, reason))
            position.quantity = 0
            return fills

        # Partial scale-out at the first objective.
        if not position.scaled_out and cfg.scale_out_fraction > 0:
            reached = high >= position.scale_level if sign > 0 else low <= position.scale_level
            if reached:
                qty = int(position.initial_quantity * cfg.scale_out_fraction)
                qty = min(qty, position.quantity)
                if qty > 0:
                    fills.append(self._close(position, position.scale_level, qty, "SCALE_OUT"))
                    position.quantity -= qty
                    position.legs += 1
                    position.scaled_out = True
                    # Risk is now off the table; protect the rest at entry.
                    position.stop = position.entry_price
                if position.quantity <= 0:
                    return fills

        # Full target on whatever remains.
        target_hit = high >= position.target if sign > 0 else low <= position.target
        if target_hit:
            fills.append(self._close(position, position.target, position.quantity, "TARGET"))
            position.quantity = 0
            return fills

        # Multi-stage ratchets, applied from the next bar onward.
        excursion = (high - position.entry_price) * sign if sign > 0 else (position.entry_price - low)
        position.peak = max(position.peak, excursion)

        # Stage 3: Lock profit once excursion reaches trail_lock_trigger_atr
        if cfg.trail_lock_trigger_atr > 0 and position.peak >= cfg.trail_lock_trigger_atr * position.atr:
            locked_stop = position.entry_price + sign * cfg.trail_lock_distance_atr * position.atr
            position.stop = (
                max(position.stop, locked_stop) if sign > 0
                else min(position.stop, locked_stop)
            )
        # Stage 2: Breakeven ratchet once excursion reaches breakeven_after_atr
        elif cfg.breakeven_after_atr > 0 and position.peak >= cfg.breakeven_after_atr * position.atr:
            position.stop = (
                max(position.stop, position.entry_price) if sign > 0
                else min(position.stop, position.entry_price)
            )

        if cfg.trail_atr > 0 and position.scaled_out:
            best = position.entry_price + sign * position.peak
            trail = best - sign * cfg.trail_atr * position.atr
            position.stop = max(position.stop, trail) if sign > 0 else min(position.stop, trail)

        if position.bars_held >= cfg.max_hold_bars:
            fills.append(self._close(position, float(bar["close"]), position.quantity, "TIME_EXIT"))
            position.quantity = 0
        return fills

    # ── main loop ────────────────────────────────────────────────────────────

    def run(
        self,
        signals: pd.DataFrame,
        frames: dict[str, pd.DataFrame],
        *,
        verbose: bool = False,
    ) -> SimResult:
        """Replay `signals` (one row per engine recommendation) through the book.

        `signals` needs columns: ts, symbol, side, atr, and optionally a
        `rank` column used to break ties when more signals arrive on one bar
        than the book can hold.  Entry, stop and target are derived here from
        ATR, so the exit design is a property of the *portfolio*, not baked
        into each signal.

        `frames` maps symbol -> 5m OHLCV frame with a tz-aware index.
        """
        cfg = self.config
        if signals.empty:
            empty = pd.DataFrame()
            return SimResult(empty, empty, empty, cfg, cfg.starting_capital, cfg.starting_capital)

        cutoff = _parse_time(cfg.no_new_entry_after)
        square_off = _parse_time(cfg.square_off_at)

        signals = signals.sort_values("ts").reset_index(drop=True)
        bar_index = sorted({ts for frame in frames.values() for ts in frame.index})
        by_bar: dict[pd.Timestamp, list[dict]] = {}
        for row in signals.to_dict("records"):
            by_bar.setdefault(row["ts"], []).append(row)

        equity = cash = float(cfg.starting_capital)
        open_positions: dict[str, Position] = {}
        trades: list[dict] = []
        curve: list[dict] = [{"timestamp": bar_index[0], "equity": equity}]
        cooldown: dict[str, pd.Timestamp] = {}
        day = None
        day_start_equity = equity
        trades_today = 0
        halted = False

        def force_close_stragglers(reason: str) -> None:
            """Close everything still open, at each symbol's last bar of its own day.

            Called when the replay crosses into a new session. A clock-based
            square-off alone is not enough: if a symbol's feed has a gap and
            simply has no bar after 15:15, the position would otherwise sit
            open and exit on the *next* session's first bar — an overnight
            hold, which an intraday MIS position cannot be. Seen on
            2026-08-13, where degraded collection truncated the session and
            INFY and LTM were carried into the next morning.
            """
            nonlocal equity
            for symbol, position in list(open_positions.items()):
                frame = frames.get(symbol)
                if frame is None or frame.empty:
                    del open_positions[symbol]
                    continue
                same_day = frame[frame.index.normalize() == position.entry_time.normalize()]
                if same_day.empty:
                    del open_positions[symbol]
                    continue
                exit_ts = same_day.index[-1]
                fill = self._close(
                    position, float(same_day.iloc[-1]["close"]), position.quantity,
                    reason, slippage_bps=cfg.square_off_slippage_bps,
                )
                equity += fill["net"]
                trades.append({
                    "entry_time": position.entry_time, "exit_time": exit_ts,
                    "symbol": symbol, "side": position.side,
                    "entry": round(position.entry_price, 2),
                    "exit": round(fill["price"], 2),
                    "quantity": fill["quantity"],
                    "gross_pnl": round(fill["gross"], 2),
                    "costs": round(fill["costs"], 2),
                    "net_pnl": round(fill["net"], 2),
                    "exit_reason": fill["reason"],
                    "bars_held": position.bars_held,
                    "atr": round(position.atr, 3),
                    "note": position.reason,
                })
                curve.append({"timestamp": exit_ts, "equity": equity})
                del open_positions[symbol]

        for ts in bar_index:
            if day != ts.normalize():
                if open_positions:
                    force_close_stragglers("SESSION_END")
                day = ts.normalize()
                day_start_equity = equity
                trades_today = 0
                halted = False

            force = ts.time() >= square_off

            # ── 1. manage open positions on this bar ─────────────────────────
            for symbol in list(open_positions):
                position = open_positions[symbol]
                frame = frames.get(symbol)
                if frame is None or ts not in frame.index:
                    continue
                for fill in self._manage(position, frame.loc[ts], ts, force):
                    cash += fill["quantity"] * fill["price"] if position.side == "LONG" else 0.0
                    equity += fill["net"]
                    position.realised += fill["net"]
                    trades.append({
                        "entry_time": position.entry_time, "exit_time": ts,
                        "symbol": symbol, "side": position.side,
                        "entry": round(position.entry_price, 2),
                        "exit": round(fill["price"], 2),
                        "quantity": fill["quantity"],
                        "gross_pnl": round(fill["gross"], 2),
                        "costs": round(fill["costs"], 2),
                        "net_pnl": round(fill["net"], 2),
                        "exit_reason": fill["reason"],
                        "bars_held": position.bars_held,
                        "atr": round(position.atr, 3),
                        "note": position.reason,
                    })
                if position.quantity <= 0:
                    cooldown[symbol] = ts + pd.Timedelta(minutes=5 * cfg.symbol_cooldown_bars)
                    del open_positions[symbol]

            curve.append({
                "timestamp": ts, "equity": equity,
                # Gross rupees at work, so a levered book can be asked the only
                # question that matters to a broker: how much margin did this
                # actually use, and when.
                "exposure": sum(p.entry_price * p.quantity for p in open_positions.values()),
            })

            if equity <= day_start_equity * (1 - cfg.daily_loss_limit_pct / 100):
                halted = True

            # ── 2. consider new entries ──────────────────────────────────────
            candidates = by_bar.get(ts, [])
            # max_trades_per_day == 0 means *unlimited*, not "no trades" — the
            # bare `>=` here silently disabled the book entirely when the cap
            # was switched off.
            if (
                not candidates or halted or force
                or ts.time() > cutoff
                or (0 < cfg.max_trades_per_day <= trades_today)
                or len(open_positions) >= cfg.max_concurrent_positions
            ):
                continue

            candidates = sorted(candidates, key=lambda r: -float(r.get("rank", 0.0)))
            deployed = sum(p.entry_price * p.quantity for p in open_positions.values())

            for signal in candidates:
                # Both caps must be re-checked per candidate, not once per bar:
                # signals arrive in clusters on the same bar, so a bar-level
                # check let an entire cluster through a 3-a-day limit.
                if len(open_positions) >= cfg.max_concurrent_positions:
                    break
                if 0 < cfg.max_trades_per_day <= trades_today:
                    break
                symbol = signal["symbol"]
                if symbol in open_positions or ts < cooldown.get(symbol, ts):
                    continue
                frame = frames.get(symbol)
                if frame is None:
                    continue
                # Fill at the OPEN of the next bar — the signal bar's close is
                # not an attainable price.
                position_of = frame.index.searchsorted(ts, side="right")
                if position_of >= len(frame):
                    continue
                fill_ts = frame.index[position_of]
                if fill_ts.normalize() != ts.normalize() or fill_ts.time() >= square_off:
                    continue
                entry = float(frame.iloc[position_of]["open"])
                atr = float(signal["atr"])
                if not np.isfinite(atr) or atr <= 0 or entry <= 0:
                    continue

                side = signal["side"]
                sign = 1 if side == "LONG" else -1
                stop = entry - sign * cfg.stop_atr * atr
                # Liquidity is read from the SIGNAL bar, never the fill bar:
                # the fill bar is still forming when the order is sent, so
                # sizing off it would be a look-ahead the live book cannot
                # reproduce.
                turnover = self._bar_turnover(frame, ts)
                quantity = self._size(equity, entry, stop, deployed, turnover)
                if quantity <= 0:
                    continue

                slip_bps = slippage_bps_for(
                    quantity * entry, turnover, atr / entry * 1e4,
                    base_bps=cfg.slippage_bps_per_leg,
                    coefficient=cfg.impact_coefficient,
                )
                open_positions[symbol] = Position(
                    symbol=symbol, side=side, entry_time=fill_ts, entry_price=entry,
                    quantity=quantity, initial_quantity=quantity, stop=stop,
                    target=entry + sign * cfg.target_atr * atr,
                    scale_level=entry + sign * cfg.scale_out_atr * atr,
                    atr=atr, reason=str(signal.get("note", "")),
                    slip_bps=slip_bps, turnover=turnover,
                )
                deployed += entry * quantity
                # The cap counts *entries*.  It previously incremented on exit
                # fills, so the book could open any number of positions in the
                # morning before the first one closed — a "3 trades a day" book
                # that quietly took six, and a partial exit counted twice.
                trades_today += 1
                if verbose:
                    print(f"  {fill_ts:%Y-%m-%d %H:%M} ENTER {side:5s} {symbol:14s} "
                          f"{quantity:5d} @ ₹{entry:,.2f}  (₹{quantity * entry:,.0f})")

        # Anything still open when the data runs out must be marked to market,
        # not dropped.  Silently discarding open positions understates losses
        # on a truncated feed — which is exactly the case mid-session, when the
        # last bar is "now" rather than the close.
        if open_positions and bar_index:
            for symbol, position in list(open_positions.items()):
                frame = frames.get(symbol)
                if frame is None or frame.empty:
                    continue
                ts = frame.index[-1]
                fill = self._close(
                    position, float(frame.iloc[-1]["close"]), position.quantity,
                    "MARK_TO_MARKET",
                )
                equity += fill["net"]
                trades.append({
                    "entry_time": position.entry_time, "exit_time": ts,
                    "symbol": symbol, "side": position.side,
                    "entry": round(position.entry_price, 2),
                    "exit": round(fill["price"], 2),
                    "quantity": fill["quantity"],
                    "gross_pnl": round(fill["gross"], 2),
                    "costs": round(fill["costs"], 2),
                    "net_pnl": round(fill["net"], 2),
                    "exit_reason": fill["reason"],
                    "bars_held": position.bars_held,
                    "atr": round(position.atr, 3),
                    "note": position.reason,
                })
                curve.append({"timestamp": ts, "equity": equity})

        equity_frame = pd.DataFrame(curve).drop_duplicates("timestamp", keep="last")
        trades_frame = pd.DataFrame(trades)
        if trades_frame.empty:
            daily = pd.DataFrame(columns=["pnl", "trades"])
        else:
            grouped = trades_frame.groupby(trades_frame["exit_time"].dt.date)
            daily = pd.DataFrame({"pnl": grouped["net_pnl"].sum(), "trades": grouped.size()})
        return SimResult(trades_frame, equity_frame, daily, cfg,
                         cfg.starting_capital, equity)


def monte_carlo_permutation(
    trades: pd.DataFrame,
    starting_capital: float = 1_000_000.0,
    num_paths: int = 1000,
    max_trades: int | None = None,
    random_seed: int = 42,
) -> dict:
    """Run Monte Carlo permutations on closed trade outcomes to evaluate drawdown distribution and risk of ruin."""
    if trades is None or trades.empty or "net_pnl" not in trades:
        return {
            "num_paths": 0,
            "starting_capital": starting_capital,
            "p5_equity": starting_capital,
            "p50_equity": starting_capital,
            "p95_equity": starting_capital,
            "max_dd_p95_pct": 0.0,
            "risk_of_ruin_pct": 0.0,
            "percentile_curves": {"p5": [starting_capital], "p50": [starting_capital], "p95": [starting_capital]},
        }

    rng = np.random.default_rng(random_seed)
    pnls = trades["net_pnl"].to_numpy(float)
    n_sample = len(pnls) if max_trades is None else min(max_trades, len(pnls))
    if n_sample <= 0:
        return {}

    paths = np.zeros((num_paths, n_sample + 1), dtype=float)
    paths[:, 0] = starting_capital
    max_drawdowns = np.zeros(num_paths, dtype=float)
    ruin_count = 0
    ruin_threshold = 0.85 * starting_capital  # 15% drawdown threshold

    for i in range(num_paths):
        resampled_pnl = rng.choice(pnls, size=n_sample, replace=True)
        equity_curve = starting_capital + np.cumsum(resampled_pnl)
        paths[i, 1:] = equity_curve

        peak = np.maximum.accumulate(np.insert(equity_curve, 0, starting_capital))
        dd = (peak - np.insert(equity_curve, 0, starting_capital)) / peak
        max_drawdowns[i] = float(np.max(dd))

        if np.any(equity_curve <= ruin_threshold):
            ruin_count += 1

    ending_equities = paths[:, -1]
    return {
        "num_paths": num_paths,
        "starting_capital": starting_capital,
        "p5_equity": float(np.percentile(ending_equities, 5)),
        "p25_equity": float(np.percentile(ending_equities, 25)),
        "p50_equity": float(np.percentile(ending_equities, 50)),
        "p75_equity": float(np.percentile(ending_equities, 75)),
        "p95_equity": float(np.percentile(ending_equities, 95)),
        "mean_ending_equity": float(np.mean(ending_equities)),
        "max_dd_p95_pct": float(np.percentile(max_drawdowns, 95) * 100.0),
        "risk_of_ruin_pct": float(ruin_count / num_paths * 100.0),
        "percentile_curves": {
            "p5": np.percentile(paths, 5, axis=0).tolist(),
            "p50": np.percentile(paths, 50, axis=0).tolist(),
            "p95": np.percentile(paths, 95, axis=0).tolist(),
        },
    }
