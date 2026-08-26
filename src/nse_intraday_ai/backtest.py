from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import time, timedelta

import pandas as pd

from nse_intraday_ai.data import DataResult, YFinanceProvider
from nse_intraday_ai.indicators import add_indicators
from nse_intraday_ai.models import Side, TradePlan
from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.strategies import EnsembleConfig, VotingSignalEngine


@dataclass(frozen=True)
class BacktestConfig:
    starting_capital: float = 100_000
    lookback_days: int = 5
    interval: str = "1m"
    cooldown_minutes: int = 60
    warmup_candles: int = 80
    max_hold_minutes: int = 90
    include_shorts: bool = True
    no_new_trade_after: str = "15:00"
    symbol_cooldown_minutes: int = 0
    stop_loss_cooldown_minutes: int = 0
    max_trades_per_day: int = 0
    daily_loss_limit_pct: float = 0.0
    estimated_cost_bps: float = 15.0
    slippage_bps: float = 3.0
    # Move the stop to entry once price has travelled this many R in favour
    # (0 disables).  Converts full-stop losses on trades that were briefly
    # green into scratch exits — the single biggest bleed in the event study
    # was stopped-out trades averaging −48 bps.
    breakeven_after_r: float = 0.8
    # Pullback limit entry: instead of buying at the signal-candle close,
    # post a limit this many ATRs below it (above for shorts) and only trade
    # if it fills within entry_fill_window_minutes.  Improves entry location
    # by construction; unfilled signals are dropped (no chase).
    entry_limit_offset_atr: float = 0.0
    entry_fill_window_minutes: int = 15
    # NSE equities are intraday-only (positions square off by close), so exits
    # are restricted to the entry day.  24-hour commodity futures have no such
    # boundary — set False so trades opened near (IST) midnight aren't clipped.
    same_day_exit_only: bool = True
    # Chandelier-style trailing stop: when > 0, the stop trails the best price
    # seen since entry by this many ATRs (using the candle's atr_14 when
    # available).  Replaces the breakeven mechanism.  The trail is evaluated
    # against the *previous* candles' peak so no intra-candle ordering is
    # assumed in our favour.
    trailing_atr_mult: float = 0.0
    # With trailing enabled, optionally drop the fixed target and let winners
    # run until the trail or the time limit ends the trade.
    trailing_replaces_target: bool = False


@dataclass(frozen=True)
class BacktestSummary:
    starting_capital: float
    ending_capital: float
    pnl: float
    pnl_pct: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    max_drawdown: float
    max_drawdown_pct: float


@dataclass(frozen=True)
class CalibrationResult:
    min_confidence: float
    min_reward_risk: float
    min_agreeing_votes: int
    min_vote_share: float
    summary: BacktestSummary


def _calibration_key(result: CalibrationResult, objective: str) -> tuple[float, ...]:
    summary = result.summary
    drawdown = max(summary.max_drawdown, 1.0)
    pnl_to_drawdown = summary.pnl / drawdown
    loss_rate = summary.losses / summary.trades if summary.trades else 1.0
    if objective == "max_pnl":
        return (
            summary.pnl,
            -summary.max_drawdown,
            summary.win_rate,
            result.min_confidence,
        )
    if objective == "low_drawdown":
        return (
            1.0 if summary.pnl > 0 else 0.0,
            -summary.max_drawdown,
            summary.pnl,
            summary.win_rate,
            result.min_confidence,
        )
    return (
        1.0 if summary.pnl > 0 else 0.0,
        pnl_to_drawdown,
        -loss_rate,
        summary.win_rate,
        summary.pnl,
        result.min_confidence,
    )


@dataclass(frozen=True)
class CandidateEvent:
    timestamp: pd.Timestamp
    symbol: str
    side: Side
    entry: float
    stop_loss: float
    target: float
    confidence: float
    reward_risk: float
    agreeing_count: int
    vote_share: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    per_share_pnl: float
    reasons: tuple[str, ...]
    agreeing_strategies: tuple[str, ...]


def _period_from_days(days: int) -> str:
    if days <= 1:
        return "1d"
    if days <= 5:
        return "5d"
    return "1mo"


def _ensure_tz_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if getattr(out.index, "tz", None) is None:
        out.index = out.index.tz_localize("Asia/Kolkata")
    else:
        out.index = out.index.tz_convert("Asia/Kolkata")
    return out


def _trim_days(frame: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized_days = pd.Index(frame.index.normalize().unique())
    selected_days = normalized_days[-lookback_days:]
    return frame[frame.index.normalize().isin(selected_days)]


def load_backtest_data(
    symbols: list[str],
    lookback_days: int,
    interval: str,
    provider: YFinanceProvider | None = None,
) -> dict[str, DataResult]:
    provider = provider or YFinanceProvider()
    results = provider.batch_history(symbols, period=_period_from_days(lookback_days), interval=interval)
    return {
        symbol: DataResult(
            symbol=result.symbol,
            frame=_trim_days(_ensure_tz_index(result.frame), lookback_days),
            source=result.source,
            warning=result.warning,
        )
        for symbol, result in results.items()
    }


def _trade_exit(
    plan: TradePlan,
    future: pd.DataFrame,
    max_hold_minutes: int,
    entry_time: pd.Timestamp,
    breakeven_after_r: float = 0.0,
    skip_target_on_first: bool = False,
    trailing_atr_mult: float = 0.0,
    trailing_replaces_target: bool = False,
) -> tuple[pd.Timestamp, float, str]:
    if future.empty or plan.entry is None or plan.stop_loss is None or plan.target is None:
        raise ValueError("Cannot exit trade without future candles and plan levels.")

    deadline = entry_time + timedelta(minutes=max_hold_minutes)
    scoped = future[future.index <= deadline]
    if scoped.empty:
        scoped = future.iloc[:1]

    stop = float(plan.stop_loss)
    original_stop = stop
    risk_dist = abs(float(plan.entry) - stop)
    trailing = trailing_atr_mult > 0
    target_active = not (trailing and trailing_replaces_target)
    peak = float(plan.entry)
    be_trigger = None
    if breakeven_after_r > 0 and risk_dist > 0 and not trailing:
        if plan.side == Side.LONG:
            be_trigger = float(plan.entry) + risk_dist * breakeven_after_r
        else:
            be_trigger = float(plan.entry) - risk_dist * breakeven_after_r
    be_armed = False

    for i, (ts, row) in enumerate(scoped.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        # On a limit-fill candle the intra-candle ordering of fill vs. target
        # touch is unknown — never award the target (or arm breakeven) there.
        first_blind = skip_target_on_first and i == 0
        # Breakeven stop is applied from the candle *after* the trigger was
        # touched, so we never assume intra-candle ordering in our favour.
        if be_armed:
            stop = max(stop, float(plan.entry)) if plan.side == Side.LONG else min(stop, float(plan.entry))
        if plan.side == Side.LONG:
            if low <= stop:
                if trailing and stop > original_stop:
                    reason = "TRAIL_STOP"
                elif be_armed and stop >= plan.entry:
                    reason = "BREAKEVEN"
                else:
                    reason = "STOP_LOSS"
                return ts, float(stop), reason
            if target_active and high >= plan.target and not first_blind:
                return ts, float(plan.target), "TARGET"
            if be_trigger is not None and high >= be_trigger and not first_blind:
                be_armed = True
        elif plan.side == Side.SHORT:
            if high >= stop:
                if trailing and stop < original_stop:
                    reason = "TRAIL_STOP"
                elif be_armed and stop <= plan.entry:
                    reason = "BREAKEVEN"
                else:
                    reason = "STOP_LOSS"
                return ts, float(stop), reason
            if target_active and low <= plan.target and not first_blind:
                return ts, float(plan.target), "TARGET"
            if be_trigger is not None and low <= be_trigger and not first_blind:
                be_armed = True
        # Trailing stop ratchets from the peak seen up to and including this
        # candle, but only takes effect from the next candle's stop check —
        # never assuming intra-candle ordering in our favour.
        if trailing and not first_blind:
            atr_now = float(row.get("atr_14", 0) or 0) or risk_dist
            if plan.side == Side.LONG:
                peak = max(peak, high)
                stop = max(stop, peak - atr_now * trailing_atr_mult)
            else:
                peak = min(peak, low)
                stop = min(stop, peak + atr_now * trailing_atr_mult)
        last_ts = ts
        last_close = close
    return last_ts, last_close, "TIME_EXIT"


def _parse_time(value: str) -> time | None:
    if not value:
        return None
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))


def _past_entry_cutoff(ts: pd.Timestamp, cutoff: str) -> bool:
    cutoff_time = _parse_time(cutoff)
    if cutoff_time is None:
        return False
    return ts.time() > cutoff_time


def _reset_day_if_needed(
    ts: pd.Timestamp,
    current_day,
    capital: float,
) -> tuple[pd.Timestamp, float, int]:
    day = ts.normalize()
    if current_day is None or day != current_day:
        return day, capital, 0
    return current_day, capital, -1


def _daily_loss_limit_hit(
    *,
    capital: float,
    day_start_capital: float,
    daily_loss_limit_pct: float,
) -> bool:
    if daily_loss_limit_pct <= 0:
        return False
    return capital <= day_start_capital * (1 - daily_loss_limit_pct / 100)


def _roundtrip_cost(
    *,
    entry: float,
    quantity: int,
    estimated_cost_bps: float,
    slippage_bps: float,
) -> float:
    return entry * quantity * (estimated_cost_bps + slippage_bps) / 10_000


def _gross_pnl(plan: TradePlan, exit_price: float) -> float:
    if plan.entry is None:
        return 0.0
    if plan.side == Side.LONG:
        gross = (exit_price - plan.entry) * plan.quantity
    elif plan.side == Side.SHORT:
        gross = (plan.entry - exit_price) * plan.quantity
    else:
        gross = 0.0
    return gross


def _quantity_for_event(
    *,
    capital: float,
    risk_per_trade_pct: float,
    max_position_pct: float,
    event: CandidateEvent,
) -> int:
    if event.side == Side.LONG:
        risk_per_share = event.entry - event.stop_loss
    elif event.side == Side.SHORT:
        risk_per_share = event.stop_loss - event.entry
    else:
        return 0
    if risk_per_share <= 0:
        return 0
    risk_budget = capital * risk_per_trade_pct / 100
    max_position_value = capital * max_position_pct / 100
    return max(0, min(int(risk_budget // risk_per_share), int(max_position_value // event.entry)))


def _summary_from_trades(
    starting_capital: float,
    capital: float,
    trades: list[dict],
    equity_curve: list[dict],
) -> tuple[BacktestSummary, pd.DataFrame, pd.DataFrame]:
    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_curve).drop_duplicates(subset=["timestamp"], keep="last")
    wins = int((trades_frame["pnl"] > 0).sum()) if not trades_frame.empty else 0
    losses = int((trades_frame["pnl"] <= 0).sum()) if not trades_frame.empty else 0
    running_max = equity_frame["equity"].cummax() if not equity_frame.empty else pd.Series([capital])
    drawdowns = running_max - equity_frame["equity"] if not equity_frame.empty else pd.Series([0])
    max_drawdown = float(drawdowns.max()) if not drawdowns.empty else 0.0
    max_equity = float(running_max.max()) if not running_max.empty else starting_capital
    summary = BacktestSummary(
        starting_capital=starting_capital,
        ending_capital=capital,
        pnl=capital - starting_capital,
        pnl_pct=((capital - starting_capital) / starting_capital * 100),
        trades=len(trades),
        wins=wins,
        losses=losses,
        win_rate=(wins / len(trades) * 100) if trades else 0.0,
        max_drawdown=max_drawdown,
        max_drawdown_pct=(max_drawdown / max_equity * 100) if max_equity else 0.0,
    )
    return summary, trades_frame, equity_frame


def _candidate_vote_stats(plan: TradePlan) -> tuple[int, float, tuple[str, ...]]:
    trade_votes = [vote for vote in plan.strategy_votes if vote.is_trade and vote.confidence >= 55]
    agreeing = [vote for vote in trade_votes if vote.side == plan.side]
    total_weight = sum(vote.weight for vote in trade_votes)
    agreeing_weight = sum(vote.weight for vote in agreeing)
    vote_share = agreeing_weight / total_weight if total_weight else 0.0
    return len(agreeing), vote_share, tuple(vote.strategy for vote in agreeing)


def _limit_fill(
    *,
    plan: TradePlan,
    history_row: pd.Series,
    future: pd.DataFrame,
    signal_ts: pd.Timestamp,
    offset_atr: float,
    fill_window_minutes: int,
) -> tuple[TradePlan, pd.Timestamp, pd.DataFrame] | None:
    """Simulate a pullback limit entry.

    Returns (adjusted_plan, fill_time, remaining_future) or None if the limit
    never fills inside the window.  The stop and target stay at their original
    structure-based levels; only the entry improves.  If the fill candle also
    breaches the stop, the trade is kept — _trade_exit will register the stop
    on that candle since it is included in remaining_future.
    """
    if plan.entry is None or plan.stop_loss is None or plan.target is None:
        return None
    atr_val = float(history_row.get("atr_14", 0) or 0)
    if atr_val <= 0:
        return None
    offset = atr_val * offset_atr
    deadline = signal_ts + timedelta(minutes=fill_window_minutes)

    if plan.side == Side.LONG:
        limit_price = float(plan.entry) - offset
        if limit_price <= float(plan.stop_loss):
            return None  # pullback would already be through the stop
    else:
        limit_price = float(plan.entry) + offset
        if limit_price >= float(plan.stop_loss):
            return None

    for idx, (fts, frow) in enumerate(future.iterrows()):
        if fts > deadline:
            return None
        touched = (
            float(frow["low"]) <= limit_price
            if plan.side == Side.LONG
            else float(frow["high"]) >= limit_price
        )
        if touched:
            risk = abs(limit_price - float(plan.stop_loss))
            reward = abs(float(plan.target) - limit_price)
            adjusted = replace(
                plan,
                entry=limit_price,
                reward_risk=reward / risk if risk > 0 else 0.0,
            )
            return adjusted, fts, future.iloc[idx:]
    return None


def generate_candidate_events(
    *,
    data: dict[str, DataResult],
    config: BacktestConfig,
    risk_per_trade_pct: float,
    max_position_pct: float,
    strategy_weights: dict[str, float] | None = None,
    context_series=None,
) -> list[CandidateEvent]:
    frames = {symbol: result.frame for symbol, result in data.items() if not result.frame.empty}
    frames = {symbol: add_indicators(frame) for symbol, frame in frames.items()}
    frames = {symbol: frame for symbol, frame in frames.items() if not frame.empty}
    if not frames:
        return []

    timestamps = sorted(set().union(*(frame.index for frame in frames.values())))
    engine = VotingSignalEngine(
        config=EnsembleConfig(
            min_agreeing_votes=1,
            min_vote_share=0.0,
            min_weighted_confidence=0.0,
        )
    )
    lenient_risk = RiskConfig(
        capital=config.starting_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_pct=max_position_pct,
        min_confidence=0.0,
        min_reward_risk=0.0,
        estimated_cost_bps=config.estimated_cost_bps,
        slippage_bps=config.slippage_bps,
    )
    events: list[CandidateEvent] = []

    for ts in timestamps:
        market_context = context_series.at(ts) if context_series is not None else None
        for symbol, frame in frames.items():
            if ts not in frame.index:
                continue
            pos = frame.index.get_loc(ts)
            if isinstance(pos, slice):
                pos = pos.stop - 1
            if pos < config.warmup_candles:
                continue
            history = frame.iloc[: pos + 1]
            plan = engine.analyze_precomputed(
                symbol, history, lenient_risk, strategy_weights or {}, market_context
            )
            if not plan.is_actionable or plan.entry is None or plan.stop_loss is None or plan.target is None:
                continue
            if not config.include_shorts and plan.side == Side.SHORT:
                continue
            future = frame.iloc[pos + 1 :]
            if config.same_day_exit_only:
                future = future[future.index.normalize() == ts.normalize()]
            if future.empty:
                continue

            entry_ts = ts
            limit_filled = False
            if config.entry_limit_offset_atr > 0:
                filled = _limit_fill(
                    plan=plan,
                    history_row=history.iloc[-1],
                    future=future,
                    signal_ts=ts,
                    offset_atr=config.entry_limit_offset_atr,
                    fill_window_minutes=config.entry_fill_window_minutes,
                )
                if filled is None:
                    continue  # limit never filled — signal skipped, no chase
                plan, entry_ts, future = filled
                limit_filled = True
                if future.empty:
                    continue

            exit_ts, exit_price, exit_reason = _trade_exit(
                plan,
                future,
                config.max_hold_minutes,
                entry_ts,
                config.breakeven_after_r,
                skip_target_on_first=limit_filled,
                trailing_atr_mult=config.trailing_atr_mult,
                trailing_replaces_target=config.trailing_replaces_target,
            )
            per_share_pnl = (
                exit_price - plan.entry
                if plan.side == Side.LONG
                else plan.entry - exit_price
            )
            agreeing_count, vote_share, agreeing_strategies = _candidate_vote_stats(plan)
            events.append(
                CandidateEvent(
                    timestamp=entry_ts,
                    symbol=symbol,
                    side=plan.side,
                    entry=float(plan.entry),
                    stop_loss=float(plan.stop_loss),
                    target=float(plan.target),
                    confidence=float(plan.confidence),
                    reward_risk=float(plan.reward_risk),
                    agreeing_count=agreeing_count,
                    vote_share=vote_share,
                    exit_time=exit_ts,
                    exit_price=float(exit_price),
                    exit_reason=exit_reason,
                    per_share_pnl=float(per_share_pnl),
                    reasons=tuple(plan.reasons),
                    agreeing_strategies=agreeing_strategies,
                )
            )
    return events


def run_backtest_from_events(
    *,
    events: list[CandidateEvent],
    starting_capital: float,
    risk_per_trade_pct: float,
    max_position_pct: float,
    cooldown_minutes: int,
    min_confidence: float,
    min_reward_risk: float,
    min_agreeing_votes: int,
    min_vote_share: float,
    config: BacktestConfig | None = None,
) -> tuple[BacktestSummary, pd.DataFrame, pd.DataFrame]:
    if not events:
        summary = BacktestSummary(
            starting_capital=starting_capital,
            ending_capital=starting_capital,
            pnl=0,
            pnl_pct=0,
            trades=0,
            wins=0,
            losses=0,
            win_rate=0,
            max_drawdown=0,
            max_drawdown_pct=0,
        )
        return summary, pd.DataFrame(), pd.DataFrame()

    by_time: dict[pd.Timestamp, list[CandidateEvent]] = {}
    for event in events:
        by_time.setdefault(event.timestamp, []).append(event)

    capital = float(starting_capital)
    next_allowed_entry = min(by_time)
    equity_curve: list[dict] = [{"timestamp": next_allowed_entry, "equity": capital}]
    trades: list[dict] = []
    current_day = None
    day_start_capital = capital
    trades_today = 0
    symbol_next_allowed: dict[str, pd.Timestamp] = {}
    controls = config or BacktestConfig(starting_capital=starting_capital, cooldown_minutes=cooldown_minutes)

    for ts in sorted(by_time):
        equity_curve.append({"timestamp": ts, "equity": capital})
        reset_day, reset_capital, reset_trades = _reset_day_if_needed(ts, current_day, capital)
        if reset_trades == 0:
            current_day = reset_day
            day_start_capital = reset_capital
            trades_today = 0
        if ts < next_allowed_entry:
            continue
        if _past_entry_cutoff(ts, controls.no_new_trade_after):
            continue
        if controls.max_trades_per_day > 0 and trades_today >= controls.max_trades_per_day:
            continue
        if _daily_loss_limit_hit(
            capital=capital,
            day_start_capital=day_start_capital,
            daily_loss_limit_pct=controls.daily_loss_limit_pct,
        ):
            continue
        candidates = [
            event
            for event in by_time[ts]
            if event.confidence >= min_confidence
            and event.reward_risk >= min_reward_risk
            and event.agreeing_count >= min_agreeing_votes
            and event.vote_share >= min_vote_share
            and ts >= symbol_next_allowed.get(event.symbol, ts)
        ]
        if not candidates:
            continue
        event = max(candidates, key=lambda item: (item.confidence, item.reward_risk))
        quantity = _quantity_for_event(
            capital=capital,
            risk_per_trade_pct=risk_per_trade_pct,
            max_position_pct=max_position_pct,
            event=event,
        )
        if quantity <= 0:
            continue
        gross_pnl = event.per_share_pnl * quantity
        costs = _roundtrip_cost(
            entry=event.entry,
            quantity=quantity,
            estimated_cost_bps=controls.estimated_cost_bps,
            slippage_bps=controls.slippage_bps,
        )
        pnl = gross_pnl - costs
        capital += pnl
        trades.append(
            {
                "entry_time": event.timestamp,
                "exit_time": event.exit_time,
                "symbol": event.symbol,
                "side": event.side.value,
                "entry": round(event.entry, 4),
                "exit": round(event.exit_price, 4),
                "stop_loss": round(event.stop_loss, 4),
                "target": round(event.target, 4),
                "quantity": quantity,
                "confidence": round(event.confidence, 2),
                "reward_risk": round(event.reward_risk, 2),
                "gross_pnl": round(gross_pnl, 2),
                "costs": round(costs, 2),
                "pnl": round(pnl, 2),
                "pnl_pct_on_capital": round(pnl / capital * 100, 4) if capital else 0,
                "exit_reason": event.exit_reason,
                "hold_minutes": round((event.exit_time - event.timestamp).total_seconds() / 60, 2),
                "reasons": " | ".join(event.reasons[:4]),
                "agreeing_strategies": ", ".join(event.agreeing_strategies),
                "capital_after": round(capital, 2),
            }
        )
        equity_curve.append({"timestamp": event.exit_time, "equity": capital})
        next_allowed_entry = ts + timedelta(minutes=cooldown_minutes)
        trades_today += 1
        symbol_lockout = controls.symbol_cooldown_minutes
        if event.exit_reason == "STOP_LOSS":
            symbol_lockout = max(symbol_lockout, controls.stop_loss_cooldown_minutes)
        if symbol_lockout > 0:
            symbol_next_allowed[event.symbol] = ts + timedelta(minutes=symbol_lockout)

    return _summary_from_trades(starting_capital, capital, trades, equity_curve)


def run_backtest(
    *,
    data: dict[str, DataResult],
    risk_config: RiskConfig,
    ensemble_config: EnsembleConfig,
    config: BacktestConfig,
    strategy_weights: dict[str, float] | None = None,
    context_series=None,
) -> tuple[BacktestSummary, pd.DataFrame, pd.DataFrame]:
    frames = {symbol: result.frame for symbol, result in data.items() if not result.frame.empty}
    frames = {symbol: add_indicators(frame) for symbol, frame in frames.items()}
    frames = {symbol: frame for symbol, frame in frames.items() if not frame.empty}
    if not frames:
        empty_summary = BacktestSummary(
            starting_capital=config.starting_capital,
            ending_capital=config.starting_capital,
            pnl=0,
            pnl_pct=0,
            trades=0,
            wins=0,
            losses=0,
            win_rate=0,
            max_drawdown=0,
            max_drawdown_pct=0,
        )
        return empty_summary, pd.DataFrame(), pd.DataFrame()

    timestamps = sorted(set().union(*(frame.index for frame in frames.values())))
    capital = float(config.starting_capital)
    equity_curve: list[dict] = []
    trades: list[dict] = []
    next_allowed_entry = timestamps[0]
    engine = VotingSignalEngine(config=ensemble_config)
    current_day = None
    day_start_capital = capital
    trades_today = 0
    symbol_next_allowed: dict[str, pd.Timestamp] = {}

    for ts in timestamps:
        equity_curve.append({"timestamp": ts, "equity": capital})
        reset_day, reset_capital, reset_trades = _reset_day_if_needed(ts, current_day, capital)
        if reset_trades == 0:
            current_day = reset_day
            day_start_capital = reset_capital
            trades_today = 0
        if ts < next_allowed_entry:
            continue
        if _past_entry_cutoff(ts, config.no_new_trade_after):
            continue
        if config.max_trades_per_day > 0 and trades_today >= config.max_trades_per_day:
            continue
        if _daily_loss_limit_hit(
            capital=capital,
            day_start_capital=day_start_capital,
            daily_loss_limit_pct=config.daily_loss_limit_pct,
        ):
            continue

        candidates: list[tuple[TradePlan, str, pd.DataFrame]] = []
        dynamic_risk = RiskConfig(
            capital=capital,
            risk_per_trade_pct=risk_config.risk_per_trade_pct,
            max_position_pct=risk_config.max_position_pct,
            min_reward_risk=risk_config.min_reward_risk,
            min_confidence=risk_config.min_confidence,
            estimated_cost_bps=risk_config.estimated_cost_bps,
            slippage_bps=risk_config.slippage_bps,
        )

        market_context = context_series.at(ts) if context_series is not None else None
        for symbol, frame in frames.items():
            if ts not in frame.index:
                continue
            pos = frame.index.get_loc(ts)
            if isinstance(pos, slice):
                pos = pos.stop - 1
            if pos < config.warmup_candles:
                continue
            history = frame.iloc[: pos + 1]
            if history.index[-1].normalize() != ts.normalize():
                continue
            plan = engine.analyze_precomputed(
                symbol, history, dynamic_risk, strategy_weights or {}, market_context
            )
            if not config.include_shorts and plan.side == Side.SHORT:
                continue
            if ts < symbol_next_allowed.get(symbol, ts):
                continue
            if plan.is_actionable and plan.quantity > 0:
                future = frame.iloc[pos + 1 :]
                if config.same_day_exit_only:
                    future = future[future.index.normalize() == ts.normalize()]
                if not future.empty:
                    candidates.append((plan, symbol, future))

        if not candidates:
            continue

        plan, symbol, future = sorted(
            candidates,
            key=lambda item: (item[0].confidence, item[0].reward_risk, item[0].reward_amount),
            reverse=True,
        )[0]
        if future.empty:
            continue

        exit_ts, exit_price, exit_reason = _trade_exit(
            plan, future, config.max_hold_minutes, ts, config.breakeven_after_r,
            trailing_atr_mult=config.trailing_atr_mult,
            trailing_replaces_target=config.trailing_replaces_target,
        )
        gross_pnl = _gross_pnl(plan, exit_price)
        costs = _roundtrip_cost(
            entry=plan.entry or 0,
            quantity=plan.quantity,
            estimated_cost_bps=risk_config.estimated_cost_bps,
            slippage_bps=risk_config.slippage_bps,
        )
        pnl = gross_pnl - costs
        capital += pnl
        trades.append(
            {
                "entry_time": ts,
                "exit_time": exit_ts,
                "symbol": symbol,
                "side": plan.side.value,
                "entry": round(plan.entry or 0, 4),
                "exit": round(exit_price, 4),
                "stop_loss": round(plan.stop_loss or 0, 4),
                "target": round(plan.target or 0, 4),
                "quantity": plan.quantity,
                "confidence": round(plan.confidence, 2),
                "reward_risk": round(plan.reward_risk, 2),
                "gross_pnl": round(gross_pnl, 2),
                "costs": round(costs, 2),
                "pnl": round(pnl, 2),
                "pnl_pct_on_capital": round(pnl / capital * 100, 4) if capital else 0,
                "exit_reason": exit_reason,
                "hold_minutes": round((exit_ts - ts).total_seconds() / 60, 2),
                "reasons": " | ".join(plan.reasons[:4]),
                "agreeing_strategies": ", ".join(
                    vote.strategy for vote in plan.strategy_votes if vote.side == plan.side
                ),
                "capital_after": round(capital, 2),
            }
        )
        next_allowed_entry = ts + timedelta(minutes=config.cooldown_minutes)
        trades_today += 1
        symbol_lockout = config.symbol_cooldown_minutes
        if exit_reason == "STOP_LOSS":
            symbol_lockout = max(symbol_lockout, config.stop_loss_cooldown_minutes)
        if symbol_lockout > 0:
            symbol_next_allowed[symbol] = ts + timedelta(minutes=symbol_lockout)
        equity_curve.append({"timestamp": exit_ts, "equity": capital})

    return _summary_from_trades(config.starting_capital, capital, trades, equity_curve)


def calibrate_backtest(
    *,
    data: dict[str, DataResult],
    starting_capital: float,
    risk_per_trade_pct: float,
    max_position_pct: float,
    config: BacktestConfig,
    target_trades: int = 10,
    objective: str = "risk_adjusted",
    confidence_grid: tuple[float, ...] = (55, 60, 65, 70, 75, 80, 85),
    reward_risk_grid: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 1.7),
    agreeing_votes_grid: tuple[int, ...] = (1, 2),
    vote_share_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.75),
    context_series=None,
) -> tuple[pd.DataFrame, CalibrationResult | None]:
    rows: list[dict] = []
    best: CalibrationResult | None = None
    events = generate_candidate_events(
        data=data,
        config=config,
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_pct=max_position_pct,
        context_series=context_series,
    )

    for min_confidence in confidence_grid:
        for min_reward_risk in reward_risk_grid:
            for min_agreeing_votes in agreeing_votes_grid:
                for min_vote_share in vote_share_grid:
                    if min_agreeing_votes == 1 and min_vote_share > 0.6:
                        continue
                    summary, _, _ = run_backtest_from_events(
                        events=events,
                        starting_capital=starting_capital,
                        risk_per_trade_pct=risk_per_trade_pct,
                        max_position_pct=max_position_pct,
                        cooldown_minutes=config.cooldown_minutes,
                        min_confidence=min_confidence,
                        min_reward_risk=min_reward_risk,
                        min_agreeing_votes=min_agreeing_votes,
                        min_vote_share=min_vote_share,
                        config=config,
                    )
                    row = {
                        "min_confidence": min_confidence,
                        "min_reward_risk": min_reward_risk,
                        "min_agreeing_votes": min_agreeing_votes,
                        "min_vote_share": min_vote_share,
                        "trades": summary.trades,
                        "ending_capital": round(summary.ending_capital, 2),
                        "pnl": round(summary.pnl, 2),
                        "pnl_pct": round(summary.pnl_pct, 3),
                        "win_rate": round(summary.win_rate, 2),
                        "max_drawdown": round(summary.max_drawdown, 2),
                        "max_drawdown_pct": round(summary.max_drawdown_pct, 3),
                        "pnl_to_drawdown": round(summary.pnl / max(summary.max_drawdown, 1), 3),
                        "risk_adjusted_score": round(
                            (summary.pnl / max(summary.max_drawdown, 1))
                            - (summary.losses / summary.trades if summary.trades else 1),
                            3,
                        ),
                    }
                    rows.append(row)
                    if summary.trades < target_trades:
                        continue
                    candidate = CalibrationResult(
                        min_confidence=min_confidence,
                        min_reward_risk=min_reward_risk,
                        min_agreeing_votes=min_agreeing_votes,
                        min_vote_share=min_vote_share,
                        summary=summary,
                    )
                    if best is None:
                        best = candidate
                        continue
                    if _calibration_key(candidate, objective) > _calibration_key(best, objective):
                        best = candidate

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["trades", "pnl", "max_drawdown"],
            ascending=[False, False, True],
        )
    return frame, best
