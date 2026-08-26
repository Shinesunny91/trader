from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_ROOT = _SRC.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_IST = ZoneInfo("Asia/Kolkata")

import nse_intraday_ai.scan_config as _sc
from nse_intraday_ai.scan_config import save as _save_scan_config
from nse_intraday_ai.signals_log import record as _log_signal, tail as _signal_log_tail
from nse_intraday_ai.backtest import (
    BacktestConfig,
    calibrate_backtest,
    load_backtest_data,
    run_backtest,
)
from nse_intraday_ai.data import (
    DEFAULT_SYMBOLS,
    DEFAULT_COMMODITY_SYMBOLS,
    DemoProvider,
    GoogleFinanceQuoteClient,
    NsePublicQuoteClient,
    UniverseResult,
    YFinanceProvider,
    load_commodity_symbols,
    load_nifty50_symbols,
    load_nifty100_symbols,
    load_nifty500_symbols,
)
from nse_intraday_ai.learning import AdaptiveWeightEngine, ShadowLearner
from nse_intraday_ai.models import Side, TradePlan
from nse_intraday_ai.scan_service import run_scan_cycle
from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.scanner import (
    ScanResult,
    leading_trade_side,
)
from nse_intraday_ai.simulator import AccountSummary, PaperTradingStore
from nse_intraday_ai.strategies import EnsembleConfig, VotingSignalEngine

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "paper_trades.sqlite3"
LEARNER_DB_PATH = ROOT / "data" / "shadow_learner.sqlite3"
NIFTY50_CACHE_PATH = ROOT / "data" / "nifty50_symbols.csv"
NIFTY100_CACHE_PATH = ROOT / "data" / "nifty100_symbols.csv"
NIFTY500_CACHE_PATH = ROOT / "data" / "nifty500_symbols.csv"
_DAEMON_STATE_PATH = ROOT / "data" / "daemon_state.json"


def _notify(title: str, body: str, urgency: str = "normal") -> None:
    env = os.environ.copy()
    if "DBUS_SESSION_BUS_ADDRESS" not in env:
        env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path=/run/user/{os.getuid()}/bus"
    if "DISPLAY" not in env:
        env["DISPLAY"] = ":1"
    try:
        subprocess.run(
            ["notify-send", title, body,
             f"--urgency={urgency}",
             "--app-name=NSE Signal Lab",
             "--icon=dialog-information",
             "--expire-time=30000"],
            env=env, timeout=5, check=False,
        )
    except Exception:
        pass


def _send_new_signal_notifications(recommendations: list) -> None:
    """Fire notify-send for any actionable signal not yet notified by the daemon."""
    if (ROOT / "data" / "notifications_paused").exists():
        return
    state: dict = {}
    if _DAEMON_STATE_PATH.exists():
        try:
            state = json.loads(_DAEMON_STATE_PATH.read_text())
        except Exception:
            pass
    today = datetime.now(_IST).date().isoformat()
    notified = {k: v for k, v in state.get("notified", {}).items() if v.get("date") == today}
    now_str = datetime.now(_IST).strftime("%H:%M")
    changed = False
    for result in recommendations:
        plan = result.plan
        if not plan.is_actionable or not plan.entry:
            continue
        key = f"{result.symbol}|{plan.side.value}|{plan.entry:.2f}"
        if key in notified:
            continue
        votes = [v for v in plan.strategy_votes if v.side == plan.side and v.is_trade]
        strats = ", ".join(v.strategy for v in votes)
        body = (
            f"{plan.side.value}  entry={plan.entry:.2f}  "
            f"SL={plan.stop_loss:.2f}  T={plan.target:.2f}  "
            f"RR={plan.reward_risk:.2f}  conf={plan.confidence:.0f}%\n"
            f"Regime: {result.regime}  Strategies: {strats}"
        )
        _notify(f"\U0001f6a8 {result.symbol}", body, urgency="critical")
        notified[key] = {"date": today, "time": now_str}
        changed = True
    if changed:
        state["notified"] = notified
        _DAEMON_STATE_PATH.write_text(json.dumps(state, indent=2))


st.set_page_config(page_title="NSE & Commodity Intraday Signal Lab", page_icon="INR", layout="wide")


@st.cache_data(ttl=3600)
def _cached_nifty50_symbols(cache_path: str) -> UniverseResult:
    return load_nifty50_symbols(Path(cache_path))


@st.cache_data(ttl=3600)
def _cached_nifty100_symbols(cache_path: str) -> UniverseResult:
    return load_nifty100_symbols(Path(cache_path))


@st.cache_data(ttl=3600)
def _cached_nifty500_symbols(cache_path: str) -> UniverseResult:
    return load_nifty500_symbols(Path(cache_path))


@st.cache_data(ttl=3600)
def _cached_commodity_symbols() -> UniverseResult:
    return load_commodity_symbols()


def _format_money(value: float) -> str:
    return f"INR {value:,.2f}"


def _entry_action(side: Side) -> str:
    if side == Side.LONG:
        return "BUY"
    if side == Side.SHORT:
        return "SELL"
    return "WAIT"


def _exit_action(side: Side) -> str:
    if side == Side.LONG:
        return "SELL"
    if side == Side.SHORT:
        return "BUY"
    return "WAIT"


def _side_explanation(side: Side) -> str:
    if side == Side.LONG:
        return "Long trade: buy first, then sell at target or stop-loss."
    if side == Side.SHORT:
        return "Short trade: sell first, then buy back at target or stop-loss."
    return "No trade."


def _render_action_banner(plan: TradePlan) -> None:
    action = _entry_action(plan.side)
    exit_action = _exit_action(plan.side)
    is_buy = plan.side == Side.LONG
    border = "#16a34a" if is_buy else "#dc2626"
    background = "#dcfce7" if is_buy else "#fee2e2"
    foreground = "#14532d" if is_buy else "#7f1d1d"
    st.markdown(
        f"""
        <div style="border:2px solid {border}; background:{background}; color:{foreground};
                    border-radius:8px; padding:14px 16px; margin:4px 0 14px 0;">
            <div style="font-size:12px; font-weight:700; text-transform:uppercase;">
                Entry action
            </div>
            <div style="font-size:34px; line-height:1.1; font-weight:800;">
                {action}
            </div>
            <div style="font-size:14px; font-weight:600; margin-top:4px;">
                Exit action: {exit_action} at target or stop-loss
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_candles(df: pd.DataFrame, symbol: str) -> go.Figure:
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(
            f"{symbol} Intraday Technical & Volume Structure",
            "Volume & Microstructure Flow",
            "Momentum & Trend Strength (RSI / ADX)",
        ),
    )

    # 1. Main Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
        ),
        row=1, col=1,
    )

    # Overlays: VWAP, EMAs, Supertrend, Volume Profile POC/VAH/VAL
    overlay_specs = [
        ("vwap", "#059669", "dash", 1.8),
        ("ema_9", "#2563eb", "solid", 1.2),
        ("ema_21", "#dc2626", "solid", 1.2),
        ("ema_50", "#9333ea", "solid", 1.2),
        ("supertrend", "#d97706", "dot", 1.5),
        ("poc", "#0891b2", "dash", 1.5),
        ("vah", "#64748b", "dot", 1.0),
        ("val", "#64748b", "dot", 1.0),
        ("vwap_u2", "#38bdf8", "dashdot", 1.0),
        ("vwap_l2", "#38bdf8", "dashdot", 1.0),
    ]
    for col, color, dash, width in overlay_specs:
        if col in df and df[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col.upper(),
                    line=dict(color=color, dash=dash, width=width),
                ),
                row=1, col=1,
            )

    # 2. Volume and Order Flow Proxy
    vol_colors = [
        "#10b981" if c >= o else "#ef4444"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker=dict(color=vol_colors, opacity=0.7),
        ),
        row=2, col=1,
    )
    if "clv_flow" in df and df["clv_flow"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["clv_flow"] * (df["volume"].mean() if "volume" in df else 1.0),
                mode="lines",
                name="CLV Flow Proxy",
                line=dict(color="#6366f1", width=1.5),
            ),
            row=2, col=1,
        )

    # 3. RSI and ADX
    if "rsi_14" in df and df["rsi_14"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rsi_14"],
                mode="lines",
                name="RSI (14)",
                line=dict(color="#2563eb", width=1.5),
            ),
            row=3, col=1,
        )
    if "adx_14" in df and df["adx_14"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["adx_14"],
                mode="lines",
                name="ADX (14)",
                line=dict(color="#e11d48", width=1.5),
            ),
            row=3, col=1,
        )

    fig.add_hline(y=70, line=dict(color="#94a3b8", dash="dot", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#94a3b8", dash="dot", width=1), row=3, col=1)

    fig.update_layout(
        height=680,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI / ADX", range=[0, 100], row=3, col=1)
    return fig


def _plot_equity_curve(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["equity"],
                mode="lines",
                name="Equity",
                line=dict(color="#2563eb"),
            )
        )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        template="plotly_white",
        yaxis_title="Capital",
    )
    return fig


def _render_account(account: AccountSummary) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity", _format_money(account.equity))
    c2.metric("Realized P&L", _format_money(account.realized_pnl))
    c3.metric("Open Risk", _format_money(account.open_risk))
    c4.metric("Closed Trades", str(account.closed_trades))
    c5.metric("Win Rate", f"{account.win_rate:.1f}%")


def _render_plan(plan: TradePlan, title: str = "Current Decision") -> None:
    st.subheader(title)
    if plan.is_actionable:
        _render_action_banner(plan)
        st.metric("Signal", f"{_entry_action(plan.side)} ({plan.side.value})", f"{plan.confidence:.1f}% confidence")
        st.caption(_side_explanation(plan.side))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", _format_money(plan.entry or 0))
        c2.metric("Stop Loss", _format_money(plan.stop_loss or 0))
        c3.metric("Target", _format_money(plan.target or 0))
        c4.metric("Quantity", f"{plan.quantity}")
        c5, c6, c7 = st.columns(3)
        c5.metric("Max Risk", _format_money(plan.risk_amount))
        c6.metric("Net Reward", _format_money(plan.reward_amount))
        c7.metric("Reward/Risk", f"{plan.reward_risk:.2f}")
    else:
        st.metric("Signal", "WAIT", f"{plan.confidence:.1f}% confidence")

    with st.expander("Why this decision", expanded=True):
        for reason in plan.reasons:
            st.write(f"- {reason}")


def _votes_frame(plan: TradePlan) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": vote.strategy,
                "side": vote.side.value,
                "confidence": round(vote.confidence, 1),
                "adaptive_weight": round(vote.weight, 2),
                "reason": vote.reason,
            }
            for vote in plan.strategy_votes
        ]
    )


def _policy_cells(
    result: ScanResult,
    learner: ShadowLearner | None,
    policy_min_samples: int,
) -> dict:
    if learner is None:
        return {}
    estimate = learner.estimate_for_result(result, min_samples=policy_min_samples)
    return {
        "rl_samples": estimate.samples,
        "rl_win_rate": round(estimate.win_rate, 1),
        "rl_avg_bps": round(estimate.avg_reward_bps, 2),
        "rl_ucb": round(estimate.ucb_score, 2),
        "rl_bonus": round(estimate.confidence_bonus, 2),
        "rl_ready": estimate.is_trained,
    }


def _policy_rank_key(
    result: ScanResult,
    learner: ShadowLearner | None,
    policy_min_samples: int,
) -> tuple[float, float, float]:
    bonus = 0.0
    if learner is not None:
        bonus = learner.estimate_for_result(result, min_samples=policy_min_samples).confidence_bonus
    return (result.plan.confidence + bonus, result.plan.reward_risk, result.plan.reward_amount)


def _policy_approved(
    result: ScanResult,
    learner: ShadowLearner | None,
    policy_min_samples: int,
) -> bool:
    if learner is None:
        return True
    estimate = learner.estimate_for_result(result, min_samples=policy_min_samples)
    return estimate.is_trained and estimate.avg_reward_bps > 0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _candidate_trade_spec(result: ScanResult, risk_config: RiskConfig) -> dict | None:
    plan = result.plan
    side = plan.side if plan.is_actionable else leading_trade_side(result)
    if side == Side.WAIT:
        return None

    usable_votes = [
        vote
        for vote in plan.strategy_votes
        if vote.side == side
        and vote.is_trade
        and vote.confidence >= 55
        and vote.entry is not None
        and vote.stop_loss is not None
        and vote.target is not None
    ]
    trade_votes = [vote for vote in plan.strategy_votes if vote.is_trade and vote.confidence >= 55]
    if not usable_votes:
        return None

    if (
        plan.entry is not None
        and plan.stop_loss is not None
        and plan.target is not None
        and (plan.side == side or plan.side == Side.WAIT)
    ):
        entry = float(plan.entry)
        stop_loss = float(plan.stop_loss)
        target = float(plan.target)
    else:
        entry = _median([float(vote.entry or 0) for vote in usable_votes])
        stops = [float(vote.stop_loss or 0) for vote in usable_votes]
        targets = [float(vote.target or 0) for vote in usable_votes]
        if side == Side.LONG:
            stop_loss = min(stops)
            target = max(targets)
        else:
            stop_loss = max(stops)
            target = min(targets)

    if side == Side.LONG:
        risk_per_share = entry - stop_loss
        gross_reward_per_share = target - entry
    else:
        risk_per_share = stop_loss - entry
        gross_reward_per_share = entry - target

    if risk_per_share <= 0 or gross_reward_per_share <= 0 or entry <= 0:
        return None

    estimated_cost_per_share = entry * (
        risk_config.estimated_cost_bps + risk_config.slippage_bps
    ) / 10_000
    net_reward_per_share = gross_reward_per_share - estimated_cost_per_share
    gross_reward_risk = gross_reward_per_share / risk_per_share
    net_reward_risk = net_reward_per_share / risk_per_share
    quantity = max(
        0,
        min(
            int(risk_config.risk_budget // risk_per_share),
            int(risk_config.max_position_value // entry),
        ),
    )

    total_weight = sum(vote.weight for vote in trade_votes)
    agreeing_weight = sum(vote.weight for vote in usable_votes)
    latest_price = result.public_ltp if result.public_ltp is not None else result.last_close
    strategies = [vote.strategy for vote in usable_votes]

    return {
        "side": side,
        "action": _entry_action(side),
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "quantity": quantity,
        "risk_amount": quantity * risk_per_share,
        "net_reward": quantity * net_reward_per_share,
        "gross_reward_risk": gross_reward_risk,
        "net_reward_risk": net_reward_risk,
        "latest_price": latest_price,
        "agreeing_votes": len(usable_votes),
        "vote_share": agreeing_weight / total_weight if total_weight else 0.0,
        "agreeing_strategies": strategies,
    }


def _strategy_performance_lookup(strategy_frame: pd.DataFrame | None) -> dict[str, dict]:
    if strategy_frame is None or strategy_frame.empty:
        return {}
    return {
        str(row["strategy"]): {
            "samples": int(row["samples"]),
            "win_rate": float(row["win_rate"]),
            "avg_reward_bps": float(row["avg_reward_bps"]),
        }
        for row in strategy_frame.to_dict("records")
    }


def _recent_strategy_cells(strategies: list[str], strategy_frame: pd.DataFrame | None) -> dict:
    lookup = _strategy_performance_lookup(strategy_frame)
    matched = [lookup[strategy] for strategy in strategies if strategy in lookup]
    if not matched:
        return {
            "recent_strategy_samples": None,
            "recent_strategy_win_rate": None,
            "recent_strategy_avg_bps": None,
        }
    samples = sum(item["samples"] for item in matched)
    win_rate = sum(item["win_rate"] * item["samples"] for item in matched) / samples
    avg_bps = sum(item["avg_reward_bps"] * item["samples"] for item in matched) / samples
    return {
        "recent_strategy_samples": samples,
        "recent_strategy_win_rate": round(win_rate, 2),
        "recent_strategy_avg_bps": round(avg_bps, 2),
    }


def _candidate_history_cells(
    result: ScanResult,
    side: Side,
    strategies: list[str],
    learner: ShadowLearner | None,
    risk_config: RiskConfig,
) -> dict:
    if learner is None:
        return {
            "symbol_30d_samples": None,
            "symbol_30d_win_rate": None,
            "symbol_30d_avg_bps": None,
        }
    history = learner.candidate_performance(
        symbol=result.symbol,
        side=side,
        strategies=strategies,
        days=30,
        cost_bps=risk_config.estimated_cost_bps + risk_config.slippage_bps,
    )
    if not history["samples"]:
        return {
            "symbol_30d_samples": None,
            "symbol_30d_win_rate": None,
            "symbol_30d_avg_bps": None,
        }
    return {
        "symbol_30d_samples": int(history["samples"]),
        "symbol_30d_win_rate": round(float(history["win_rate"]), 2),
        "symbol_30d_avg_bps": round(float(history["avg_reward_bps"]), 2),
    }


def _recommendations_frame(
    recommendations: list[ScanResult],
    learner: ShadowLearner | None = None,
    policy_min_samples: int = 20,
) -> pd.DataFrame:
    rows = []
    for idx, result in enumerate(recommendations, start=1):
        plan = result.plan
        agreeing = [vote.strategy for vote in plan.strategy_votes if vote.side == plan.side]
        latest_price = result.public_ltp if result.public_ltp is not None else result.last_close
        m_rank = result.model_rank or idx
        vz_val = round(float(result.frame["volume_z"].iloc[-1]), 1) if result.frame is not None and "volume_z" in result.frame else None
        clv_val = round(float(result.frame["clv_flow"].iloc[-1]), 2) if result.frame is not None and "clv_flow" in result.frame else None
        pred_bps = f"{result.predicted_net_bps:+.1f} bps" if result.predicted_net_bps is not None else ("—" if result.rank_score is None else f"{result.rank_score:+.1f} pts")

        row = {
                "rank": f"#{m_rank}",
                "model_edge": pred_bps,
                "action": _entry_action(plan.side),
                "symbol": result.symbol,
                "side": plan.side.value,
                "exit_action": _exit_action(plan.side),
                "confidence": round(plan.confidence, 1),
                "policy_confidence": round(
                    plan.confidence
                    + _policy_cells(result, learner, policy_min_samples).get("rl_bonus", 0),
                    1,
                ),
                "vol_z": vz_val,
                "clv_flow": clv_val,
                "entry": round(plan.entry or 0, 2),
                "stop_loss": round(plan.stop_loss or 0, 2),
                "target": round(plan.target or 0, 2),
                "quantity": plan.quantity,
                "max_risk": round(plan.risk_amount, 2),
                "net_reward": round(plan.reward_amount, 2),
                "reward_risk": round(plan.reward_risk, 2),
                "latest_price": round(latest_price, 2) if latest_price is not None else None,
                "public_ltp": round(result.public_ltp, 2) if result.public_ltp is not None else None,
                "quote_age_sec": (
                    round(result.quote_age_seconds)
                    if result.quote_age_seconds is not None
                    else None
                ),
                "regime": result.regime or "—",
                "agreeing_strategies": ", ".join(agreeing),
                "timestamp": plan.timestamp,
            }
        row.update(_policy_cells(result, learner, policy_min_samples))
        rows.append(row)
    return pd.DataFrame(rows)


def _render_execution_tickets(recommendations: list[ScanResult], risk_config: RiskConfig) -> None:
    """The actionable panel: exactly the trade the 49-session study measured.

    The recommendation table below this shows the raw engine plan, whose entry
    is the signal bar's close (unattainable) and whose stop/target come from
    each strategy's own levels rather than the validated 1.5/3.0 ATR design.
    Trading that table while believing the simulation means trading a setup
    that was never tested, so the tradeable version gets top billing and the
    raw one is labelled as reference.
    """
    import json
    from datetime import datetime

    from nse_intraday_ai.execution_plan import (
        MAX_TRADES_PER_DAY,
        build_execution_plan,
        expectancy_note,
    )

    st.subheader("Order Tickets — the validated trade")

    # Prefer tickets published by the daily book (scripts/sim_today.py). That
    # script *is* the validated path — same ranking model, same portfolio
    # rules — so reading its output keeps the screen and the measured book from
    # drifting apart, which is exactly how the 2026-07-07 config incident
    # happened. Fall back to building tickets inline only if it has not run for
    # this session.
    published = ROOT / "data" / "today_tickets.json"
    payload = None
    if published.exists():
        try:
            candidate = json.loads(published.read_text())
            if candidate.get("session") == datetime.now(_IST).date().isoformat():
                payload = candidate
        except Exception:
            payload = None

    if payload is not None:
        st.warning(payload.get("expectancy", expectancy_note()))
        cap = payload.get("daily_cap", MAX_TRADES_PER_DAY)
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Trades taken today",
            f"{payload.get('trades_taken', 0)}" + ("" if cap <= 0 else f" / {cap}"),
            help="No daily cap is set — the book is bounded by concurrency and capital."
            if cap <= 0 else None,
        )
        c2.metric("Open slots", str(payload.get("slots_remaining", 0)))
        c3.metric("Session P&L", f"₹{payload.get('session_pnl', 0):+,.0f}")
        if cap <= 0:
            st.error(
                "**Daily trade cap is OFF.** Measured over 34 held-out sessions on the "
                "same signals: capping at 3 trades/day returned +5.22%; uncapped "
                "returned −13.53% on 897 positions instead of 102. Gross edge falls "
                "23.0 → 3.3 bps because the extra trades are lower-ranked, while each "
                "still pays its full ~8 bps round trip. Restore the cap by setting "
                "`MAX_TRADES_PER_DAY = 1` in `src/nse_intraday_ai/execution_plan.py` "
                "— 1 is the validated setting since 2026-08-17, not 3."
            )
        st.caption(
            f"Published {payload.get('generated_at', '?')} — refreshes every 5 minutes "
            f"while the market is open. {payload.get('gated_signals', 0):,} signals "
            f"ranked today (the book takes the best one; filtering before ranking was "
            f"measured worse). Capital ₹{payload.get('capital', 0):,.0f}."
        )
        if not payload.get("tickets"):
            st.info(payload.get("status", "No tradable ticket right now."))
        else:
            st.success(payload.get("status", ""))
        for entry in payload["tickets"]:
            st.code(entry["ticket"], language="text")
            st.caption(
                f"signal fired {entry.get('age_minutes', 0):.0f} min ago "
                f"({entry.get('signal_time', '')[11:16]}), ranked by "
                f"{entry.get('ranked_by', '?')}"
            )
        return

    st.warning(expectancy_note())
    st.caption(
        "Built inline from the live scan — the daily book has not published "
        "tickets for this session yet (`python scripts/sim_today.py`)."
    )
    if not recommendations:
        st.info("No ranked signal is fresh enough to act on. A flat book is a position.")
        return

    st.caption(
        f"Capital ₹{risk_config.capital:,.0f} from the sidebar. The study capped the "
        f"book at {MAX_TRADES_PER_DAY} trades a day — taking more was measurably "
        f"worse, not better — so only the top {MAX_TRADES_PER_DAY} are ticketed."
    )
    for index, result in enumerate(recommendations[:MAX_TRADES_PER_DAY]):
        plan = result.plan
        atr = None
        if result.frame is not None and not result.frame.empty and "atr_14" in result.frame:
            atr = float(result.frame["atr_14"].iloc[-1])
        ticket = build_execution_plan(
            symbol=result.symbol,
            side=plan.side.value,
            signal_price=float(plan.entry or result.last_close or 0.0),
            atr=atr or 0.0,
            capital=risk_config.capital,
            taken_today=index,
            predicted_net_bps=result.predicted_net_bps,
            model_rank=result.model_rank or (index + 1),
        )
        st.code(ticket.order_ticket(), language="text")
        import urllib.parse
        from nse_intraday_ai.alerts import (
            send_order_ticket_telegram, telegram_configured,
        )
        wa_text = f"🚨 *NSE TRADE TICKET*\n{ticket.order_ticket()}\n\n📱 Live Terminal: https://were-grid-residents-others.trycloudflare.com"
        wa_url = f"https://api.whatsapp.com/send?phone=918123157952&text={urllib.parse.quote(wa_text)}"
        col_wa, col_tg = st.columns(2)
        with col_wa:
            st.markdown(f'<a href="{wa_url}" target="_blank" style="display:inline-block;padding:8px 16px;background-color:#25D366;color:white;text-decoration:none;border-radius:6px;font-weight:bold;margin-bottom:12px;">📲 WhatsApp</a>', unsafe_allow_html=True)
        with col_tg:
            if telegram_configured():
                tg_key = f"tg_send_{result.symbol}_{index}"
                if st.button(f"📨 Send to Telegram", key=tg_key):
                    ok = send_order_ticket_telegram(ticket.order_ticket())
                    if ok:
                        st.success("✅ Sent to Telegram!")
                    else:
                        st.error("❌ Telegram send failed")
            else:
                st.caption("⚙️ Telegram not configured — see Settings")
        # Auto-push to Telegram if configured
        if telegram_configured():
            _tg_sent_key = f"_tg_auto_{result.symbol}_{index}"
            if _tg_sent_key not in st.session_state:
                send_order_ticket_telegram(ticket.order_ticket())
                st.session_state[_tg_sent_key] = True


def _near_misses_frame(
    candidates: list[ScanResult],
    risk_config: RiskConfig,
    learner: ShadowLearner | None = None,
    policy_min_samples: int = 20,
    strategy_performance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for result in candidates:
        spec = _candidate_trade_spec(result, risk_config)
        if spec is None:
            continue
        side = spec["side"]
        agreeing = spec["agreeing_strategies"]
        policy = _policy_cells(result, learner, policy_min_samples)
        row = {
                "possible_action": _entry_action(side),
                "symbol": result.symbol,
                "possible_side": side.value,
                "confidence": round(result.plan.confidence, 1),
                "policy_confidence": round(
                    result.plan.confidence + policy.get("rl_bonus", 0),
                    1,
                ),
                "entry": round(spec["entry"], 2),
                "stop_loss": round(spec["stop_loss"], 2),
                "target": round(spec["target"], 2),
                "quantity": spec["quantity"],
                "max_risk": round(spec["risk_amount"], 2),
                "net_reward": round(spec["net_reward"], 2),
                "net_reward_risk": round(spec["net_reward_risk"], 2),
                "gross_reward_risk": round(spec["gross_reward_risk"], 2),
                "agreeing_votes": spec["agreeing_votes"],
                "vote_share": round(spec["vote_share"], 2),
                "latest_price": (
                    round(spec["latest_price"], 2) if spec["latest_price"] is not None else None
                ),
                "public_ltp": round(result.public_ltp, 2) if result.public_ltp is not None else None,
                "quote_age_sec": (
                    round(result.quote_age_seconds)
                    if result.quote_age_seconds is not None
                    else None
                ),
                "why_blocked": " | ".join(result.plan.reasons[:2]),
                "agreeing_strategies": ", ".join(agreeing),
            }
        row.update(policy)
        row.update(_recent_strategy_cells(agreeing, strategy_performance))
        row.update(_candidate_history_cells(result, side, agreeing, learner, risk_config))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and "recent_strategy_avg_bps" in frame:
        frame = frame.sort_values(
            ["recent_strategy_avg_bps", "confidence", "net_reward_risk"],
            ascending=[False, False, False],
            na_position="last",
        )
    return frame


def _style_action_frame(frame: pd.DataFrame):
    action_col = "action" if "action" in frame.columns else "possible_action"
    if frame.empty or action_col not in frame.columns:
        return frame

    def row_style(row):
        action = row[action_col]
        if action == "BUY":
            cell_style = "background-color: #dcfce7; color: #14532d; font-weight: 800;"
        elif action == "SELL":
            cell_style = "background-color: #fee2e2; color: #7f1d1d; font-weight: 800;"
        else:
            cell_style = ""
        return [cell_style if column == action_col else "" for column in row.index]

    # `.style` needs jinja2 at a version pandas is happy with; when it is
    # missing or too old pandas raises AttributeError and the entire scanner
    # view dies. Row colouring is decoration — the numbers are the product —
    # so fall back to the plain frame rather than taking the tab down.
    try:
        return frame.style.apply(row_style, axis=1)
    except (AttributeError, ImportError):
        return frame


def _relaxation_suggestions(
    *,
    near_misses: list[ScanResult],
    min_confidence: float,
    min_reward_risk: float,
    min_agreeing_votes: int,
    min_vote_share: float,
    policy_assist: bool,
    require_policy_approval: bool,
    learner: ShadowLearner | None,
    policy_min_samples: int,
) -> list[str]:
    suggestions: list[str] = []
    if not near_misses:
        suggestions.extend(
            [
                f"Try minimum confidence around {max(55, min_confidence - 5):.0f}%.",
                f"Try minimum reward/risk around {max(0.8, min_reward_risk - 0.2):.1f}.",
                "Try 5-minute candles if 1-minute candles are too noisy.",
            ]
        )
        return suggestions

    closest = near_misses[0]
    side = leading_trade_side(closest)
    agreeing = [vote for vote in closest.plan.strategy_votes if vote.side == side]
    action = _entry_action(side)

    suggestions.append(
        f"Closest setup is {action} {closest.symbol} with {closest.plan.confidence:.1f}% confidence."
    )
    if min_agreeing_votes > len(agreeing):
        suggestions.append(
            f"Reduce minimum agreeing strategies from {min_agreeing_votes} to {len(agreeing)}."
        )
    if closest.plan.confidence < min_confidence:
        suggestions.append(
            f"Reduce minimum confidence from {min_confidence:.0f}% to about {max(55, closest.plan.confidence - 1):.0f}%."
        )
    if min_vote_share > 0.5:
        suggestions.append(f"Reduce weighted vote share from {min_vote_share:.2f} to 0.50.")
    if min_reward_risk > 1.0:
        suggestions.append(f"Reduce minimum reward/risk from {min_reward_risk:.1f} to 1.0.")

    if policy_assist and require_policy_approval and learner is not None:
        estimate = learner.estimate_for_result(closest, min_samples=policy_min_samples)
        if not estimate.is_trained:
            suggestions.append("Disable 'Require trained positive policy' until this state has enough samples.")
        elif estimate.avg_reward_bps <= 0:
            suggestions.append(
                "Disable 'Require trained positive policy' only for paper trading; the trained policy is negative for this state."
            )

    if len(suggestions) == 1:
        suggestions.append("Lower confidence or reward/risk one step at a time to force at least one paper-trade candidate.")
    return suggestions


def _render_no_reco_guidance(
    *,
    near_misses: list[ScanResult],
    risk_config: RiskConfig,
    ensemble_config: EnsembleConfig,
    policy_assist: bool,
    require_policy_approval: bool,
    learner: ShadowLearner | None,
    policy_min_samples: int,
) -> None:
    st.subheader("How To Force One Paper Candidate")
    st.warning("Use these only to explore or paper trade. Lowering thresholds increases false signals.")
    suggestions = _relaxation_suggestions(
        near_misses=near_misses,
        min_confidence=risk_config.min_confidence,
        min_reward_risk=risk_config.min_reward_risk,
        min_agreeing_votes=ensemble_config.min_agreeing_votes,
        min_vote_share=ensemble_config.min_vote_share,
        policy_assist=policy_assist,
        require_policy_approval=require_policy_approval,
        learner=learner,
        policy_min_samples=policy_min_samples,
    )
    for suggestion in suggestions:
        st.write(f"- {suggestion}")


def _provider_factory(data_mode: str):
    return DemoProvider if data_mode == "Demo" else YFinanceProvider


def _refresh_seconds_for_interval(interval: str) -> int:
    defaults = {
        "1m": 60,
        "2m": 120,
        "5m": 300,
        "15m": 900,
    }
    return defaults.get(interval, 120)


def _auto_refresh_seconds(interval: str) -> int:
    """Fastest *useful* poll for the 'Auto' refresh mode.

    Signals only change when a bar closes, but the unified scan service's
    new-bar gate makes intermediate polls cheap cache reads — so we can poll
    every ~15-30s to catch each newly closed bar within one tick, while the
    single post-close tick does the network fetch.  Polling faster than the
    cache-read cost would just thrash, so this is the floor, not zero."""
    return {"1m": 15, "2m": 15, "5m": 20, "15m": 30}.get(interval, 20)


def _render_closed_toasts(closed: list[dict]) -> None:
    for item in closed:
        st.toast(
            f"Closed #{item['id']} {item['symbol']} {item['close_reason']}: "
            f"{_format_money(float(item['pnl']))}"
        )


def _scanner_state_key(
    *,
    universe_label: str,
    data_mode: str,
    period: str,
    interval: str,
    quote_source_label: str,
    scan_limit: int,
) -> str:
    clean_label = universe_label.replace(" ", "_").lower()
    return (
        f"scanner_batches_{clean_label}_{data_mode}_{period}_{interval}_"
        f"{quote_source_label}_{scan_limit}"
    )


def _render_scanner(
    *,
    store: PaperTradingStore,
    universe: UniverseResult,
    universe_label: str,
    data_mode: str,
    period: str,
    interval: str,
    risk_config: RiskConfig,
    ensemble_config: EnsembleConfig,
    strategy_weights: dict[str, float],
    simulation_mode: bool,
    auto_paper_trade: bool,
    max_workers: int,
    scan_limit: int,
    top_n: int,
    quote_client_factory,
    quote_source_label: str,
    quote_max_age_seconds: int,
    shadow_learning: bool,
    policy_assist: bool,
    require_policy_approval: bool,
    policy_min_samples: int,
    run_scan: bool,
    manual_scan: bool = False,
) -> dict[str, float]:
    if universe.warning:
        st.warning(universe.warning)
    st.caption(f"Universe source: {universe.source}")

    symbols = universe.symbols
    # The unified scan service covers the whole universe every cycle (one bulk
    # fetch + cache), so there is no per-cycle symbol batching any more.
    st.write(f"Scanner universe: {len(symbols)} symbols (full-universe scan each cycle)")

    if not run_scan:
        st.info(f"Press Scan {universe_label} now to run the parallel market search.")
        return {}

    universe_key = "commodity" if "commodit" in universe_label.lower() else "nse"
    latest_prices: dict[str, float] = {}
    closed: list[dict] = []
    status = st.empty()

    # Unified pipeline: run_scan_cycle is the SAME code the daemon runs —
    # incremental cache-first fetch (whole universe in one cycle, no more
    # 30-symbol batching), closed-bar evaluation with a catch-up cursor,
    # shared policy + meta veto chain.
    with st.spinner(f"Scanning all {len(symbols)} {universe_label} symbols (unified pipeline)..."):
        cycle = run_scan_cycle(
            universe_key,
            source="app",
            provider_factory=_provider_factory(data_mode),
            quote_client_factory=quote_client_factory,
            strategy_weights=strategy_weights,
            require_policy_approval=bool(policy_assist and require_policy_approval),
            policy_min_samples=policy_min_samples,
            record_shadow=bool(shadow_learning or policy_assist),
            log_signals=False,  # the app logs below with per-session dedup
            # A manual "Scan now" forces a network refresh (the user may not
            # have the daemon running); auto-refresh uses the new-bar gate.
            force_fetch=manual_scan,
        )
        st.session_state["last_market_context"] = cycle.market_context
        latest_prices = dict(cycle.latest_prices)
        for result in cycle.results:
            quote = cycle.quotes.get(result.symbol)
            if quote is not None:
                latest_prices[result.symbol] = float(quote.last_price)
            if (
                result.last_high is not None
                and result.last_low is not None
                and result.last_close is not None
            ):
                closed.extend(
                    store.update_open_trades_from_bar(
                        symbol=result.symbol,
                        high=result.last_high,
                        low=result.last_low,
                        close=result.last_close,
                    )
                )
        results = cycle.results
        status.write(
            f"Scanned {cycle.symbols_with_data}/{cycle.symbols_total} symbols in "
            f"{cycle.cycle_seconds:.0f}s — {cycle.evaluated_bars} newly closed bars evaluated "
            f"(signals are computed on closed {cycle.universe} bars only)."
        )

    _render_closed_toasts(closed)
    learner = ShadowLearner(LEARNER_DB_PATH) if shadow_learning or policy_assist else None
    if cycle.inserted or cycle.evaluated_pending:
        st.caption(
            f"Shadow learner updated: {cycle.inserted} new samples, "
            f"{cycle.evaluated_pending} evaluated."
        )

    # The unified gate chain (rank -> policy veto -> meta veto -> policy-
    # adjusted sort) already ran inside run_scan_cycle — byte-identical to
    # what the daemon notifies on.
    recommendations = cycle.recommendations
    near_misses = cycle.near_misses
    policy_blocked: list[ScanResult] = list(cycle.policy_blocked) + list(cycle.unproven)
    if cycle.meta_vetoed:
        st.caption(
            f"Meta-label veto dropped {len(cycle.meta_vetoed)} signal(s) scoring below "
            "the trained cut (evidence-gated model, scripts/train_meta_model.py)."
        )
    if cycle.stale_signals:
        st.caption(
            "Late signals from catch-up bars (recorded for the learner, too old to enter): "
            + ", ".join(
                f"{plan.symbol} {plan.side.value}@{ts.strftime('%H:%M')}"
                for ts, plan in cycle.stale_signals[:6]
            )
        )
    _send_new_signal_notifications(recommendations)
    performance_learner = learner or ShadowLearner(LEARNER_DB_PATH)
    strategy_performance = performance_learner.strategy_performance_frame(
        days=5,
        min_samples=5,
        limit=10,
        cost_bps=risk_config.estimated_cost_bps + risk_config.slippage_bps,
    )
    no_data = sum(1 for result in results if result.rows == 0)
    errors = sum(1 for result in results if result.error)
    cached_symbol_count = len(results)
    stale_quotes = sum(
        1
        for result in results
        if result.quote_age_seconds is not None
        and result.quote_age_seconds > quote_max_age_seconds
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Display Coverage", f"{cached_symbol_count}/{len(symbols)}")
    m2.metric("Very Confident Setups", str(len(recommendations)))
    m3.metric("No Data", str(no_data))
    m4.metric("Errors", str(errors))
    m5.metric(
        f"Quotes >{quote_max_age_seconds}s Old",
        str(stale_quotes) if quote_client_factory else "Off",
    )
    if quote_client_factory:
        st.caption(f"Latest price overlay: {quote_source_label}")
    if policy_blocked:
        st.warning(
            f"{len(policy_blocked)} actionable setup(s) were blocked because the trained policy "
            "did not show positive historical reward for that state."
        )

    if not recommendations:
        st.info(f"No {universe_label} instrument currently passed the configured filters.")
        _render_no_reco_guidance(
            near_misses=near_misses,
            risk_config=risk_config,
            ensemble_config=ensemble_config,
            policy_assist=policy_assist,
            require_policy_approval=require_policy_approval,
            learner=learner,
            policy_min_samples=policy_min_samples,
        )
        if policy_blocked:
            st.subheader("Policy-Blocked Candidates")
            st.caption("These passed rule filters but were rejected by the trained historical policy.")
            st.dataframe(
                _style_action_frame(_recommendations_frame(policy_blocked[:top_n], learner, policy_min_samples)),
                width="stretch",
                hide_index=True,
            )
        if near_misses:
            fallback_frame = _near_misses_frame(
                near_misses[:top_n],
                risk_config,
                learner,
                policy_min_samples,
                strategy_performance,
            )
            if not fallback_frame.empty:
                st.subheader("Next Ranked Paper Candidates")
                st.caption(
                    "These are below the configured trade filters. Levels are reconstructed from "
                    "the leading strategy votes and reward/risk includes configured costs."
                )
                st.dataframe(
                    _style_action_frame(fallback_frame),
                    width="stretch",
                    hide_index=True,
                )
        if not strategy_performance.empty:
            st.subheader("Strategies Working Recently")
            st.caption("Evaluated samples from the last 5 days, ranked by average net reward after costs.")
            st.dataframe(strategy_performance, width="stretch", hide_index=True)
        return latest_prices

    visible = recommendations[:top_n]
    # Log actionable signals — dedup within session so each unique signal logs once
    _logged_keys = st.session_state.setdefault("app_logged_signal_keys", set())
    for _r in visible:
        _p = _r.plan
        if _p.is_actionable and _p.entry:
            _sig_key = f"{_r.symbol}|{_p.side.value}|{_p.entry:.2f}"
            if _sig_key not in _logged_keys:
                _votes = [v for v in _p.strategy_votes if v.side == _p.side and v.is_trade]
                _log_signal(
                    symbol=_r.symbol, side=_p.side.value,
                    confidence=_p.confidence, entry=float(_p.entry),
                    stop_loss=float(_p.stop_loss or 0), target=float(_p.target or 0),
                    reward_risk=_p.reward_risk, regime=_r.regime,
                    strategies=", ".join(v.strategy for v in _votes),
                    source="app",
                )
                _logged_keys.add(_sig_key)
    if simulation_mode and auto_paper_trade:
        opened = [store.open_trade(result.plan) for result in visible]
        opened_count = sum(1 for trade_id in opened if trade_id is not None)
        if opened_count:
            st.info(f"Auto paper-trading recorded or refreshed {opened_count} virtual trades.")

    # ── Performance dashboard ─────────────────────────────────────────────────
    _shadow = ShadowLearner(LEARNER_DB_PATH)
    _shadow_stats = _shadow.stats()
    _mctx_disp = st.session_state.get("last_market_context")
    _d1, _d2, _d3, _d4, _d5 = st.columns(5)
    _d1.metric("Shadow WR", f"{_shadow_stats.win_rate:.1f}%",
               delta=f"{'▲' if _shadow_stats.win_rate>50 else '▼'}{abs(_shadow_stats.win_rate-50):.1f}pp vs 50%")
    _d2.metric("Avg Reward", f"{_shadow_stats.avg_reward_bps:+.1f} bps")
    _d3.metric("Signals Today", str(len(visible)))
    _d4.metric("NIFTY Regime",
               getattr(_mctx_disp, "index_regime", "–") if _mctx_disp else "–")
    _d5.metric("VIX",
               f"{_mctx_disp.vix_value:.1f} ({_mctx_disp.vix_level})" if _mctx_disp and _mctx_disp.vix_value else "–")
    st.divider()

    _render_execution_tickets(visible, risk_config)

    st.subheader("Very Confident Recommendations")
    st.caption(
        "Reference view of the raw engine output. The **entry** column is the signal "
        "bar's close — a price that has already gone — and the stop/target come from "
        "each strategy's own levels, not the validated exit design. Trade the order "
        "tickets above, not this table."
    )
    reco_df = _recommendations_frame(visible, learner, policy_min_samples)
    st.dataframe(
        _style_action_frame(reco_df),
        width="stretch",
        hide_index=True,
    )
    if not reco_df.empty:
        from datetime import datetime as _dt
        csv_data = reco_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Recommendations to CSV",
            data=csv_data,
            file_name=f"nse_intraday_signals_{_dt.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="export_reco_csv",
        )

    selected_symbol = st.selectbox(
        "Recommendation details",
        [result.symbol for result in visible],
        index=0,
    )
    selected = next(result for result in visible if result.symbol == selected_symbol)

    left, right = st.columns([1.7, 1])
    with left:
        if selected.frame is not None and not selected.frame.empty:
            st.plotly_chart(_plot_candles(selected.frame.tail(180), selected.symbol), width="stretch")
        else:
            st.info("Chart data was not retained for this recommendation.")
    with right:
        _render_plan(selected.plan, title=f"{selected.symbol} Trade Spec")
        if simulation_mode and not auto_paper_trade:
            if st.button("Place selected virtual trade"):
                trade_id = store.open_trade(selected.plan)
                st.success(f"Virtual trade #{trade_id} recorded.")
                st.rerun()

    st.subheader("Selected Strategy Votes")
    st.dataframe(_votes_frame(selected.plan), width="stretch", hide_index=True)
    if near_misses:
        st.subheader("Closest Blocked Candidates")
        st.caption("These are not trade calls. They show what nearly passed and why it was blocked.")
        blocked_frame = _near_misses_frame(
            near_misses[:top_n],
            risk_config,
            learner,
            policy_min_samples,
            strategy_performance,
        )
        st.dataframe(
            _style_action_frame(blocked_frame),
            width="stretch",
            hide_index=True,
        )
    if not strategy_performance.empty:
        st.subheader("Strategies Working Recently")
        st.caption("Evaluated samples from the last 5 days, ranked by average net reward after costs.")
        st.dataframe(strategy_performance, width="stretch", hide_index=True)

    st.subheader("Live Adaptive Strategy Weights")
    st.caption(
        "Combines shadow-learner reward signal (stop/target outcomes) with paper-trade win rate. "
        "Weights update every scan cycle. Strategies below 1.0 have been underperforming; above 1.0 are favoured by the vote."
    )
    _weight_engine_display = AdaptiveWeightEngine(performance_learner)
    _weight_frame = _weight_engine_display.weight_frame(paper_weights=store.strategy_weights())
    st.dataframe(_weight_frame, width="stretch", hide_index=True)

    return latest_prices


def _render_single_symbol(
    *,
    store: PaperTradingStore,
    data_mode: str,
    symbol: str,
    period: str,
    interval: str,
    risk_config: RiskConfig,
    ensemble_config: EnsembleConfig,
    strategy_weights: dict[str, float],
    simulation_mode: bool,
    auto_paper_trade: bool,
) -> dict[str, float]:
    provider = _provider_factory(data_mode)()
    result = provider.history(symbol, period=period, interval=interval)
    if result.warning:
        st.warning(result.warning)

    engine = VotingSignalEngine(config=ensemble_config)
    df, plan = engine.analyze(result.symbol, result.frame, risk_config, strategy_weights)

    closed = store.update_open_trades(df, result.symbol)
    _render_closed_toasts(closed)

    if simulation_mode and auto_paper_trade and plan.is_actionable:
        trade_id = store.open_trade(plan)
        if trade_id:
            st.info(f"Paper trade active/opened as trade #{trade_id}.")

    latest_prices = {}
    if not df.empty:
        latest_prices[result.symbol] = float(df.iloc[-1]["close"])

    left, right = st.columns([1.7, 1])
    with left:
        if not df.empty:
            st.plotly_chart(_plot_candles(df.tail(180), result.symbol), width="stretch")
        else:
            st.info("No candles available.")
    with right:
        _render_plan(plan)
        if simulation_mode and plan.is_actionable and not auto_paper_trade:
            if st.button("Place virtual trade now"):
                trade_id = store.open_trade(plan)
                st.success(f"Virtual trade #{trade_id} recorded.")
                st.rerun()

    st.subheader("Strategy Votes")
    votes_df = _votes_frame(plan)
    if not votes_df.empty:
        st.dataframe(votes_df, width="stretch", hide_index=True)

    return latest_prices


def _render_backtest(
    *,
    universe: UniverseResult,
    universe_label: str,
    lookback_days: int,
    interval: str,
    risk_config: RiskConfig,
    ensemble_config: EnsembleConfig,
    starting_capital: float,
    cooldown_minutes: int,
    max_hold_minutes: int,
    include_shorts: bool,
    no_new_trade_after: str,
    symbol_cooldown_minutes: int,
    stop_loss_cooldown_minutes: int,
    max_trades_per_day: int,
    daily_loss_limit_pct: float,
    calibrate_thresholds: bool,
    calibration_objective: str,
    target_trades: int,
    run_clicked: bool,
) -> None:
    if universe.warning:
        st.warning(universe.warning)
    st.caption(f"Backtest universe source: {universe.source}")

    if not run_clicked:
        st.info(f"Press Run backtest to replay historical {universe_label} candles.")
        return

    with st.spinner("Downloading candles and replaying historical signals..."):
        data = load_backtest_data(universe.symbols, lookback_days=lookback_days, interval=interval)
        is_commodity_universe = "commodit" in universe_label.lower()
        from nse_intraday_ai.context_series import build_vwap_breadth_series, fetch_context_series
        context_series = fetch_context_series(
            period=f"{max(lookback_days + 2, 3)}d",
            for_commodities=is_commodity_universe,
        )
        if not is_commodity_universe:
            context_series.vwap_breadth = build_vwap_breadth_series(
                {symbol: result.frame for symbol, result in data.items()}
            )
        backtest_config = BacktestConfig(
            starting_capital=starting_capital,
            lookback_days=lookback_days,
            interval=interval,
            cooldown_minutes=cooldown_minutes,
            max_hold_minutes=max_hold_minutes,
            include_shorts=include_shorts,
            no_new_trade_after="" if is_commodity_universe else no_new_trade_after,
            symbol_cooldown_minutes=symbol_cooldown_minutes,
            stop_loss_cooldown_minutes=stop_loss_cooldown_minutes,
            max_trades_per_day=max_trades_per_day,
            daily_loss_limit_pct=daily_loss_limit_pct,
            estimated_cost_bps=risk_config.estimated_cost_bps,
            slippage_bps=risk_config.slippage_bps,
            same_day_exit_only=not is_commodity_universe,
        )
        calibration_frame = pd.DataFrame()
        best_calibration = None
        if calibrate_thresholds:
            calibration_frame, best_calibration = calibrate_backtest(
                data=data,
                starting_capital=starting_capital,
                risk_per_trade_pct=risk_config.risk_per_trade_pct,
                max_position_pct=risk_config.max_position_pct,
                config=backtest_config,
                target_trades=target_trades,
                objective=calibration_objective,
                context_series=context_series,
            )
            if best_calibration is not None:
                risk_config = RiskConfig(
                    capital=starting_capital,
                    risk_per_trade_pct=risk_config.risk_per_trade_pct,
                    max_position_pct=risk_config.max_position_pct,
                    min_confidence=best_calibration.min_confidence,
                    min_reward_risk=best_calibration.min_reward_risk,
                    estimated_cost_bps=risk_config.estimated_cost_bps,
                    slippage_bps=risk_config.slippage_bps,
                )
                ensemble_config = EnsembleConfig(
                    min_agreeing_votes=best_calibration.min_agreeing_votes,
                    min_vote_share=best_calibration.min_vote_share,
                    min_weighted_confidence=best_calibration.min_confidence,
                )

        summary, trades, equity = run_backtest(
            data=data,
            risk_config=risk_config,
            ensemble_config=ensemble_config,
            config=backtest_config,
            context_series=context_series,
        )

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Start Capital", _format_money(summary.starting_capital))
    b2.metric("End Capital", _format_money(summary.ending_capital), f"{summary.pnl_pct:.2f}%")
    b3.metric("P&L", _format_money(summary.pnl))
    b4.metric("Trades", str(summary.trades))
    b5.metric("Win Rate", f"{summary.win_rate:.1f}%")

    d1, d2, d3 = st.columns(3)
    d1.metric("Wins", str(summary.wins))
    d2.metric("Losses", str(summary.losses))
    d3.metric("Max Drawdown", f"{_format_money(summary.max_drawdown)} ({summary.max_drawdown_pct:.2f}%)")

    st.plotly_chart(_plot_equity_curve(equity), width="stretch")

    if calibrate_thresholds:
        st.subheader("Calibration")
        if best_calibration is None:
            st.warning(f"No threshold set reached {target_trades} trades.")
        else:
            st.success(
                "Selected thresholds: "
                f"confidence >= {best_calibration.min_confidence}, "
                f"reward/risk >= {best_calibration.min_reward_risk}, "
                f"agreeing votes >= {best_calibration.min_agreeing_votes}, "
                f"vote share >= {best_calibration.min_vote_share}"
            )
        if not calibration_frame.empty:
            st.dataframe(calibration_frame.head(30), width="stretch", hide_index=True)

    if trades.empty:
        st.info("No trades were taken by the configured rules.")
    else:
        st.subheader("Backtest Trades")
        st.dataframe(trades, width="stretch", hide_index=True)

    missing = [
        {"symbol": symbol, "warning": result.warning}
        for symbol, result in data.items()
        if result.frame.empty
    ]
    if missing:
        with st.expander("Symbols Without Candle Data"):
            st.dataframe(pd.DataFrame(missing), width="stretch", hide_index=True)



# ── Intra-week (swing) book & Recommendation Workbench ─────────────────────
# Rendered in the shared Streamlit app so the desktop browser and the Android
# WebView show identical features — the phone is a client of this page, not a
# separate product, so anything added here appears on both by construction.

def _render_recommend_workbench() -> None:
    from nse_intraday_ai import recommend_ui
    recommend_ui.render()


def _render_candidate_paper() -> None:
    from nse_intraday_ai import candidate_ui
    candidate_ui.render()


@st.cache_data(ttl=900, show_spinner=False)
def _swing_panel(universe: str):
    """Daily panel for the swing book. Cached: it reads ~1M rows."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from swing_backtest import load_frames
    from nse_intraday_ai.swing import build_panel
    frames = load_frames(universe)
    return build_panel(frames), {k: v for k, v in frames.items()}


def _render_swing() -> None:
    from nse_intraday_ai.costs import segment_round_trip_bps
    from nse_intraday_ai.swing import SCORES, _position_size
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from swing_backtest import config_for
    from swing_today import EXPECTANCY, expectancy_note

    st.subheader("Intra-week book")
    st.caption(
        "Hold one name for a few sessions instead of a few bars. Overnight "
        "positions settle as DELIVERY, where STT is 0.1% on both legs — the "
        "round trip is ~30 bps, not ~10."
    )

    col1, col2, col3, col4 = st.columns(4)
    universe = col1.selectbox("Universe", ["nse", "commodity"], index=0,
                              format_func=lambda u: "NSE 500" if u == "nse" else "Commodities")
    default_strategy = "rsi_oversold" if universe == "commodity" else "trend_pullback"
    names = [k for k in SCORES if k != "random"]
    strategy = col2.selectbox("Strategy", names, index=names.index(default_strategy))
    hold = col3.selectbox("Hold (sessions)", [5, 10, 20, 40], index=0)
    top = col4.number_input("Candidates", 1, 20, 5)

    capital = st.session_state.get("swing_capital", 10_00_000.0)
    capital = st.number_input("Capital", 50_000.0, 5_00_00_000.0, float(capital), step=50_000.0)
    st.session_state["swing_capital"] = capital
    risk_pct = st.slider("Risk per trade (% of capital)", 0.5, 5.0, 2.0, 0.5)

    note = expectancy_note(universe, int(hold), strategy)
    if universe == "nse" and int(hold) == 5:
        st.error(f"Measured expectancy: {note}")
    elif universe == "commodity":
        st.success(f"Measured expectancy: {note}")
    else:
        st.warning(f"Measured expectancy: {note}")

    if not st.button("Find candidates", type="primary"):
        st.info("Ten-year test: `python scripts/swing_backtest.py " + universe + "`")
        return

    with st.spinner("Loading 10 years of daily bars..."):
        panel, _frames = _swing_panel(universe)
    if panel.empty:
        st.warning("No daily data cached. Run `python scripts/fetch_daily.py` first.")
        return

    cfg = config_for(universe, hold_days=int(hold), positions=1,
                     capital=float(capital), risk_per_trade_pct=float(risk_pct))
    latest = panel.date.max()
    rows = panel[(panel.date == latest) & (panel.turnover_20d >= cfg.min_turnover)]
    if rows.empty:
        st.warning("Nothing liquid enough on the latest bar.")
        return

    ranked = rows.assign(_score=SCORES[strategy](rows)).dropna(subset=["_score"])
    ranked = ranked.sort_values("_score", ascending=False).head(int(top))

    records = []
    for _, r in ranked.iterrows():
        entry, atr = float(r["close"]), float(r["atr"])
        qty = _position_size(entry, atr, cfg)
        if qty <= 0:
            continue
        records.append({
            "Symbol": r["symbol"],
            "Ref close": round(entry, 2),
            "Stop": round(entry - cfg.stop_atr * atr, 2),
            "Qty": qty,
            "Value": round(qty * entry, 0),
            "Risk": round(qty * cfg.stop_atr * atr, 0),
            "Cost bps": round(segment_round_trip_bps(
                entry, qty, segment=cfg.segment,
                slippage_bps_per_leg=cfg.slippage_bps_per_leg, symbol=r["symbol"]), 1),
        })
    if not records:
        st.warning("No candidate clears the sizing rules at this capital.")
        return

    st.caption(f"Data through {latest.date()} · {cfg.segment.value} · "
               f"stop {cfg.stop_atr} ATR · hold {hold} sessions")
    st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
    st.caption(
        "The stop is quoted off the **fill**, which is the next session's open "
        "and is not known yet — not off the reference close above."
    )


def main() -> None:
    st.title("NSE & Commodity Intraday Signal Lab")
    st.caption(
        "Research and paper-trading tool. It is not a profit guarantee or personal financial advice. "
        "Use broker/licensed feeds before any real-money workflow."
    )

    with st.sidebar:
        # ── Daemon notification toggle ────────────────────────────────────────
        _pause_flag = ROOT / "data" / "notifications_paused"
        _notif_on = not _pause_flag.exists()
        _notif_new = st.toggle(
            "Desktop signal alerts (daemon)",
            value=_notif_on,
            key="notif_top_toggle",
            help="Enable/disable desktop popup notifications from the background scanner daemon",
        )
        if _notif_new != _notif_on:
            if _notif_new:
                _pause_flag.unlink(missing_ok=True)
                st.toast("Desktop notifications ENABLED")
            else:
                _pause_flag.touch()
                st.toast("Desktop notifications PAUSED")

        # ── Telegram Bot Setup ────────────────────────────────────────────────
        from nse_intraday_ai.alerts import (
            telegram_configured, save_telegram_config,
            send_telegram, telegram_setup_interactive, _load_telegram_config,
        )
        with st.expander("📨 Telegram Alerts Setup", expanded=not telegram_configured()):
            if telegram_configured():
                st.success("✅ Telegram connected!")
                cfg = _load_telegram_config()
                st.caption(f"Chat ID: `{cfg.get('chat_id', '?')}`")
                if st.button("🧪 Send Test Alert", key="tg_test_btn"):
                    ok = send_telegram(
                        "🧪 *Test Alert from NSE Quant Terminal*\n\n"
                        "Telegram alerts are working! ✅"
                    )
                    if ok:
                        st.success("Test message sent! Check Telegram.")
                    else:
                        st.error("Failed to send. Check bot token.")
            else:
                st.markdown(
                    "**Free setup (2 minutes):**\n"
                    "1. Open Telegram → search **@BotFather**\n"
                    "2. Send `/newbot` → follow prompts\n"
                    "3. Copy the **Bot Token** and paste below\n"
                    "4. Open your new bot in Telegram, send `/start`\n"
                    "5. Click **Connect** below"
                )
                tg_token = st.text_input(
                    "Bot Token", type="password", key="tg_token_input",
                    placeholder="123456:ABCdefGHIjklMNO..."
                )
                if st.button("🔗 Connect Telegram", key="tg_connect_btn"):
                    if not tg_token:
                        st.error("Please paste your bot token first.")
                    else:
                        with st.spinner("Detecting chat_id..."):
                            chat_id = telegram_setup_interactive(tg_token)
                        if chat_id:
                            st.success(f"✅ Connected! chat_id = {chat_id}")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(
                                "No messages found. Make sure you:\n"
                                "1. Opened your bot in Telegram\n"
                                "2. Sent /start\n"
                                "Then click Connect again."
                            )
        st.divider()

        st.header("Market")
        workspace_mode = st.radio(
            "Mode",
            ["NIFTY 500 scanner", "NIFTY 100 scanner", "NIFTY 50 scanner", "Commodity scanner",
             "Recommendation workbench", "Candidate paper track", "Intra-week book", "Backtest", "Single symbol"],
            index=0,
        )
        scanner_mode = workspace_mode in {"NIFTY 500 scanner", "NIFTY 100 scanner", "NIFTY 50 scanner", "Commodity scanner"}
        commodity_mode = workspace_mode == "Commodity scanner"
        data_mode = st.radio("Data source", ["Yahoo Finance", "Demo"], horizontal=True)
        # 5m/5d defaults: 5m is the validated signal timeframe (Jul-2026 study;
        # 1m edges were smaller than costs), and at 5m the strategies need
        # 35-70 bars of history to vote, so a 1d lookback starves them until
        # mid-afternoon.
        period = st.selectbox("Lookback", ["1d", "5d"], index=1)
        interval = st.selectbox("Candle interval", ["1m", "2m", "5m", "15m"], index=2)
        # Auto is the default: fastest useful refresh. The scan service's
        # new-bar gate keeps most auto polls to cheap cache reads and only
        # fetches when a bar actually closes, so fast polling is safe.
        refresh_mode = st.selectbox(
            "Refresh mode",
            ["Auto (fastest)", "Custom interval", "Off"],
            index=0,
            help="Auto polls every ~15-30s (per candle interval) to catch each "
                 "newly closed bar within a tick; a network fetch only runs when "
                 "a bar actually closes. Custom lets you set a fixed interval.",
        )
        if refresh_mode == "Off":
            auto_refresh = False
            refresh_seconds = _refresh_seconds_for_interval(interval)
        elif refresh_mode == "Custom interval":
            auto_refresh = True
            refresh_seconds = st.slider(
                "Refresh seconds",
                10,
                900,
                _refresh_seconds_for_interval(interval),
                step=5,
                key=f"refresh_seconds_{interval}",
            )
        else:  # Auto (fastest)
            auto_refresh = True
            refresh_seconds = _auto_refresh_seconds(interval)
            st.caption(f"Auto refresh: every {refresh_seconds}s (fastest useful cadence).")

        if workspace_mode == "Single symbol":
            single_symbol_choices = [*DEFAULT_SYMBOLS, *DEFAULT_COMMODITY_SYMBOLS]
            symbol = st.selectbox("Symbol", single_symbol_choices, index=1)
            custom_symbol = st.text_input("Or custom symbol", placeholder="Example: INFY or GC=F")
            if custom_symbol.strip():
                symbol = custom_symbol.strip()
        elif workspace_mode == "Backtest":
            symbol = ""
            run_backtest_clicked = st.button("Run backtest", type="primary")
            backtest_universe_label = st.selectbox("Backtest universe", ["NIFTY 500", "NIFTY 100", "NIFTY 50", "Commodities"], index=1)
            backtest_days = st.slider("Backtest days", 1, 30, 5, step=1)
            backtest_cooldown = st.slider("Trade cooldown minutes", 10, 240, 10, step=10)
            backtest_max_hold = st.slider("Max hold minutes", 15, 240, 90, step=15)
            backtest_include_shorts = st.toggle("Include short trades", value=True)
            st.subheader("Loss Controls")
            backtest_no_new_trade_after_choice = st.selectbox(
                "No new trades after",
                ["14:30", "14:45", "15:00", "15:15", "Off"],
                index=1,
            )
            backtest_no_new_trade_after = (
                "" if backtest_no_new_trade_after_choice == "Off" else backtest_no_new_trade_after_choice
            )
            backtest_symbol_cooldown = st.slider(
                "Same-symbol cooldown minutes",
                0,
                240,
                30,
                step=10,
            )
            backtest_stop_loss_cooldown = st.slider(
                "Same-symbol lockout after stop-loss minutes",
                0,
                240,
                60,
                step=10,
            )
            backtest_max_trades_per_day = st.slider("Max trades per day, 0 = off", 0, 50, 0, step=1)
            backtest_daily_loss_limit_pct = st.slider(
                "Daily loss limit %, 0 = off",
                0.0,
                5.0,
                1.0,
                step=0.25,
            )
            calibrate_thresholds = st.toggle("Auto-calibrate thresholds", value=True)
            calibration_objective_choice = st.selectbox(
                "Calibration objective",
                [
                    "Risk-adjusted",
                    "Lowest drawdown",
                    "Maximum P&L",
                ],
                index=0,
            )
            calibration_objective = {
                "Risk-adjusted": "risk_adjusted",
                "Lowest drawdown": "low_drawdown",
                "Maximum P&L": "max_pnl",
            }[calibration_objective_choice]
            target_trades = st.slider("Target trades", 1, 50, 10, step=1)
            quote_max_age_seconds = 120
            shadow_learning = False
            policy_assist = False
            policy_min_samples = 20
        else:
            symbol = ""
            if commodity_mode:
                scan_label = "commodities"
            elif workspace_mode == "NIFTY 500 scanner":
                scan_label = "NIFTY 500"
            elif workspace_mode == "NIFTY 100 scanner":
                scan_label = "NIFTY 100"
            else:
                scan_label = "NIFTY 50"
            scan_clicked = st.button(f"Scan {scan_label} now", type="primary")
            auto_scan = st.toggle(
                "Run scanner on each refresh",
                value=True,
                key="run_scanner_on_refresh_v2",
            )
            if commodity_mode:
                quote_source = "Off"
                quote_client_factory = None
                quote_max_age_seconds = 120
                st.caption("Commodity scanner uses Yahoo latest candle data; quote freshness is off.")
            else:
                quote_source = st.selectbox(
                    "Latest price source",
                    ["Google Finance", "NSE snapshot", "Off"],
                    index=0,
                )
                quote_client_factory = {
                    "Google Finance": GoogleFinanceQuoteClient,
                    "NSE snapshot": NsePublicQuoteClient,
                    "Off": None,
                }[quote_source]
                quote_max_age_seconds = st.slider(
                    "Max quote age seconds",
                    30,
                    900,
                    600 if quote_source == "Google Finance" else 120,
                    step=30,
                    key=f"quote_age_{quote_source}",
                )
            shadow_learning = st.toggle("Shadow learner", value=True)
            policy_assist = st.toggle("Policy assist ranking", value=True)
            require_policy_approval = st.toggle("Require trained positive policy", value=commodity_mode)
            policy_min_samples = st.slider("Policy min samples", 5, 100, 20, step=5)
            max_workers = st.slider("Parallel workers", 2, 16, 8, step=2)
            scan_limit = st.number_input(
                "Batch size, 0 = all",
                0,
                50,
                20,
                step=5,
                key="scan_batch_size_v1",
            )
            top_n = st.slider("Show top recommendations", 3, 20, 10, step=1)
        if workspace_mode == "Single symbol":
            quote_max_age_seconds = 120
            quote_source = "Off"
            quote_client_factory = None
            shadow_learning = False
            policy_assist = False
            require_policy_approval = False
            policy_min_samples = 20
        if workspace_mode != "Backtest":
            run_backtest_clicked = False
            backtest_universe_label = "NIFTY 50"
            backtest_days = 5
            backtest_cooldown = 60
            backtest_max_hold = 90
            backtest_include_shorts = True
            backtest_no_new_trade_after = "15:00"
            backtest_symbol_cooldown = 0
            backtest_stop_loss_cooldown = 0
            backtest_max_trades_per_day = 0
            backtest_daily_loss_limit_pct = 0.0
            calibrate_thresholds = False
            calibration_objective = "risk_adjusted"
            target_trades = 10
            quote_source = "Off"
            quote_client_factory = None

        st.header("Risk")
        scanner_defaults = scanner_mode
        backtest_defaults = workspace_mode == "Backtest"
        commodity_defaults = commodity_mode or (
            workspace_mode == "Backtest" and backtest_universe_label == "Commodities"
        )
        # Defaults match the validated portfolio configuration (execution_plan.py
        # / sim_today.py): ₹10L, 1% risk, 33% max position.  They used to be
        # ₹1L / 0.5% / 25%, which silently sized every order ticket for a
        # tenth of the account being traded.
        capital = st.number_input(
            "Trading capital", 10_000, 50_000_000, 1_000_000, step=50_000,
            help="Order tickets are sized off this. Set it to the account you actually trade.",
        )
        risk_pct = st.slider("Risk per trade %", 0.1, 3.0, 1.0, step=0.1)
        max_pos_pct = st.slider("Max position %", 5.0, 100.0, 33.0, step=1.0)
        # Futures costs are flat per lot (≈2–6 bps of notional round-trip on MCX/CME),
        # not equity-style percentage brokerage — don't apply 0.5% to commodities.
        # NSE default recalibrated 0.5 -> 0.15 (Jul-2026): discount-broker
        # intraday round trip is ~15 bps; the old 50 bps assumption made the
        # net reward/risk gate reject essentially every equity signal.
        commission_pct = st.slider(
            "Round-trip commission %",
            0.0,
            2.0,
            0.05 if commodity_defaults else 0.15,
            step=0.01,
        )
        slippage_bps = st.slider("Slippage bps", 0.0, 50.0, 3.0, step=1.0)
        st.caption("Reward/risk and backtest P&L are calculated after commission and slippage.")
        # Scanner gate defaults recalibrated by the Jul-2026 two-week event
        # study: conf 70 / rr 1.5 / 2 votes (on 5m bars with market context +
        # event windows) was positive in both study weeks for NSE equities.
        min_conf = st.slider(
            "Minimum confidence %",
            55.0,
            98.0,
            85.0 if commodity_defaults else 70.0 if scanner_defaults else 86.0 if backtest_defaults else 78.0,
            step=1.0,
        )
        min_rr = st.slider(
            "Minimum net reward/risk",
            0.8,
            4.0,
            1.5 if commodity_defaults else 1.5 if scanner_defaults else 1.7 if backtest_defaults else 1.5,
            step=0.1,
        )

        st.header("Voting")
        # Commodities dropped from 2 required votes to 1: the opening-range
        # strategy no longer votes on 24-hour futures, so 2-vote agreement
        # starved the scanner (Jun-2026 study: conf>=85 alone selects a
        # positive-expectancy subset; the 2-vote gate left ~0 trades/week).
        min_votes = st.slider(
            "Minimum agreeing strategies",
            1,
            4,
            1 if commodity_defaults else 2,
        )
        min_vote_share = st.slider(
            "Minimum weighted vote share",
            0.50,
            1.00,
            0.70 if commodity_defaults else 0.50 if scanner_defaults else 0.75 if backtest_defaults else 0.62,
            step=0.01,
        )

        st.header("Market Context")
        _mctx = st.session_state.get("last_market_context")
        if _mctx is not None:
            _idx_color = {"TRENDING_UP": "🟢", "TRENDING_DOWN": "🔴", "RANGING": "🟡", "HIGH_VOL": "🟠"}.get(_mctx.index_regime, "⚪")
            _vix_color = {"LOW": "🟢", "NORMAL": "🟡", "ELEVATED": "🟠", "HIGH": "🔴"}.get(_mctx.vix_level, "⚪")
            _brd_color = {"BULL_DAY": "🟢", "BEAR_DAY": "🔴", "NEUTRAL": "🟡"}.get(_mctx.breadth_signal, "⚪")
            st.caption(f"{_idx_color} NIFTY: **{_mctx.index_regime}** ({_mctx.index_long_adj:+.0f}L / {_mctx.index_short_adj:+.0f}S)")
            vix_str = f"{_mctx.vix_value:.1f}" if _mctx.vix_value else "N/A"
            st.caption(f"{_vix_color} VIX: **{vix_str}** ({_mctx.vix_level})")
            st.caption(f"{_brd_color} Breadth: **{_mctx.breadth_signal}** ({_mctx.breadth_up_pct:.0%} trending up)")
            _risk_color = "🟢" if _mctx.global_risk > 0.15 else "🔴" if _mctx.global_risk < -0.15 else "🟡"
            st.caption(f"{_risk_color} Global risk (S&P fut + Asia + Europe): **{_mctx.global_risk:+.2f}**")
            if _mctx.vwap_breadth_pct is not None:
                _bw = _mctx.vwap_breadth_pct
                _bw_color = "🟢" if _bw >= 0.65 else "🔴" if _bw <= 0.35 else "🟡"
                st.caption(f"{_bw_color} VWAP breadth: **{_bw:.0%}** of universe above VWAP")
            from nse_intraday_ai.event_risk import upcoming_event as _upcoming_event
            import pandas as _pd_ev
            _event_note = _upcoming_event(_pd_ev.Timestamp.now(tz="Asia/Kolkata"))
            if _event_note:
                st.caption(f"⚠️ {_event_note}")
            _globals = []
            if _mctx.dxy_change_pct is not None:
                _globals.append(f"DXY {_mctx.dxy_change_pct:+.2f}%")
            if _mctx.usdinr_change_pct is not None:
                _globals.append(f"USDINR {_mctx.usdinr_change_pct:+.2f}%")
            if _mctx.us10y_change_pct is not None:
                _globals.append(f"US10Y {_mctx.us10y_change_pct:+.2f}%")
            if _mctx.us_vix_value is not None:
                _globals.append(f"VIX(US) {_mctx.us_vix_value:.1f}")
            if _globals:
                st.caption("Macro: " + " · ".join(_globals))
            _sector_idx = _mctx.sector_index_regimes or _mctx.sector_regimes
            if _sector_idx:
                _top_sectors = sorted(_sector_idx.items(), key=lambda x: x[1])
                st.caption("Sectors: " + "  ".join(f"{s}={r[:4]}" for s, r in _top_sectors[:4]))
        else:
            st.caption("Run a scan to see market context.")

        st.header("Signal Log")
        _recent = _signal_log_tail(10)
        if _recent:
            import pandas as _pd2
            _log_df = _pd2.DataFrame(_recent)[["ts","source","symbol","side","confidence","regime"]]
            st.dataframe(_log_df, hide_index=True, use_container_width=True)
            st.caption("'app' = webpage · 'daemon' = notification — should match for same symbol/time")
        else:
            st.caption("No signals logged yet today.")

        st.header("Simulation")
        simulation_mode = st.toggle("Simulation mode", value=True)
        if scanner_mode:
            auto_paper_trade = st.toggle("Auto paper-trade shown recommendations", value=False)
        else:
            auto_paper_trade = st.toggle("Auto paper-trade actionable signals", value=False)
        reset_clicked = st.button("Reset paper trades")

    store = PaperTradingStore(DB_PATH, starting_capital=float(capital))
    if reset_clicked:
        store.reset()
        st.success("Paper trading history reset.")

    risk_config = RiskConfig(
        capital=float(capital),
        risk_per_trade_pct=float(risk_pct),
        max_position_pct=float(max_pos_pct),
        min_reward_risk=float(min_rr),
        min_confidence=float(min_conf),
        estimated_cost_bps=float(commission_pct) * 100,
        slippage_bps=float(slippage_bps),
    )
    ensemble_config = EnsembleConfig(
        min_agreeing_votes=int(min_votes),
        min_vote_share=float(min_vote_share),
        min_weighted_confidence=float(min_conf),
    )
    # Persist settings so scanner daemon and monitor loop stay in sync with the
    # app — into THIS universe's section only, so a commodity session can never
    # clobber the NSE daemon's gates again (2026-07-07 incident).
    _cfg_universe = "commodity" if "commodit" in str(workspace_mode).lower() else "nse"
    _save_scan_config({
        "capital": float(capital),
        "risk_per_trade_pct": float(risk_pct),
        "max_position_pct": float(max_pos_pct),
        "min_confidence": float(min_conf),
        "min_reward_risk": float(min_rr),
        "estimated_cost_bps": float(commission_pct) * 100,
        "slippage_bps": float(slippage_bps),
        "min_agreeing_votes": int(min_votes),
        "min_vote_share": float(min_vote_share),
        "period": str(period),
        "interval": str(interval),
    }, universe=_cfg_universe)

    refresh_interval = int(refresh_seconds) if auto_refresh and workspace_mode != "Backtest" else None
    if refresh_interval is not None:
        st.caption(f"Auto update active: refreshing this view every {refresh_interval} seconds.")

    @st.fragment(run_every=refresh_interval)
    def _render_refreshing_body() -> None:
        _learner_for_weights = ShadowLearner(LEARNER_DB_PATH)
        _weight_engine = AdaptiveWeightEngine(_learner_for_weights)
        weights = _weight_engine.weights(paper_weights=store.strategy_weights())
        if workspace_mode in {"NIFTY 500 scanner", "NIFTY 100 scanner", "NIFTY 50 scanner", "Commodity scanner"}:
            run_scan = scan_clicked or (auto_refresh and auto_scan)
            if workspace_mode == "Commodity scanner":
                scanner_universe = _cached_commodity_symbols()
                scanner_label = "commodities"
            elif workspace_mode == "NIFTY 500 scanner":
                scanner_universe = _cached_nifty500_symbols(str(NIFTY500_CACHE_PATH))
                scanner_label = "NIFTY 500"
            elif workspace_mode == "NIFTY 100 scanner":
                scanner_universe = _cached_nifty100_symbols(str(NIFTY100_CACHE_PATH))
                scanner_label = "NIFTY 100"
            else:
                scanner_universe = _cached_nifty50_symbols(str(NIFTY50_CACHE_PATH))
                scanner_label = "NIFTY 50"
            latest_prices = _render_scanner(
                store=store,
                universe=scanner_universe,
                universe_label=scanner_label,
                data_mode=data_mode,
                period=period,
                interval=interval,
                risk_config=risk_config,
                ensemble_config=ensemble_config,
                strategy_weights=weights,
                simulation_mode=simulation_mode,
                auto_paper_trade=auto_paper_trade,
                max_workers=int(max_workers),
                scan_limit=int(scan_limit),
                top_n=int(top_n),
                quote_client_factory=quote_client_factory,
                quote_source_label=str(quote_source),
                quote_max_age_seconds=int(quote_max_age_seconds),
                shadow_learning=bool(shadow_learning),
                policy_assist=bool(policy_assist),
                require_policy_approval=bool(require_policy_approval),
                policy_min_samples=int(policy_min_samples),
                run_scan=run_scan,
                manual_scan=bool(scan_clicked),
            )
        elif workspace_mode == "Recommendation workbench":
            latest_prices = {}
            _render_recommend_workbench()
        elif workspace_mode == "Candidate paper track":
            latest_prices = {}
            _render_candidate_paper()
        elif workspace_mode == "Intra-week book":
            latest_prices = {}
            _render_swing()
        elif workspace_mode == "Backtest":
            latest_prices = {}
            if backtest_universe_label == "Commodities":
                backtest_universe = _cached_commodity_symbols()
            elif backtest_universe_label == "NIFTY 500":
                backtest_universe = _cached_nifty500_symbols(str(NIFTY500_CACHE_PATH))
            elif backtest_universe_label == "NIFTY 100":
                backtest_universe = _cached_nifty100_symbols(str(NIFTY100_CACHE_PATH))
            else:
                backtest_universe = _cached_nifty50_symbols(str(NIFTY50_CACHE_PATH))
            _render_backtest(
                universe=backtest_universe,
                universe_label=backtest_universe_label,
                lookback_days=int(backtest_days),
                interval=interval,
                risk_config=risk_config,
                ensemble_config=ensemble_config,
                starting_capital=float(capital),
                cooldown_minutes=int(backtest_cooldown),
                max_hold_minutes=int(backtest_max_hold),
                include_shorts=bool(backtest_include_shorts),
                no_new_trade_after=str(backtest_no_new_trade_after),
                symbol_cooldown_minutes=int(backtest_symbol_cooldown),
                stop_loss_cooldown_minutes=int(backtest_stop_loss_cooldown),
                max_trades_per_day=int(backtest_max_trades_per_day),
                daily_loss_limit_pct=float(backtest_daily_loss_limit_pct),
                calibrate_thresholds=bool(calibrate_thresholds),
                calibration_objective=str(calibration_objective),
                target_trades=int(target_trades),
                run_clicked=bool(run_backtest_clicked),
            )
        else:
            latest_prices = _render_single_symbol(
                store=store,
                data_mode=data_mode,
                symbol=symbol,
                period=period,
                interval=interval,
                risk_config=risk_config,
                ensemble_config=ensemble_config,
                strategy_weights=weights,
                simulation_mode=simulation_mode,
                auto_paper_trade=auto_paper_trade,
            )

        st.caption(f"Last refreshed: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%H:%M:%S')}")

        st.subheader("Paper Trading Account")
        _render_account(store.summary(latest_prices))

        st.subheader("Paper Trades")
        trades = store.trades_frame()
        if trades.empty:
            st.info("No virtual trades yet.")
        else:
            st.dataframe(trades, width="stretch", hide_index=True)

        st.subheader("Adaptive Strategy Performance")
        stats = store.strategy_stats_frame()
        if stats.empty:
            st.info("Weights remain neutral until at least 5 closed samples per strategy.")
        else:
            st.dataframe(stats, width="stretch", hide_index=True)
            st.json(weights)

        st.subheader("Shadow Learner")
        learner = ShadowLearner(LEARNER_DB_PATH)
        learner.refresh_policy()
        shadow_stats = learner.stats()
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Pending Samples", str(shadow_stats.pending))
        l2.metric("Evaluated Samples", str(shadow_stats.evaluated))
        l3.metric("Shadow Win Rate", f"{shadow_stats.win_rate:.1f}%")
        l4.metric("Avg Reward", f"{shadow_stats.avg_reward_bps:.1f} bps")
        policy = learner.policy_frame()
        if policy.empty:
            st.info("Shadow learner needs future scans to evaluate delayed outcomes.")
        else:
            st.dataframe(policy, width="stretch", hide_index=True)

        st.subheader("Recent Strategy Performance")
        strategy_performance = learner.strategy_performance_frame(
            days=5,
            min_samples=5,
            limit=10,
            cost_bps=risk_config.estimated_cost_bps + risk_config.slippage_bps,
        )
        if strategy_performance.empty:
            st.info("No strategy has enough evaluated samples in the last 5 days.")
        else:
            st.caption("Average reward is net of the configured commission and slippage.")
            st.dataframe(strategy_performance, width="stretch", hide_index=True)

    _render_refreshing_body()


if __name__ == "__main__":
    main()
