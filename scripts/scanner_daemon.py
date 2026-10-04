"""Headless scan daemon — thin wrapper around the unified scan service.

All scanning/gating/veto/learning logic lives in
nse_intraday_ai.scan_service.run_scan_cycle — the SAME code path the
Streamlit app renders, so the notification you get on the desktop and the
recommendation you see in the browser can never diverge again.

This wrapper only decides WHEN to scan (market hours per universe) and HOW to
notify (desktop notifications with per-day dedup).  Runs one cycle per
invocation; systemd's nse-scanner.timer provides the cadence (every minute —
the service's bar cursor makes each closed 5m bar evaluated exactly once, and
catch-up logic covers any missed cycles).

Scanner signals are research output — the voting engine has no edge after
costs — so they go to the desktop only.  The phone channel (ntfy) is reserved
for the validated gap-reversal book (scripts/gap_reversal.py).  Two env vars
restore the old behaviour:

    NSE_DAEMON_UNIVERSES=nse,commodity   also scan commodities (default: nse)
    NSE_PUSH_SCANNER_ALERTS=1            also push scanner signals to the phone
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.data import GoogleFinanceQuoteClient  # noqa: E402
from nse_intraday_ai.scan_service import run_scan_cycle  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
STATE_PATH = ROOT / "data" / "daemon_state.json"
UNIVERSES = [u.strip() for u in os.environ.get("NSE_DAEMON_UNIVERSES", "nse").split(",") if u.strip()]
PUSH_SCANNER_ALERTS = os.environ.get("NSE_PUSH_SCANNER_ALERTS") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scanner-daemon")


# ── Per-universe market hours ─────────────────────────────────────────────

def nse_market_open(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    m = now.hour * 60 + now.minute
    return 555 <= m <= 930            # 9:15 – 15:30 IST


def commodity_market_open(now: datetime) -> bool:
    # Globex/MCX trade nearly 24h on weekdays; skip only the weekend and the
    # dead 05:00–09:00 IST stretch (thin Globex, no MCX).
    if now.weekday() >= 5:
        return False
    return not (5 <= now.hour < 9)


# ── Desktop notification ──────────────────────────────────────────────────

def _notifications_paused() -> bool:
    return (ROOT / "data" / "notifications_paused").exists()


def notify(title: str, body: str, urgency: str = "normal") -> None:
    """Send a desktop notification via notify-send."""
    if _notifications_paused():
        log.info("Notifications paused — skipping: %s", title)
        return
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
        if urgency in ("critical", "normal"):
            try:
                subprocess.run(
                    ["canberra-gtk-play", "-i", "message-new-instant"],
                    env=env, timeout=2, check=False,
                )
            except Exception:
                pass
    except Exception as exc:
        log.warning("notify-send failed: %s", exc)

    if not PUSH_SCANNER_ALERTS:
        return
    try:
        from nse_intraday_ai.alerts import send_ntfy
        send_ntfy(
            title=title,
            message=body,
            priority="high" if urgency == "critical" else "default",
            tags="rotating_light,chart_with_upwards_trend" if urgency == "critical" else "bell",
        )
    except Exception as exc:
        log.warning("ntfy push failed: %s", exc)


# ── Signal dedup state ────────────────────────────────────────────────────

def _load_state() -> dict:
    from nse_intraday_ai.atomic_io import atomic_read_json
    return atomic_read_json(STATE_PATH, default={"notified": {}})


def _save_state(state: dict) -> None:
    from nse_intraday_ai.atomic_io import atomic_write_json
    atomic_write_json(STATE_PATH, state)


def _signal_key(symbol: str, side: str, entry: float) -> str:
    return f"{symbol}|{side}|{entry:.2f}"


MAX_NOTIFICATIONS_PER_SCAN = 5


def _notify_cycle(cycle, now: datetime) -> None:
    state = _load_state()
    today = now.date().isoformat()
    notified = {k: v for k, v in state.get("notified", {}).items() if v.get("date") == today}

    sent = []
    for result in cycle.recommendations:
        plan = result.plan
        key = _signal_key(result.symbol, plan.side.value, plan.entry or 0)
        if key in notified:
            continue
        if len(sent) >= MAX_NOTIFICATIONS_PER_SCAN:
            log.info("Notification cap reached (%d), skipping remaining.", MAX_NOTIFICATIONS_PER_SCAN)
            break
        votes = [v for v in plan.strategy_votes if v.side == plan.side and v.is_trade]
        est = cycle.estimates.get(id(result))
        history_line = (
            f"\nLearned: {est.win_rate:.0f}% WR, {est.avg_reward_bps:+.0f} bps net ({est.samples} samples)"
            if est is not None and est.is_trained
            else ""
        )
        body = (
            f"{plan.side.value}  entry={plan.entry:.2f}  "
            f"SL={plan.stop_loss:.2f}  T={plan.target:.2f}  "
            f"RR={plan.reward_risk:.2f}  conf={plan.confidence:.0f}%\n"
            f"Regime: {result.regime}  Strategies: {', '.join(v.strategy for v in votes)}"
            f"{history_line}"
        )
        notify(f"🚨 {result.symbol}", body, urgency="critical")
        log.info("Notified: %s %s conf=%.0f%%", result.symbol, plan.side.value, plan.confidence)
        notified[key] = {"date": today, "time": now.strftime("%H:%M")}
        sent.append(result.symbol)

    # Late signals from catch-up bars: informational, never critical, and capped
    # — after a reboot the catch-up can surface dozens at once (2026-09-28: a
    # burst of commodity catch-up alerts got the ntfy topic rate-limited).
    late = 0
    for bar_ts, plan in cycle.stale_signals:
        key = _signal_key(plan.symbol, plan.side.value, plan.entry or 0)
        if key in notified:
            continue
        if late >= MAX_NOTIFICATIONS_PER_SCAN:
            notified[key] = {"date": today, "time": now.strftime("%H:%M")}
            continue
        late += 1
        notify(
            f"⏱ late signal {plan.symbol}",
            f"{plan.side.value} formed on the {bar_ts.strftime('%H:%M')} bar "
            f"(conf={plan.confidence:.0f}%) — window likely passed, logged for the learner.",
            urgency="low",
        )
        notified[key] = {"date": today, "time": now.strftime("%H:%M")}

    state["notified"] = notified
    _save_state(state)
    if sent:
        log.info("New notifications sent: %s", sent)


# ── Entry point: one cycle per universe per invocation ───────────────────

def run() -> None:
    now = datetime.now(IST)
    # NOTE: paused notifications must NOT skip the scan — the scan keeps the
    # candle cache fresh and the shadow learner evaluating outcomes.
    hours = {"nse": nse_market_open, "commodity": commodity_market_open}
    for universe in UNIVERSES:
        open_fn = hours.get(universe)
        if open_fn is None:
            log.warning("unknown universe %r in NSE_DAEMON_UNIVERSES", universe)
            continue
        if not open_fn(now):
            log.info("%s market closed (%s), skipping.", universe, now.strftime("%H:%M IST"))
            continue
        try:
            cycle = run_scan_cycle(
                universe,
                source="daemon",
                quote_client_factory=GoogleFinanceQuoteClient,
            )
        except Exception:
            log.exception("%s scan cycle failed", universe)
            continue
        log.info("Scan done: %s", cycle.summary)
        _notify_cycle(cycle, now)


if __name__ == "__main__":
    run()
