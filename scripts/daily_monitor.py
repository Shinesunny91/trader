#!/usr/bin/env python3
"""Daily monitor of the live gap-reversal book (run after the 15:40 record).

Answers, from local files only (no network unless --push):
  * how did the last session go (₹, bps/trade, winners, stops, unfilled limits)?
  * how does the live record compare with the walk-forward expectation?
  * which expert ranked the list, and how did each expert's own top-8 do
    (rule vs ranker vs open ranker, from data/models/meta_state.json)?
  * is the drift alarm (CUSUM) close to firing?  is the model stale?

    PYTHONPATH=src .venv/bin/python scripts/daily_monitor.py [--push]

Writes data/daily_monitor_report.json; --push sends a 4-line summary via ntfy.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from nse_intraday_ai.atomic_io import atomic_read_json, atomic_write_json  # noqa: E402

DATA = ROOT / "data"
BOOK = DATA / "gap_reversal" / "paper_book.csv"
MODELS = DATA / "models"
REPORT = DATA / "daily_monitor_report.json"
STALE_MODEL_DAYS = 10          # the weekly retrain should never be older than this


def load_book() -> pd.DataFrame:
    if not BOOK.exists():
        return pd.DataFrame()
    df = pd.read_csv(BOOK)
    df["notional"] = (df["entry"] * df["quantity"]).where(df["quantity"] > 0)
    df["net_bps"] = df["net"] / df["notional"] * 1e4
    return df


def session_summary(df: pd.DataFrame, session: str) -> dict:
    day = df[df["session"] == session]
    filled = day[day["quantity"] > 0]
    return {
        "session": session,
        "trades": int(len(filled)), "not_filled": int((day["exit_reason"] == "NO_FILL").sum()),
        "net_inr": round(float(day["net"].sum()), 0),
        "net_bps_per_trade": round(float(filled["net_bps"].mean()), 1) if len(filled) else None,
        "winners": int((filled["net"] > 0).sum()),
        "stops": int((filled["exit_reason"] == "STOP").sum()),
        "best": (filled.nlargest(1, "net")[["symbol", "net"]].to_dict("records") or [None])[0],
        "worst": (filled.nsmallest(1, "net")[["symbol", "net"]].to_dict("records") or [None])[0],
    }


def live_vs_expected(df: pd.DataFrame, expected_bps: float) -> dict:
    filled = df[df["quantity"] > 0]
    per_day = filled.groupby("session")["net_bps"].mean()
    n = len(per_day)
    out = {"sessions": n, "expected_bps": expected_bps,
           "cumulative_net_inr": round(float(df["net"].sum()), 0)}
    if n:
        out["live_bps_per_trade"] = round(float(per_day.mean()), 1)
        out["green_days_pct"] = round(float((per_day > 0).mean() * 100), 0)
        if n >= 2 and per_day.std() > 0:
            # How unusual is the live mean if the expectation were true?
            out["t_vs_expected"] = round(float((per_day.mean() - expected_bps)
                                               / (per_day.std() / n ** 0.5)), 2)
    return out


def experts(state: dict, last_n: int = 20) -> dict:
    track = state.get("track", [])[-last_n:]
    if not track:
        return {}
    frame = pd.DataFrame(track).set_index("session")
    return {"window": f"last {len(frame)} sessions",
            "mean_bps": {k: round(float(v), 1) for k, v in frame.mean().items()},
            "last": {k: round(float(v), 1) for k, v in frame.iloc[-1].items()}}


def model_info() -> dict:
    champ = atomic_read_json(MODELS / "champion.json", default={})
    if not champ:
        return {"champion": None}
    trained = champ.get("directory", "").rsplit("_", 1)[-1]
    out = {"champion": champ.get("directory"), "promoted_at": champ.get("promoted_at")}
    try:
        out["age_days"] = (date.today() - date.fromisoformat(trained)).days
    except ValueError:
        pass
    ev = champ.get("evidence", {})
    if ev:
        out["evidence"] = {k: ev.get(k, {}).get("net_bps_per_trade") for k in ("rule", "ranker", "ranker_open")}
        out["vs_rule_t"] = ev.get("vs_rule", {}).get("t_stat")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", action="store_true", help="send the summary to the phone (ntfy)")
    args = ap.parse_args()

    df = load_book()
    state = atomic_read_json(MODELS / "meta_state.json", default={})
    picks = atomic_read_json(DATA / "gap_reversal" / "picks.json", default={})
    expected = float(state.get("expected_bps", 20.0))
    report: dict = {"at": datetime.now().isoformat(timespec="seconds")}
    if not df.empty:
        report["last_session"] = session_summary(df, df["session"].max())
        report["live"] = live_vs_expected(df, expected)
    report["experts"] = experts(state)
    report["model"] = model_info()
    report["drift"] = {"cusum": round(float(state.get("cusum", 0.0)), 1),
                       "threshold": state.get("threshold_bps", 400.0),
                       "alarm": bool(state.get("alarm")), "cautious_mode": bool(state.get("cautious_mode"))}
    report["list"] = {"session": picks.get("session"), "ranked_by": picks.get("ranked_by"),
                      "source": picks.get("source"), "fallback": picks.get("fallback")}

    alerts = []
    ls = report.get("last_session", {})
    if ls.get("trades") and ls["stops"] / ls["trades"] > 0.5:
        alerts.append(f"{ls['stops']}/{ls['trades']} trades stopped out")
    if report["drift"]["cusum"] > 0.8 * report["drift"]["threshold"]:
        alerts.append(f"CUSUM at {report['drift']['cusum']:.0f}/{report['drift']['threshold']:.0f}")
    if report["drift"]["alarm"]:
        alerts.append("DRIFT ALARM: live book persistently below expectation")
    if report["model"].get("age_days", 0) > STALE_MODEL_DAYS:
        alerts.append(f"model is {report['model']['age_days']} days old (weekly retrain missed?)")
    if report["list"].get("fallback"):
        alerts.append(f"last list used the fallback path: {report['list']['fallback']}")
    report["alerts"] = alerts
    atomic_write_json(REPORT, report)

    lines = []
    if ls:
        lines.append(f"{ls['session']}: ₹{ls['net_inr']:+,.0f}, {ls['winners']}/{ls['trades']} winners, "
                     f"{ls['stops']} stops, {ls['not_filled']} unfilled"
                     + (f", {ls['net_bps_per_trade']:+.0f} bps/trade" if ls['net_bps_per_trade'] is not None else ""))
    lv = report.get("live", {})
    if lv:
        lines.append(f"Live {lv['sessions']} sessions: ₹{lv['cumulative_net_inr']:+,.0f}, "
                     f"{lv.get('live_bps_per_trade', float('nan')):+.0f} bps/trade vs {expected:.0f} expected")
    ex = report["experts"]
    if ex:
        lines.append("Experts (" + ex["window"] + "): "
                     + ", ".join(f"{k} {v:+.0f}" for k, v in ex["mean_bps"].items()) + " bps/trade")
    lines.append("Alerts: " + ("; ".join(alerts) if alerts else "none"))
    print("\n".join(lines))
    print(f"report: {REPORT}")
    if args.push:
        from nse_intraday_ai.alerts import send_ntfy
        send_ntfy("\n".join(lines), title="Gap book — daily monitor",
                  priority="high" if alerts else "default", tags="bar_chart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
