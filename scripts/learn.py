"""Weekly self-improvement: retrain the ranker, re-validate it, promote only on evidence.

    python scripts/learn.py                      # full cycle (~25 min, ~8 GB RAM)
    python scripts/learn.py --push               # and send the verdict to the phone
    python scripts/learn.py --if-needed --push   # the timer: Saturdays, on a drift alarm,
                                                 # or if no review has ever run

Steps: refresh the NSE/F&O/global data, rebuild the point-in-time history,
walk-forward the ranker over the last two years (quarterly refits), compare its
out-of-sample book with the gap rule's, fit a challenger on everything, and
promote it only if it beat the rule with t >= 2.  The verdict and the evidence
are written to data/models/weekly_report.json for the app.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai import pipeline as P  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--if-needed", action="store_true",
                        help="only retrain on Saturdays, when the drift alarm is up, or if no "
                             "review has ever run (the timer runs this every evening)")
    args = parser.parse_args()

    if args.if_needed:
        from nse_intraday_ai import meta as M
        from nse_intraday_ai import ranker as R

        reasons = []
        if date.today().weekday() == 5:
            reasons.append("weekly schedule")
        if M.MetaState.load().alarm:
            reasons.append("drift alarm")
        if not (R.MODELS / "weekly_report.json").exists():
            reasons.append("never reviewed")
        if not reasons:
            print("no retrain needed today")
            return
        print("retraining: " + ", ".join(reasons))

    if not args.no_fetch:
        P.refresh(date.today() - timedelta(days=1), days=30)
    report = P.weekly(log=lambda m: print(m, flush=True))
    r, b, o = report["ranker"], report["rule"], report["ranker_open"]
    v, vo = report["vs_rule"], report["open_vs_ranker"]
    verdict = lambda n: "PROMOTED" if report["verdicts"][n]["promoted"] else "kept previous"  # noqa: E731
    body = (f"Walk-forward {report['window'][0]}..{report['window'][1]} (net bps/trade):\n"
            f"rule {b.get('net_bps_per_trade', float('nan')):+.1f} | 08:45 ranker "
            f"{r.get('net_bps_per_trade', float('nan')):+.1f} | 09:09 open ranker "
            f"{o.get('net_bps_per_trade', float('nan')):+.1f}\n"
            f"ranker - rule {v['mean_diff_bps']:+.1f}/day (t {v['t_stat']}): {verdict('ranker')}\n"
            f"open - ranker {vo['mean_diff_bps']:+.1f}/day (t {vo['t_stat']}): {verdict('ranker_open')}")
    print(body)
    if args.push:
        from nse_intraday_ai.alerts import send_ntfy
        send_ntfy(body, title="🧠 Weekly model review", priority="default", tags="brain")


if __name__ == "__main__":
    main()
