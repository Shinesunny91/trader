"""Weekly self-improvement: retrain the ranker, re-validate it, promote only on evidence.

    python scripts/learn.py            # full cycle (~15 min, ~8 GB RAM)
    python scripts/learn.py --push     # and send the verdict to the phone

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
    args = parser.parse_args()

    if not args.no_fetch:
        P.refresh(date.today() - timedelta(days=1), days=30)
    report = P.weekly(log=lambda m: print(m, flush=True))
    r, b, v = report["ranker"], report["rule"], report["vs_rule"]
    verdict = "PROMOTED" if report["promoted"] else "kept the previous model"
    body = (f"Walk-forward {report['window'][0]}..{report['window'][1]}:\n"
            f"ranker {r.get('net_bps_per_trade', float('nan')):+.1f} bps/trade (t {r.get('t_stat')}), "
            f"rule {b.get('net_bps_per_trade', float('nan')):+.1f} (t {b.get('t_stat')})\n"
            f"difference {v['mean_diff_bps']:+.1f} bps/day, t {v['t_stat']} over {v['sessions']} sessions\n"
            f"challenger {report['challenger']}: {verdict}")
    print(body)
    if args.push:
        from nse_intraday_ai.alerts import send_ntfy
        send_ntfy(body, title="🧠 Weekly model review", priority="default", tags="brain")


if __name__ == "__main__":
    main()
