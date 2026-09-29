"""One-shot (resumable) download of everything the ranker learns from.

    python scripts/backfill_nse.py                  # 2016-06 .. yesterday, ~45 min first time
    python scripts/backfill_nse.py --since 2026-09-01

Fetches only what is missing, so it doubles as a repair tool after downtime:
  NSE equity bhavcopy (+MTO)   data/nse_bhav/
  NSE F&O bhavcopy             data/nse_fo/
  NSE crowding / bans / shorts data/nse_extra/
  NSE participant OI           data/nse_flows/
  global + Indian index series and ADRs, 10 years daily   candle cache
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai import features as FT  # noqa: E402
from nse_intraday_ai import gap_reversal as G  # noqa: E402
from nse_intraday_ai import nse_bhav, nse_extra, nse_fo  # noqa: E402
from nse_intraday_ai.nse_flows import load_history  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2016-06-01")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    start, end = date.fromisoformat(args.since), date.today() - timedelta(days=1)
    t = time.time()
    for name, mod in (("equity bhavcopy", nse_bhav), ("F&O bhavcopy", nse_fo),
                      ("crowding/bans/short sales", nse_extra)):
        ok, hol = mod.backfill(start, end, workers=args.workers)
        print(f"{name:28s} {ok:5d} new sessions, {hol:4d} holidays  ({time.time() - t:.0f}s)", flush=True)
    load_history(start, end, ROOT / "data" / "nse_flows", what="oi")
    print(f"{'participant OI':28s} done  ({time.time() - t:.0f}s)", flush=True)
    period = "10y" if start.year <= 2017 else "2y"
    got = G.refresh(list(FT.MARKET_SERIES) + list(FT.ADRS), interval="1d", period=period)
    print(f"{'global/index series':28s} {len(got)}/{len(FT.MARKET_SERIES) + len(FT.ADRS)}  "
          f"({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
