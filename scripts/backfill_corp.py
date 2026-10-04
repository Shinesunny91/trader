"""Resumable backfill of NSE corporate events into data/nse_corp/.

    PYTHONPATH=src .venv/bin/python scripts/backfill_corp.py --start 2016-01-01 --end today \
        --what bm,results,ann

Months already complete (fetched after they ended) are skipped, and an
announcements month restarts from the last week it stored, so the job can be
killed and relaunched at any time.  Order: board meetings and results first
(~250 requests each), announcements last (~560 weekly requests).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from nse_intraday_ai import nse_corp

ALIASES = {"bm": ["board_meetings"], "board_meetings": ["board_meetings"],
           "results": ["results", "integrated"], "res": ["results", "integrated"],
           "integrated": ["integrated"], "ann": ["announcements"], "announcements": ["announcements"]}


def _day(text: str) -> date:
    return date.today() if text == "today" else date.fromisoformat(text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="today")
    p.add_argument("--what", default="bm,results,ann",
                   help="comma list of bm, results, ann (pr is not implemented)")
    a = p.parse_args(argv)
    kinds: list[str] = []
    for w in a.what.split(","):
        w = w.strip().lower()
        if w == "pr":
            print("pr: daily PR archives are not implemented yet; skipped", flush=True)
            continue
        if w not in ALIASES:
            p.error(f"unknown --what item {w!r}")
        kinds += [k for k in ALIASES[w] if k not in kinds]
    log = lambda m: print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {m}", flush=True)  # noqa: E731
    log(f"backfill {kinds} {a.start}..{a.end}")
    try:
        nse_corp.refresh_symbol_changes()
    except Exception as exc:                                      # noqa: BLE001
        log(f"symbol changes download failed: {exc}")
    stats = nse_corp.backfill(_day(a.start), _day(a.end), kinds, log=log)
    log(f"done: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
