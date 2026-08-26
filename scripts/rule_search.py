"""Search for ONE simple, stateable intraday rule per universe.

The brief: something a person can hold in their head — "at 09:20, if the Asian
tape is up, buy X" — that wins most of the time.

The danger with that brief is that a search over enough simple rules will always
produce one that looks excellent on the data it was chosen from.  This workspace
has been burned by exactly that (foreign indices: +0.09 in sample, ~0 out; a
pred_bps threshold picked on 17 sessions lost 614 bps on the next 17).  So every
rule here is:

  * defined only from information available at the signal bar,
  * scored on a CALIBRATION window,
  * then re-scored, untouched, on a later VALIDATION window,
  * and reported with both numbers plus the count of independent sessions.

A rule is only reported as usable if it keeps its sign in both windows AND
clears its universe's round-trip cost in the validation window.  Everything else
is printed too, because knowing what failed is the point.

    python scripts/rule_search.py nse
    python scripts/rule_search.py commodity
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.learning import COMMODITY_COST_BPS, ASSUMED_COST_BPS  # noqa: E402

DB = ROOT / "data" / "candles.sqlite3"
LEARNER = ROOT / "data" / "shadow_learner.sqlite3"
IST = "Asia/Kolkata"

# Macro series the rules may reference.  These are the only ones with enough
# history in the cache to be usable, and each is read *as of* the signal bar.
MACRO = {
    "nikkei":  "^N225",
    "hangseng": "^HSI",
    "spfut":   "ES=F",
    "nifty":   "^NSEI",
    "vix":     "^INDIAVIX",
    "usdinr":  "USDINR=X",
    "dxy":     "DX-Y.NYB",
    "crude":   "CL=F",
}


def load_macro(lookback_bars: int = 12) -> pd.DataFrame:
    """Percent change of each macro series over the trailing `lookback_bars`.

    Forward-filled onto a 5-minute grid, then shifted so a signal at time t can
    only ever see the bar that closed at or before t.
    """
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    frames = {}
    for name, sym in MACRO.items():
        d = pd.read_sql_query(
            "SELECT ts, close FROM candles WHERE symbol=? AND interval='5m' ORDER BY ts",
            con, params=(sym,),
        )
        if d.empty:
            continue
        d["ts"] = pd.to_datetime(d["ts"], utc=True, format="ISO8601").dt.tz_convert(IST)
        s = d.set_index("ts")["close"].astype(float)
        s = s[~s.index.duplicated(keep="last")]
        frames[name] = (s / s.shift(lookback_bars) - 1.0) * 100.0
        frames[f"{name}_lvl"] = s
    con.close()
    grid = pd.DataFrame(frames).sort_index()
    return grid


def load_observations(universe: str) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{LEARNER}?mode=ro", uri=True)
    op = "LIKE" if universe == "commodity" else "NOT LIKE"
    d = pd.read_sql_query(
        f"""SELECT observed_at, symbol, side, confidence, reward_risk, vote_count, reward_bps
            FROM shadow_observations
            WHERE status='EVALUATED' AND reward_bps IS NOT NULL
              AND source='live' AND symbol {op} '%=F'""",
        con,
    )
    con.close()
    d["ts"] = pd.to_datetime(d["observed_at"], utc=True, format="ISO8601").dt.tz_convert(IST)
    d["date"] = d["ts"].dt.date
    d["minute"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    return d.sort_values("ts").drop(columns=["observed_at"])


def attach_macro(obs: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """As-of join: each signal sees only macro bars already closed, max 45m stale."""
    macro = macro.sort_index()
    out = pd.merge_asof(
        obs.sort_values("ts"), macro, left_on="ts", right_index=True,
        direction="backward", tolerance=pd.Timedelta("45min"),
    )
    return out


# ── rule vocabulary ────────────────────────────────────────────────────────
# Each rule is (label, predicate).  Predicates use only columns present at the
# signal bar.  Deliberately small and readable: this is meant to produce a rule
# a person can follow, not a model.

def time_windows(universe: str) -> list[tuple[str, int, int]]:
    if universe == "commodity":
        return [("00:00-05:00 post-US", 0, 300), ("05:00-09:00 Globex", 300, 540),
                ("09:00-14:00 Asia/EU", 540, 840), ("14:00-19:00 EU/US-pre", 840, 1140),
                ("19:00-24:00 US RTH", 1140, 1441)]
    return [("09:15-09:45 open", 555, 585), ("09:45-10:30", 585, 630),
            ("10:30-12:00 mid", 630, 720), ("12:00-14:00 lunch", 720, 840),
            ("14:00-15:15 close", 840, 915)]


def macro_conditions(universe: str) -> list[tuple[str, object]]:
    c: list[tuple[str, object]] = [("any tape", lambda d: pd.Series(True, index=d.index))]
    for name in ("nikkei", "hangseng", "spfut", "nifty", "crude", "dxy", "usdinr"):
        if name not in ("nikkei", "hangseng", "spfut", "nifty", "crude", "dxy", "usdinr"):
            continue
        c.append((f"{name} up",   lambda d, n=name: d[n] > 0))
        c.append((f"{name} down", lambda d, n=name: d[n] < 0))
    c.append(("asia up (N225 & HSI)",   lambda d: (d["nikkei"] > 0) & (d["hangseng"] > 0)))
    c.append(("asia down (N225 & HSI)", lambda d: (d["nikkei"] < 0) & (d["hangseng"] < 0)))
    return c


def evaluate(frame: pd.DataFrame, cost: float, min_n: int = 30) -> pd.DataFrame:
    rows = []
    for wlabel, lo, hi in time_windows(ARGS.universe):
        inwin = (frame["minute"] >= lo) & (frame["minute"] < hi)
        for clabel, pred in macro_conditions(ARGS.universe):
            try:
                mask = inwin & pred(frame).fillna(False)
            except KeyError:
                continue
            for side in ("LONG", "SHORT"):
                sel = frame[mask & (frame["side"] == side)]
                if len(sel) < min_n:
                    continue
                rows.append({
                    "window": wlabel, "condition": clabel, "side": side,
                    "n": len(sel), "sessions": sel["date"].nunique(),
                    "gross": sel["reward_bps"].mean(),
                    "net": sel["reward_bps"].mean() - cost,
                    "winpct": (sel["reward_bps"] > cost).mean() * 100,
                })
    return pd.DataFrame(rows)


def main() -> None:
    global ARGS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("universe", choices=["nse", "commodity"])
    p.add_argument("--min-n", type=int, default=30, help="min signals per cell")
    ARGS = p.parse_args()

    cost = COMMODITY_COST_BPS if ARGS.universe == "commodity" else ASSUMED_COST_BPS
    obs = attach_macro(load_observations(ARGS.universe), load_macro())
    if obs.empty:
        print("no live observations for this universe")
        return

    days = sorted(obs["date"].unique())
    cut = days[len(days) // 2]
    cal, val = obs[obs["date"] < cut], obs[obs["date"] >= cut]
    print(f"{ARGS.universe}: {len(obs):,} live signals over {len(days)} sessions "
          f"({days[0]} .. {days[-1]}), round trip {cost:.0f} bps")
    print(f"  calibrate < {cut}  ({len(cal):,} signals, {cal['date'].nunique()} sessions)")
    print(f"  validate >= {cut}  ({len(val):,} signals, {val['date'].nunique()} sessions)\n")

    a = evaluate(cal, cost, ARGS.min_n)
    b = evaluate(val, cost, ARGS.min_n)
    if a.empty or b.empty:
        print("not enough data to form any cell at this --min-n")
        return
    m = a.merge(b, on=["window", "condition", "side"], suffixes=("_cal", "_val"))
    m["holds"] = (np.sign(m["net_cal"]) == np.sign(m["net_val"])) & (m["net_val"] > 0)

    print(f"{len(m)} rules had >= {ARGS.min_n} signals in BOTH windows.\n")
    show = m.sort_values("net_cal", ascending=False)
    hdr = (f"{'window':22s}{'condition':24s}{'side':6s}"
           f"{'cal n':>7s}{'cal net':>9s}{'val n':>7s}{'val net':>9s}{'val win%':>10s}  ")
    print("── best 12 by CALIBRATION net (i.e. what a naive search would pick) ──")
    print(hdr)
    for _, r in show.head(12).iterrows():
        flag = "HOLDS" if r["holds"] else ""
        print(f"{r['window']:22s}{r['condition']:24s}{r['side']:6s}"
              f"{r['n_cal']:7.0f}{r['net_cal']:9.1f}{r['n_val']:7.0f}{r['net_val']:9.1f}"
              f"{r['winpct_val']:10.1f}  {flag}")

    keep = m[m["holds"]].sort_values("net_val", ascending=False)
    print(f"\n── survived out of sample: {len(keep)} of {len(m)} ──")
    if keep.empty:
        print("  none. every rule that looked good on the calibration window either")
        print("  flipped sign or failed to clear costs on the validation window.")
    else:
        print(hdr)
        for _, r in keep.iterrows():
            print(f"{r['window']:22s}{r['condition']:24s}{r['side']:6s}"
                  f"{r['n_cal']:7.0f}{r['net_cal']:9.1f}{r['n_val']:7.0f}{r['net_val']:9.1f}"
                  f"{r['winpct_val']:10.1f}")
        print(f"\n  Expect {len(m)} * 0.25 ~= {len(m) * 0.25:.0f} rules to pass this test by")
        print("  chance alone (sign agreement is a coin flip, and >0 is roughly another).")
        print("  Treat any survivor as a hypothesis to paper-trade, not a discovery.")


if __name__ == "__main__":
    main()
