"""Does the overnight gap predict the next session's intraday direction?

`cross_section_study.py` reported a long-short decile spread on the gap of
+0.189% per trade, t = 12.7, over 2,454 sessions -- by a wide margin the
strongest thing this repo has measured.  This script tries to break it.

The reason it is worth taking seriously in the first place is what it does
*not* share.  The classic same-day gap fade regresses `close(t)/open(t)` on
`open(t)/close(t-1)`, and both sides contain `open(t)`.  Any error in the
opening print -- a stale quote, a wide auction, a single odd-lot trade --
inflates the gap and deflates the same-day return by construction, producing
reversal out of pure noise.  The spread measured here instead predicts
`close(t+1)/open(t+1)` from `open(t)/close(t-1)`, which have no price in
common, so that artifact cannot generate it.

Both versions are reported below so the size of the artifact is visible rather
than argued about.

Checks applied: decile monotonicity (a real effect is ordered, not just extreme),
year by year (an effect that lives in two years is a regime), liquidity terciles
(an effect confined to illiquid names is untradeable), and a block bootstrap
that respects the fact that all names share a market factor each day.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.costs import round_trip_bps  # noqa: E402

PANEL = ROOT / "data" / "daily_panel.parquet"
INTRADAY_BPS = round_trip_bps(1000.0, 300)
DECILE = 0.1


def daily_spread(d: pd.DataFrame, sig: str, fwd: str, q: float = DECILE) -> pd.Series:
    d = d.dropna(subset=[sig, fwd])
    r = d.groupby("day")[sig].rank(pct=True)
    top = d[r >= 1 - q].groupby("day")[fwd].mean()
    bot = d[r <= q].groupby("day")[fwd].mean()
    return (bot - top)          # reversal: long the gap-downs, short the gap-ups


def block_bootstrap(v: np.ndarray, block: int = 21, draws: int = 20000,
                    seed: int = 0) -> tuple[float, float, float]:
    """CI that resamples contiguous blocks of sessions.

    Independent daily resampling would treat 2,454 sessions as 2,454
    independent facts.  They are not: every name shares that day's market, and
    a strategy's good and bad stretches cluster.  Month-long blocks keep that
    structure in the resample.
    """
    rng = np.random.default_rng(seed)
    n = v.size
    nb = max(1, n // block)
    starts = np.arange(0, n - block + 1)
    means = np.empty(draws)
    for i in range(draws):
        pick = rng.choice(starts, nb, replace=True)
        means[i] = np.concatenate([v[s:s + block] for s in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(means <= 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", type=Path, default=PANEL)
    args = ap.parse_args()

    d = pd.read_parquet(args.panel)
    d["oc_same_day"] = (d["close"] / d["open"] - 1) * 100
    print(f"{len(d):,} rows   {d['day'].nunique():,} sessions   "
          f"intraday round trip {INTRADAY_BPS:.2f} bps\n")

    print("SHARED-PRICE ARTIFACT — the same signal against two forward windows")
    print(f"  {'forward window':<44}{'gross %':>10}{'t':>8}{'ann %':>9}")
    print("  " + "-" * 69)
    for fwd, label in [("oc_same_day", "same day open->close (shares open(t))"),
                       ("fwd_1d", "next day open->close (no shared price)")]:
        s = daily_spread(d, "gap", fwd).dropna()
        v = s.to_numpy(float)
        t = v.mean() / (v.std() / np.sqrt(v.size))
        print(f"  {label:<44}{v.mean():>10.3f}{t:>8.1f}{v.mean() * 252:>9.1f}")
    print("  The gap between these two is the artifact. Only the second is usable.\n")

    s = daily_spread(d, "gap", "fwd_1d").dropna()
    v = s.to_numpy(float)
    net = v - INTRADAY_BPS / 100
    lo, hi, p = block_bootstrap(net)
    print("NET OF COSTS — long gap-downs, short gap-ups, at the next open")
    print(f"  gross {v.mean():+.3f}%/day   cost {INTRADAY_BPS / 100:.3f}%   "
          f"net {net.mean():+.3f}%/day")
    print(f"  annualised net {net.mean() * 252:+.1f}%   "
          f"days up {(net > 0).mean() * 100:.1f}%   sessions {v.size:,}")
    print(f"  block bootstrap 95% CI on net {lo:+.3f} to {hi:+.3f} %/day   "
          f"P(<=0) = {p:.4f}\n")

    print("DECILE MONOTONICITY — a real effect is ordered, not just extreme")
    dd = d.dropna(subset=["gap", "fwd_1d"])
    dec = dd.groupby("day")["gap"].rank(pct=True)
    dd = dd.assign(dec=np.ceil(dec * 10).clip(1, 10).astype(int))
    prof = dd.groupby("dec")["fwd_1d"].mean()
    for k, val in prof.items():
        bar = "#" * int(abs(val) * 300)
        print(f"  decile {k:>2} (gap {'low' if k == 1 else 'high' if k == 10 else '   '}) "
              f"{val:>+8.3f}%  {bar}")
    print(f"  monotone decreasing: {bool((prof.diff().dropna() < 0).all())}\n")

    print("BY YEAR — an effect that lives in two years is a regime, not an edge")
    yr = pd.DataFrame({"net": net}, index=s.index)
    yr["year"] = yr.index.year
    g = yr.groupby("year")["net"]
    print(f"  {'year':<8}{'sessions':>10}{'net %/day':>12}{'ann %':>9}{'up%':>8}")
    for year, grp in g:
        print(f"  {year:<8}{len(grp):>10}{grp.mean():>12.3f}{grp.mean() * 252:>9.1f}"
              f"{(grp > 0).mean() * 100:>8.1f}")
    print(f"  years positive: {(g.mean() > 0).sum()}/{g.ngroups}\n")

    print("BY LIQUIDITY TERCILE — is it only in names you cannot trade?")
    dd = dd.copy()
    dd["liq"] = dd.groupby("day")["turnover_cr"].transform(
        lambda x: pd.qcut(x, 3, labels=["thin", "mid", "liquid"], duplicates="drop"))
    print(f"  {'tercile':<10}{'gross %':>10}{'net %':>10}{'ann net %':>12}{'t':>8}")
    for name, grp in dd.groupby("liq", observed=True):
        sp = daily_spread(grp, "gap", "fwd_1d").dropna()
        if sp.size < 100:
            continue
        w = sp.to_numpy(float)
        n = w - INTRADAY_BPS / 100
        t = n.mean() / (n.std() / np.sqrt(n.size))
        print(f"  {str(name):<10}{w.mean():>10.3f}{n.mean():>10.3f}"
              f"{n.mean() * 252:>12.1f}{t:>8.1f}")


if __name__ == "__main__":
    main()
