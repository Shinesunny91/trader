"""The overnight-drift anomaly in the NIFTY 500, and whether it is tradeable.

This is the first effect measured in this repo that is not conditioned on
`VotingSignalEngine` firing.  Every earlier study asked "the engine fired --
can the outcome be ranked?", which made every negative result a verdict on the
engine rather than on the market, and confined the evidence to the 48 sessions
the 5-minute cache holds.  The daily cache holds 2,479 sessions of 500 names.

What that data says, decomposed over ten years:

    overnight gap        +0.2135 %/day     (+53.8 %/yr)
    intraday open->close -0.1182 %/day     (-29.8 %/yr)
    close to close       +0.0923 %/day     (+23.3 %/yr)

Indian equities earn their return overnight and give part of it back during the
session -- the same pattern documented in US and European markets.  The give-back
is not uniform: it is concentrated in names that gapped up recently.  Sorting on
that gap and shorting the top decile intraday is the strategy measured here.

**The measurement trap.** The obvious version regresses `close(t)/open(t)` on
`open(t)/close(t-1)`.  Both contain `open(t)`, so any error in the opening print
inflates the gap and deflates the same-day return, manufacturing the effect out
of noise.  That version reports +0.602 %/day at t = 26.3.  The version here
predicts session t+1 from the gap on session t -- no shared price -- and reports
+0.258 %/day at t = 11.0.  The difference between those two numbers is the
artifact, and it is more than half the apparent edge.

**Survivorship bias is present and works in this strategy's favour**, which is
worth stating plainly rather than quietly banking: the universe is today's
NIFTY 500, so names that were delisted or demoted are missing, and those are
disproportionately names that fell. A short book would have done *better* with
them included, so the bias understates rather than flatters this result.

    python scripts/overnight_drift_study.py
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
from nse_intraday_ai.execution_plan import intraday_go_live  # noqa: E402

PANEL = ROOT / "data" / "daily_panel.parquet"
DECILE = 0.1
# Cells examined across cross_section_study.py (7 signals x 4 holds) before this
# one was singled out.  Declared so the gate can price the search honestly.
CELLS_SCANNED = 28


def cost_pct(slippage_bps_per_leg: float = 2.5) -> float:
    return round_trip_bps(1000.0, 300, slippage_bps_per_leg=slippage_bps_per_leg) / 100


def short_leg(panel: pd.DataFrame, q: float = DECILE) -> pd.Series:
    """Daily gross % from shorting the top gap decile on the following session."""
    d = panel.dropna(subset=["gap", "fwd_1d"])
    r = d.groupby("day")["gap"].rank(pct=True)
    return (-d[r >= 1 - q].groupby("day")["fwd_1d"].mean()).dropna()


def block_bootstrap(v: np.ndarray, block: int = 21, draws: int = 20000,
                    seed: int = 0) -> tuple[float, float, float]:
    """CI resampling contiguous month-long blocks of sessions.

    Daily resampling would treat 2,449 sessions as independent facts. They share
    a market factor and the strategy's good and bad stretches cluster, so blocks
    are the honest unit.
    """
    rng = np.random.default_rng(seed)
    starts = np.arange(0, v.size - block + 1)
    nb = max(1, v.size // block)
    means = np.array([np.concatenate([v[s:s + block]
                                      for s in rng.choice(starts, nb, replace=True)]).mean()
                      for _ in range(draws)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(means <= 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slippage", type=float, default=2.5)
    args = ap.parse_args()

    d = pd.read_parquet(PANEL)
    d["oc_same"] = (d["close"] / d["open"] - 1) * 100
    gross = short_leg(d)
    cost = cost_pct(args.slippage)
    net = (gross - cost).to_numpy(float)

    print(f"{len(d):,} rows   {d['day'].nunique():,} sessions   "
          f"{str(d['day'].min())[:10]} -> {str(d['day'].max())[:10]}\n")

    print("DECOMPOSITION")
    print(f"  overnight gap         {d['gap'].mean():+.4f} %/day   "
          f"({d['gap'].mean() * 252:+.1f} %/yr)")
    print(f"  intraday open->close  {d['oc_same'].mean():+.4f} %/day   "
          f"({d['oc_same'].mean() * 252:+.1f} %/yr)\n")

    print("THE ARTIFACT — same signal, two forward windows")
    biased = d.dropna(subset=["gap", "oc_same"])
    rb = biased.groupby("day")["gap"].rank(pct=True)
    bs = (-biased[rb >= 1 - DECILE].groupby("day")["oc_same"].mean()).dropna()
    for label, s in [("gap(t) -> oc(t)   shares open(t)", bs),
                     ("gap(t) -> oc(t+1)  no shared price", gross)]:
        v = s.to_numpy(float)
        print(f"  {label:<38}{v.mean():>9.3f} %/day   t = "
              f"{v.mean() / (v.std() / np.sqrt(v.size)):>5.1f}")
    print()

    lo, hi, p = block_bootstrap(net)
    print(f"CLEAN STRATEGY — short top gap decile, session t+1, open to close")
    print(f"  gross {gross.mean():+.3f} %/day   cost {cost:.3f} %   "
          f"net {net.mean():+.3f} %/day   ({net.mean() * 252:+.1f} %/yr)")
    print(f"  sessions {net.size:,}   days up {(net > 0).mean() * 100:.1f}%")
    print(f"  block bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}] %/day   P(<=0) = {p:.4f}\n")

    yr = pd.DataFrame({"net": net}, index=gross.index)
    g = yr.groupby(yr.index.year)["net"]
    print("BY YEAR (annualised net %)")
    print("  " + "  ".join(f"{y}:{v * 252:+5.0f}" for y, v in g.mean().items()))
    print(f"  positive in {(g.mean() > 0).sum()}/{g.ngroups} years   "
          f"— but decaying: {g.mean().iloc[:3].mean() * 252:+.0f}% early "
          f"vs {g.mean().iloc[-3:].mean() * 252:+.0f}% recent\n")

    print("SLIPPAGE BREAK-EVEN")
    for s in (2.5, 5.0, 7.5, 10.0, 12.5):
        n = gross.mean() - cost_pct(s)
        print(f"  {s:>4.1f} bps/leg -> round trip {cost_pct(s) * 100:>5.1f} bps   "
              f"net {n:>+7.3f} %/day   {n * 252:>+7.1f} %/yr")

    half = net.size // 2
    ok, checks = intraday_go_live(
        net_bps=net.mean() * 100, sessions=net.size, p_value=p,
        half1_bps=net[:half].mean() * 100, half2_bps=net[half:].mean() * 100,
        cells_scanned=CELLS_SCANNED,
    )
    print(f"\nGO-LIVE GATE: {'CLEARED' if ok else 'NOT CLEARED'}")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()
