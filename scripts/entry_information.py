"""Does the entry signal know which way the price will go?

Everything else in this repo -- exits, sizing, ranking, cost work -- is
downstream of one question, and it had never been asked directly.  Every study
here labelled outcomes with triple barriers, and a triple barrier mixes two
things together: whether the signal predicted direction, and whether the
particular stop/target geometry happened to suit the path.  A 2:1 target/stop
produces a small positive *gross* number on pure noise, which is exactly how
the population came to look like it had +0.8 bps of edge to rank.

Stripping the barriers away leaves the raw question: taking the signal's own
side, what is the forward return at a fixed horizon?  If the entry carries
directional information, that number is positive and grows with the horizon,
because information about direction compounds as the move plays out.  If it is
flat at zero across every horizon, the entry is a coin flip, and no exit rule,
position sizer, or ranking model can extract anything from it -- there is
nothing there to extract.

Run:
    python scripts/entry_information.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.costs import round_trip_bps  # noqa: E402

COST = round_trip_bps(1000.0, 300)
HORIZONS = [("ret6_bps", "30 min"), ("ret12_bps", "60 min"),
            ("ret24_bps", "120 min"), ("ret_eod_bps", "to close")]
BOOTSTRAP = 20_000


def session_ci(per_session: pd.Series, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI over *sessions*, not trades.

    Trades inside one session share that session's market, so they are nowhere
    near independent; resampling them individually would report a confidence
    interval several times tighter than the truth.
    """
    rng = np.random.default_rng(seed)
    vals = per_session.to_numpy()
    boot = np.array([rng.choice(vals, vals.size, replace=True).mean()
                     for _ in range(BOOTSTRAP)])
    return tuple(np.percentile(boot, [2.5, 97.5]))


def main() -> None:
    d = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    d["day"] = pd.to_datetime(d["ts"]).dt.normalize()
    print(f"{len(d):,} signals over {d.day.nunique()} sessions   "
          f"cost {COST:.2f} bps per round trip\n")

    print("Forward return taking the signal's own side (gross bps, no barriers):")
    header = f"  {'horizon':<11}{'mean':>9}{'median':>9}{'win%':>8}{'sd':>8}{'95% CI on session mean':>26}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for col, label in HORIZONS:
        s = d[col].dropna()
        lo, hi = session_ci(d.groupby("day")[col].mean().dropna())
        print(f"  {label:<11}{s.mean():>9.2f}{s.median():>9.2f}{(s > 0).mean() * 100:>8.1f}"
              f"{s.std():>8.1f}{f'[{lo:+.2f}, {hi:+.2f}]':>26}")

    print("\n  Direction pays nothing at any horizon, and a real edge would grow"
          "\n  with horizon rather than stay flat.  The cost line is 10.10 bps.")

    # The same question asked of the model-ranked subset: ranking can only help
    # if the thing it ranks contains information in the first place.
    oos = ROOT / "data" / "intraday_oos.parquet"
    if oos.exists():
        o = pd.read_parquet(oos)
        o["day"] = pd.to_datetime(o["ts"]).dt.normalize()
        key = ["symbol", "ts", "side"]
        merged = o.merge(d[key + ["ret_eod_bps"]], on=key, how="inner")
        print(f"\nSame question of the model-ranked subset ({len(merged):,} matched rows):")
        print(f"  {'selection':<26}{'trades':>8}{'ret_eod':>10}{'win%':>8}")
        print("  " + "-" * 52)
        base = merged.groupby("day").ret_eod_bps.mean().mean()
        print(f"  {'every signal':<26}{len(merged):>8,}{base:>10.2f}"
              f"{(merged.ret_eod_bps > 0).mean() * 100:>8.1f}")
        for q in (0.10, 0.02, 0.01):
            means, n = [], 0
            for _, s in merged.groupby("day"):
                k = max(1, int(round(len(s) * q)))
                top = s.nlargest(k, "p_rf")
                means.append(top.ret_eod_bps.mean())
                n += k
            wins = pd.concat([s.nlargest(max(1, int(round(len(s) * q))), "p_rf")
                              for _, s in merged.groupby("day")])
            print(f"  {f'p_rf top {q:.0%} per session':<26}{n:>8,}"
                  f"{np.mean(means):>10.2f}{(wins.ret_eod_bps > 0).mean() * 100:>8.1f}")
        print("\n  Ranking cannot add information that the population does not carry.")


if __name__ == "__main__":
    main()
