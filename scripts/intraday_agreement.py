"""Does agreement between independent models find the signals one model cannot?

The motivation is that a single model's top pick is a single model's opinion,
and `scripts/train_model.py` showed how easily that opinion is noise dressed as
skill.  Four families that see the data differently — a linear model, two
boosted-tree depths, a bagged forest — agreeing on the same signal is a
different and stronger claim than any of them ranking it first alone.

So this asks three things of `data/intraday_oos.parquet`:

1. **How correlated are the four?**  If they rank almost identically then
   "agreement" is one opinion counted four times and the rest of this script is
   meaningless.
2. **Does the intersection of their top-K beat each of them?**  Reported against
   the population mean, because concentrating into a smaller, worse subset is a
   real and common outcome.
3. **Does any of it hold in time?**  Every cell is re-reported on a
   calibrate/validate split of the held-out sessions.  A rule found by scanning
   many cells will produce winners in-sample by construction; the split is what
   separates them from findings.

Nothing here is allowed to select a winner and report its number as the result.
The scan width is printed alongside, because a table of 40 cells containing one
good cell is a table of noise containing noise.

    python scripts/intraday_agreement.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STORE = ROOT / "data" / "intraday_oos.parquet"


def load(path: Path) -> tuple[pd.DataFrame, list[str]]:
    d = pd.read_parquet(path)
    models = sorted(c[2:] for c in d.columns if c.startswith("p_"))
    d = d.dropna(subset=[f"p_{m}" for m in models])
    d["day"] = d["ts"].dt.normalize()
    return d, models


def rank_agreement(d: pd.DataFrame, models: list[str]) -> None:
    """Spearman correlation of the models' within-session rankings."""
    ranks = {m: d.groupby("day")[f"p_{m}"].rank(pct=True) for m in models}
    frame = pd.DataFrame(ranks)
    print("\nwithin-session rank correlation between models")
    print(frame.corr(method="spearman").to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n(near-1.0 everywhere means agreement adds no information — the models\n"
          " are one opinion in four costumes.)")


def per_model(d: pd.DataFrame, models: list[str], ks: list[int]) -> pd.DataFrame:
    rows = []
    for m in models:
        for k in ks:
            top = d.sort_values(f"p_{m}", ascending=False).groupby("day").head(k)
            per = top.groupby("day")["label_bps"].mean()
            rows.append({"selector": m, "k": k, "trades": len(top),
                         "net_bps": top["label_bps"].mean(),
                         "sessions_up": f"{int((per > 0).sum())}/{len(per)}"})
    return pd.DataFrame(rows)


def unanimous(d: pd.DataFrame, models: list[str], ks: list[int]) -> pd.DataFrame:
    """Signals every model puts inside its own top-K for that session."""
    rows = []
    for k in ks:
        picks = None
        for m in models:
            ranked = d.groupby("day")[f"p_{m}"].rank(ascending=False, method="first")
            inside = set(d.index[ranked <= k])
            picks = inside if picks is None else (picks & inside)
        sel = d.loc[sorted(picks)]
        if sel.empty:
            rows.append({"selector": f"unanimous top-{k}", "k": k, "trades": 0,
                         "net_bps": np.nan, "sessions_up": "0/0"})
            continue
        per = sel.groupby("day")["label_bps"].mean()
        rows.append({"selector": f"unanimous top-{k}", "k": k, "trades": len(sel),
                     "net_bps": sel["label_bps"].mean(),
                     "sessions_up": f"{int((per > 0).sum())}/{len(per)}"})
    return pd.DataFrame(rows)


def mean_rank(d: pd.DataFrame, models: list[str], ks: list[int]) -> pd.DataFrame:
    """Rank-average across models — the soft version of unanimity."""
    avg = sum(d.groupby("day")[f"p_{m}"].rank(pct=True) for m in models) / len(models)
    d = d.assign(_avg=avg)
    rows = []
    for k in ks:
        top = d.sort_values("_avg", ascending=False).groupby("day").head(k)
        per = top.groupby("day")["label_bps"].mean()
        rows.append({"selector": "rank-average", "k": k, "trades": len(top),
                     "net_bps": top["label_bps"].mean(),
                     "sessions_up": f"{int((per > 0).sum())}/{len(per)}"})
    return pd.DataFrame(rows)


def split_check(d: pd.DataFrame, models: list[str], ks: list[int]) -> pd.DataFrame:
    """Re-report every selector on each half of the held-out sessions."""
    days = np.array(sorted(d["day"].unique()))
    mid = days[len(days) // 2]
    halves = {"H1": d[d["day"] < mid], "H2": d[d["day"] >= mid]}
    out = None
    for name, part in halves.items():
        table = pd.concat([per_model(part, models, ks),
                           unanimous(part, models, ks),
                           mean_rank(part, models, ks)])
        table = table.set_index(["selector", "k"])[["net_bps"]].rename(
            columns={"net_bps": name})
        out = table if out is None else out.join(table)
    out["both_positive"] = (out["H1"] > 0) & (out["H2"] > 0)
    return out.reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, default=STORE)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    args = p.parse_args()

    if not args.store.exists():
        print(f"{args.store} not found — run scripts/intraday_oos.py first")
        return

    d, models = load(args.store)
    print("=" * 88)
    print(f"INTRADAY MODEL AGREEMENT — {len(d):,} scored signals, "
          f"{d['day'].nunique()} held-out sessions, {len(models)} models")
    print("=" * 88)
    print(f"population net edge {d['label_bps'].mean():+.2f} bps  "
          f"(this is the bar every selector below must beat)")

    rank_agreement(d, models)

    table = pd.concat([per_model(d, models, args.ks),
                       unanimous(d, models, args.ks),
                       mean_rank(d, models, args.ks)])
    print("\nselector performance over all held-out sessions\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    split = split_check(d, models, args.ks)
    print("\nsame selectors, re-reported on each half of the held-out window\n")
    print(split.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    survivors = split[split["both_positive"]]
    cells = len(split)
    print("\n" + "=" * 88)
    print(f"{len(survivors)} of {cells} selectors are positive in BOTH halves. "
          f"At a coin-flip rate\nyou would expect about {cells / 4:.0f} by chance, "
          f"so treat this as a screen, not a result.")
    if not survivors.empty:
        print()
        print(survivors.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main()
