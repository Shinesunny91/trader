"""Forward-test the one causal cell that survived scrutiny, on unseen sessions.

`causal_selectivity.py` scanned 48 (model, percentile, cap) combinations and one
of them -- p_ridge, trade above the 99th percentile of the trailing score
distribution, one trade a day -- returned +67 bps over 14 sessions.  Unlike
everything else tried in this repo it then survived a best-of-48 permutation
null, a time-matched control, a both-halves split and dropping its two best
sessions.

None of that makes it real.  It was still *chosen* by looking at those sessions,
and 14 trades is not a sample.  The only test that carries weight now is the one
the cell cannot have been fitted to: sessions that did not exist when it was
selected.  This script trains ridge on everything up to a cutoff and applies the
identical rule -- same percentile, same cap, same first-signal-in-time
tie-break -- to the sessions after it.

    python scripts/forward_test_cell.py --cutoff 2026-08-17
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from train_model import prepare  # noqa: E402

POSITION = 300_000.0
PERCENTILE = 99.0
CAP = 1


def session_dates(ts: pd.Series) -> pd.Series:
    """Calendar date of each timestamp, tz dropped.

    The dataset stores tz-aware IST timestamps, so comparing them against a
    plain `pd.Timestamp("2026-08-25")` silently matches nothing rather than
    raising -- which looks exactly like "that session has no signals".
    """
    return pd.to_datetime(ts).dt.tz_localize(None).dt.normalize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", default="2026-08-17")
    ap.add_argument("--live", type=Path, default=ROOT / "data" / "dataset_live.parquet")
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "dataset.parquet")
    ap.add_argument("--barrier", default="1.5_3.0_12")
    ap.add_argument("--universe", type=int, default=150)
    args = ap.parse_args()

    cutoff = pd.Timestamp(args.cutoff)
    train_raw = pd.read_parquet(args.train)
    live_raw = pd.read_parquet(args.live)
    if args.universe:
        from intraday_sim import liquid_symbols
        liquid = liquid_symbols(args.universe)
        train_raw = train_raw[train_raw["symbol"].isin(liquid)]
        live_raw = live_raw[live_raw["symbol"].isin(liquid)]

    train_raw = train_raw[session_dates(train_raw["ts"]) <= cutoff]
    live_raw = live_raw[session_dates(live_raw["ts"]) > cutoff]
    if live_raw.empty:
        raise SystemExit(f"no sessions after {args.cutoff} in {args.live.name}")

    tf, Xtr, ytr = prepare(train_raw, args.barrier)
    lf, Xte, yte = prepare(live_raw, args.barrier)
    lf = lf.reset_index(drop=True)
    lf["day"] = session_dates(lf["ts"])
    lf["label_bps"] = yte

    scaler = StandardScaler().fit(Xtr)
    model = Ridge(alpha=10.0).fit(scaler.transform(Xtr), ytr)
    train_scores = model.predict(scaler.transform(Xtr))
    thresh = float(np.percentile(train_scores, PERCENTILE))
    lf["score"] = model.predict(scaler.transform(Xte))

    print(f"train {len(tf):,} signals through {cutoff.date()}   "
          f"p{PERCENTILE} threshold = {thresh:.2f} bps")
    print(f"test  {len(lf):,} signals over {lf['day'].nunique()} unseen sessions\n")

    header = f"  {'session':<12}{'symbol':<15}{'side':<7}{'score':>8}{'net_bps':>9}{'rupees':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    taken = []
    for day, s in lf.sort_values("ts").groupby("day"):
        hits = s[s["score"] >= thresh].head(CAP)
        if hits.empty:
            print(f"  {str(day.date()):<12}{'(no signal cleared the threshold)':<39}")
            continue
        for _, r in hits.iterrows():
            print(f"  {str(day.date()):<12}{r['symbol']:<15}{r['side']:<7}"
                  f"{r['score']:>8.1f}{r['label_bps']:>9.1f}"
                  f"{r['label_bps'] / 1e4 * POSITION:>10,.0f}")
            taken.append(r["label_bps"])

    if not taken:
        print("\n  the rule fired on no session — nothing to score")
        return
    t = np.asarray(taken)
    print(f"\n  {len(t)} trades   mean {t.mean():+.2f} bps   "
          f"up {(t > 0).sum()}/{len(t)}   total Rs {t.sum() / 1e4 * POSITION:+,.0f}")
    print(f"  in-sample cell was +67.44 bps over 14 sessions")


if __name__ == "__main__":
    main()
