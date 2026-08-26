"""What would the intraday book have done on one specific session?

Trains the four model families on every session in the shipped dataset -- all
of which end before the session under test -- and scores that session's signals
with them.  Nothing about the test day touches the fit, so the answer is a real
out-of-sample result for one day rather than a backtest cell.

The output is deliberately blunt: how many signals printed, what the whole
population did, what the daily cap would have picked, and what that comes to in
rupees at the position size the sizer actually takes.

    python scripts/score_session.py --day 2026-08-25
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

from sklearn.preprocessing import StandardScaler  # noqa: E402

from intraday_oos import families  # noqa: E402
from nse_intraday_ai.execution_plan import MAX_TRADES_PER_DAY  # noqa: E402
from train_model import prepare  # noqa: E402

POSITION = 300_000.0


def session_dates(ts: pd.Series) -> pd.Series:
    """Calendar date of each timestamp, tz dropped.

    The dataset stores tz-aware IST timestamps, so comparing them against a
    plain `pd.Timestamp("2026-08-25")` silently matches nothing rather than
    raising -- which looks exactly like "that session has no signals".
    """
    return pd.to_datetime(ts).dt.tz_localize(None).dt.normalize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", required=True)
    ap.add_argument("--live", type=Path, default=ROOT / "data" / "dataset_live.parquet")
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "dataset.parquet")
    ap.add_argument("--barrier", default="1.5_3.0_12")
    ap.add_argument("--universe", type=int, default=150)
    args = ap.parse_args()

    day = pd.Timestamp(args.day)
    train_raw = pd.read_parquet(args.train)
    live_raw = pd.read_parquet(args.live)
    live_raw = live_raw[session_dates(live_raw["ts"]) == day]
    if live_raw.empty:
        raise SystemExit(f"no signals for {args.day} in {args.live}")

    if args.universe:
        from intraday_sim import liquid_symbols
        liquid = liquid_symbols(args.universe)
        train_raw = train_raw[train_raw["symbol"].isin(liquid)]
        live_raw = live_raw[live_raw["symbol"].isin(liquid)]

    # Guard the whole point of the exercise: if the training file already
    # contains the test day, the result is not out of sample and the number
    # below would be meaningless in a way that is easy to miss.
    train_days = session_dates(train_raw["ts"])
    if (train_days == day).any():
        raise SystemExit(f"{args.day} is present in {args.train.name} — not out of sample")

    tf, Xtr, ytr = prepare(train_raw, args.barrier)
    lf, Xte, yte = prepare(live_raw, args.barrier)
    print(f"train: {len(tf):,} signals over {train_days.nunique()} sessions "
          f"(through {train_days.max().date()})")
    print(f"test:  {len(lf):,} signals on {day.date()}, {lf['symbol'].nunique()} symbols\n")

    print(f"whole population that session: net {yte.mean():+.2f} bps, "
          f"{(yte > 0).mean():.1%} of signals profitable")
    print(f"                               gross {lf[f'gross_bps_{args.barrier}'].mean():+.2f} bps\n")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    header = f"  {'model':<14}{'pick':<16}{'side':<7}{'pred':>8}{'actual':>9}{'rupees':>10}"
    print(f"top-{MAX_TRADES_PER_DAY} pick per model (cap = MAX_TRADES_PER_DAY):")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, factory in families().items():
        model = factory()
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        idx = np.argsort(-pred)[:MAX_TRADES_PER_DAY]
        for i in idx:
            row = lf.iloc[i]
            rupees = yte[i] / 1e4 * POSITION
            print(f"  {name:<14}{row['symbol']:<16}{row['side']:<7}"
                  f"{pred[i]:>8.1f}{yte[i]:>9.1f}{rupees:>10,.0f}")

    print(f"\n  'actual' is net bps after the {10.10:.2f} bps round trip; "
          f"rupees at a Rs {POSITION:,.0f} position.")


if __name__ == "__main__":
    main()
