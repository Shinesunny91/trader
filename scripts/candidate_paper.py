"""Run the surviving intraday candidate on paper and accumulate its record.

The candidate is the only rule this repo has produced that passed a properly
constructed null: ridge score above the 99th percentile of the trailing score
distribution, one trade per session, taken in time order as signals print.  It
returned +67.44 bps over 14 in-sample sessions and then **lost** 23.51 bps over
the first four forward trades, which is why this script writes a paper log
instead of an order ticket.

Four forward trades settle nothing in either direction.  `intraday_go_live()`
asks for 60 sessions, both halves positive, and p <= 0.01 before real money
moves, and the only way to get there is to run the rule forward, session by
session, and record what it does.  That is all this does.  It never places an
order and it never sizes against real capital -- the rupee column is what the
trade *would* have made at the standard position, so the log can be scored
later against the gate.

    python scripts/candidate_paper.py --day 2026-08-25
    python scripts/candidate_paper.py --backfill 2026-08-18 2026-08-25
    python scripts/candidate_paper.py --status
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

from nse_intraday_ai.execution_plan import (  # noqa: E402
    GO_LIVE_GATE,
    build_execution_plan,
    intraday_go_live,
)
from train_model import prepare  # noqa: E402

LOG = ROOT / "data" / "candidate_paper.csv"
BARRIER = "1.5_3.0_12"
PERCENTILE = 99.0
UNIVERSE = 150
CAPITAL = 1_000_000.0
POSITION = 300_000.0


def session_dates(ts: pd.Series) -> pd.Series:
    """Calendar date of each timestamp, tz dropped.

    The dataset stores tz-aware IST timestamps, so comparing them against a
    plain `pd.Timestamp("2026-08-25")` silently matches nothing rather than
    raising -- which looks exactly like "that session has no signals".
    """
    return pd.to_datetime(ts).dt.tz_localize(None).dt.normalize()


def liquid() -> set[str]:
    from intraday_sim import liquid_symbols
    return liquid_symbols(UNIVERSE)


def fit_through(train_raw: pd.DataFrame, cutoff: pd.Timestamp):
    """Ridge fitted on every labelled session strictly before `cutoff`.

    Returns the model, its scaler and the score threshold, so the caller can
    score a session without any of it having touched that session's rows.
    """
    hist = train_raw[session_dates(train_raw["ts"]) < cutoff]
    if hist.empty:
        raise SystemExit(f"no training sessions before {cutoff.date()}")
    _, X, y = prepare(hist, BARRIER)
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=10.0).fit(scaler.transform(X), y)
    threshold = float(np.percentile(model.predict(scaler.transform(X)), PERCENTILE))
    return model, scaler, threshold, len(hist)


def run_day(day: pd.Timestamp, train_raw: pd.DataFrame, live_raw: pd.DataFrame,
            keep: set[str]) -> dict | None:
    session = live_raw[session_dates(live_raw["ts"]) == day]
    session = session[session["symbol"].isin(keep)]
    if session.empty:
        print(f"{day.date()}: no signals in the dataset")
        return None

    model, scaler, threshold, n_train = fit_through(train_raw, day)
    frame, X, y = prepare(session, BARRIER)
    frame = frame.reset_index(drop=True)
    frame["score"] = model.predict(scaler.transform(X))
    frame["net_bps"] = y

    hits = frame.sort_values("ts")
    hits = hits[hits["score"] >= threshold]
    if hits.empty:
        print(f"{day.date()}: nothing cleared p{PERCENTILE} (threshold "
              f"{threshold:+.2f} bps) — no trade, which is a real outcome")
        return {"day": day.date(), "symbol": "", "side": "", "score": np.nan,
                "threshold": round(threshold, 2), "net_bps": np.nan,
                "rupees": 0.0, "traded": 0, "n_train": n_train}

    r = hits.iloc[0]
    plan = build_execution_plan(
        symbol=r["symbol"], side=r["side"], signal_price=float(r["fill"]),
        atr=float(r["atr"]), capital=CAPITAL,
    )
    print(f"\n{day.date()}  {r['symbol']} {r['side']}   score {r['score']:+.2f} "
          f"(threshold {threshold:+.2f})   trained on {n_train:,} signals")
    print(f"  {plan.order_ticket()}")
    print(f"  stop {plan.stop_from_fill}   target {plan.target_from_fill}   "
          f"square off {plan.square_off_at}")
    if not np.isnan(r["net_bps"]):
        print(f"  outcome: {r['net_bps']:+.1f} bps net  ->  "
              f"Rs {r['net_bps'] / 1e4 * POSITION:+,.0f} at a Rs {POSITION:,.0f} position")
    return {"day": day.date(), "symbol": r["symbol"], "side": r["side"],
            "score": round(float(r["score"]), 2), "threshold": round(threshold, 2),
            "net_bps": round(float(r["net_bps"]), 2),
            "rupees": round(float(r["net_bps"]) / 1e4 * POSITION, 0),
            "traded": 1, "n_train": n_train}


def append(rows: list[dict]) -> None:
    rows = [r for r in rows if r]
    if not rows:
        return
    new = pd.DataFrame(rows)
    if LOG.exists():
        old = pd.read_csv(LOG)
        new = pd.concat([old[~old["day"].astype(str).isin(new["day"].astype(str))], new])
    new.sort_values("day").to_csv(LOG, index=False)
    print(f"\nwrote {LOG.relative_to(ROOT)} ({len(new)} sessions)")


def status() -> None:
    if not LOG.exists():
        print(f"no paper log yet — run --backfill or --day first")
        return
    d = pd.read_csv(LOG)
    traded = d[d["traded"] == 1].dropna(subset=["net_bps"])
    print(f"{len(d)} sessions observed, {len(traded)} with a trade, "
          f"{len(d) - len(traded)} with none\n")
    if traded.empty:
        return
    print(traded[["day", "symbol", "side", "score", "net_bps", "rupees"]]
          .to_string(index=False))
    v = traded["net_bps"].to_numpy(float)
    h = len(v) // 2
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(v, v.size, replace=True).mean() for _ in range(20000)])
    p_value = float(np.mean(boot <= 0))
    print(f"\nmean {v.mean():+.2f} bps   up {(v > 0).sum()}/{len(v)}   "
          f"total Rs {traded['rupees'].sum():+,.0f}")
    ok, checks = intraday_go_live(
        net_bps=float(v.mean()), sessions=len(d), p_value=p_value,
        half1_bps=float(v[:h].mean()) if h else 0.0,
        half2_bps=float(v[h:].mean()), cells_scanned=1,
    )
    print(f"\ngo-live gate: {'CLEARED' if ok else 'NOT CLEARED'}")
    for k, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {k}")
    print(f"\n{max(0, GO_LIVE_GATE['min_sessions'] - len(d))} more sessions "
          f"needed before the sample question is even answerable.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day")
    ap.add_argument("--backfill", nargs=2, metavar=("FROM", "TO"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--live", type=Path, default=ROOT / "data" / "dataset_live.parquet")
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "dataset.parquet")
    args = ap.parse_args()

    if args.status:
        status()
        return
    if not (args.day or args.backfill):
        ap.error("give --day, --backfill FROM TO, or --status")

    train_raw = pd.read_parquet(args.train)
    live_raw = pd.read_parquet(args.live)
    keep = liquid()
    train_raw = train_raw[train_raw["symbol"].isin(keep)]

    if args.day:
        days = [pd.Timestamp(args.day)]
    else:
        lo, hi = (pd.Timestamp(x) for x in args.backfill)
        have = sorted(session_dates(live_raw["ts"]).unique())
        days = [d for d in have if lo <= d <= hi]
    append([run_day(d, train_raw, live_raw, keep) for d in days])
    print()
    status()


if __name__ == "__main__":
    main()
