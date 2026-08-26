"""Measure all four horizons the same way, for NSE and commodities.

The point of running them together is the cost/edge ratio.  Transaction costs
are roughly fixed per round trip while the move being chased grows with the
square root of holding time, so the drag falls monotonically with horizon.  That
is the single strongest force in this system and it is invisible unless the
horizons are measured on one bench.

Every result is reported against four things:

  * a **random-pick baseline** run many times — the noise floor of the procedure
  * **buy and hold** — a long book in a rising market makes money for reasons
    that have nothing to do with the signal
  * an **out-of-sample time split** — a rule that works in one half of a decade
    is a rule that does not work
  * the count of **independent observations**, because a 250-session hold over
    ten years is ten trades, not 2,500, and every statistic must be read
    against that.

Learned ranking uses purged, embargoed walk-forward with uniqueness weighting
(`nse_intraday_ai.ml`).  At these horizons labels overlap heavily and an
ordinary walk-forward leaks the answer into the training set.

    python scripts/horizon_study.py                       # everything
    python scripts/horizon_study.py --universe commodity
    python scripts/horizon_study.py --horizon intramonth --model
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

from nse_intraday_ai import horizons as H  # noqa: E402
from nse_intraday_ai.candle_cache import CandleCache  # noqa: E402
from nse_intraday_ai.ml import walk_forward_predict  # noqa: E402
from nse_intraday_ai.swing import SCORES, SwingConfig, backtest, build_panel  # noqa: E402
from swing_backtest import load_frames  # noqa: E402

CACHE = CandleCache(ROOT / "data" / "candles.sqlite3")

# Features the learned ranker may use. All are computed from closed bars only.
ML_FEATURES = [
    "r_ret_1w", "r_ret_4w", "r_ret_12w", "r_ret_12w_skip1w", "r_ret_26w",
    "r_rsi_14", "r_atr_pct", "r_volume_ratio", "r_dist_52w_high",
    "r_pct_of_20d_range", "r_turnover_20d",
    "above_sma50", "above_sma200", "dist_sma50", "dist_sma200", "sma50_slope",
    "vol_20d", "atr_pct",
]


def config_for(universe: str, horizon: H.Horizon, **kw) -> SwingConfig:
    base = dict(
        segment=horizon.segment_for(universe),
        min_turnover=0.0 if universe == "commodity" else 5e7,
        positions=1,
        hold_days=horizon.hold_sessions,
        stop_atr=horizon.stop_atr,
        target_atr=horizon.target_atr,
        entry_weekday=0 if horizon.hold_sessions <= 5 else None,
    )
    base.update(kw)
    return SwingConfig(**base)


def benchmark(start, end) -> float:
    f = CACHE.load("^NSEI", "1d", limit=4000)
    if f.empty:
        return float("nan")
    f = f.loc[(f.index >= start) & (f.index <= end)]
    return (f["close"].iloc[-1] / f["close"].iloc[0] - 1) * 100 if len(f) > 1 else float("nan")


def add_forward_label(panel: pd.DataFrame, frames: dict, hold: int) -> pd.DataFrame:
    """Net forward return over the holding period — the thing to predict."""
    out = []
    for symbol, rows in panel.groupby("symbol", sort=False):
        bars = frames.get(symbol)
        if bars is None or bars.empty:
            continue
        closes = bars["close"]
        pos = closes.index.get_indexer(rows["date"], method="nearest")
        fwd = np.full(len(rows), np.nan)
        arr = closes.to_numpy()
        for i, p in enumerate(pos):
            j = p + hold
            if 0 <= p < len(arr) and j < len(arr) and arr[p] > 0:
                fwd[i] = (arr[j] / arr[p] - 1.0) * 1e4        # bps
        r = rows.copy()
        r["fwd_bps"] = fwd
        r["bar_pos"] = pos
        out.append(r)
    return pd.concat(out) if out else pd.DataFrame()


def model_scores(panel: pd.DataFrame, frames: dict, horizon: H.Horizon) -> pd.Series | None:
    """Purged walk-forward predictions, aligned to `panel`."""
    labelled = add_forward_label(panel, frames, horizon.hold_sessions)
    if labelled.empty:
        return None
    feats = [c for c in ML_FEATURES if c in labelled.columns]
    frame = labelled.dropna(subset=feats + ["fwd_bps", "bar_pos"])
    if len(frame) < 500:
        return None
    frame = frame.sort_values("date")
    start = frame["bar_pos"].astype(int).reset_index(drop=True)
    end = start + horizon.hold_sessions
    pred = walk_forward_predict(
        frame[feats].reset_index(drop=True),
        frame["fwd_bps"].reset_index(drop=True),
        start, end,
        n_splits=6,
        embargo=max(horizon.hold_sessions // 2, 2),
    )
    return pd.Series(pred.to_numpy(), index=frame.index).reindex(panel.index)


def run_one(universe: str, horizon: H.Horizon, frames: dict, panel: pd.DataFrame,
            trials: int, use_model: bool) -> None:
    cfg = config_for(universe, horizon)
    start, end = panel.date.min(), panel.date.max()
    sessions = panel.date.nunique()
    indep = horizon.independent_obs(sessions)

    print(f"\n{'=' * 92}")
    print(f"{universe.upper()}  ·  {horizon.label}  ({horizon.hold_sessions}-session hold, "
          f"{cfg.segment.value})")
    print(f"{'=' * 92}")
    print(f"  {horizon.note}")
    print(f"  {sessions} sessions of data -> ~{indep} INDEPENDENT observations at this horizon")

    hdr = (f"  {'ranking':24s}{'trades':>7s}{'net %':>10s}{'avg bps':>10s}"
           f"{'cost bps':>10s}{'win %':>8s}{'PF':>7s}{'maxDD':>8s}")
    print("\n" + hdr)

    results = {}
    for name in SCORES:
        if name == "random":
            continue
        res = backtest(panel, frames, SCORES[name], cfg)
        s = res.summary()
        if not s["trades"]:
            continue
        results[name] = (res, s)
        print(f"  {name:24s}{s['trades']:7d}{s['net_pct']:10.1f}{s['avg_bps']:10.1f}"
              f"{res.trades.cost_bps.mean():10.1f}{s['win_pct']:8.1f}"
              f"{s['profit_factor']:7.2f}{s['max_dd_pct']:8.1f}")

    if use_model:
        preds = model_scores(panel, frames, horizon)
        if preds is not None and preds.notna().sum() > 100:
            lookup = preds
            res = backtest(panel, frames, lambda rows: lookup.reindex(rows.index), cfg)
            s = res.summary()
            if s["trades"]:
                results["MODEL (purged wf)"] = (res, s)
                print(f"  {'MODEL (purged wf)':24s}{s['trades']:7d}{s['net_pct']:10.1f}"
                      f"{s['avg_bps']:10.1f}{res.trades.cost_bps.mean():10.1f}"
                      f"{s['win_pct']:8.1f}{s['profit_factor']:7.2f}{s['max_dd_pct']:8.1f}")
        else:
            print(f"  {'MODEL (purged wf)':24s}   not enough labelled samples at this horizon")

    outs = []
    for i in range(trials):
        np.random.seed(500 + i)
        outs.append(backtest(panel, frames, SCORES["random"], cfg).summary().get("net_pct", 0.0))
    rmean, rsd = float(np.mean(outs)), float(np.std(outs) or 1e-9)
    print(f"  {'random (x' + str(trials) + ')':24s}{'':7s}{rmean:10.1f}"
          f"{'':10s}{'':10s}{'':8s}{'':7s}   sd {rsd:.1f}")
    bh = benchmark(start, end)
    print(f"  {'NIFTY buy & hold':24s}{'':7s}{bh:10.1f}")

    mid = start + (end - start) / 2
    print(f"\n  {'ranking':24s}{'vs noise':>11s}{'H1 net %':>11s}{'H2 net %':>11s}   verdict")
    for name, (res, s) in sorted(results.items(), key=lambda kv: -kv[1][1]["net_pct"]):
        z = (s["net_pct"] - rmean) / rsd
        if name.startswith("MODEL"):
            h1 = h2 = float("nan")
        else:
            h1 = backtest(panel[panel.date < mid], frames, SCORES[name], cfg).summary().get("net_pct", 0)
            h2 = backtest(panel[panel.date >= mid], frames, SCORES[name], cfg).summary().get("net_pct", 0)
        stable = (h1 > 0 and h2 > 0)
        verdict = ("beats noise + stable" if z >= 2 and stable else
                   "beats noise" if z >= 2 else
                   "marginal" if z >= 1 else "indistinguishable")
        if s["net_pct"] < bh:
            verdict += ", below buy&hold"
        print(f"  {name:24s}{z:+11.2f}{h1:11.1f}{h2:11.1f}   {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", choices=["nse", "commodity", "both"], default="both")
    p.add_argument("--horizon", choices=H.ORDER + ["all"], default="all")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--model", action="store_true", help="also fit the learned ranker")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    universes = ["nse", "commodity"] if args.universe == "both" else [args.universe]
    keys = H.ORDER if args.horizon == "all" else [args.horizon]

    for universe in universes:
        frames = load_frames(universe, args.limit)
        panel = build_panel(frames)
        if panel.empty:
            print(f"{universe}: no daily data — run scripts/fetch_daily.py")
            continue
        for key in keys:
            horizon = H.get(key)
            if horizon.interval != "1d":
                print(f"\n{'=' * 92}\n{universe.upper()}  ·  {horizon.label}\n{'=' * 92}")
                print("  Intraday runs on 5-minute bars through the separate engine:")
                print("    python scripts/sim_today.py           (today's book)")
                print("    python scripts/swing_backtest.py      (daily horizons)")
                print("  Measured there: no rule-based gate was profitable, and the +27.6")
                print("  bps/trade model-ranked book once claimed here has been RETRACTED —")
                print("  it was an argmax over four model families on their own test set.")
                print("  Gross intraday edge is +0.8 bps against a 10.1 bps round trip:")
                print("    python scripts/intraday_edge_audit.py")
                continue
            run_one(universe, horizon, frames, panel, args.trials, args.model)


if __name__ == "__main__":
    main()
