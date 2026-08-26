"""Deep test of the intra-week book over 10 years of daily bars.

Every result here is reported against three things that kill most backtests:

  1. **A random-pick baseline.**  The same book, same sizing, same costs, same
     dates — picking uniformly at random.  Run many times, it is the noise floor
     of the *procedure*, not of the market.  A strategy that cannot beat it by
     several standard deviations has shown nothing.
  2. **Buy and hold.**  A long-only book in a market that rose over the sample
     will make money for reasons that have nothing to do with the signal.  NIFTY
     over the same window is the honest benchmark.
  3. **Out-of-sample time splits.**  Scores are fixed formulas, not fitted
     models, so there is nothing to train — but a rule that only works in one
     half of a decade is still a rule that does not work.

    python scripts/swing_backtest.py nse
    python scripts/swing_backtest.py commodity
    python scripts/swing_backtest.py nse --sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_intraday_ai.candle_cache import CandleCache  # noqa: E402
from nse_intraday_ai.costs import Segment  # noqa: E402
from nse_intraday_ai.data import DEFAULT_COMMODITY_SYMBOLS  # noqa: E402
from nse_intraday_ai.swing import SCORES, SwingConfig, backtest, build_panel  # noqa: E402

CACHE = CandleCache(ROOT / "data" / "candles.sqlite3")
BENCHMARK = "^NSEI"


def load_frames(universe: str, limit: int | None = None) -> dict[str, pd.DataFrame]:
    import sqlite3
    con = sqlite3.connect(f"file:{ROOT / 'data' / 'candles.sqlite3'}?mode=ro", uri=True)
    like = "%=F" if universe == "commodity" else "%.NS"
    symbols = [r[0] for r in con.execute(
        "SELECT symbol, COUNT(*) n FROM candles WHERE interval='1d' AND symbol LIKE ?"
        " GROUP BY symbol HAVING n >= 400 ORDER BY n DESC", (like,))]
    con.close()
    if limit:
        symbols = symbols[:limit]
    frames = {}
    for s in symbols:
        f = CACHE.load(s, "1d", limit=4000)
        if not f.empty:
            frames[s] = f
    return frames


def config_for(universe: str, **kw) -> SwingConfig:
    """Universe defaults; any SwingConfig field can be overridden by keyword."""
    base = dict(
        segment=Segment.COMMODITY_FUTURES if universe == "commodity" else Segment.EQUITY_DELIVERY,
        min_turnover=0.0 if universe == "commodity" else 5e7,
        positions=1, hold_days=5, entry_weekday=0,
    )
    base.update(kw)
    return SwingConfig(**base)


def benchmark_return(start, end) -> float:
    f = CACHE.load(BENCHMARK, "1d", limit=4000)
    if f.empty:
        return float("nan")
    f = f.loc[(f.index >= start) & (f.index <= end)]
    if len(f) < 2:
        return float("nan")
    return (f["close"].iloc[-1] / f["close"].iloc[0] - 1) * 100


def run(panel, frames, name, cfg):
    res = backtest(panel, frames, SCORES[name], cfg)
    s = res.summary()
    s["name"] = name
    return res, s


def random_baseline(panel, frames, cfg, trials: int) -> tuple[float, float, list[float]]:
    outs = []
    for i in range(trials):
        np.random.seed(1000 + i)
        r = backtest(panel, frames, SCORES["random"], cfg)
        outs.append(r.summary().get("net_pct", 0.0))
    return float(np.mean(outs)), float(np.std(outs) or 1e-9), outs


def fmt(s: dict) -> str:
    return (f"{s['trades']:6d}{s['net_pct']:10.1f}{s['avg_bps']:11.1f}"
            f"{s['win_pct']:9.1f}{s['profit_factor']:9.2f}{s['max_dd_pct']:9.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("universe", choices=["nse", "commodity"])
    p.add_argument("--hold", type=int, default=5)
    p.add_argument("--positions", type=int, default=1)
    p.add_argument("--stop-atr", type=float, default=2.5)
    p.add_argument("--target-atr", type=float, default=0.0)
    p.add_argument("--trials", type=int, default=25, help="random baseline runs")
    p.add_argument("--limit", type=int, default=None, help="cap symbols (for a fast pass)")
    p.add_argument("--sweep", action="store_true", help="hold/positions/stop grid")
    args = p.parse_args()

    print(f"loading {args.universe} daily frames...", flush=True)
    frames = load_frames(args.universe, args.limit)
    print(f"  {len(frames)} symbols")
    panel = build_panel(frames)
    if panel.empty:
        raise SystemExit("no panel — run scripts/fetch_daily.py first")
    start, end = panel.date.min(), panel.date.max()
    years = (end - start).days / 365.25
    print(f"  panel {len(panel):,} rows, {start.date()} -> {end.date()} ({years:.1f}y)\n")

    cfg = config_for(args.universe, hold_days=args.hold, positions=args.positions,
                     stop_atr=args.stop_atr, target_atr=args.target_atr)
    print(f"config: {args.positions} position(s), {args.hold}-session hold, "
          f"stop {args.stop_atr} ATR, target {args.target_atr or 'none'}, "
          f"{cfg.segment.value}\n")

    hdr = f"{'strategy':24s}{'trades':>6s}{'net %':>10s}{'avg bps':>11s}{'win %':>9s}{'PF':>9s}{'maxDD':>9s}"
    print("── full sample ──"); print(hdr)
    results = {}
    for name in SCORES:
        if name == "random":
            continue
        res, s = run(panel, frames, name, cfg)
        results[name] = (res, s)
        print(f"{name:24s}{fmt(s)}")

    rmean, rsd, routs = random_baseline(panel, frames, cfg, args.trials)
    print(f"{'random (x' + str(args.trials) + ')':24s}{'':6s}{rmean:10.1f}"
          f"{'':11s}{'':9s}{'':9s}   sd {rsd:.1f}")
    bh = benchmark_return(start, end)
    print(f"{'NIFTY buy & hold':24s}{'':6s}{bh:10.1f}")

    print(f"\n── vs the noise floor (net %, ({'x'}-random)/sd) ──")
    for name, (_, s) in sorted(results.items(), key=lambda kv: -kv[1][1]["net_pct"]):
        z = (s["net_pct"] - rmean) / rsd
        verdict = "beats noise" if z >= 2 else ("marginal" if z >= 1 else "indistinguishable")
        print(f"  {name:24s} {s['net_pct']:8.1f}%   z = {z:+5.2f}   {verdict}")

    # ── out-of-sample halves ───────────────────────────────────────────────
    mid = start + (end - start) / 2
    print(f"\n── time split (first half < {mid.date()} <= second half) ──")
    print(f"{'strategy':24s}{'H1 net %':>11s}{'H2 net %':>11s}{'both +':>9s}")
    for name in results:
        h1 = backtest(panel[panel.date < mid], frames, SCORES[name], cfg).summary()
        h2 = backtest(panel[panel.date >= mid], frames, SCORES[name], cfg).summary()
        both = "yes" if h1.get("net_pct", 0) > 0 and h2.get("net_pct", 0) > 0 else ""
        print(f"{name:24s}{h1.get('net_pct',0):11.1f}{h2.get('net_pct',0):11.1f}{both:>9s}")

    # ── per-year, the real robustness test ────────────────────────────────
    best = max(results.items(), key=lambda kv: kv[1][1]["net_pct"])[0]
    print(f"\n── {best}: year by year ──")
    res = results[best][0]
    t = res.trades.copy()
    t["year"] = pd.to_datetime(t.exit_date).dt.year
    for year, g in t.groupby("year"):
        pnl = g.net_pnl.sum() / cfg.capital * 100
        bar = "#" * min(40, int(abs(pnl) * 2))
        print(f"  {year}  {len(g):4d} trades  {pnl:+7.2f}%  {bar}")

    per_year = t.groupby("year").net_pnl.sum() / cfg.capital * 100
    print(f"\n  years positive: {(per_year > 0).sum()}/{len(per_year)}")
    boot = [np.random.default_rng(i).choice(t.net_bps.values, len(t), replace=True).mean()
            for i in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  bootstrap 95% CI on net bps/trade: [{lo:+.1f}, {hi:+.1f}]"
          + ("  (spans zero)" if lo < 0 < hi else "  (excludes zero)"))
    print(f"  cost drag: {t.cost_bps.mean():.1f} bps/trade of "
          f"{t.gross_bps.mean():+.1f} bps gross")

    if args.sweep:
        print("\n── parameter sweep on " + best + " ──")
        print(f"{'hold':>6s}{'pos':>5s}{'stop':>7s}{'trades':>8s}{'net %':>10s}{'avg bps':>10s}{'maxDD':>8s}")
        for hold in (3, 5, 10, 15):
            for positions in (1, 3, 5):
                for stop in (2.5, 4.0, 0.0):
                    c = config_for(args.universe, hold_days=hold, positions=positions,
                                   stop_atr=stop, target_atr=args.target_atr)
                    s = backtest(panel, frames, SCORES[best], c).summary()
                    if not s["trades"]:
                        continue
                    print(f"{hold:6d}{positions:5d}{stop:7.1f}{s['trades']:8d}"
                          f"{s['net_pct']:10.1f}{s['avg_bps']:10.1f}{s['max_dd_pct']:8.1f}")


if __name__ == "__main__":
    main()
