#!/usr/bin/env python3
"""Walk-forward A/B of ranker variants, faithful to the live 08:45 book.

Every variant goes through the same expanding-window, quarterly-refit
walk-forward as `pipeline.weekly()`, and is booked exactly as the live list is
traded: top-8 of the 300 most liquid names, trade-for-trade names removed,
pre-open SELL LIMIT entry (unfilled picks earn 0), 13 bps costs.

A variant replaces the baseline only if its paired daily edge has t >= 3 after
a Holm correction across the variants in the run (Harvey, Liu & Zhu 2016); the
Deflated Sharpe charges for every trial made so far.

    PYTHONPATH=src .venv/bin/python scripts/compare_rankers.py                 # default set
    PYTHONPATH=src .venv/bin/python scripts/compare_rankers.py morning morning_corp
    PYTHONPATH=src .venv/bin/python scripts/compare_rankers.py --rebuild-panel

Run long jobs under systemd so they survive restarts:
    systemd-run --user --unit=compare-rankers -p Nice=5 -p MemoryMax=10G \
        --working-directory=$PWD --setenv=PYTHONPATH=src \
        .venv/bin/python scripts/compare_rankers.py
Outputs: data/ranker_comparison/<variant>.parquet (OOS scores + daily book),
data/ranker_comparison.json (report).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from nse_intraday_ai import features as FT  # noqa: E402
from nse_intraday_ai.atomic_io import atomic_write_json  # noqa: E402
from nse_intraday_ai.gap_reversal import GapReversalConfig  # noqa: E402
from nse_intraday_ai.pipeline import BOOK_UNIVERSE, COST_BPS  # noqa: E402
from nse_intraday_ai.ranker import (  # noqa: E402
    DEFAULT_EXCLUDE, RankerConfig, book, deflated_sharpe, holm, paired, paired_pvalue,
    summarize, walk_forward,
)

OUT = ROOT / "data" / "ranker_comparison"
PANEL = OUT / "panel_v2.parquet"          # v2: built with the corp features
REPORT = ROOT / "data" / "ranker_comparison.json"
START = "2019-01-01"
# Variants already tried in docs/research-log.md; the DSR charges for them.
TRIALS_SO_FAR = 33

MORNING = DEFAULT_EXCLUDE + FT.OPEN_FEATURES          # what the 08:45 model may not see
VARIANTS: dict[str, RankerConfig] = {
    # the production champion's recipe
    "morning": RankerConfig(exclude=MORNING + FT.CORP_FEATURES),
    # + catalyst features (results / announcements / exchange queries / gap z)
    "morning_corp": RankerConfig(exclude=MORNING),
    # optional, not in the default run (already found neutral on the open panel)
    "morning_lgbm_reg": RankerConfig(ranker_type="lgbm_reg", exclude=MORNING + FT.CORP_FEATURES),
    "morning_3seed": RankerConfig(seeds=(0, 42, 137), exclude=MORNING + FT.CORP_FEATURES),
}
DEFAULT_RUN = ("morning", "morning_corp")
BASELINE = "morning"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_panel(rebuild: bool) -> pd.DataFrame:
    if PANEL.exists() and not rebuild:
        return pd.read_parquet(PANEL)
    log("building the full point-in-time panel (same code path as pipeline.weekly) ...")
    panel = FT.history(with_open=True)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL)
    return panel


def books(panel: pd.DataFrame, score: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(live book with the pre-open limit, the same picks entered at the open)."""
    clean = panel["was_be_20"].fillna(0) == 0
    s = score.where(clean)
    limit = GapReversalConfig().entry_limit_pct
    return (book(panel, s, universe=BOOK_UNIVERSE, cost_bps=COST_BPS, entry_limit=limit),
            book(panel, s, universe=BOOK_UNIVERSE, cost_bps=COST_BPS))


def run_variant(name: str, panel: pd.DataFrame, start: str) -> pd.DataFrame:
    path = OUT / f"{name}.parquet"
    if path.exists():
        log(f"  {name}: checkpoint")
        return pd.read_parquet(path)
    cfg = VARIANTS[name]
    log(f"--- {name}: {cfg.ranker_type}, seeds={cfg.seeds}, {len(cfg.exclude)} excluded ---")
    t0 = time.time()
    scores = walk_forward(panel, cfg, start=start, refit="QS", log=log)
    limit_book, open_book = books(panel, scores)
    daily = pd.DataFrame({"net_bps": limit_book, "net_bps_at_open": open_book})
    daily.attrs["elapsed"] = time.time() - t0
    daily.to_parquet(path)
    scores.dropna().rename("score").to_frame().to_parquet(OUT / f"{name}_scores.parquet")
    log(f"  {name}: {limit_book.mean():.1f} bps/day-slot, {time.time() - t0:.0f}s")
    return daily


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="*", help=f"subset of {sorted(VARIANTS)}")
    ap.add_argument("--start", default=START, help="first out-of-sample session")
    ap.add_argument("--rebuild-panel", action="store_true")
    args = ap.parse_args()
    names = args.variants or list(DEFAULT_RUN)
    unknown = set(names) - set(VARIANTS)
    if unknown:
        ap.error(f"unknown variants {sorted(unknown)}")
    if BASELINE not in names:
        names.insert(0, BASELINE)

    panel = load_panel(args.rebuild_panel)
    log(f"panel {len(panel):,} rows, {panel.index.get_level_values('session').nunique()} sessions, "
        f"{len(panel.columns)} columns")
    rule_limit, rule_open = books(panel, panel["gap1"])
    daily = {"gap_rule": rule_limit}
    at_open = {"gap_rule": rule_open}
    for n in names:
        d = run_variant(n, panel, args.start)
        daily[n], at_open[n] = d["net_bps"], d["net_bps_at_open"]

    stats = {n: summarize(d, args.start) for n, d in daily.items()}
    tested = [n for n in names if n != BASELINE]
    pairs = {n: paired(daily[n], daily[BASELINE], args.start) for n in tested + ["gap_rule"]}
    adj = holm({n: paired_pvalue(pairs[n]["t_stat"]) for n in tested}) if tested else {}
    for n in tested:
        pairs[n]["holm_p"] = adj[n]
    sharpes = [s["sharpe"] / 250 ** 0.5 for s in stats.values() if "sharpe" in s]
    dsr = {n: deflated_sharpe(d[d.index.astype(str) >= args.start], TRIALS_SO_FAR + len(names), sharpes)
           for n, d in daily.items()}
    limit_effect = {n: paired(daily[n], at_open[n], args.start) for n in daily}

    print("\n" + "=" * 92)
    print(f"{'variant':18s} {'bps/slot':>9s} {'t':>7s} {'Sharpe':>7s} {'maxDD%':>7s} {'DSR':>6s} "
          f"{'vs base':>8s} {'t':>6s} {'holm_p':>7s} {'limit Δ':>8s} {'t':>6s}")
    for n in daily:
        s, p, le = stats[n], pairs.get(n, {}), limit_effect[n]
        print(f"{n:18s} {s.get('net_bps_per_trade', float('nan')):9.2f} {s.get('t_stat', 0):7.2f} "
              f"{s.get('sharpe', 0):7.2f} {s.get('max_drawdown_pct', 0):7.2f} {dsr[n]['dsr']:6.3f} "
              f"{p.get('mean_diff_bps', 0):+8.2f} {p.get('t_stat', 0):+6.2f} {p.get('holm_p', float('nan')):7.3f} "
              f"{le['mean_diff_bps']:+8.2f} {le['t_stat']:+6.2f}")
    print("=" * 92)
    print("bps/slot = net bps per book slot per session (unfilled limit orders count as 0).")
    print("limit Δ  = live limit entry minus the same picks entered at the open.\n")

    winners = [n for n in tested if pairs[n]["t_stat"] >= 3 and pairs[n]["holm_p"] < 0.05
               and pairs[n]["mean_diff_bps"] > 0]
    verdict = (f"ADOPT {max(winners, key=lambda n: pairs[n]['t_stat'])}" if winners
               else f"KEEP {BASELINE} (no variant survives t>=3 after Holm)")
    log("RECOMMENDATION: " + verdict)
    atomic_write_json(REPORT, {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "start": args.start, "baseline": BASELINE,
        "stats": stats, "paired_vs_baseline": pairs, "deflated_sharpe": dsr,
        "limit_vs_open": limit_effect, "recommendation": verdict,
    })
    log(f"report: {REPORT}")


if __name__ == "__main__":
    main()
