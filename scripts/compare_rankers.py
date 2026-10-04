#!/usr/bin/env python3
"""Walk-forward comparison of ranker variants on the 10-year gap-reversal dataset.

Runs each variant through the same expanding-window quarterly-refit walk-forward
used by the live system, then compares net bps/trade, Sharpe, win-rate, and
paired t-stats against the baseline.

Usage:
    cd trading-workspace
    PYTHONPATH=src .venv/bin/python scripts/compare_rankers.py
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from nse_intraday_ai.features import history
from nse_intraday_ai.ranker import (
    RankerConfig, walk_forward, book, summarize, paired, feature_columns,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_variant(name, panel, config, **wf_kwargs):
    """Run walk-forward + book simulation for one ranker variant."""
    log(f"--- {name} ---")
    log(f"  config: ranker_type={config.ranker_type}, seeds={config.seeds}, "
        f"halflife={config.halflife_years}, target_kind={config.target_kind}")
    t0 = time.time()
    scores = walk_forward(panel, config, log=lambda m: log(m), **wf_kwargs)
    elapsed = time.time() - t0
    daily = book(panel, scores, k=8, cost_bps=13.0)
    stats = summarize(daily)
    stats["variant"] = name
    stats["elapsed_sec"] = round(elapsed, 1)
    log(f"  {name}: {stats['net_bps_per_trade']:.1f} bps/trade, "
        f"t={stats.get('t_stat', 'N/A')}, sharpe={stats.get('sharpe', 'N/A')}, "
        f"elapsed={elapsed:.0f}s")
    return scores, daily, stats


def main():
    log("Loading 10-year feature panel...")
    cache = ROOT / "data" / "ranker_comparison" / "panel.parquet"
    if cache.exists():
        panel = pd.read_parquet(cache)
    else:
        panel = history(with_open=True)   # same panel as pipeline.weekly()
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache)
    log(f"Panel: {len(panel):,} rows, {panel.index.get_level_values('session').nunique()} sessions, "
        f"{len(panel.columns)} columns")
    
    # Define variants to test
    variants = {
        # 1. BASELINE: Current production config
        "baseline_hgb": RankerConfig(),
        
        # 2. HGB with multi-seed ensemble (diversification)
        "hgb_3seed": RankerConfig(seeds=(0, 42, 137)),
        
        # 3. HGB with recency weighting (3-year halflife)
        "hgb_recency": RankerConfig(halflife_years=3.0),
        
        # 4. HGB with both multi-seed + recency
        "hgb_3seed_recency": RankerConfig(seeds=(0, 42, 137), halflife_years=3.0),
        
        # 5. LightGBM Regressor (same objective, faster/different trees)
        "lgbm_reg": RankerConfig(ranker_type="lgbm_reg"),
        
        # 6. LightGBM Regressor with multi-seed
        "lgbm_reg_3seed": RankerConfig(ranker_type="lgbm_reg", seeds=(0, 42, 137)),
        
        # 7. LambdaMART LTR (ranking-native objective, NDCG@8)
        "lgbm_rank": RankerConfig(ranker_type="lgbm_rank"),
        
        # 8. LambdaMART with multi-seed
        "lgbm_rank_3seed": RankerConfig(ranker_type="lgbm_rank", seeds=(0, 42, 137)),
        
        # 9. LambdaMART with recency weighting
        "lgbm_rank_recency": RankerConfig(ranker_type="lgbm_rank", halflife_years=3.0),
        
        # 10. Rank-based targets instead of demeaned
        "hgb_rank_target": RankerConfig(target_kind="rank"),
        
        # 11. LambdaMART with rank targets
        "lgbm_rank_ranktarget": RankerConfig(ranker_type="lgbm_rank", target_kind="rank"),
    }
    
    # Also include the simple gap rule as baseline reference
    log("Computing gap rule baseline...")
    gap_scores = panel["gap1"] if "gap1" in panel.columns else None
    if gap_scores is not None:
        gap_daily = book(panel, gap_scores, k=8, cost_bps=13.0)
        gap_stats = summarize(gap_daily)
        gap_stats["variant"] = "gap_rule"
        gap_stats["elapsed_sec"] = 0
        log(f"  gap_rule: {gap_stats['net_bps_per_trade']:.1f} bps/trade, t={gap_stats.get('t_stat')}")
    else:
        gap_daily = None
        gap_stats = {"variant": "gap_rule", "sessions": 0}
    
    # Run all variants.  Each finished variant's daily series is checkpointed
    # so an interrupted run resumes where it stopped instead of starting over.
    ckpt = ROOT / "data" / "ranker_comparison"
    ckpt.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    results = {}
    all_stats = [gap_stats] if gap_stats else []
    all_daily = {"gap_rule": gap_daily} if gap_daily is not None else {}

    for name, config in variants.items():
        if only and name not in only:
            continue
        path = ckpt / f"{name}.parquet"
        if path.exists():
            daily = pd.read_parquet(path)["net_bps"]
            stats = summarize(daily)
            stats["variant"] = name
            stats["elapsed_sec"] = "cached"
            all_daily[name] = daily
            all_stats.append(stats)
            log(f"  {name}: loaded checkpoint ({stats['net_bps_per_trade']:.1f} bps/trade)")
            continue
        try:
            scores, daily, stats = run_variant(name, panel, config)
            results[name] = scores
            all_daily[name] = daily
            all_stats.append(stats)
            daily.rename("net_bps").to_frame().to_parquet(path)
        except Exception as e:
            log(f"  {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({"variant": name, "error": str(e)})
    
    # Build comparison table
    log("\n" + "=" * 80)
    log("RESULTS COMPARISON")
    log("=" * 80)
    
    df = pd.DataFrame(all_stats)
    cols = ["variant", "net_bps_per_trade", "t_stat", "sharpe", "up_day_pct",
            "max_drawdown_pct", "sessions", "elapsed_sec"]
    display_cols = [c for c in cols if c in df.columns]
    print(df[display_cols].to_string(index=False))
    
    # Paired t-tests against baseline
    baseline_name = "baseline_hgb"
    if baseline_name in all_daily and all_daily[baseline_name] is not None:
        log("\n" + "-" * 80)
        log("PAIRED T-TESTS vs BASELINE (baseline_hgb)")
        log("-" * 80)
        baseline_daily = all_daily[baseline_name]
        for name, daily in all_daily.items():
            if name == baseline_name or daily is None:
                continue
            p = paired(daily, baseline_daily)
            verdict = "BETTER ✓" if p["t_stat"] > 2.0 else (
                "WORSE ✗" if p["t_stat"] < -2.0 else "NOT SIGNIFICANT")
            log(f"  {name:30s} diff={p['mean_diff_bps']:+.1f} bps  t={p['t_stat']:+.2f}  "
                f"{p['a_better_pct']:.0f}% better  [{verdict}]")
    
    # Save results
    output_path = ROOT / "data" / "ranker_comparison.json"
    import json
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "panel_shape": list(panel.shape),
        "results": all_stats,
    }
    # Add paired tests
    if baseline_name in all_daily:
        paired_results = {}
        for name, daily in all_daily.items():
            if name == baseline_name or daily is None:
                continue
            paired_results[name] = paired(daily, all_daily[baseline_name])
        output["paired_vs_baseline"] = paired_results
    
    output_path.write_text(json.dumps(output, indent=2, default=str))
    log(f"\nResults saved to {output_path}")
    
    # Final recommendation
    log("\n" + "=" * 80)
    log("RECOMMENDATION")
    log("=" * 80)
    valid = [s for s in all_stats if "net_bps_per_trade" in s and s.get("sessions", 0) > 100]
    if valid:
        best = max(valid, key=lambda s: s["net_bps_per_trade"])
        log(f"Best variant: {best['variant']} at {best['net_bps_per_trade']:.1f} bps/trade "
            f"(t={best.get('t_stat', 'N/A')}, sharpe={best.get('sharpe', 'N/A')})")
        baseline = next((s for s in valid if s["variant"] == baseline_name), None)
        if baseline:
            improvement = best["net_bps_per_trade"] - baseline["net_bps_per_trade"]
            log(f"Improvement over baseline: {improvement:+.1f} bps/trade")


if __name__ == "__main__":
    main()
