#!/usr/bin/env python3
"""Nightly learning job — retrains the shadow policy against the CURRENT engine.

Runs after market close (systemd timer, weekdays 17:00 IST):
 1. Refreshes the 1m candle cache for the NIFTY-100 universe (last 8 days).
 2. Replays the last N cached trading days through the live signal engine,
    producing candidate events with the exact strategies/exits in the code
    right now.
 3. Replaces the 'historical_backtest' observations in the shadow learner
    with those events and rebuilds the policy + adaptive weights.

This keeps the learner honest after strategy-code changes: live observations
capture what the old code did; this job re-labels history with what the
current code WOULD have done, so the policy veto and weights track the
deployed engine rather than its ancestors.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import warnings
warnings.filterwarnings("ignore")

from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
LOG_PATH = ROOT / "data" / "nightly_learn.log"
LEARNER_DB = ROOT / "data" / "shadow_learner.sqlite3"
REPLAY_DAYS = 5  # trading days replayed through the current engine

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def refresh_cache(symbols: list[str]) -> None:
    from nse_intraday_ai.data import YFinanceProvider

    provider = YFinanceProvider()
    chunk = 25
    fetched = 0
    for i in range(0, len(symbols), chunk):
        try:
            results = provider.batch_history(symbols[i : i + chunk], period="8d", interval="1m")
            fetched += sum(1 for r in results.values() if not r.frame.empty)
        except Exception as exc:
            log.warning("cache refresh chunk %d failed: %s", i // chunk, exc)
        time.sleep(1)
    log.info("cache refresh: %d/%d symbols", fetched, len(symbols))


def replay_days(symbols: list[str]) -> list:
    import pandas as pd

    from nse_intraday_ai.backtest import BacktestConfig, generate_candidate_events
    from nse_intraday_ai.candle_cache import CandleCache
    from nse_intraday_ai.data import DataResult

    cache = CandleCache(ROOT / "data" / "candles.sqlite3")
    since = datetime.now(IST) - timedelta(days=14)
    frames = {}
    for s in symbols:
        df = cache.load(s, "1m", since=since, limit=50000)
        if not df.empty:
            frames[s] = df
    all_days = sorted({d for df in frames.values() for d in set(df.index.normalize())})
    days = all_days[-REPLAY_DAYS:]
    log.info("replaying %d days over %d symbols: %s", len(days), len(frames), [str(d.date()) for d in days])

    config = BacktestConfig()  # current engine defaults (cost 15, breakeven on)

    # Equal-weight market proxy per minute — used to label each event with
    # the index regime so the policy can be conditioned on it.  Coarser than
    # the live daemon's ADX-based ^NSEI regime, but the same UP/DOWN/RANGING
    # vocabulary: return from session open beyond ±15 bps picks the trend.
    proxy = pd.DataFrame({s: df["close"] for s, df in frames.items()}).mean(axis=1)

    def regime_at(ts) -> str | None:
        day_mask = proxy.index.normalize() == ts.normalize()
        day_proxy = proxy[day_mask]
        if day_proxy.empty or ts not in day_proxy.index:
            return None
        ret = day_proxy.at[ts] / day_proxy.iloc[0] - 1
        if ret > 0.0015:
            return "TRENDING_UP"
        if ret < -0.0015:
            return "TRENDING_DOWN"
        return "RANGING"

    events = []
    for day in days:
        data = {}
        for s, df in frames.items():
            day_df = df[df.index.normalize() == day]
            if len(day_df) >= 100:
                data[s] = DataResult(s, day_df, "cache", None)
        if not data:
            continue
        day_events = generate_candidate_events(
            data=data, config=config, risk_per_trade_pct=0.5, max_position_pct=25.0
        )
        events.extend(day_events)
        log.info("%s: %d events", day.date(), len(day_events))
    return events, regime_at


def main() -> None:
    import pandas as pd

    from nse_intraday_ai.learning import AdaptiveWeightEngine, ShadowLearner

    symbols = (
        pd.read_csv(ROOT / "data" / "nifty100_symbols.csv")["Symbol"]
        .dropna().astype(str).str.strip().tolist()
    )
    symbols = [s + ".NS" for s in symbols]

    refresh_cache(symbols)
    events, regime_at = replay_days(symbols)
    if not events:
        log.warning("no events generated — skipping retrain")
        return

    learner = ShadowLearner(LEARNER_DB)
    inserted = learner.record_historical_events(
        events, min_confidence=55.0, source="historical_backtest", replace_source=True,
        index_regime_fn=lambda e: regime_at(e.timestamp),
    )
    stats = learner.stats()
    weights = AdaptiveWeightEngine(learner).weights()
    log.info(
        "retrained: %d events inserted | evaluated=%d win_rate=%.1f%% avg=%.1f bps",
        inserted, stats.evaluated, stats.win_rate, stats.avg_reward_bps,
    )
    log.info("weights: %s", {k: round(v, 2) for k, v in sorted(weights.items())})


if __name__ == "__main__":
    main()
