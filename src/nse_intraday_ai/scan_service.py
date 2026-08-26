"""Unified scan pipeline — the ONE code path behind both the Streamlit app
and the scanner daemon.

Before 2026-07-07 the app and daemon wired their own pipelines around
scan_universe_batch and drifted (different gates, different vetoes, a shared
flat config file they clobbered for each other).  This module owns the whole
cycle; callers only render/notify.

Sampling model ("don't miss anything"):
- Signals are evaluated on CLOSED bars only (the timeframe the July-2026
  studies validated).  A forming candle is never scored.
- A per-universe bar cursor (data/scan_state.json) tracks the last evaluated
  bar.  Each cycle evaluates every newly closed bar since the cursor — if a
  cycle is late, the missed bars are still evaluated, recorded to the shadow
  learner and signal log, and surfaced as `stale` (too old to enter, never
  silently dropped).
- Each cycle fetches only ~1 day per symbol in one bulk request (the candle
  cache supplies the rest of the lookback), so a 500-symbol cycle stays fast
  enough to run every minute.
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import nse_intraday_ai.scan_config as scan_config
from nse_intraday_ai.data import (
    DEFAULT_COMMODITY_SYMBOLS,
    YFinanceProvider,
    load_nifty500_symbols,
)
from nse_intraday_ai.indicators import add_indicators, vwap as _vwap
from nse_intraday_ai.learning import AdaptiveWeightEngine, PolicyEstimate, ShadowLearner
from nse_intraday_ai.meta_model import features_from_plan, load_if_available
from nse_intraday_ai.models import TradePlan
from nse_intraday_ai.scanner import ScanResult, rank_actionable, rank_near_misses
from nse_intraday_ai.signal_model import score_and_rank_scan_results
from nse_intraday_ai.signals_log import record as _log_signal
from nse_intraday_ai.strategies import VotingSignalEngine

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "scan_state.json"
LEARNER_DB = ROOT / "data" / "shadow_learner.sqlite3"
NIFTY500_CACHE = ROOT / "data" / "nifty500_symbols.csv"

_INTERVAL_MINUTES = {"1m": 1, "2m": 2, "5m": 5, "15m": 15}
# A signal older than this many bars is still logged/learned but flagged
# stale — too old to act on as a fresh entry.
_STALE_AFTER_BARS = 2


@dataclass
class ScanCycle:
    universe: str
    results: list[ScanResult] = field(default_factory=list)
    recommendations: list[ScanResult] = field(default_factory=list)   # final, ranked
    estimates: dict[int, PolicyEstimate] = field(default_factory=dict)  # id(result) -> estimate
    near_misses: list[ScanResult] = field(default_factory=list)
    policy_blocked: list[ScanResult] = field(default_factory=list)
    unproven: list[ScanResult] = field(default_factory=list)
    meta_vetoed: list[ScanResult] = field(default_factory=list)
    # Signals dropped by the Aug-2026 gates, kept separately so the UI can show
    # *why* a candidate was refused rather than just hiding it.
    quality_vetoed: list[tuple[ScanResult, str]] = field(default_factory=list)
    news_vetoed: list[tuple[ScanResult, str]] = field(default_factory=list)
    news_size_multiplier: float = 1.0
    # (bar_ts, plan) for actionable signals found on catch-up bars — evaluated
    # and recorded, but too old to enter now.
    stale_signals: list[tuple[pd.Timestamp, TradePlan]] = field(default_factory=list)
    # Fresh LTP overlay for display, keyed by symbol (never mutates results).
    quotes: dict[str, object] = field(default_factory=dict)
    latest_prices: dict[str, float] = field(default_factory=dict)
    market_context: object | None = None
    evaluated_bars: int = 0
    inserted: int = 0
    evaluated_pending: int = 0
    cycle_seconds: float = 0.0
    symbols_with_data: int = 0
    symbols_total: int = 0
    fetched: bool = False   # did this cycle hit the network (vs cache-only)?

    @property
    def summary(self) -> str:
        return (
            f"{self.universe}: {self.symbols_with_data}/{self.symbols_total} symbols, "
            f"{self.evaluated_bars} new bars, {len(self.recommendations)} actionable "
            f"({len(self.policy_blocked)} policy-vetoed, {len(self.meta_vetoed)} meta-vetoed, "
            f"{len(self.quality_vetoed)} quality-vetoed, {len(self.news_vetoed)} news-vetoed, "
            f"{len(self.unproven)} unproven, {len(self.stale_signals)} stale), "
            f"{self.inserted} learned, {'fetch' if self.fetched else 'cache'} {self.cycle_seconds:.0f}s"
        )


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def universe_symbols(universe: str) -> list[str]:
    if universe == "commodity":
        return list(DEFAULT_COMMODITY_SYMBOLS)
    return load_nifty500_symbols(NIFTY500_CACHE).symbols


def _closed_bars(frame: pd.DataFrame, interval: str, now: pd.Timestamp) -> pd.DataFrame:
    """Bars whose window has fully closed (bar label = window start)."""
    width = pd.Timedelta(minutes=_INTERVAL_MINUTES.get(interval, 5))
    return frame[frame.index + width <= now]


def _stitched_frames(
    provider,
    symbols: list[str],
    *,
    interval: str,
    period: str,
    do_fetch: bool,
) -> dict[str, pd.DataFrame]:
    """Assemble per-symbol frames for evaluation.

    When ``do_fetch`` is set we pull ~1 day of candles from the network in one
    bulk request (retry_missing off — flaky symbols are filled from cache /
    the next tick, so a single slow symbol can't stall the cycle) and let the
    provider cache them; otherwise we skip the network entirely and serve from
    the local candle cache (cheap).  The bulk lookback (`period`) still comes
    from cache so strategies see full history either way."""
    fetched: dict = {}
    if do_fetch:
        try:
            fetched = provider.batch_history(
                symbols, period="1d", interval=interval, retry_missing=False
            )
        except TypeError:
            fetched = provider.batch_history(symbols, period="1d", interval=interval)
    cache = getattr(provider, "_cache", None)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = pd.DataFrame()
        if cache is not None:
            try:
                frame = cache.load_period(symbol, interval, period)
            except Exception:
                frame = pd.DataFrame()
        if frame.empty:
            result = fetched.get(symbol)
            if result is not None and not result.frame.empty:
                frame = result.frame
                if frame.index.tz is None:
                    frame = frame.tz_localize(IST)
        if not frame.empty:
            frames[symbol] = frame
    return frames


def _latest_closed_bar_label(now: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Label of the most recently *closed* bar as of ``now`` (bar T closes at
    T + width, so the newest closed label is floor(now) - width)."""
    width = pd.Timedelta(minutes=_INTERVAL_MINUTES.get(interval, 5))
    return now.floor(f"{_INTERVAL_MINUTES.get(interval, 5)}min") - width


def _quote_overlay(quote_client_factory, symbols: list[str]) -> dict:
    """Fresh LTPs for a SMALL display set (recommendations + near misses)."""
    if quote_client_factory is None or not symbols:
        return {}
    try:
        client = quote_client_factory()
        batch = getattr(client, "index_quotes", None)
        if batch is not None:
            return batch(symbols)
        return {s: client.quote(s) for s in symbols}
    except Exception:
        return {}


def run_scan_cycle(
    universe: str = "nse",
    *,
    source: str = "daemon",
    provider_factory=YFinanceProvider,
    quote_client_factory=None,
    cfg: dict | None = None,
    strategy_weights: dict[str, float] | None = None,
    require_policy_approval: bool | None = None,
    policy_min_samples: int = 20,
    record_shadow: bool = True,
    log_signals: bool = True,
    state_path: Path | str | None = None,
    now: pd.Timestamp | None = None,
    market_context=None,
    symbols: list[str] | None = None,
    fetch: bool = True,
    force_fetch: bool = False,
) -> ScanCycle:
    """Run one unified scan cycle for a universe. Returns everything the app
    renders and the daemon notifies on.

    Sampling economics: signals only change when a bar closes, so the
    expensive network fetch is gated on a new bar actually being due since the
    cursor.  On a 1-minute timer / 5-minute timeframe that means ~4 of every 5
    ticks are cheap cache reads; the tick after a bar closes does the fetch and
    catches it within ~a minute.  Pass force_fetch=True to always hit the
    network (e.g. a manual "Scan now")."""
    started = _time.monotonic()
    now = now or pd.Timestamp.now(tz=IST)
    cfg = cfg or scan_config.load(universe)
    risk = scan_config.to_risk_config(cfg)
    ensemble = scan_config.to_ensemble_config(cfg)
    interval = str(cfg.get("interval", "5m"))
    period = str(cfg.get("period", "5d"))
    if require_policy_approval is None:
        require_policy_approval = bool(cfg.get("require_policy_approval", True))

    cycle = ScanCycle(universe=universe)
    if symbols is None:
        symbols = universe_symbols(universe)
    cycle.symbols_total = len(symbols)

    # ── Cursor + new-bar gate (read BEFORE any fetch) ───────────────────────
    state_file = Path(state_path) if state_path else STATE_PATH
    state = _load_state(state_file)
    cursor_key = f"{universe}:{interval}"
    last_bar = pd.Timestamp(state[cursor_key]) if cursor_key in state else None
    bar_width = pd.Timedelta(minutes=_INTERVAL_MINUTES.get(interval, 5))
    stale_cutoff = now - _STALE_AFTER_BARS * bar_width
    latest_due = _latest_closed_bar_label(now, interval)
    new_bar_due = last_bar is None or latest_due > last_bar
    do_fetch = fetch and (new_bar_due or force_fetch)
    cycle.fetched = do_fetch

    learner = ShadowLearner(LEARNER_DB) if record_shadow else None
    if strategy_weights is None and learner is not None:
        strategy_weights = AdaptiveWeightEngine(learner).weights()
    strategy_weights = strategy_weights or {}

    if market_context is None:
        try:
            from nse_intraday_ai.market_context import fetch_index_vix_context
            market_context = fetch_index_vix_context(for_commodities=universe == "commodity")
        except Exception:
            market_context = None
    cycle.market_context = market_context

    provider = provider_factory()
    frames = _stitched_frames(provider, symbols, interval=interval, period=period, do_fetch=do_fetch)
    # Cold cache: if we skipped the fetch but have nothing to show, fetch once.
    if not frames and fetch and not do_fetch:
        cycle.fetched = True
        frames = _stitched_frames(provider, symbols, interval=interval, period=period, do_fetch=True)
    cycle.symbols_with_data = len(frames)
    if not frames:
        cycle.cycle_seconds = _time.monotonic() - started
        return cycle

    frames = {s: add_indicators(f) for s, f in frames.items()}
    frames = {s: f for s, f in frames.items() if not f.empty}

    # Live VWAP breadth from the same frames, before any signal is evaluated.
    if market_context is not None:
        above = total = 0
        for frame in frames.values():
            if len(frame) < 5:
                continue
            try:
                last_vwap = float(_vwap(frame).iloc[-1])
                last_close = float(frame["close"].iloc[-1])
            except Exception:
                continue
            total += 1
            above += 1 if last_close > last_vwap else 0
        if total >= 10:
            market_context.vwap_breadth_pct = above / total

    # ── Bar cursor: evaluate every newly closed bar exactly once ───────────
    # (cursor + gate already read at the top, before the fetch decision)
    engine = VotingSignalEngine(config=ensemble)
    # Minimal ScanResults for every newly evaluated bar (fresh AND catch-up)
    # so the shadow learner records each bar exactly once — no duplicates at
    # fast cycles, no gaps at slow ones.
    new_bar_records: list[ScanResult] = []
    max_bar_seen = last_bar

    for symbol, frame in frames.items():
        closed = _closed_bars(frame, interval, now)
        if closed.empty:
            continue
        latest_closed_ts = closed.index[-1]
        if max_bar_seen is None or latest_closed_ts > max_bar_seen:
            max_bar_seen = latest_closed_ts
        new_ts = closed.index if last_bar is None else closed.index[closed.index > last_bar]
        # First run has no cursor: evaluate only the latest bar, not history.
        if last_bar is None:
            new_ts = closed.index[-1:]

        latest_plan: TradePlan | None = None
        for ts in new_ts:
            pos = closed.index.get_loc(ts)
            history = closed.iloc[: pos + 1]
            if len(history) < 15:
                continue
            plan = engine.analyze_precomputed(symbol, history, risk, strategy_weights, market_context)
            cycle.evaluated_bars += 1
            if ts == latest_closed_ts:
                latest_plan = plan
            if plan.is_actionable and ts < stale_cutoff:
                cycle.stale_signals.append((ts, plan))
            row = history.iloc[-1]
            new_bar_records.append(
                ScanResult(
                    symbol=symbol,
                    plan=plan,
                    rows=len(history),
                    source=getattr(provider, "name", "unknown"),
                    last_high=float(row["high"]),
                    last_low=float(row["low"]),
                    last_close=float(row["close"]),
                    quote_age_seconds=max(0.0, (now - (ts + bar_width)).total_seconds()),
                    regime=str(row["regime"]) if "regime" in history.columns else None,
                )
            )

        # Display state must reflect the latest closed bar even when the
        # cursor already covered it (e.g. two cycles inside one 5m bar).
        if latest_plan is None:
            if len(closed) >= 15:
                latest_plan = engine.analyze_precomputed(
                    symbol, closed, risk, strategy_weights, market_context
                )
            else:
                continue

        last_row = closed.iloc[-1]
        signal_age = (now - latest_closed_ts).total_seconds()
        cycle.latest_prices[symbol] = float(last_row["close"])
        cycle.results.append(
            ScanResult(
                symbol=symbol,
                plan=latest_plan,
                rows=len(closed),
                source=getattr(provider, "name", "unknown"),
                warning=None,
                error=None,
                last_high=float(last_row["high"]),
                last_low=float(last_row["low"]),
                last_close=float(last_row["close"]),
                quote_age_seconds=signal_age,
                frame=closed if latest_plan.is_actionable else None,
                regime=str(last_row["regime"]) if "regime" in closed.columns else None,
            )
        )

    # Sector clustering + breadth adjustments (post-scan context)
    if market_context is not None and cycle.results:
        import dataclasses
        from nse_intraday_ai.market_context import apply_postscan_context
        apply_postscan_context(market_context, cycle.results)
        cycle.results = [
            dataclasses.replace(r, market_adj=adj) if (adj := market_context.total_postscan_adj(r.symbol, r.plan.side.value)) != 0.0 else r
            for r in cycle.results
        ]

    # ── Shared gate chain (identical for app and daemon) ────────────────────
    recommendations = rank_actionable(cycle.results)
    cycle.near_misses = rank_near_misses(cycle.results, min_confidence=55)

    index_regime = getattr(market_context, "index_regime", None)
    if learner is not None:
        kept: list[tuple[ScanResult, PolicyEstimate | None]] = []
        for result in recommendations:
            est = learner.estimate_for_result(
                result, min_samples=policy_min_samples, index_regime=index_regime
            )
            if require_policy_approval and not est.is_trained:
                cycle.unproven.append(result)
            elif require_policy_approval and est.avg_reward_bps <= 0:
                cycle.policy_blocked.append(result)
            else:
                kept.append((result, est))
    else:
        kept = [(result, None) for result in recommendations]

    # ── Entry-quality + macro-alignment gate (Aug-2026 study) ───────────────
    # These replace confidence as the selection criterion for NSE equities.
    # Measured over 280K candidate signals / 49 sessions: confidence correlated
    # +0.009 with forward return and the shipped conf>=70 gate selected
    # *below*-average signals out of sample, while volume expansion and impulse
    # size were the only entry features positive in both halves of the window.
    if universe != "commodity" and (cfg.get("entry_quality_gate") or cfg.get("macro_gate")):
        from nse_intraday_ai.entry_quality import (
            GateConfig, compute_entry_quality, macro_alignment, passes_entry_gate,
        )

        gate_cfg = GateConfig(
            require_conviction=bool(cfg.get("entry_quality_gate", True)),
            min_macro_score=float(cfg.get("min_macro_score", 0.0)),
            require_macro_coverage=bool(cfg.get("macro_gate", True)),
        )
        surviving = []
        for result, est in kept:
            quality = compute_entry_quality(result.frame, result.plan.side.value)
            macro = macro_alignment(
                result.plan.side.value,
                nifty_change_pct=getattr(market_context, "nifty_change_pct", None),
                usdinr_change_pct=getattr(market_context, "usdinr_change_pct", None),
                crude_change_pct=getattr(market_context, "crude_change_pct", None),
            ) if cfg.get("macro_gate", True) else None
            verdict = passes_entry_gate(quality, macro, config=gate_cfg)
            if verdict.allow:
                surviving.append((result, est))
            else:
                cycle.quality_vetoed.append((result, verdict.reason))
        kept = surviving

    # ── News risk gate ──────────────────────────────────────────────────────
    # Deliberately an abstention mechanism, not a direction signal: a free RSS
    # headline arrives after the move, but it still tells us the ATR-derived
    # stop no longer describes the next hour.  See news_feed.py.
    if cfg.get("news_gate") and kept:
        try:
            from nse_intraday_ai.news_feed import refresh_news_context

            news = refresh_news_context([r.symbol for r, _ in kept][:12])
            surviving = []
            for result, est in kept:
                allow, size_multiplier, reason = news.entry_gate(result.symbol)
                if allow:
                    surviving.append((result, est))
                    cycle.news_size_multiplier = min(cycle.news_size_multiplier, size_multiplier)
                else:
                    cycle.news_vetoed.append((result, reason or "news veto"))
            kept = surviving
        except Exception as exc:   # a news outage must never block a scan
            cycle.news_vetoed.append(
                (kept[0][0], f"news feed unavailable ({type(exc).__name__})")
            ) if kept else None

    meta = load_if_available(universe) if cfg.get("meta_veto", True) else None
    if meta is not None:
        frac = float(cfg.get("meta_veto_fraction", 0.5))
        surviving = []
        for result, est in kept:
            feats = features_from_plan(result.plan)
            if feats is None or meta.passes(feats, frac):
                surviving.append((result, est))
            else:
                cycle.meta_vetoed.append(result)
        kept = surviving

    import dataclasses

    # Score surviving candidates with the Random Forest signal model (or composite fallback)
    raw_candidates = [r for r, _ in kept]
    scored_candidates = score_and_rank_scan_results(raw_candidates, market_context=cycle.market_context)
    scored_map = {r.symbol: r for r in scored_candidates}

    kept = [
        (scored_map.get(r.symbol, r), est)
        for r, est in kept
    ]

    # Rank by model predicted score (predicted_net_bps / composite rank_score)
    kept.sort(
        key=lambda pair: (
            (pair[0].rank_score if pair[0].rank_score is not None else (pair[0].plan.confidence / 10.0))
            + (pair[1].confidence_bonus if pair[1] is not None else 0.0)
        ),
        reverse=True,
    )

    # Cross-sectional correlation diversification filter (prevents collinear cluster concentration)
    from nse_intraday_ai.correlation import return_matrix, rolling_corr, pick_diversified

    candidate_frames = {r.symbol: r.frame for r, _ in kept if r.frame is not None and not r.frame.empty}
    if len(candidate_frames) >= 2:
        try:
            ret_mat = return_matrix(candidate_frames)
            corr_mat = rolling_corr(ret_mat, window=60, min_periods=15)
            if not corr_mat.empty:
                ranked_df = pd.DataFrame([
                    {"symbol": r.symbol, "_score": r.rank_score if r.rank_score is not None else (r.plan.confidence / 10.0)}
                    for r, _ in kept
                ])
                picks = pick_diversified(ranked_df, corr_mat, n=len(kept), max_corr=0.70)
                picked_syms = {p.symbol for p in picks}
                diverse_kept = [pair for pair in kept if pair[0].symbol in picked_syms]
                demoted_kept = [pair for pair in kept if pair[0].symbol not in picked_syms]
                kept = diverse_kept + demoted_kept
        except Exception:
            pass

    final_recos = []
    for rank_idx, (r, _) in enumerate(kept, start=1):
        final_recos.append(dataclasses.replace(r, model_rank=rank_idx))

    cycle.recommendations = final_recos
    cycle.estimates = {id(r): e for (r, _), e in zip(kept, [e for _, e in kept]) if e is not None}

    # Optional fresh-LTP overlay for the small display set only — kept in a
    # parallel dict so result identity (and the estimates id-map) is stable.
    if quote_client_factory is not None:
        display = [r.symbol for r in cycle.recommendations + cycle.near_misses[:10]]
        cycle.quotes = {
            symbol: quote
            for symbol, quote in _quote_overlay(quote_client_factory, display).items()
            if quote is not None and getattr(quote, "last_price", None) is not None
        }

    # ── Learning + logging (every new bar, fresh or stale — nothing missed) ─
    if learner is not None:
        cycle.evaluated_pending = learner.evaluate_pending(cycle.latest_prices)
        if new_bar_records:
            # enforce_session guards against recording stale NSE candles after
            # close; 24-hour commodity futures trade (and cluster their edge)
            # in the IST evening, so the guard must not apply there.
            cycle.inserted = learner.record_scan_results(
                new_bar_records,
                min_confidence=65,
                enforce_session=universe != "commodity",
                index_regime=index_regime,
            )
        if cycle.inserted or cycle.evaluated_pending:
            learner.refresh_policy()

    def _log(result_symbol, plan, regime, log_source):
        _log_signal(
            symbol=result_symbol,
            side=plan.side.value,
            confidence=plan.confidence,
            entry=plan.entry or 0.0,
            stop_loss=plan.stop_loss or 0.0,
            target=plan.target or 0.0,
            reward_risk=plan.reward_risk,
            regime=regime,
            strategies=", ".join(
                v.strategy for v in plan.strategy_votes if v.side == plan.side and v.is_trade
            ),
            source=log_source,
        )

    if log_signals:
        for result in cycle.recommendations:
            _log(result.symbol, result.plan, result.regime, source)
        for bar_ts, plan in cycle.stale_signals:
            _log(plan.symbol, plan, None, f"{source}-late@{bar_ts.strftime('%H:%M')}")

    if max_bar_seen is not None:
        state[cursor_key] = max_bar_seen.isoformat()
        _save_state(state, state_file)

    cycle.cycle_seconds = _time.monotonic() - started
    return cycle
