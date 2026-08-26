from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd

from nse_intraday_ai.data import DataResult
from nse_intraday_ai.models import Side, TradePlan
from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.strategies import EnsembleConfig, VotingSignalEngine


ProviderFactory = Callable[[], object]
QuoteClientFactory = Callable[[], object]


@dataclass(frozen=True)
class ScanResult:
    symbol: str
    plan: TradePlan
    rows: int = 0
    source: str = ""
    warning: str | None = None
    error: str | None = None
    last_high: float | None = None
    last_low: float | None = None
    last_close: float | None = None
    public_ltp: float | None = None
    quote_time: str | None = None
    quote_age_seconds: float | None = None
    quote_warning: str | None = None
    frame: pd.DataFrame | None = None
    regime: str | None = None
    market_adj: float = 0.0   # post-scan breadth + sector confidence adjustment
    predicted_net_bps: float | None = None
    model_rank: int | None = None
    rank_score: float | None = None


def _provider_history(
    provider: object, symbol: str, period: str, interval: str
) -> DataResult:
    history = getattr(provider, "history")
    return history(symbol, period=period, interval=interval)


def _public_quote(quote_client: object, symbol: str):
    quote = getattr(quote_client, "quote")
    return quote(symbol)


def _apply_fresh_quote_to_frame(
    data: DataResult,
    quote_result,
    fresh_quote_max_age_seconds: float,
) -> DataResult:
    if (
        quote_result is not None
        and quote_result.last_price is not None
        and quote_result.age_seconds is not None
        and quote_result.age_seconds <= fresh_quote_max_age_seconds
        and not data.frame.empty
    ):
        data_frame = data.frame.copy()
        last_index = data_frame.index[-1]
        ltp = float(quote_result.last_price)
        data_frame.loc[last_index, "close"] = ltp
        data_frame.loc[last_index, "high"] = max(float(data_frame.loc[last_index, "high"]), ltp)
        data_frame.loc[last_index, "low"] = min(float(data_frame.loc[last_index, "low"]), ltp)
        return DataResult(data.symbol, data_frame, data.source, data.warning)
    return data


def _build_scan_result(
    *,
    data: DataResult,
    quote_result,
    risk_config: RiskConfig,
    ensemble_config: EnsembleConfig,
    strategy_weights: dict[str, float] | None,
    keep_frame_for_actionable: bool,
    market_context=None,
) -> ScanResult:
    engine = VotingSignalEngine(config=ensemble_config)
    from nse_intraday_ai.indicators import add_indicators
    df = add_indicators(data.frame)
    plan = engine.analyze_precomputed(data.symbol, df, risk_config, strategy_weights or {}, market_context)
    if df.empty:
        return ScanResult(
            symbol=data.symbol,
            plan=plan,
            rows=0,
            source=data.source,
            warning=data.warning,
            error=None,
            public_ltp=quote_result.last_price if quote_result else None,
            quote_time=quote_result.last_update_time if quote_result else None,
            quote_age_seconds=quote_result.age_seconds if quote_result else None,
            quote_warning=quote_result.warning if quote_result else None,
        )

    last = df.iloc[-1]
    regime = str(last["regime"]) if "regime" in df.columns else None
    return ScanResult(
        symbol=data.symbol,
        plan=plan,
        rows=len(data.frame),
        source=data.source,
        warning=data.warning,
        error=None,
        last_high=float(last["high"]),
        last_low=float(last["low"]),
        last_close=float(last["close"]),
        public_ltp=quote_result.last_price if quote_result else None,
        quote_time=quote_result.last_update_time if quote_result else None,
        quote_age_seconds=quote_result.age_seconds if quote_result else None,
        quote_warning=quote_result.warning if quote_result else None,
        frame=df if keep_frame_for_actionable and plan.is_actionable else None,
        regime=regime,
    )


def scan_symbol(
    *,
    symbol: str,
    provider_factory: ProviderFactory,
    period: str,
    interval: str,
    risk_config: RiskConfig,
    ensemble_config: EnsembleConfig,
    strategy_weights: dict[str, float] | None = None,
    quote_client_factory: QuoteClientFactory | None = None,
    fresh_quote_max_age_seconds: float = 30.0,
    keep_frame_for_actionable: bool = True,
    market_context=None,
) -> ScanResult:
    try:
        provider = provider_factory()
        data = _provider_history(provider, symbol, period, interval)
        quote_result = None
        if quote_client_factory is not None:
            quote_result = _public_quote(quote_client_factory(), data.symbol)
            data = _apply_fresh_quote_to_frame(data, quote_result, fresh_quote_max_age_seconds)
        return _build_scan_result(
            data=data,
            quote_result=quote_result,
            risk_config=risk_config,
            ensemble_config=ensemble_config,
            strategy_weights=strategy_weights,
            keep_frame_for_actionable=keep_frame_for_actionable,
            market_context=market_context,
        )
    except Exception as exc:  # pragma: no cover - defensive around provider failures
        empty_plan = VotingSignalEngine(config=ensemble_config).analyze(
            symbol, pd.DataFrame(), risk_config, strategy_weights or {}
        )[1]
        return ScanResult(
            symbol=symbol,
            plan=empty_plan,
            rows=0,
            source="scanner",
            error=str(exc),
        )


def iter_scan_universe(
    *,
    symbols: Iterable[str],
    provider_factory: ProviderFactory,
    period: str,
    interval: str,
    risk_config: RiskConfig,
    ensemble_config: EnsembleConfig,
    strategy_weights: dict[str, float] | None = None,
    quote_client_factory: QuoteClientFactory | None = None,
    fresh_quote_max_age_seconds: float = 30.0,
    max_workers: int = 12,
    keep_frame_for_actionable: bool = True,
    market_context=None,
) -> Iterator[ScanResult]:
    symbol_list = list(symbols)
    workers = max(1, min(max_workers, len(symbol_list) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                scan_symbol,
                symbol=symbol,
                provider_factory=provider_factory,
                period=period,
                interval=interval,
                risk_config=risk_config,
                ensemble_config=ensemble_config,
                strategy_weights=strategy_weights,
                quote_client_factory=quote_client_factory,
                fresh_quote_max_age_seconds=fresh_quote_max_age_seconds,
                keep_frame_for_actionable=keep_frame_for_actionable,
                market_context=market_context,
            )
            for symbol in symbol_list
        ]
        for future in as_completed(futures):
            yield future.result()


def scan_universe_batch(
    *,
    symbols: Iterable[str],
    provider_factory: ProviderFactory,
    period: str,
    interval: str,
    risk_config: RiskConfig,
    ensemble_config: EnsembleConfig,
    strategy_weights: dict[str, float] | None = None,
    quote_client_factory: QuoteClientFactory | None = None,
    fresh_quote_max_age_seconds: float = 30.0,
    max_workers: int = 12,
    keep_frame_for_actionable: bool = True,
    market_context=None,
) -> list[ScanResult]:
    import dataclasses
    symbol_list = list(symbols)
    provider = provider_factory()
    batch_history = getattr(provider, "batch_history", None)
    if batch_history is None:
        results = list(
            iter_scan_universe(
                symbols=symbol_list,
                provider_factory=provider_factory,
                period=period,
                interval=interval,
                risk_config=risk_config,
                ensemble_config=ensemble_config,
                strategy_weights=strategy_weights,
                quote_client_factory=quote_client_factory,
                fresh_quote_max_age_seconds=fresh_quote_max_age_seconds,
                max_workers=max_workers,
                keep_frame_for_actionable=keep_frame_for_actionable,
                market_context=market_context,
            )
        )
    else:
        data_by_symbol: dict[str, DataResult] = batch_history(symbol_list, period=period, interval=interval)
        # Live VWAP breadth: fraction of the universe above session VWAP,
        # computed from the batch frames BEFORE signals are evaluated so the
        # engine sees same-cycle breadth (no one-scan lag).
        if market_context is not None and data_by_symbol:
            from nse_intraday_ai.indicators import vwap as _vwap
            above = total = 0
            for _data in data_by_symbol.values():
                frame = _data.frame
                if frame.empty or len(frame) < 5 or "close" not in frame.columns:
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
        quote_by_symbol = {}
        if quote_client_factory is not None and data_by_symbol:
            quote_client = quote_client_factory()
            batch_quotes = getattr(quote_client, "index_quotes", None)
            if batch_quotes is not None:
                quote_by_symbol = batch_quotes(list(data_by_symbol))
            else:
                workers = max(1, min(max_workers, len(data_by_symbol)))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_public_quote, quote_client_factory(), symbol): symbol
                        for symbol in data_by_symbol
                    }
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            quote_by_symbol[symbol] = future.result()
                        except Exception:  # pragma: no cover
                            quote_by_symbol[symbol] = None

        results = []
        for symbol, data in data_by_symbol.items():
            quote_result = quote_by_symbol.get(symbol)
            data = _apply_fresh_quote_to_frame(data, quote_result, fresh_quote_max_age_seconds)
            results.append(
                _build_scan_result(
                    data=data,
                    quote_result=quote_result,
                    risk_config=risk_config,
                    ensemble_config=ensemble_config,
                    strategy_weights=strategy_weights,
                    keep_frame_for_actionable=keep_frame_for_actionable,
                    market_context=market_context,
                )
            )

    # Post-scan: apply sector clustering + breadth adjustments stored as market_adj
    if market_context is not None:
        from nse_intraday_ai.market_context import apply_postscan_context
        apply_postscan_context(market_context, results)
        enriched = []
        for r in results:
            adj = market_context.total_postscan_adj(r.symbol, r.plan.side.value)
            enriched.append(dataclasses.replace(r, market_adj=adj) if adj != 0.0 else r)
        return enriched

    return results


def _effective_confidence(result: ScanResult) -> float:
    return result.plan.confidence + result.market_adj


def rank_actionable(
    results: Iterable[ScanResult], *, max_quote_age_seconds: float | None = None
) -> list[ScanResult]:
    actionable = [result for result in results if result.plan.is_actionable]
    if max_quote_age_seconds is not None:
        actionable = [
            result
            for result in actionable
            if result.quote_age_seconds is not None
            and result.quote_age_seconds <= max_quote_age_seconds
        ]
    return sorted(
        actionable,
        key=lambda result: (
            _effective_confidence(result),
            result.plan.reward_risk,
            result.plan.reward_amount,
        ),
        reverse=True,
    )


def leading_trade_side(result: ScanResult) -> Side:
    scores: dict[Side, float] = {Side.LONG: 0.0, Side.SHORT: 0.0}
    for vote in result.plan.strategy_votes:
        if vote.side in scores and vote.entry is not None:
            scores[vote.side] += vote.weight * vote.confidence
    side, score = max(scores.items(), key=lambda item: item[1])
    return side if score > 0 else Side.WAIT


def rank_near_misses(
    results: Iterable[ScanResult],
    *,
    min_confidence: float = 55.0,
    max_quote_age_seconds: float | None = None,
) -> list[ScanResult]:
    candidates: list[ScanResult] = []
    for result in results:
        if result.plan.is_actionable:
            continue
        if result.plan.confidence < min_confidence:
            continue
        if leading_trade_side(result) == Side.WAIT:
            continue
        if (
            max_quote_age_seconds is not None
            and result.quote_age_seconds is not None
            and result.quote_age_seconds > max_quote_age_seconds
        ):
            continue
        candidates.append(result)

    return sorted(
        candidates,
        key=lambda result: (
            result.plan.confidence,
            sum(1 for vote in result.plan.strategy_votes if vote.side == leading_trade_side(result)),
        ),
        reverse=True,
    )
