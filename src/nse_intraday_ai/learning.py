from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nse_intraday_ai.models import Side
from nse_intraday_ai.scanner import ScanResult, leading_trade_side

IST = ZoneInfo("Asia/Kolkata")

# Estimated roundtrip cost (flat-fee broker charges + slippage) in basis
# points.  Outcomes are labelled WIN only when the move beats this — the old
# gross ">0 bps" labelling taught the learner that trades losing money after
# charges were wins.
#
# Per-universe since 2026-08-17.  A single 18 bps constant was being charged to
# commodity futures whose round trip is ~8 bps (5 bps fees + 3 bps slippage,
# `scan_config.DEFAULTS_BY_UNIVERSE`), so every futures arm was handicapped by
# ~10 bps against a measured gross edge of ~5 bps.  Measured effect on the
# 3,365 evaluated commodity observations: the shared 18 bps table blocked 94.6%
# of them, and that blocked set had actually returned +4.98 bps.
ASSUMED_COST_BPS = 18.0                      # NSE equities (legacy default)
COMMODITY_COST_BPS = 8.0


def is_commodity(symbol: str) -> bool:
    """Yahoo futures roots carry '=' (GC=F, CL=F); NSE equities never do."""
    return "=" in symbol


def cost_bps_for(symbol: str) -> float:
    return COMMODITY_COST_BPS if is_commodity(symbol) else ASSUMED_COST_BPS


# Same rule expressed for SQL, so aggregate queries charge each row the cost of
# its own universe instead of a single blended constant.
_COST_SQL = f"(CASE WHEN symbol LIKE '%=%' THEN {COMMODITY_COST_BPS} ELSE {ASSUMED_COST_BPS} END)"


def now_ist() -> datetime:
    return datetime.now(IST)


def in_market_hours(dt: datetime) -> bool:
    """True if dt (IST-aware or naive-IST) falls inside the NSE session."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(IST)
    if dt.weekday() >= 5:
        return False
    m = dt.hour * 60 + dt.minute
    return 555 <= m <= 930  # 09:15–15:30


@dataclass(frozen=True)
class ShadowStats:
    pending: int
    evaluated: int
    win_rate: float
    avg_reward_bps: float


@dataclass(frozen=True)
class PolicyEstimate:
    state_key: str
    samples: int
    win_rate: float
    avg_reward_bps: float
    ucb_score: float
    confidence_bonus: float
    is_trained: bool


KNOWN_STRATEGIES = (
    "trend_continuation",
    "vwap_mean_reversion",
    "opening_range_breakout",
    "volatility_compression_breakout",
    "ema_scalp",
    "vwap_bounce_scalp",
    "momentum_burst_scalp",
    "supertrend",
    "bb_kc_squeeze",
    "rsi_divergence",
    "fair_value_gap",
    "liquidity_sweep_reversal",
    "volume_profile_poc",
    "opening_drive_momentum",
    "vwap_band_mean_reversion",
    "opening_range_expansion",
    "mtf_trend_alignment",
)


class ShadowLearner:
    """Delayed-reward shadow learner for candidate outcomes.

    This is intentionally not allowed to place trades. It collects state/action/reward
    samples so an RL or contextual-bandit policy can be trained after enough evidence.
    """

    def __init__(self, db_path: Path | str, candle_cache=None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if candle_cache is None:
            try:
                from nse_intraday_ai.candle_cache import CandleCache
                candle_cache = CandleCache(self.db_path.parent / "candles.sqlite3")
            except Exception:
                candle_cache = None
        self._candle_cache = candle_cache
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reward_risk REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    quote_age_seconds REAL,
                    vote_count INTEGER NOT NULL,
                    state_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evaluated_at TEXT,
                    future_price REAL,
                    reward_bps REAL,
                    outcome TEXT,
                    reasons_json TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'live',
                    external_id TEXT
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(shadow_observations)").fetchall()
            }
            if "source" not in columns:
                conn.execute(
                    "ALTER TABLE shadow_observations "
                    "ADD COLUMN source TEXT NOT NULL DEFAULT 'live'"
                )
            if "external_id" not in columns:
                conn.execute("ALTER TABLE shadow_observations ADD COLUMN external_id TEXT")
            if "stop_loss" not in columns:
                conn.execute("ALTER TABLE shadow_observations ADD COLUMN stop_loss REAL")
            if "target" not in columns:
                conn.execute("ALTER TABLE shadow_observations ADD COLUMN target REAL")
            if "agreeing_strategies_json" not in columns:
                conn.execute(
                    "ALTER TABLE shadow_observations ADD COLUMN agreeing_strategies_json TEXT"
                )
            if "regime" not in columns:
                # Per-symbol market regime at observation time — collected so
                # future analysis can condition edges on regime (e.g. the
                # laggard-short hypothesis from the 2026-07 study).
                conn.execute("ALTER TABLE shadow_observations ADD COLUMN regime TEXT")
            if "index_regime" not in columns:
                # NIFTY-wide regime at observation time.  2026-07-02 showed
                # why this matters: SHORT states learned on a bearish week
                # stayed "net-positive" and fired into a bullish tape.
                conn.execute("ALTER TABLE shadow_observations ADD COLUMN index_regime TEXT")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shadow_pending
                ON shadow_observations(status, observed_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shadow_state_regime
                ON shadow_observations(state_key, index_regime)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_shadow_external_id
                ON shadow_observations(external_id)
                WHERE external_id IS NOT NULL
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_policy (
                    state_key TEXT PRIMARY KEY,
                    samples INTEGER NOT NULL,
                    wins INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    avg_reward_bps REAL NOT NULL,
                    ucb_score REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _tod_bucket(ts: datetime | None = None, *, symbol: str | None = None) -> str:
        """Bucket an IST timestamp into named session windows for the state key.

        The timestamp of the *observation* must be used — the old version
        always read the wall clock, so historical backfills were all filed
        under whatever bucket the training run happened to execute in.

        Equities use the NSE session; 24-hour futures use the global futures
        clock instead.  Applying the NSE windows to a future filed everything
        from 14:00 IST to 09:59 the next morning — 20 of 24 hours, including
        the whole US overlap that `strategies._commodity_time_multiplier`
        identifies as the liquid stretch — into a single "close" bucket shared
        with Indian afternoon equity data.
        """
        ts = ts or datetime.now(IST)
        if ts.tzinfo is not None:
            ts = ts.astimezone(IST)
        m = ts.hour * 60 + ts.minute
        if symbol and is_commodity(symbol):
            if m < 300:   return "us_late"   # 00:00–05:00 post-US settle
            if m < 540:   return "globex"    # 05:00–09:00 thin Asian Globex
            if m < 840:   return "asia_eu"   # 09:00–14:00 Asia into EU open
            if m < 1140:  return "eu_us"     # 14:00–19:00 EU / US pre-open
            return "us_rth"                  # 19:00–24:00 US regular hours
        if m < 600:   return "open"    # 9:15–10:00 gap fill / ORB window
        if m < 720:   return "mid"     # 10:00–12:00 prime trend window
        if m < 840:   return "lunch"   # 12:00–14:00 low volume
        return "close"                 # 14:00–15:30 afternoon / late

    def _state_key_from_features(
        self,
        *,
        side: Side,
        confidence: float,
        reward_risk: float,
        vote_count: int,
        ts: datetime | None = None,
        symbol: str | None = None,
    ) -> str:
        # Coarse buckets on purpose: the old 5-pt confidence x 0.1 RR x raw
        # vote-count grid produced 2,784 states of which only 190 ever
        # reached 20 samples — the policy was untrained almost everywhere.
        confidence_bucket = min(int(confidence // 10 * 10), 90)
        rr_bucket = min(int(max(reward_risk, 0) * 2) / 2, 3.0)
        votes = min(int(vote_count), 3)
        tod = self._tod_bucket(ts, symbol=symbol)
        # Universe prefix (2026-08-17).  Without it both scanners wrote to the
        # same arms and 121,023 NSE observations averaging -1.8 bps set the
        # verdict for 3,365 commodity observations averaging +4.7 bps — a 36:1
        # pooling that vetoed 94.6% of commodity signals.  Equity keys keep
        # their historical shape so the existing table stays valid.
        prefix = "C|" if (symbol and is_commodity(symbol)) else ""
        return f"{prefix}{side.value}|conf{confidence_bucket}|rr{rr_bucket}|votes{votes}|{tod}"

    def _state_key(self, result: ScanResult, side: Side) -> str:
        votes = sum(1 for vote in result.plan.strategy_votes if vote.side == side)
        return self._state_key_from_features(
            side=side,
            confidence=result.plan.confidence,
            reward_risk=result.plan.reward_risk,
            vote_count=votes,
            symbol=result.symbol,
        )

    def record_historical_events(
        self,
        events,
        *,
        min_confidence: float = 55.0,
        source: str = "historical_backtest",
        replace_source: bool = True,
        index_regime_fn=None,
    ) -> int:
        """Train the policy table from already-known historical candidate outcomes.

        index_regime_fn: optional callable(event) -> str | None giving the
        market-wide regime at the event's timestamp (stored for
        regime-conditional policy estimates).
        """
        inserted = 0
        with self._connect() as conn:
            if replace_source:
                conn.execute("DELETE FROM shadow_observations WHERE source = ?", (source,))

            for event in events:
                if event.confidence < min_confidence or event.entry <= 0:
                    continue
                reward_bps = event.per_share_pnl / event.entry * 10_000
                state_key = self._state_key_from_features(
                    side=event.side,
                    confidence=event.confidence,
                    reward_risk=event.reward_risk,
                    vote_count=event.agreeing_count,
                    ts=event.timestamp.to_pydatetime() if hasattr(event.timestamp, "to_pydatetime") else event.timestamp,
                    symbol=event.symbol,
                )
                event_cost = cost_bps_for(event.symbol)
                external_id = (
                    f"{source}|{event.timestamp.isoformat()}|{event.symbol}|"
                    f"{event.side.value}|{event.entry:.4f}|{event.exit_time.isoformat()}"
                )
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO shadow_observations (
                        observed_at, symbol, side, confidence, reward_risk, entry_price,
                        quote_age_seconds, vote_count, state_key, status, evaluated_at,
                        future_price, reward_bps, outcome, reasons_json, source, external_id,
                        agreeing_strategies_json, index_regime
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, 'EVALUATED', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.timestamp.isoformat(),
                        event.symbol,
                        event.side.value,
                        event.confidence,
                        event.reward_risk,
                        float(event.entry),
                        int(event.agreeing_count),
                        state_key,
                        event.exit_time.isoformat(),
                        float(event.exit_price),
                        float(reward_bps),
                        "WIN" if reward_bps > event_cost else "LOSS",
                        json.dumps(list(event.reasons)),
                        source,
                        external_id,
                        json.dumps(list(event.agreeing_strategies)),
                        index_regime_fn(event) if index_regime_fn is not None else None,
                    ),
                )
                inserted += max(cursor.rowcount, 0)
        self.refresh_policy()
        return inserted

    def record_scan_results(
        self,
        results: list[ScanResult],
        min_confidence: float = 55.0,
        enforce_session: bool = True,
        index_regime: str | None = None,
    ) -> int:
        now = now_ist()
        if enforce_session and not in_market_hours(now):
            # Off-session scans see stale candles; recording them produced
            # thousands of zero-move "LOSS" rows that poisoned the policy.
            return 0
        observed_at = now.isoformat(timespec="seconds")
        inserted = 0
        with self._connect() as conn:
            for result in results:
                side = result.plan.side if result.plan.is_actionable else leading_trade_side(result)
                price = result.public_ltp if result.public_ltp is not None else result.last_close
                if side == Side.WAIT or price is None or result.plan.confidence < min_confidence:
                    continue
                votes = sum(1 for vote in result.plan.strategy_votes if vote.side == side)
                agreeing_strategies = [
                    vote.strategy
                    for vote in result.plan.strategy_votes
                    if vote.side == side and vote.is_trade
                ]
                conn.execute(
                    """
                    INSERT INTO shadow_observations (
                        observed_at, symbol, side, confidence, reward_risk, entry_price,
                        quote_age_seconds, vote_count, state_key, status, reasons_json,
                        stop_loss, target, agreeing_strategies_json, regime, index_regime
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observed_at,
                        result.symbol,
                        side.value,
                        result.plan.confidence,
                        result.plan.reward_risk,
                        float(price),
                        result.quote_age_seconds,
                        votes,
                        self._state_key(result, side),
                        json.dumps(list(result.plan.reasons)),
                        float(result.plan.stop_loss) if result.plan.stop_loss is not None else None,
                        float(result.plan.target) if result.plan.target is not None else None,
                        json.dumps(agreeing_strategies),
                        result.regime,
                        index_regime,
                    ),
                )
                inserted += 1
        return inserted

    def _path_outcome(
        self,
        *,
        symbol: str,
        side: str,
        entry: float,
        stop,
        target,
        observed_at: datetime,
        horizon_minutes: int,
    ) -> tuple[float, float] | None:
        """Walk cached 1m candles from observation time; return (reward_bps, exit_price).

        Replicates the backtest exit model (stop checked before target within
        a candle, time-exit at the horizon) instead of the old single-price
        snapshot, which could not see stop-outs that happened along the way.
        """
        if self._candle_cache is None:
            return None
        try:
            # Cache timestamps are stored as UTC ISO strings and compared
            # lexically — the query bound must be UTC too, not IST.
            from datetime import timezone as _tz
            since_utc = observed_at.astimezone(_tz.utc)
            frame = self._candle_cache.load(symbol, "1m", since=since_utc, limit=horizon_minutes + 120)
        except Exception:
            return None
        if frame is None or frame.empty:
            return None
        deadline = observed_at + timedelta(minutes=horizon_minutes)
        path = frame[(frame.index > observed_at) & (frame.index <= deadline)]
        if path.empty:
            return None
        stop_f = float(stop) if stop is not None else None
        tgt_f = float(target) if target is not None else None
        long = side == Side.LONG.value
        for _, row in path.iterrows():
            high, low = float(row["high"]), float(row["low"])
            if long:
                if stop_f is not None and low <= stop_f:
                    return (stop_f - entry) / entry * 10_000, stop_f
                if tgt_f is not None and high >= tgt_f:
                    return (tgt_f - entry) / entry * 10_000, tgt_f
            else:
                if stop_f is not None and high >= stop_f:
                    return (entry - stop_f) / entry * 10_000, stop_f
                if tgt_f is not None and low <= tgt_f:
                    return (entry - tgt_f) / entry * 10_000, tgt_f
        close = float(path.iloc[-1]["close"])
        reward = (close - entry) if long else (entry - close)
        return reward / entry * 10_000, close

    def evaluate_pending(
        self,
        latest_prices: dict[str, float],
        horizon_minutes: int = 30,
        enforce_session: bool = True,
    ) -> int:
        cutoff = now_ist() - timedelta(minutes=horizon_minutes)
        evaluated = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM shadow_observations WHERE status = 'PENDING'"
            ).fetchall()
            for row in rows:
                observed_at = datetime.fromisoformat(row["observed_at"])
                if observed_at > cutoff:
                    continue
                if observed_at.tzinfo is None:
                    observed_at = observed_at.replace(tzinfo=IST)
                if enforce_session and not in_market_hours(observed_at):
                    # Stale off-session signal — discard instead of labelling
                    # a zero-move as LOSS.
                    conn.execute(
                        "UPDATE shadow_observations SET status = 'DISCARDED' WHERE id = ?",
                        (row["id"],),
                    )
                    continue
                entry = float(row["entry_price"])
                stop = row["stop_loss"]
                tgt = row["target"]

                path = self._path_outcome(
                    symbol=row["symbol"],
                    side=row["side"],
                    entry=entry,
                    stop=stop,
                    target=tgt,
                    observed_at=observed_at,
                    horizon_minutes=horizon_minutes,
                )
                if path is not None:
                    reward_bps, future_price = path
                else:
                    # Fallback: single-price snapshot from the latest scan.
                    future_price = latest_prices.get(row["symbol"])
                    if future_price is None:
                        continue
                    if row["side"] == Side.LONG.value:
                        reward_bps = (future_price - entry) / entry * 10_000
                    else:
                        reward_bps = (entry - future_price) / entry * 10_000
                outcome = "WIN" if reward_bps > cost_bps_for(row["symbol"]) else "LOSS"
                conn.execute(
                    """
                    UPDATE shadow_observations
                    SET status = 'EVALUATED', evaluated_at = ?, future_price = ?,
                        reward_bps = ?, outcome = ?
                    WHERE id = ?
                    """,
                    (
                        now_ist().isoformat(timespec="seconds"),
                        float(future_price),
                        float(reward_bps),
                        outcome,
                        row["id"],
                    ),
                )
                evaluated += 1
        if evaluated:
            self.refresh_policy()
        return evaluated

    def refresh_policy(
        self, exploration: float = 8.0, decay_half_life_days: float = 7.0
    ) -> None:
        """Rebuild the policy table with exponential time-decay on observations.

        Recent wins/losses are weighted more heavily than older ones.  A 7-day
        half-life means an observation from 7 days ago is worth half as much as
        one from today.  Raw sample counts (for min-samples gating) are kept
        separate from the decayed win-rate and average reward.
        """
        now = now_ist()
        _ln2 = math.log(2)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT state_key, observed_at,
                       CASE WHEN reward_bps > {cost} THEN 1 ELSE 0 END AS is_win,
                       reward_bps - {cost} AS reward_bps
                FROM shadow_observations
                WHERE status = 'EVALUATED'
                """.format(cost=_COST_SQL),
                (),
            ).fetchall()

        state_data: dict[str, dict] = defaultdict(
            lambda: {"raw_n": 0, "raw_wins": 0, "w_sum": 0.0, "w_win": 0.0, "w_reward": 0.0}
        )
        for row in rows:
            try:
                obs_at = datetime.fromisoformat(row["observed_at"])
                if obs_at.tzinfo is None:
                    obs_at = obs_at.replace(tzinfo=IST)
            except ValueError:
                obs_at = now
            age_days = max(0.0, (now - obs_at).total_seconds() / 86_400.0)
            w = math.exp(-_ln2 * age_days / max(decay_half_life_days, 0.5))
            d = state_data[row["state_key"]]
            d["raw_n"] += 1
            d["raw_wins"] += int(row["is_win"])
            d["w_sum"] += w
            d["w_win"] += w * int(row["is_win"])
            d["w_reward"] += w * float(row["reward_bps"] or 0)

        total_w = sum(d["w_sum"] for d in state_data.values()) or 1.0

        with self._connect() as conn:
            for sk, d in state_data.items():
                n = d["raw_n"]
                wins = d["raw_wins"]
                wsum = max(d["w_sum"], 1e-9)
                win_rate = d["w_win"] / wsum * 100
                avg_reward_bps = d["w_reward"] / wsum
                # UCB exploration bonus — still relative to effective weight mass
                bonus = exploration * (total_w / max(wsum, 1e-9)) ** 0.5
                ucb_score = avg_reward_bps + bonus
                conn.execute(
                    """
                    INSERT INTO shadow_policy (
                        state_key, samples, wins, win_rate, avg_reward_bps,
                        ucb_score, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(state_key) DO UPDATE SET
                        samples = excluded.samples,
                        wins = excluded.wins,
                        win_rate = excluded.win_rate,
                        avg_reward_bps = excluded.avg_reward_bps,
                        ucb_score = excluded.ucb_score,
                        updated_at = excluded.updated_at
                    """,
                    (
                        sk, n, wins,
                        round(win_rate, 4),
                        round(avg_reward_bps, 4),
                        round(ucb_score, 4),
                        now.isoformat(timespec="seconds"),
                    ),
                )

    def estimate_for_result(
        self,
        result: ScanResult,
        min_samples: int = 20,
        index_regime: str | None = None,
    ) -> PolicyEstimate:
        side = result.plan.side if result.plan.is_actionable else leading_trade_side(result)
        if side == Side.WAIT:
            return PolicyEstimate("WAIT", 0, 0, 0, 0, 0, False)

        state_key = self._state_key(result, side)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM shadow_policy WHERE state_key = ?", (state_key,)
            ).fetchone()

            # Regime-aware estimate hierarchy.  Regime-labelled observations
            # come from the current engine (live scans + nightly replay), so
            # they are preferred over the global policy row, which still
            # mixes in stale pre-rework history:
            #   1. (state, current index regime) with enough samples
            #   2. (state, any labelled regime)  with enough samples
            #   3. global policy row (fallback)
            # 2026-07-02 case study: a SHORT state carried a positive global
            # score from stale data while every labelled regime bucket was
            # negative — this hierarchy vetoes it, the global row does not.
            since_labelled = (now_ist() - timedelta(days=45)).isoformat(timespec="seconds")

            def _labelled_stats(match_regime: str | None):
                clause = "AND index_regime = ?" if match_regime else "AND index_regime IS NOT NULL"
                cost = _COST_SQL
                params = [state_key]
                if match_regime:
                    params.append(match_regime)
                params.append(since_labelled)
                return conn.execute(
                    f"""
                    SELECT COUNT(*) AS n,
                           AVG(reward_bps) - AVG({cost}) AS avg_net,
                           100.0 * SUM(CASE WHEN reward_bps > {cost} THEN 1 ELSE 0 END) / COUNT(*) AS wr
                    FROM shadow_observations
                    WHERE status = 'EVALUATED' AND state_key = ? {clause}
                      AND observed_at >= ?
                    """,
                    params,
                ).fetchone()

            candidates = []
            if index_regime:
                candidates.append((_labelled_stats(index_regime), max(10, min_samples // 2), f"|idx={index_regime}"))
            candidates.append((_labelled_stats(None), min_samples, "|idx=any"))
            for cond, needed, suffix in candidates:
                if cond and int(cond["n"] or 0) >= needed:
                    samples = int(cond["n"])
                    avg_reward_bps = float(cond["avg_net"] or 0)
                    win_rate = float(cond["wr"] or 0)
                    raw_bonus = avg_reward_bps / 2 + (win_rate - 50) / 8
                    return PolicyEstimate(
                        state_key=f"{state_key}{suffix}",
                        samples=samples,
                        win_rate=win_rate,
                        avg_reward_bps=avg_reward_bps,
                        ucb_score=avg_reward_bps,
                        confidence_bonus=max(-8.0, min(8.0, raw_bonus)),
                        is_trained=True,
                    )

        if row is None:
            return PolicyEstimate(state_key, 0, 0, 0, 0, 0, False)

        samples = int(row["samples"])
        avg_reward_bps = float(row["avg_reward_bps"] or 0)
        win_rate = float(row["win_rate"] or 0)
        ucb_score = float(row["ucb_score"] or 0)
        is_trained = samples >= min_samples
        if not is_trained:
            confidence_bonus = 0.0
        else:
            raw_bonus = avg_reward_bps / 2 + (win_rate - 50) / 8
            confidence_bonus = max(-8.0, min(8.0, raw_bonus))
        return PolicyEstimate(
            state_key=state_key,
            samples=samples,
            win_rate=win_rate,
            avg_reward_bps=avg_reward_bps,
            ucb_score=ucb_score,
            confidence_bonus=confidence_bonus,
            is_trained=is_trained,
        )

    def purge_and_migrate(self) -> dict:
        """One-time data hygiene pass over accumulated observations.

        1. DISCARD observations recorded outside NSE market hours (stale-cache
           junk — 8,842 such rows existed as of 2026-07-02, all labelled LOSS
           at 0 bps, dragging every strategy's stats down).
        2. Relabel outcomes net of ASSUMED_COST_BPS.
        3. Recompute state keys with the coarser bucketing and the
           observation's own time-of-day (the old code used the wall clock).
        4. Rebuild the policy table from scratch.
        """
        discarded = relabelled = rekeyed = 0
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, observed_at, symbol, side, confidence, reward_risk,
                       vote_count, reward_bps, outcome, status
                FROM shadow_observations
                WHERE status IN ('EVALUATED', 'PENDING')
                """
            ).fetchall()
            for row in rows:
                try:
                    observed_at = datetime.fromisoformat(row["observed_at"])
                except ValueError:
                    continue
                if observed_at.tzinfo is None:
                    observed_at = observed_at.replace(tzinfo=IST)
                # 24-hour futures legitimately trade outside the NSE session;
                # applying the equity clock here DISCARDED 9,385 commodity
                # observations in 30 days, including the whole US-overlap
                # evening where the measured futures edge sits.
                if not is_commodity(row["symbol"]) and not in_market_hours(observed_at):
                    conn.execute(
                        "UPDATE shadow_observations SET status = 'DISCARDED' WHERE id = ?",
                        (row["id"],),
                    )
                    discarded += 1
                    continue
                updates: dict[str, object] = {}
                if row["status"] == "EVALUATED" and row["reward_bps"] is not None:
                    outcome = ("WIN" if float(row["reward_bps"]) > cost_bps_for(row["symbol"])
                               else "LOSS")
                    if outcome != row["outcome"]:
                        updates["outcome"] = outcome
                        relabelled += 1
                try:
                    side = Side(row["side"])
                except ValueError:
                    side = Side.WAIT
                new_key = self._state_key_from_features(
                    side=side,
                    confidence=float(row["confidence"] or 0),
                    reward_risk=float(row["reward_risk"] or 0),
                    vote_count=int(row["vote_count"] or 0),
                    ts=observed_at,
                    symbol=row["symbol"],
                )
                updates["state_key"] = new_key
                rekeyed += 1
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                conn.execute(
                    f"UPDATE shadow_observations SET {set_clause} WHERE id = ?",
                    (*updates.values(), row["id"]),
                )
            conn.execute("DELETE FROM shadow_policy")
        self.refresh_policy()
        return {"discarded": discarded, "relabelled": relabelled, "rekeyed": rekeyed}

    def stats(self) -> ShadowStats:
        with self._connect() as conn:
            pending = conn.execute(
                "SELECT COUNT(*) AS n FROM shadow_observations WHERE status = 'PENDING'"
            ).fetchone()["n"]
            rows = conn.execute(
                "SELECT reward_bps, outcome FROM shadow_observations WHERE status = 'EVALUATED'"
            ).fetchall()
        evaluated = len(rows)
        wins = sum(1 for row in rows if row["outcome"] == "WIN")
        avg_reward = sum(float(row["reward_bps"]) for row in rows) / evaluated if evaluated else 0.0
        return ShadowStats(
            pending=int(pending),
            evaluated=evaluated,
            win_rate=(wins / evaluated * 100) if evaluated else 0.0,
            avg_reward_bps=avg_reward,
        )

    def policy_frame(self, limit: int = 20) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT state_key, samples, wins,
                       ROUND(win_rate, 2) AS win_rate,
                       ROUND(avg_reward_bps, 2) AS avg_reward_bps,
                       ROUND(ucb_score, 2) AS ucb_score,
                       updated_at
                FROM shadow_policy
                ORDER BY samples DESC, ucb_score DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def strategy_performance_frame(
        self,
        *,
        days: int = 5,
        min_samples: int = 5,
        limit: int = 20,
        cost_bps: float = 0.0,
    ) -> pd.DataFrame:
        since = now_ist() - timedelta(days=days)
        rows: list[dict] = []
        with self._connect() as conn:
            observations = conn.execute(
                """
                SELECT observed_at, reward_bps, outcome, reasons_json
                FROM shadow_observations
                WHERE status = 'EVALUATED'
                """
            ).fetchall()

        for row in observations:
            try:
                observed_at = datetime.fromisoformat(row["observed_at"])
            except ValueError:
                continue
            if observed_at < since:
                continue
            try:
                reasons = json.loads(row["reasons_json"])
            except json.JSONDecodeError:
                reasons = []
            reason_text = " ".join(str(reason) for reason in reasons)
            matched = [strategy for strategy in KNOWN_STRATEGIES if strategy in reason_text]
            for strategy in matched:
                reward_bps = float(row["reward_bps"] or 0) - cost_bps
                rows.append(
                    {
                        "strategy": strategy,
                        "reward_bps": reward_bps,
                        "win": reward_bps > 0,
                        "observed_at": observed_at.isoformat(timespec="seconds"),
                    }
                )

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows)
        grouped = (
            frame.groupby("strategy")
            .agg(
                samples=("reward_bps", "size"),
                wins=("win", "sum"),
                avg_reward_bps=("reward_bps", "mean"),
                total_reward_bps=("reward_bps", "sum"),
                latest_observed_at=("observed_at", "max"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["samples"] >= min_samples]
        if grouped.empty:
            return pd.DataFrame()

        grouped["win_rate"] = grouped["wins"] / grouped["samples"] * 100
        grouped = grouped.sort_values(
            ["avg_reward_bps", "win_rate", "samples"],
            ascending=[False, False, False],
        )
        grouped = grouped.head(limit)
        for column in ("avg_reward_bps", "total_reward_bps", "win_rate"):
            grouped[column] = grouped[column].round(2)
        return grouped[
            [
                "strategy",
                "samples",
                "win_rate",
                "avg_reward_bps",
                "total_reward_bps",
                "latest_observed_at",
            ]
        ]

    def candidate_performance(
        self,
        *,
        symbol: str,
        side: Side,
        strategies: list[str],
        days: int = 30,
        cost_bps: float = 0.0,
    ) -> dict:
        since = now_ist() - timedelta(days=days)
        wanted = set(strategies)
        rewards: list[float] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT observed_at, reward_bps, reasons_json
                FROM shadow_observations
                WHERE status = 'EVALUATED'
                  AND symbol = ?
                  AND side = ?
                """,
                (symbol, side.value),
            ).fetchall()

        for row in rows:
            try:
                observed_at = datetime.fromisoformat(row["observed_at"])
            except ValueError:
                continue
            if observed_at < since:
                continue
            try:
                reasons = json.loads(row["reasons_json"])
            except json.JSONDecodeError:
                reasons = []
            reason_text = " ".join(str(reason) for reason in reasons)
            if wanted and not any(strategy in reason_text for strategy in wanted):
                continue
            rewards.append(float(row["reward_bps"] or 0) - cost_bps)

        if not rewards:
            return {"samples": 0, "win_rate": 0.0, "avg_reward_bps": 0.0}

        wins = sum(1 for reward in rewards if reward > 0)
        return {
            "samples": len(rewards),
            "win_rate": wins / len(rewards) * 100,
            "avg_reward_bps": sum(rewards) / len(rewards),
        }


class AdaptiveWeightEngine:
    """Online strategy weight adapter — closes the learn→vote feedback loop.

    Each time a scan runs, shadow observations accumulate.  After outcomes are
    evaluated (stop/target hit, or horizon elapsed), this engine converts the
    per-strategy win-rate and average reward into multipliers for the
    VotingSignalEngine.  Strategies that have been consistently right earn
    higher voting weight; persistently wrong ones shrink.

    Usage::

        engine = AdaptiveWeightEngine(learner)
        weights = engine.weights(paper_weights=store.strategy_weights())
        # pass weights to VotingSignalEngine or scan_universe_batch

    Weights are bounded [0.3, 2.0] so no strategy is ever silenced or
    blindly dominant.  Adaptation only kicks in after *min_shadow_samples*
    evaluated observations per strategy so early noise is ignored.
    """

    def __init__(self, learner: ShadowLearner) -> None:
        self.learner = learner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def weights(
        self,
        paper_weights: dict[str, float] | None = None,
        min_shadow_samples: int = 15,
    ) -> dict[str, float]:
        """Return per-strategy weights suitable for VotingSignalEngine.

        Combines shadow-learner reward signal with optional paper-trade
        win rates via geometric mean, then clamps to [0.3, 2.0].
        """
        shadow = self._shadow_weights(min_shadow_samples)
        paper = paper_weights or {}
        result: dict[str, float] = {}
        for strategy in KNOWN_STRATEGIES:
            sw = shadow.get(strategy, 1.0)
            pw = paper.get(strategy, 1.0)
            combined = (sw * pw) ** 0.5  # geometric mean keeps both signals balanced
            result[strategy] = max(0.3, min(2.0, combined))
        return result

    def weight_frame(self, paper_weights: dict[str, float] | None = None) -> pd.DataFrame:
        """DataFrame of per-strategy weights for display in the UI."""
        combined = self.weights(paper_weights)
        shadow = self._shadow_weights(15)
        paper = paper_weights or {}
        rows = []
        for strategy in KNOWN_STRATEGIES:
            rows.append(
                {
                    "strategy": strategy,
                    "combined_weight": round(combined.get(strategy, 1.0), 3),
                    "shadow_weight": round(shadow.get(strategy, 1.0), 3),
                    "paper_weight": round(paper.get(strategy, 1.0), 3),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _shadow_weights(self, min_samples: int) -> dict[str, float]:
        """Derive per-strategy weights from evaluated shadow observations with 2026 recency weighting."""
        strategy_weighted_rewards: dict[str, list[tuple[float, float]]] = {s: [] for s in KNOWN_STRATEGIES}

        with self.learner._connect() as conn:
            rows = conn.execute(
                """
                SELECT agreeing_strategies_json, reasons_json, reward_bps, observed_at, symbol
                FROM shadow_observations
                WHERE status = 'EVALUATED'
                  AND observed_at >= ?
                """,
                ((now_ist() - timedelta(days=30)).isoformat(timespec="seconds"),),
            ).fetchall()

        now_dt = now_ist()
        for row in rows:
            reward = float(row["reward_bps"] or 0) - cost_bps_for(row["symbol"])
            try:
                obs_dt = datetime.fromisoformat(row["observed_at"])
                if obs_dt.tzinfo is None:
                    obs_dt = obs_dt.replace(tzinfo=IST)
                days_ago = max(0.0, (now_dt - obs_dt).total_seconds() / 86400.0)
                decay_wt = math.exp(-days_ago / 14.0)
                if obs_dt.year >= 2026:
                    decay_wt *= 1.5
            except Exception:
                decay_wt = 1.0

            item = (reward, decay_wt)
            strats_json = row["agreeing_strategies_json"]
            if strats_json:
                try:
                    for s in json.loads(strats_json):
                        if s in strategy_weighted_rewards:
                            strategy_weighted_rewards[s].append(item)
                    continue
                except json.JSONDecodeError:
                    pass
            try:
                reasons = json.loads(row["reasons_json"])
            except json.JSONDecodeError:
                continue
            reason_text = " ".join(str(r) for r in reasons)
            for s in KNOWN_STRATEGIES:
                if s in reason_text:
                    strategy_weighted_rewards[s].append(item)

        weights: dict[str, float] = {}
        for strategy, items in strategy_weighted_rewards.items():
            if len(items) < min_samples:
                weights[strategy] = 1.0
                continue
            total_wt = sum(wt for _, wt in items)
            if total_wt <= 0:
                weights[strategy] = 1.0
                continue
            weighted_win = sum(wt for r, wt in items if r > 0) / total_wt
            weighted_avg_reward = sum(r * wt for r, wt in items) / total_wt
            score = 1.0 + (weighted_win - 0.5) * 1.2 + max(-0.3, min(0.3, weighted_avg_reward / 50.0))
            weights[strategy] = max(0.3, min(2.0, score))
        return weights
