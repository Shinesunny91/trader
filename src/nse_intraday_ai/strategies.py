from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from zoneinfo import ZoneInfo as _ZI

import numpy as np
import pandas as pd

from nse_intraday_ai.indicators import add_indicators
from nse_intraday_ai.models import Side, StrategySignal, TradePlan
from nse_intraday_ai.risk import RiskConfig, build_trade_plan


_IST = _ZI("Asia/Kolkata")

# Which market regimes each strategy is designed for.
# Signals from a strategy operating outside its preferred regime get a
# confidence penalty, making it harder for them to win the ensemble vote.
REGIME_AFFINITY: dict[str, frozenset[str]] = {
    "trend_continuation":              frozenset({"TRENDING_UP", "TRENDING_DOWN"}),
    "vwap_mean_reversion":             frozenset({"RANGING"}),
    "opening_range_breakout":          frozenset({"TRENDING_UP", "TRENDING_DOWN", "HIGH_VOL"}),
    "volatility_compression_breakout": frozenset({"TRENDING_UP", "TRENDING_DOWN", "HIGH_VOL"}),
    "ema_scalp":                       frozenset({"TRENDING_UP", "TRENDING_DOWN"}),
    "vwap_bounce_scalp":               frozenset({"RANGING", "TRENDING_UP", "TRENDING_DOWN"}),
    "momentum_burst_scalp":            frozenset({"TRENDING_UP", "TRENDING_DOWN", "HIGH_VOL"}),
    "supertrend":                      frozenset({"TRENDING_UP", "TRENDING_DOWN"}),
    "bb_kc_squeeze":                   frozenset({"TRENDING_UP", "TRENDING_DOWN", "HIGH_VOL"}),
    "rsi_divergence":                  frozenset({"RANGING", "TRENDING_UP", "TRENDING_DOWN", "HIGH_VOL"}),
    "fair_value_gap":                  frozenset({"TRENDING_UP", "TRENDING_DOWN"}),
    "liquidity_sweep_reversal":        frozenset({"RANGING", "HIGH_VOL", "TRENDING_UP", "TRENDING_DOWN"}),
    "volume_profile_poc":              frozenset({"RANGING", "TRENDING_UP", "TRENDING_DOWN", "HIGH_VOL"}),
    "opening_drive_momentum":          frozenset({"TRENDING_UP", "TRENDING_DOWN", "HIGH_VOL"}),
    "vwap_band_mean_reversion":        frozenset({"RANGING", "HIGH_VOL"}),
    "opening_range_expansion":         frozenset({"TRENDING_UP", "TRENDING_DOWN", "HIGH_VOL"}),
    "mtf_trend_alignment":             frozenset({"TRENDING_UP", "TRENDING_DOWN"}),
}

_REGIME_PENALTY = 18.0  # raised from 12 — today's data showed 12pt wasn't enough to suppress contrarian signals in strong trends

# Escape hatch for A/B backtesting: NSE_AI_LEGACY=1 restores the pre-2026-07
# engine behavior (NSE session multiplier and opening-range breakout applied
# to commodities as well).
_LEGACY_ENGINE = os.environ.get("NSE_AI_LEGACY") == "1"

# Default vote weights for commodity futures, used when no learned weight
# exists for the strategy.  Jun-2026 2-week event study: fast scalps built for
# NSE 1m equity flow were negative in BOTH weeks on 5m 24-hour futures
# (ema_scalp −14.6/−1.7 bps, momentum_burst −6.5/−2.3, trend_continuation
# −2.4/−4.9); down-weight conservatively rather than disable.
COMMODITY_WEIGHT_DEFAULTS: dict[str, float] = {
    "ema_scalp": 0.5,
    "momentum_burst_scalp": 0.7,
    "trend_continuation": 0.8,
}


def _commodity_time_multiplier(df: pd.DataFrame) -> float:
    """Confidence multiplier for 24-hour commodity futures (IST clock).

    Liquidity on CME/ICE (and MCX, which shadows them) peaks in the
    US-overlap evening; the IST early-morning Globex stretch is thin and
    breakout-hostile.  Conservative multipliers only.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return 1.0
    ts = df.index[-1]
    try:
        ts = ts.tz_convert(_IST) if ts.tzinfo else ts.tz_localize(_IST)
    except Exception:
        return 1.0
    m = ts.hour * 60 + ts.minute
    if m < 300:      # 00:00–05:00 late US close / thin Globex
        return 0.90
    if m < 540:      # 05:00–09:00 dead zone
        return 0.85
    if m < 900:      # 09:00–15:00 Asian session, modest
        return 0.95
    if m < 1080:     # 15:00–18:00 Europe open
        return 1.00
    if m < 1410:     # 18:00–23:30 US overlap — deepest liquidity
        return 1.05
    return 0.95      # 23:30–24:00 US midday wind-down


def _time_of_day_multiplier(df: pd.DataFrame) -> float:
    """Confidence multiplier based on NSE session time (IST).

    Calibrated against evaluated shadow-learner outcomes (Mar–Jun 2026) and
    the 2026-07-02 candidate-event study.

    Re-measured Aug 2026 on 280K candidate signals over 49 sessions, split in
    half to check stability.  Only four of the seven windows kept their sign
    across the split, so the table is left alone except where it was
    demonstrably backwards: **10:00–11:00 was negative in both halves**
    (−1.42 and −2.19 bps) and the table was boosting it 1.06x, the single
    largest miscalibration in the schedule.  The windows this study liked best
    (09:15–09:30 and 13:15–14:30) flipped sign between halves and are
    therefore *not* boosted — fitting them would be fitting noise.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return 1.0
    ts = df.index[-1]
    try:
        ts = ts.tz_convert(_IST) if ts.tzinfo else ts.tz_localize(_IST)
    except Exception:
        return 1.0
    m = ts.hour * 60 + ts.minute
    if m < 555 or m >= 930:   # outside NSE session — historical/backtest data
        return 1.0
    if m < 570:    # 9:15–9:30  opening rush, gap fills
        return 0.85
    if m < 600:    # 9:30–10:00 gap fill resolving
        return 0.98
    if m < 660:    # 10:00–11:00 negative in both halves of the 49-session study
        return 0.92
    if m < 795:    # 11:00–13:15 lunch chop — worst observed window, suppress
        return 0.88
    if m < 870:    # 13:15–14:30 trends resume — near break-even observed
        return 1.02
    if m < 905:    # 14:30–15:05 afternoon positioning
        return 0.97
    return 0.80    # 15:05–15:30 close squaring, avoid new entries


def _timestamp(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    value = df.index[-1]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _clamp_confidence(value: float) -> float:
    return float(max(0, min(100, value)))


def _plan_levels(side: Side, price: float, atr: float, reward_multiple: float = 1.8) -> tuple[float, float]:
    stop_distance = max(atr * 1.5, price * 0.003)  # wider stop: 1.5×ATR, min 0.3% of price
    if side == Side.LONG:
        return price - stop_distance, price + stop_distance * reward_multiple
    if side == Side.SHORT:
        return price + stop_distance, price - stop_distance * reward_multiple
    return price, price


class BaseStrategy:
    name = "base"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        raise NotImplementedError


class TrendContinuationStrategy(BaseStrategy):
    """Pullback-resume entry within an established EMA trend.

    The old version fired continuously whenever the EMA stack was aligned —
    a state that persists for hours — so it flooded the ensemble with late,
    momentum-chasing entries (97% of gated signals on 2026-07-02; −19 bps
    average).  Now it only fires on the *event* of a pullback to the fast
    EMA resolving back in the trend direction: better entry location, and
    at most a couple of signals per trend leg.
    """

    name = "trend_continuation"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 55:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 55 candles.")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        recent = df.iloc[-7:-1]  # pullback lookback window (6 bars before current)
        price = float(row["close"])
        atr = float(row["atr_14"])
        bull_stack = row["ema_9"] > row["ema_21"] > row["ema_50"]
        bear_stack = row["ema_9"] < row["ema_21"] < row["ema_50"]
        volume_ok = row["volume_z"] > -1.5
        rsi = float(row["rsi_14"])

        regime = str(row.get("regime", "")) if "regime" in df.columns else ""
        rsi_ceiling = 74 if regime == "TRENDING_UP" else 68
        rsi_floor = 26 if regime == "TRENDING_DOWN" else 32

        # Pullback: some recent bar tagged the fast EMA; resume: current bar
        # closes back in trend direction beyond the prior bar's extreme while
        # the prior bar was still part of the pullback (freshness guard).
        pulled_back_long = bool((recent["low"] <= recent["ema_9"] * 1.0005).any())
        resume_long = (
            float(row["close"]) > float(row["open"])
            and price > float(prev["high"])
            and (float(prev["close"]) <= float(prev["open"]) or float(prev["low"]) <= float(prev["ema_9"]) * 1.0005)
        )
        if (
            bull_stack
            and price > float(row["vwap"])
            and pulled_back_long
            and resume_long
            and 40 <= rsi <= rsi_ceiling
            and volume_ok
        ):
            stop, target = _plan_levels(Side.LONG, price, atr)
            stop = min(stop, float(recent["low"].min()) - atr * 0.1)
            regime_bonus = 6.0 if regime == "TRENDING_UP" else 0.0
            adx_bonus = min(8.0, max(0.0, (float(row.get("adx_14", 0)) - 20) * 0.4))
            confidence = 62 + adx_bonus + min(8, max(0, row["volume_z"]) * 3) + regime_bonus
            return StrategySignal(
                self.name,
                Side.LONG,
                _clamp_confidence(confidence),
                price,
                stop,
                target,
                "Uptrend pullback to EMA-9 resolved upward (resume candle).",
            )

        pulled_back_short = bool((recent["high"] >= recent["ema_9"] * 0.9995).any())
        resume_short = (
            float(row["close"]) < float(row["open"])
            and price < float(prev["low"])
            and (float(prev["close"]) >= float(prev["open"]) or float(prev["high"]) >= float(prev["ema_9"]) * 0.9995)
        )
        if (
            bear_stack
            and price < float(row["vwap"])
            and pulled_back_short
            and resume_short
            and rsi_floor <= rsi <= 60
            and volume_ok
        ):
            stop, target = _plan_levels(Side.SHORT, price, atr)
            stop = max(stop, float(recent["high"].max()) + atr * 0.1)
            regime_bonus = 6.0 if regime == "TRENDING_DOWN" else 0.0
            adx_bonus = min(8.0, max(0.0, (float(row.get("adx_14", 0)) - 20) * 0.4))
            confidence = 62 + adx_bonus + min(8, max(0, row["volume_z"]) * 3) + regime_bonus
            return StrategySignal(
                self.name,
                Side.SHORT,
                _clamp_confidence(confidence),
                price,
                stop,
                target,
                "Downtrend pullback to EMA-9 resolved downward (resume candle).",
            )

        return StrategySignal(self.name, Side.WAIT, 15, None, None, None, "No fresh pullback-resume in trend.")


class VwapMeanReversionStrategy(BaseStrategy):
    name = "vwap_mean_reversion"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 35:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 35 candles.")

        row = df.iloc[-1]
        price = float(row["close"])
        atr = float(row["atr_14"])
        distance_atr = (price - row["vwap"]) / max(atr, price * 0.001)
        trend_flat = abs(row["ema_21"] - row["ema_50"]) / price < 0.0045
        regime = str(row.get("regime", "RANGING")) if "regime" in df.columns else "RANGING"
        # Mean reversion only works in RANGING or mild trend — not in strong trending markets
        regime_ok = regime in ("RANGING", "HIGH_VOL")

        # Strict conditions: RSI must be deeply extreme, deviation > 2.0×ATR, volume confirming.
        # A confirming reversal candle is required — entering while the candle
        # is still falling is catching a knife, not fading a stretch.
        volume_spike = row["volume_z"] > 0.5
        bullish_reversal = float(row["close"]) > float(row["open"])
        bearish_reversal = float(row["close"]) < float(row["open"])
        if trend_flat and regime_ok and distance_atr < -2.0 and row["rsi_14"] < 28 and volume_spike and bullish_reversal:
            stop, target = _plan_levels(Side.LONG, price, atr, reward_multiple=1.8)
            target = min(target, float(row["vwap"]))
            confidence = 60 + min(20, abs(distance_atr) * 6) + max(0, 30 - row["rsi_14"]) * 0.8
            return StrategySignal(
                self.name,
                Side.LONG,
                _clamp_confidence(confidence),
                price,
                stop,
                target,
                "Deep VWAP stretch in ranging regime: RSI<28, deviation>2×ATR, volume spike.",
            )

        if trend_flat and regime_ok and distance_atr > 2.0 and row["rsi_14"] > 72 and volume_spike and bearish_reversal:
            stop, target = _plan_levels(Side.SHORT, price, atr, reward_multiple=1.8)
            target = max(target, float(row["vwap"]))
            confidence = 60 + min(20, abs(distance_atr) * 6) + max(0, row["rsi_14"] - 70) * 0.8
            return StrategySignal(
                self.name,
                Side.SHORT,
                _clamp_confidence(confidence),
                price,
                stop,
                target,
                "Price is stretched above VWAP in a ranging regime with overbought RSI.",
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No clean VWAP reversion.")


class OpeningRangeBreakoutStrategy(BaseStrategy):
    name = "opening_range_breakout"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if isinstance(df.index, pd.DatetimeIndex):
            current_day = df.index[-1].normalize()
            session = df[df.index.normalize() == current_day]
        else:
            session = df

        # Adapt to bar timeframe (on 5m candles, 6 bars = 30 min opening range; on 1m, 30 bars = 30 min)
        bar_step = (
            (session.index[1] - session.index[0]).total_seconds()
            if len(session) >= 2 and isinstance(session.index, pd.DatetimeIndex)
            else 300
        )
        is_5m = bar_step >= 240
        or_bars = 6 if is_5m else 30
        min_bars = or_bars + 1

        if len(session) < min_bars:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, f"Need at least {min_bars} session candles.")

        opening = session.iloc[:or_bars]
        row = session.iloc[-1]
        prev = session.iloc[-2]
        price = float(row["close"])
        atr = float(row["atr_14"])
        high = float(opening["high"].max())
        low = float(opening["low"].min())
        range_size = max(high - low, price * 0.001)
        volume_break = row["volume_z"] > 0.6

        regime = str(row.get("regime", "")) if "regime" in df.columns else ""
        # Gate direction to regime — don't take LONG breakouts against a TRENDING_DOWN tape
        long_ok  = regime != "TRENDING_DOWN"
        short_ok = regime != "TRENDING_UP"

        # Fresh-cross requirement: the breakout must have happened on this
        # candle (prev close still inside the range).  Without this the
        # signal re-fires every minute for as long as price stays beyond the
        # range — 648 duplicate signals on 2026-07-02 alone.
        fresh_break_up = float(prev["close"]) <= high
        fresh_break_dn = float(prev["close"]) >= low

        if long_ok and fresh_break_up and price > high and volume_break and row["ema_9"] > row["ema_21"]:
            stop = max(high - range_size * 0.45, price - atr * 1.1)
            target = price + max(range_size, atr) * 1.6
            regime_bonus = 5.0 if regime == "TRENDING_UP" else 0.0
            confidence = 60 + min(20, ((price - high) / range_size) * 22) + min(10, row["volume_z"] * 3) + regime_bonus
            return StrategySignal(
                self.name,
                Side.LONG,
                _clamp_confidence(confidence),
                price,
                stop,
                target,
                "Price broke above the opening range with volume confirmation.",
            )

        if short_ok and fresh_break_dn and price < low and volume_break and row["ema_9"] < row["ema_21"]:
            stop = min(low + range_size * 0.45, price + atr * 1.1)
            target = price - max(range_size, atr) * 1.6
            regime_bonus = 5.0 if regime == "TRENDING_DOWN" else 0.0
            confidence = 60 + min(20, ((low - price) / range_size) * 22) + min(10, row["volume_z"] * 3) + regime_bonus
            return StrategySignal(
                self.name,
                Side.SHORT,
                _clamp_confidence(confidence),
                price,
                stop,
                target,
                "Price broke below the opening range with volume confirmation.",
            )

        return StrategySignal(
            self.name, Side.WAIT, 12, None, None, None, "No confirmed opening range breakout."
        )


class VolatilityCompressionBreakoutStrategy(BaseStrategy):
    name = "volatility_compression_breakout"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 70:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 70 candles.")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(row["close"])
        atr = float(row["atr_14"])
        recent_high = float(df["high"].iloc[-21:-1].max())
        recent_low = float(df["low"].iloc[-21:-1].min())
        compression = df["range_pct"].iloc[-20:-3].median() < df["range_pct"].iloc[-65:-20].median() * 0.75
        volume_expansion = row["volume_z"] > 0.8
        # Fresh-cross: prev close must still be inside the range so the
        # signal fires once per breakout, not continuously afterwards.
        fresh_up = float(prev["close"]) <= recent_high
        fresh_dn = float(prev["close"]) >= recent_low

        if compression and volume_expansion and fresh_up and price > recent_high and row["close"] > prev["close"]:
            stop, target = _plan_levels(Side.LONG, price, atr, reward_multiple=2.0)
            confidence = 63 + min(22, row["volume_z"] * 3.5)
            return StrategySignal(
                self.name,
                Side.LONG,
                _clamp_confidence(confidence),
                price,
                stop,
                target,
                "Volatility compressed, then price expanded above the recent range.",
            )

        if compression and volume_expansion and fresh_dn and price < recent_low and row["close"] < prev["close"]:
            stop, target = _plan_levels(Side.SHORT, price, atr, reward_multiple=2.0)
            confidence = 63 + min(22, row["volume_z"] * 3.5)
            return StrategySignal(
                self.name,
                Side.SHORT,
                _clamp_confidence(confidence),
                price,
                stop,
                target,
                "Volatility compressed, then price expanded below the recent range.",
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No compression breakout.")


def _scalp_levels(side: Side, price: float, atr: float) -> tuple[float, float]:
    """Tighter stop (0.8×ATR) and 2× reward for quick scalp trades."""
    stop_distance = max(atr * 0.8, price * 0.002)
    if side == Side.LONG:
        return price - stop_distance, price + stop_distance * 2.0
    if side == Side.SHORT:
        return price + stop_distance, price - stop_distance * 2.0
    return price, price


class EmaScalpStrategy(BaseStrategy):
    """Fast EMA 5/9 cross scalp aligned with EMA-9/21 trend direction.

    Catches early-stage momentum within an established trend.  The stop is
    tighter than swing strategies (0.8×ATR) so the trade resolves quickly.
    """

    name = "ema_scalp"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 15:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 15 candles.")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(row["close"])
        atr = float(row["atr_14"])

        ema5_now = float(row["ema_5"])
        ema9_now = float(row["ema_9"])
        ema5_prev = float(prev["ema_5"])
        ema9_prev = float(prev["ema_9"])

        volume_ok = row["volume_z"] > -2.0  # relaxed: morning volume below 2d rolling avg
        rsi = float(row["rsi_14"])

        bull_cross = ema5_prev <= ema9_prev and ema5_now > ema9_now
        trend_up = ema9_now > float(row["ema_21"])
        if bull_cross and trend_up and 42 <= rsi <= 68 and volume_ok and row["return_1"] > 0:
            stop, target = _scalp_levels(Side.LONG, price, atr)
            confidence = 62 + min(14, row["volume_z"] * 4) + min(8, max(0, row["return_1"]) * 500)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(confidence), price, stop, target,
                "EMA-5 crossed above EMA-9 with trend aligned upward."
            )

        bear_cross = ema5_prev >= ema9_prev and ema5_now < ema9_now
        trend_down = ema9_now < float(row["ema_21"])
        if bear_cross and trend_down and 32 <= rsi <= 58 and volume_ok and row["return_1"] < 0:
            stop, target = _scalp_levels(Side.SHORT, price, atr)
            confidence = 62 + min(14, row["volume_z"] * 4) + min(8, max(0, -row["return_1"]) * 500)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(confidence), price, stop, target,
                "EMA-5 crossed below EMA-9 with trend aligned downward."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No clean EMA-5/9 cross scalp.")


class VwapBounceScalpStrategy(BaseStrategy):
    """Scalp the first clean candle that recovers across VWAP after a brief touch.

    Unlike VwapMeanReversion (which requires >1.2 ATR stretch), this strategy
    requires only a single candle whose *low* tagged or broke VWAP and whose
    *close* is back above it — confirming the bounce within 1-3 candles.
    """

    name = "vwap_bounce_scalp"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 20:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 20 candles.")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(row["close"])
        atr = float(row["atr_14"])
        vwap_val = float(row["vwap"])
        distance_atr = (price - vwap_val) / max(atr, price * 0.001)

        volume_ok = row["volume_z"] > -1.5  # relaxed: morning volume below 2d rolling avg
        rsi = float(row["rsi_14"])
        bullish_candle = float(row["close"]) > float(row["open"])
        prev_low_tagged = float(prev["low"]) <= vwap_val * 1.001

        if (
            prev_low_tagged
            and bullish_candle
            and price > vwap_val
            and 0.0 <= distance_atr <= 0.9
            and rsi < 62
            and volume_ok
        ):
            stop_lvl = min(float(prev["low"]) - atr * 0.1, price - atr * 0.6)
            target = price + atr * 1.3
            confidence = 60 + min(12, row["volume_z"] * 3.5) + min(10, max(0, 62 - rsi) * 0.25)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(confidence), price, stop_lvl, target,
                "Price dipped to VWAP last candle then recovered — VWAP bounce long."
            )

        bearish_candle = float(row["close"]) < float(row["open"])
        prev_high_tagged = float(prev["high"]) >= vwap_val * 0.999

        if (
            prev_high_tagged
            and bearish_candle
            and price < vwap_val
            and -0.9 <= distance_atr <= 0.0
            and rsi > 38
            and volume_ok
        ):
            stop_lvl = max(float(prev["high"]) + atr * 0.1, price + atr * 0.6)
            target = price - atr * 1.3
            confidence = 60 + min(12, row["volume_z"] * 3.5) + min(10, max(0, rsi - 38) * 0.25)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(confidence), price, stop_lvl, target,
                "Price popped to VWAP last candle then rejected — VWAP rejection short."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No clean VWAP bounce setup.")


class MomentumBurstScalpStrategy(BaseStrategy):
    """Enter on continuation of a strong impulse candle backed by a volume spike.

    Looks for a single candle that moved ≥0.25% with volume z-score >1.0,
    is price-location-aware (above VWAP for longs, below for shorts), and
    has trend alignment.  Tight 0.8×ATR stop means the trade either runs
    quickly or exits fast.
    """

    name = "momentum_burst_scalp"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 20:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 20 candles.")

        row = df.iloc[-1]
        price = float(row["close"])
        atr = float(row["atr_14"])
        rsi = float(row["rsi_14"])
        ret1 = float(row["return_1"])
        vz = float(row["volume_z"])
        above_vwap = price > float(row["vwap"])

        strong_vol = vz > 1.0
        trend_up = float(row["ema_9"]) > float(row["ema_21"])
        trend_dn = float(row["ema_9"]) < float(row["ema_21"])

        if ret1 > 0.0025 and strong_vol and trend_up and above_vwap and 45 <= rsi <= 70:
            stop, target = _scalp_levels(Side.LONG, price, atr)
            confidence = 61 + min(18, ret1 * 3000) + min(10, vz * 2.5)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(confidence), price, stop, target,
                "Strong bullish impulse candle with volume spike above VWAP."
            )

        if ret1 < -0.0025 and strong_vol and trend_dn and not above_vwap and 30 <= rsi <= 55:
            stop, target = _scalp_levels(Side.SHORT, price, atr)
            confidence = 61 + min(18, abs(ret1) * 3000) + min(10, vz * 2.5)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(confidence), price, stop, target,
                "Strong bearish impulse candle with volume spike below VWAP."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No momentum burst setup.")


class SupertrendStrategy(BaseStrategy):
    """Supertrend trend-following strategy (period=7, multiplier=3).

    Popular in NSE intraday trading.  A fresh direction flip (bearish→bullish
    or vice-versa) is the primary entry trigger; continuation entries are
    taken when price pulls back within one ATR of the Supertrend line in the
    same direction.  VWAP and RSI act as secondary filters.
    """

    name = "supertrend"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 20:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 20 candles.")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(row["close"])
        atr_val = float(row["atr_14"])
        curr_dir = float(row["supertrend_dir"])
        prev_dir = float(prev["supertrend_dir"])
        st_line = float(row["supertrend"]) if not pd.isna(row["supertrend"]) else price

        if curr_dir == 0 or prev_dir == 0:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Supertrend not yet established.")

        fresh_bull = prev_dir < 0 and curr_dir > 0
        # Pullback continuation must be an event: the *previous* candle
        # touched near the Supertrend line and the current one closes back
        # in trend direction.  The old "price within 1.2 ATR of the line"
        # zone test stayed true for hours and re-fired continuously.
        bull_touch = float(prev["low"]) <= st_line + atr_val * 0.35
        bull_pullback = (
            curr_dir > 0
            and prev_dir > 0
            and bull_touch
            and float(row["close"]) > float(row["open"])
            and price > st_line
        )
        rsi = float(row["rsi_14"])

        if (fresh_bull or bull_pullback) and price > float(row["vwap"]) and rsi < 72:
            stop = min(st_line - atr_val * 0.1, price - atr_val * 0.75)
            target = price + (price - stop) * 2.0
            base = 70 if fresh_bull else 64
            confidence = base + min(12, float(row["volume_z"]) * 3)
            reason = "Supertrend flipped bullish." if fresh_bull else "Supertrend bullish — pullback to support."
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(confidence), price, stop, target, reason
            )

        fresh_bear = prev_dir > 0 and curr_dir < 0
        bear_touch = float(prev["high"]) >= st_line - atr_val * 0.35
        bear_pullback = (
            curr_dir < 0
            and prev_dir < 0
            and bear_touch
            and float(row["close"]) < float(row["open"])
            and price < st_line
        )

        if (fresh_bear or bear_pullback) and price < float(row["vwap"]) and rsi > 28:
            stop = max(st_line + atr_val * 0.1, price + atr_val * 0.75)
            target = price - (stop - price) * 2.0
            base = 70 if fresh_bear else 64
            confidence = base + min(12, float(row["volume_z"]) * 3)
            reason = "Supertrend flipped bearish." if fresh_bear else "Supertrend bearish — pullback to resistance."
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(confidence), price, stop, target, reason
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No Supertrend signal.")


class BollingerKeltnerSqueezeStrategy(BaseStrategy):
    """TTM-Squeeze style: Bollinger Bands inside Keltner Channels = compression.

    When BB (20, 2σ) contracts inside KC (20, 1.5×ATR), the market is coiling.
    The moment BB expands outside KC the squeeze fires.  Direction is confirmed
    by close vs. Bollinger midband, VWAP side, and a minimum volume expansion.
    """

    name = "bb_kc_squeeze"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 25:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 25 candles.")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(row["close"])
        atr_val = float(row["atr_14"])

        for col in ("bb_upper", "bb_lower", "bb_mid", "kc_upper", "kc_lower"):
            if pd.isna(row[col]) or pd.isna(prev[col]):
                return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Indicator warmup.")

        bb_u, bb_l = float(row["bb_upper"]), float(row["bb_lower"])
        kc_u, kc_l = float(row["kc_upper"]), float(row["kc_lower"])
        prev_squeeze = float(prev["bb_upper"]) < float(prev["kc_upper"]) and float(prev["bb_lower"]) > float(prev["kc_lower"])
        curr_released_up = prev_squeeze and bb_u >= kc_u
        curr_released_dn = prev_squeeze and bb_l <= kc_l
        above_mid = price > float(row["bb_mid"])
        rsi = float(row["rsi_14"])
        vz = float(row["volume_z"])
        volume_ok = vz > 0.3

        if curr_released_up and above_mid and rsi > 45 and volume_ok:
            sd = max(atr_val * 0.85, price * 0.002)
            stop, target = price - sd, price + sd * 2.1
            confidence = 65 + min(15, vz * 4)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(confidence), price, stop, target,
                "BB-KC squeeze released upward — volatility expansion after compression."
            )

        if curr_released_dn and not above_mid and rsi < 55 and volume_ok:
            sd = max(atr_val * 0.85, price * 0.002)
            stop, target = price + sd, price - sd * 2.1
            confidence = 65 + min(15, vz * 4)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(confidence), price, stop, target,
                "BB-KC squeeze released downward — volatility expansion after compression."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No squeeze breakout.")


class RsiDivergenceScalpStrategy(BaseStrategy):
    """RSI divergence: price makes a new swing extreme but RSI fails to confirm.

    Bullish divergence — price at/near the lookback low while RSI is already
    higher than it was at that low.  Bearish divergence — the mirror.
    Requires a confirming candle body in the expected direction.
    """

    name = "rsi_divergence"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 20:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 20 candles.")

        row = df.iloc[-1]
        price = float(row["close"])
        atr_val = float(row["atr_14"])
        rsi_now = float(row["rsi_14"])
        vz = float(row["volume_z"])
        regime = str(row.get("regime", "RANGING")) if "regime" in df.columns else "RANGING"
        lookback = min(15, len(df) - 2)
        window = df.iloc[-(lookback + 1):-1]

        # Bullish divergence — requires stronger RSI gap in downtrends, blocked entirely in strong down
        low_idx = window["close"].idxmin()
        price_low = float(window.loc[low_idx, "close"])
        rsi_at_low = float(window.loc[low_idx, "rsi_14"])
        # In TRENDING_DOWN, divergence must be much stronger (5pt gap vs 3pt) to overcome trend
        min_rsi_gap = 5.0 if regime == "TRENDING_DOWN" else 3.0
        if (
            price <= price_low * 1.005
            and rsi_now > rsi_at_low + min_rsi_gap
            and rsi_now < 45                  # tighter RSI ceiling (was 48)
            and float(row["close"]) > float(row["open"])
            and vz > 0.2                      # require volume (was 0.0)
            and (regime != "TRENDING_DOWN" or rsi_now < 35)
        ):
            sd = max(atr_val * 0.85, price * 0.002)
            stop, target = price - sd, price + sd * 1.9
            conf = 63 + min(14, (rsi_now - rsi_at_low) * 0.7) + min(7, vz * 2)
            # Penalty for going against a downtrend — divergences fail more often
            if regime == "TRENDING_DOWN":
                conf = max(0, conf - 8)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(conf), price, stop, target,
                "Bullish RSI divergence: price at swing low but RSI trending up."
            )

        # Bearish divergence — same logic mirrored
        high_idx = window["close"].idxmax()
        price_high = float(window.loc[high_idx, "close"])
        rsi_at_high = float(window.loc[high_idx, "rsi_14"])
        min_rsi_gap_bear = 5.0 if regime == "TRENDING_UP" else 3.0
        if (
            price >= price_high * 0.995
            and rsi_now < rsi_at_high - min_rsi_gap_bear
            and rsi_now > 55                  # tighter RSI floor (was 52)
            and float(row["close"]) < float(row["open"])
            and vz > 0.2
            and (regime != "TRENDING_UP" or rsi_now > 65)
        ):
            sd = max(atr_val * 0.85, price * 0.002)
            stop, target = price + sd, price - sd * 1.9
            conf = 63 + min(14, (rsi_at_high - rsi_now) * 0.7) + min(7, vz * 2)
            if regime == "TRENDING_UP":
                conf = max(0, conf - 8)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(conf), price, stop, target,
                "Bearish RSI divergence: price at swing high but RSI trending down."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No RSI divergence setup.")


class FairValueGapScalpStrategy(BaseStrategy):
    """ICT Fair Value Gap (FVG) / imbalance fill strategy.

    A bullish FVG forms when candle[A].high < candle[C].low across three
    consecutive candles — a price zone with no overlap, implying an imbalance.
    Price tends to retrace into these gaps.  We enter when price re-enters
    the gap zone and bet on continuation in the impulse direction.

    Bearish FVG is the mirror.  Gap must be at least 0.3×ATR to be tradable.
    Searches up to 10 bars back for unmitigated gaps.
    """

    name = "fair_value_gap"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 12:
            return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "Need at least 12 candles.")

        row = df.iloc[-1]
        prev_close = float(df.iloc[-2]["close"])
        price = float(row["close"])
        atr_val = float(row["atr_14"])
        rsi_now = float(row["rsi_14"])
        vz = float(row["volume_z"])
        # Require larger gap (0.5×ATR) and volume confirmation — filters out shallow noise gaps
        min_gap = atr_val * 0.5
        regime = str(row.get("regime", "RANGING")) if "regime" in df.columns else "RANGING"
        vol_ok = vz > 0.2   # require above-average volume at entry

        for k in range(3, min(11, len(df) - 1)):
            c_a = df.iloc[-(k + 2)]   # candle before impulse
            c_c = df.iloc[-k]         # candle after impulse

            # Bullish FVG — blocked in TRENDING_DOWN (gap fills fail against downtrend).
            # Fires only on first entry into the gap (prev close outside it),
            # otherwise it re-signals every minute price sits in the zone.
            gap_lo = float(c_a["high"])
            gap_hi = float(c_c["low"])
            if (gap_hi > gap_lo + min_gap and gap_lo <= price <= gap_hi
                    and not (gap_lo <= prev_close <= gap_hi)
                    and rsi_now < 60 and vol_ok
                    and regime != "TRENDING_DOWN"):
                stop = gap_lo - atr_val * 0.25
                target = price + (price - stop) * 2.0
                conf = 63 + min(14, (gap_hi - gap_lo) / atr_val * 6)
                if regime == "TRENDING_UP":
                    conf = min(100, conf + 6)   # bonus for trend alignment
                return StrategySignal(
                    self.name, Side.LONG, _clamp_confidence(conf), price, stop, target,
                    f"Price filling bullish Fair Value Gap {gap_lo:.2f}–{gap_hi:.2f}."
                )

            # Bearish FVG — blocked in TRENDING_UP
            gap_hi_b = float(c_a["low"])
            gap_lo_b = float(c_c["high"])
            if (gap_hi_b > gap_lo_b + min_gap and gap_lo_b <= price <= gap_hi_b
                    and not (gap_lo_b <= prev_close <= gap_hi_b)
                    and rsi_now > 40 and vol_ok
                    and regime != "TRENDING_UP"):
                stop = gap_hi_b + atr_val * 0.25
                target = price - (stop - price) * 2.0
                conf = 63 + min(14, (gap_hi_b - gap_lo_b) / atr_val * 6)
                if regime == "TRENDING_DOWN":
                    conf = min(100, conf + 6)
                return StrategySignal(
                    self.name, Side.SHORT, _clamp_confidence(conf), price, stop, target,
                    f"Price filling bearish Fair Value Gap {gap_lo_b:.2f}–{gap_hi_b:.2f}."
                )

        return StrategySignal(self.name, Side.WAIT, 0, None, None, None, "No FVG.")


class LiquiditySweepReversalStrategy(BaseStrategy):
    """Institutional Liquidity Sweep & Displacement Reversal (ICT/SMC).

    Detects when price sweeps beyond a recent 20-bar / session swing extreme
    (triggering resting stop-loss orders) and immediately displaces back inside
    with high close-location value (CLV) and volume expansion.
    """
    name = "liquidity_sweep_reversal"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 25:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Insufficient data.")

        row = df.iloc[-1]
        price = float(row["close"])
        atr_val = float(row.get("atr_14", 0.0))
        if not np.isfinite(atr_val) or atr_val <= 0:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Invalid ATR.")

        lookback = df.iloc[-25:-1]
        prior_low_min = float(lookback["low"].min())
        prior_high_max = float(lookback["high"].max())

        cur_low = float(row["low"])
        cur_high = float(row["high"])
        cur_clv = float(row.get("clv", 0.0))
        cur_vz = float(row.get("volume_z", 0.0))
        cur_lower_wick = float(row.get("lower_wick_ratio", 0.0))
        cur_upper_wick = float(row.get("upper_wick_ratio", 0.0))

        # Bullish Liquidity Sweep
        if cur_low < prior_low_min and price > prior_low_min and (cur_clv >= 0.20 or cur_lower_wick >= 0.35) and cur_vz >= 0.5:
            stop = cur_low - atr_val * 0.25
            target = price + max(atr_val * 3.0, (price - stop) * 2.5)
            conf = 68.0 + min(18.0, max(0.0, cur_clv) * 12.0 + max(0.0, cur_vz) * 3.0 + cur_lower_wick * 10.0)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(conf), price, stop, target,
                f"Bullish liquidity sweep below {prior_low_min:.2f} with strong displacement (CLV={cur_clv:+.2f}, vol_z={cur_vz:+.1f})."
            )

        # Bearish Liquidity Sweep
        if cur_high > prior_high_max and price < prior_high_max and (cur_clv <= -0.20 or cur_upper_wick >= 0.35) and cur_vz >= 0.5:
            stop = cur_high + atr_val * 0.25
            target = price - max(atr_val * 3.0, (stop - price) * 2.5)
            conf = 68.0 + min(18.0, max(0.0, -cur_clv) * 12.0 + max(0.0, cur_vz) * 3.0 + cur_upper_wick * 10.0)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(conf), price, stop, target,
                f"Bearish liquidity sweep above {prior_high_max:.2f} with strong displacement (CLV={cur_clv:+.2f}, vol_z={cur_vz:+.1f})."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No liquidity sweep detected.")


class VolumeProfilePocStrategy(BaseStrategy):
    """Intraday Volume Profile Point of Control (POC) & Value Area (VAH/VAL) Trading.

    Trades rejections from Value Area Extremes (VAL/VAH) back toward POC, or
    breakouts/acceptance away from POC in trending conditions.
    """
    name = "volume_profile_poc"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 20 or "poc" not in df or "vah" not in df or "val" not in df:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Volume profile data not available.")

        row = df.iloc[-1]
        price = float(row["close"])
        poc = float(row["poc"])
        vah = float(row["vah"])
        val = float(row["val"])
        atr_val = float(row.get("atr_14", 0.0))

        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(poc):
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Invalid ATR or POC.")

        clv_val = float(row.get("clv", 0.0))
        vz = float(row.get("volume_z", 0.0))
        regime = str(row.get("regime", "RANGING"))

        # Setup 1: VAL Rejection Long
        if price <= val + 0.35 * atr_val and clv_val >= 0.25 and vz >= 0.0:
            stop = min(val, float(row["low"])) - 0.3 * atr_val
            target = max(poc, price + 2.0 * atr_val)
            conf = 66.0 + min(18.0, max(0.0, clv_val) * 12.0 + max(0.0, vz) * 3.0)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(conf), price, stop, target,
                f"Long rejection from Value Area Low {val:.2f} towards POC {poc:.2f}."
            )

        # Setup 2: VAH Rejection Short
        if price >= vah - 0.35 * atr_val and clv_val <= -0.25 and vz >= 0.0:
            stop = max(vah, float(row["high"])) + 0.3 * atr_val
            target = min(poc, price - 2.0 * atr_val)
            conf = 66.0 + min(18.0, max(0.0, -clv_val) * 12.0 + max(0.0, vz) * 3.0)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(conf), price, stop, target,
                f"Short rejection from Value Area High {vah:.2f} towards POC {poc:.2f}."
            )

        # Setup 3: Trending POC Breakout / Acceptance
        if regime in ("TRENDING_UP", "HIGH_VOL") and price > poc + 0.3 * atr_val and vz >= 1.2 and clv_val >= 0.3:
            stop = poc - 0.2 * atr_val
            target = price + max(vah - price, 2.5 * atr_val)
            conf = 68.0 + min(16.0, max(0.0, vz) * 4.0)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(conf), price, stop, target,
                f"Bullish POC acceptance breakout above {poc:.2f}."
            )

        if regime in ("TRENDING_DOWN", "HIGH_VOL") and price < poc - 0.3 * atr_val and vz >= 1.2 and clv_val <= -0.3:
            stop = poc + 0.2 * atr_val
            target = price - max(price - val, 2.5 * atr_val)
            conf = 68.0 + min(16.0, max(0.0, vz) * 4.0)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(conf), price, stop, target,
                f"Bearish POC acceptance breakdown below {poc:.2f}."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No Volume Profile setup.")


class OpeningDriveMomentumStrategy(BaseStrategy):
    """Institutional Opening Drive Momentum.

    Fires during the first 90 minutes of the session on high volume expansion
    with decisive candlestick body driving away from session open & VWAP.
    """
    name = "opening_drive_momentum"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 6:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Insufficient data.")

        # Check session time (first 90 min of session: e.g. 09:15 to 10:45 IST)
        if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
            ts = df.index[-1]
            try:
                ts = ts.tz_convert(_IST) if ts.tzinfo else ts.tz_localize(_IST)
                m = ts.hour * 60 + ts.minute
                if m < 555 or m > 645:
                    return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Outside opening drive window.")
            except Exception:
                pass

        row = df.iloc[-1]
        price = float(row["close"])
        vwap_val = float(row.get("vwap", price))
        atr_val = float(row.get("atr_14", 0.0))
        vz = float(row.get("volume_z", 0.0))
        body_r = float(row.get("body_ratio", 0.0))
        streak_val = float(row.get("streak", 0.0))

        if not np.isfinite(atr_val) or atr_val <= 0:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Invalid ATR.")

        ema9 = float(row.get("ema_9", price))
        if vz >= 1.5 and body_r >= 0.50 and price > vwap_val and price > ema9 and streak_val >= 1:
            stop = max(vwap_val - 0.2 * atr_val, price - 1.8 * atr_val)
            target = price + (price - stop) * 2.2
            conf = 70.0 + min(18.0, max(0.0, vz) * 4.0 + body_r * 8.0)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(conf), price, stop, target,
                f"Bullish opening drive impulse (vol_z={vz:+.1f}, body={body_r:.0%})."
            )

        if vz >= 1.5 and body_r >= 0.50 and price < vwap_val and price < ema9 and streak_val <= -1:
            stop = min(vwap_val + 0.2 * atr_val, price + 1.8 * atr_val)
            target = price - (stop - price) * 2.2
            conf = 70.0 + min(18.0, max(0.0, vz) * 4.0 + body_r * 8.0)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(conf), price, stop, target,
                f"Bearish opening drive impulse (vol_z={vz:+.1f}, body={body_r:.0%})."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No opening drive setup.")


class VwapBandMeanReversionStrategy(BaseStrategy):
    """Institutional VWAP 2.0-Sigma Volatility Band Mean Reversion."""
    name = "vwap_band_mean_reversion"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 15:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Insufficient data.")

        row = df.iloc[-1]
        price = float(row["close"])
        vwap_val = float(row.get("vwap", price))
        vu2 = float(row.get("vwap_u2", price * 1.02))
        vl2 = float(row.get("vwap_l2", price * 0.98))
        atr_val = float(row.get("atr_14", 0.0))
        lower_wick = float(row.get("lower_wick_ratio", 0.0))
        upper_wick = float(row.get("upper_wick_ratio", 0.0))
        clv_val = float(row.get("clv", 0.0))

        if not np.isfinite(atr_val) or atr_val <= 0:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Invalid ATR.")

        # Oversold below Lower Band 2 with bullish wick rejection
        if price <= vl2 and (lower_wick >= 0.25 or clv_val >= 0.20):
            stop = min(float(row["low"]) - 0.2 * atr_val, price - 1.8 * atr_val)
            target = vwap_val
            if target > price + 1.0 * atr_val:
                conf = 72.0 + min(16.0, lower_wick * 20.0 + max(0.0, clv_val) * 10.0)
                return StrategySignal(
                    self.name, Side.LONG, _clamp_confidence(conf), price, stop, target,
                    f"Exhaustion at VWAP -2σ band with bullish rejection wick ({lower_wick:.0%})."
                )

        # Overbought above Upper Band 2 with bearish wick rejection
        if price >= vu2 and (upper_wick >= 0.25 or clv_val <= -0.20):
            stop = max(float(row["high"]) + 0.2 * atr_val, price + 1.8 * atr_val)
            target = vwap_val
            if target < price - 1.0 * atr_val:
                conf = 72.0 + min(16.0, upper_wick * 20.0 + max(0.0, -clv_val) * 10.0)
                return StrategySignal(
                    self.name, Side.SHORT, _clamp_confidence(conf), price, stop, target,
                    f"Exhaustion at VWAP +2σ band with bearish rejection wick ({upper_wick:.0%})."
                )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No VWAP Band exhaustion setup.")


class OpeningRangeBreakoutExpansionStrategy(BaseStrategy):
    """Institutional Opening Range 15-Minute Breakout & Volatility Expansion."""
    name = "opening_range_expansion"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 6:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Insufficient data.")

        if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No session timestamps.")

        idx = df.index
        day_mask = idx.normalize() == idx[-1].normalize()
        sess_df = df[day_mask]
        if len(sess_df) < 4:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Opening range forming.")

        or_df = sess_df.iloc[:3]
        or_high = float(or_df["high"].max())
        or_low = float(or_df["low"].min())
        or_range = or_high - or_low

        row = df.iloc[-1]
        price = float(row["close"])
        atr_val = float(row.get("atr_14", 0.0))
        vz = float(row.get("volume_z", 0.0))
        clv_val = float(row.get("clv", 0.0))

        if not np.isfinite(atr_val) or atr_val <= 0 or or_range <= 0:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Invalid ATR or OR range.")

        # Bullish ORB expansion
        if price > or_high and vz >= 0.8 and clv_val >= 0.10:
            stop = max(or_high - 0.5 * or_range, price - 1.8 * atr_val)
            target = price + (price - stop) * 2.2
            conf = 70.0 + min(18.0, max(0.0, vz) * 4.0 + clv_val * 10.0)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(conf), price, stop, target,
                f"15m Opening Range expansion breakout (vol_z={vz:+.1f}, clv={clv_val:+.2f})."
            )

        # Bearish ORB expansion
        if price < or_low and vz >= 0.8 and clv_val <= -0.10:
            stop = min(or_low + 0.5 * or_range, price + 1.8 * atr_val)
            target = price - (stop - price) * 2.2
            conf = 70.0 + min(18.0, max(0.0, vz) * 4.0 + (-clv_val) * 10.0)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(conf), price, stop, target,
                f"15m Opening Range expansion breakdown (vol_z={vz:+.1f}, clv={clv_val:+.2f})."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Within opening range.")


class MultiTimeframeTrendAlignmentStrategy(BaseStrategy):
    """Institutional Multi-Horizon EMA Trend Alignment."""
    name = "mtf_trend_alignment"

    def evaluate(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 25:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Insufficient data.")

        row = df.iloc[-1]
        price = float(row["close"])
        ema9 = float(row.get("ema_9", price))
        ema21 = float(row.get("ema_21", price))
        ema50 = float(row.get("ema_50", price))
        vwap_val = float(row.get("vwap", price))
        adx_val = float(row.get("adx_14", 0.0))
        plus_di = float(row.get("plus_di_14", 0.0))
        minus_di = float(row.get("minus_di_14", 0.0))
        atr_val = float(row.get("atr_14", 0.0))

        if not np.isfinite(atr_val) or atr_val <= 0:
            return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "Invalid ATR.")

        # Bullish MTF Alignment (with anti-exhaustion filter: price cannot be overextended > 0.85 ATR from EMA9)
        if (
            price > ema9 > ema21 > ema50
            and price > vwap_val
            and adx_val >= 20.0
            and plus_di > minus_di
            and (price - ema9) <= 0.85 * atr_val
        ):
            stop = max(ema21 - 0.2 * atr_val, price - 1.8 * atr_val)
            target = price + (price - stop) * 2.5
            conf = 72.0 + min(16.0, (adx_val - 20.0) * 0.8)
            return StrategySignal(
                self.name, Side.LONG, _clamp_confidence(conf), price, stop, target,
                f"Full MTF trend alignment (Price > EMA9 > EMA21 > EMA50 > VWAP, ADX={adx_val:.1f})."
            )

        # Bearish MTF Alignment (with anti-exhaustion filter: price cannot be overextended > 0.85 ATR below EMA9)
        if (
            price < ema9 < ema21 < ema50
            and price < vwap_val
            and adx_val >= 20.0
            and minus_di > plus_di
            and (ema9 - price) <= 0.85 * atr_val
        ):
            stop = min(ema21 + 0.2 * atr_val, price + 1.8 * atr_val)
            target = price - (stop - price) * 2.5
            conf = 72.0 + min(16.0, (adx_val - 20.0) * 0.8)
            return StrategySignal(
                self.name, Side.SHORT, _clamp_confidence(conf), price, stop, target,
                f"Full MTF trend alignment (Price < EMA9 < EMA21 < EMA50 < VWAP, ADX={adx_val:.1f})."
            )

        return StrategySignal(self.name, Side.WAIT, 10, None, None, None, "No MTF alignment setup.")


DEFAULT_STRATEGIES: tuple[BaseStrategy, ...] = (
    TrendContinuationStrategy(),
    VwapMeanReversionStrategy(),
    OpeningRangeBreakoutStrategy(),
    VolatilityCompressionBreakoutStrategy(),
    EmaScalpStrategy(),
    VwapBounceScalpStrategy(),
    MomentumBurstScalpStrategy(),
    SupertrendStrategy(),
    BollingerKeltnerSqueezeStrategy(),
    RsiDivergenceScalpStrategy(),
    FairValueGapScalpStrategy(),
    LiquiditySweepReversalStrategy(),
    VolumeProfilePocStrategy(),
    OpeningDriveMomentumStrategy(),
    VwapBandMeanReversionStrategy(),
    OpeningRangeBreakoutExpansionStrategy(),
    MultiTimeframeTrendAlignmentStrategy(),
)


@dataclass
class EnsembleConfig:
    min_agreeing_votes: int = 1
    min_vote_share: float = 0.50
    min_weighted_confidence: float = 65.0


class VotingSignalEngine:
    def __init__(
        self,
        strategies: tuple[BaseStrategy, ...] = DEFAULT_STRATEGIES,
        config: EnsembleConfig | None = None,
    ) -> None:
        self.strategies = strategies
        self.config = config or EnsembleConfig()

    def analyze(
        self,
        symbol: str,
        raw_df: pd.DataFrame,
        risk_config: RiskConfig,
        strategy_weights: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, TradePlan]:
        df = add_indicators(raw_df)
        plan = self.analyze_precomputed(symbol, df, risk_config, strategy_weights)
        return df, plan

    def analyze_precomputed(
        self,
        symbol: str,
        df: pd.DataFrame,
        risk_config: RiskConfig,
        strategy_weights: dict[str, float] | None = None,
        market_context=None,
    ) -> TradePlan:
        if df.empty:
            return build_trade_plan(
                symbol=symbol,
                side=Side.WAIT,
                confidence=0,
                entry=None,
                stop_loss=None,
                target=None,
                reasons=["No usable candle data."],
                strategy_votes=[],
                timestamp="",
                config=risk_config,
            )

        from nse_intraday_ai.market_context import is_commodity_symbol
        is_commodity = is_commodity_symbol(symbol) and not _LEGACY_ENGINE
        current_regime = str(df["regime"].iloc[-1]) if "regime" in df.columns else "RANGING"
        tod_mult = _commodity_time_multiplier(df) if is_commodity else _time_of_day_multiplier(df)

        weights = strategy_weights or {}
        votes: list[StrategySignal] = []
        for strategy in self.strategies:
            # Opening-range breakout needs a defined session open; on 24-hour
            # futures the "opening range" is an arbitrary midnight slice.
            if is_commodity and strategy.name == "opening_range_breakout":
                continue
            signal = strategy.evaluate(df)
            default_weight = COMMODITY_WEIGHT_DEFAULTS.get(signal.strategy, 1.0) if is_commodity else 1.0
            weight = max(0.2, min(2.5, float(weights.get(signal.strategy, default_weight))))
            # Penalise signals fired in an off-regime (e.g. mean-reversion in a trend)
            affinity = REGIME_AFFINITY.get(signal.strategy)
            if affinity and signal.is_trade and current_regime not in affinity:
                adj_conf = max(0.0, signal.confidence - _REGIME_PENALTY)
            else:
                adj_conf = signal.confidence
            # Apply India VIX strategy modifier (low VIX boosts mean-rev, high VIX
            # boosts breakouts).  Equities only — India VIX says nothing about
            # gold or crude volatility.
            if market_context is not None and signal.is_trade and not is_commodity:
                adj_conf = max(0.0, adj_conf + market_context.vix_strategy_adj(signal.strategy))
            votes.append(
                StrategySignal(
                    signal.strategy,
                    signal.side,
                    adj_conf,
                    signal.entry,
                    signal.stop_loss,
                    signal.target,
                    signal.reason,
                    weight,
                )
            )

        trade_votes = [vote for vote in votes if vote.is_trade and vote.confidence >= 55]
        if not trade_votes:
            return build_trade_plan(
                symbol=symbol,
                side=Side.WAIT,
                confidence=0,
                entry=None,
                stop_loss=None,
                target=None,
                reasons=["No strategy produced a tradable setup."],
                strategy_votes=votes,
                timestamp=_timestamp(df),
                config=risk_config,
            )

        side_counts = Counter(vote.side for vote in trade_votes)
        side = side_counts.most_common(1)[0][0]
        agreeing = [vote for vote in trade_votes if vote.side == side]
        opposing = [vote for vote in trade_votes if vote.side != side]

        total_weight = sum(vote.weight for vote in trade_votes)
        agreeing_weight = sum(vote.weight for vote in agreeing)
        vote_share = agreeing_weight / total_weight if total_weight else 0
        weighted_conf = sum(vote.confidence * vote.weight for vote in agreeing) / max(agreeing_weight, 0.01)
        disagreement_penalty = min(18, len(opposing) * 7 + (1 - vote_share) * 20)
        confidence = _clamp_confidence(weighted_conf - disagreement_penalty)
        confidence = _clamp_confidence(confidence * tod_mult)
        # Apply market context: NIFTY 50 alignment for equities (not commodities),
        # plus symbol-aware adjustments (sector index, global risk, DXY/USDINR).
        if market_context is not None:
            ctx_adj = 0.0
            if not is_commodity:
                ctx_adj += market_context.index_adj(side.value)
            ctx_adj += market_context.extra_symbol_adj(symbol, side.value)
            confidence = _clamp_confidence(confidence + ctx_adj)
        # Scheduled macro event windows (EIA/claims/NFP, NSE expiry) — clock
        # based, so live scans and backtests apply the identical de-rating.
        from nse_intraday_ai.event_risk import event_risk_penalty
        event_ts = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
        event_penalty, event_label = event_risk_penalty(event_ts, symbol)
        if event_penalty:
            confidence = _clamp_confidence(confidence - event_penalty)

        enough_votes = len(agreeing) >= self.config.min_agreeing_votes
        enough_share = vote_share >= self.config.min_vote_share
        enough_conf = confidence >= self.config.min_weighted_confidence

        if not (enough_votes and enough_share and enough_conf):
            reasons = [
                f"Ensemble withheld trade: votes={len(agreeing)}, vote_share={vote_share:.2f}, confidence={confidence:.1f}%.",
                "Multi-strategy voting is useful here because it filters single-strategy false positives.",
            ]
            return build_trade_plan(
                symbol=symbol,
                side=Side.WAIT,
                confidence=confidence,
                entry=None,
                stop_loss=None,
                target=None,
                reasons=reasons,
                strategy_votes=votes,
                timestamp=_timestamp(df),
                config=risk_config,
            )

        entries = np.array([vote.entry for vote in agreeing if vote.entry is not None], dtype=float)
        stops = np.array([vote.stop_loss for vote in agreeing if vote.stop_loss is not None], dtype=float)
        targets = np.array([vote.target for vote in agreeing if vote.target is not None], dtype=float)
        entry = float(np.median(entries))

        if side == Side.LONG:
            stop_loss = float(min(stops))
            target = float(max(targets))
        else:
            stop_loss = float(max(stops))
            target = float(min(targets))

        reasons = [
            f"{len(agreeing)} strategies agree on {side.value}; weighted vote share {vote_share:.2f}.",
            f"Market regime: {current_regime} | Session multiplier: {tod_mult:.2f}x.",
            *([f"Event window: {event_label} (−{event_penalty:.0f} confidence)."] if event_penalty else []),
            *[f"{vote.strategy}: {vote.reason}" for vote in agreeing],
        ]
        return build_trade_plan(
            symbol=symbol,
            side=side,
            confidence=confidence,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            reasons=reasons,
            strategy_votes=votes,
            timestamp=_timestamp(df),
            config=risk_config,
        )
