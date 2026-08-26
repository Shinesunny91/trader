"""Entry-quality and macro-alignment gating — the measured replacement for
the engine's confidence gate.

What the 280K-event study (scripts/entry_timing_study.py, 49 sessions, 505
symbols) actually found:

* The engine's own **confidence does not rank outcomes**: corr(confidence,
  forward 6-bar return) = +0.009, and the ≥85% bucket was the *worst* of all
  (−3.58 bps).  Out of sample, `conf>=70` — the shipped gate — selected signals
  averaging −1.20 bps against an all-signal baseline of −1.03.  Requiring
  multiple agreeing strategies was worse still (−1.73 bps).  Both gates were
  selecting slightly *below-average* signals while cutting volume by 90%.
* What does rank, in both halves of the window: **volume expansion** (vol_z)
  and **impulse size** (run6, the 6-bar move in ATR units).  Signals with
  vol_z > 3 or run6 > 2.5 were positive in-sample *and* out-of-sample — the
  only entry features that were.
* Signals fired in the mushy middle — below-average volume, no impulse — are
  the bleed: 166K of 280K events had vol_z ≤ 0 and averaged −1.20 bps.

And on the macro side (scripts/macro_feature_study.py):

* **Foreign equity markets do not help at this horizon.** S&P futures, Nikkei,
  DAX and Hang Seng all showed large in-sample correlations (+0.09 to +0.10)
  and *nothing* out of sample (−0.003 to +0.0004); DAX, Nikkei and gold
  flipped sign.  They set the gap, and the gap is already in the tape by 09:15.
* **Three macro readings survive**: NIFTY's own 1-hour momentum, USDINR
  (INR strength helps equities), and crude (a spike hurts an oil importer).
  Their combined alignment score beat its own bottom decile in 8 of 10 rolling
  walk-forward weeks, median spread +9 bps, top-decile lift +5.3 bps.

So this module gates on *those* and nothing else.  Adding the features as
confidence bonuses was not attempted: the repo has already rejected that
pattern repeatedly, because a bonus pushes marginal signals over a threshold
instead of removing bad ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Thresholds are the study's out-of-sample survivors, deliberately kept coarse.
# Anything finer would be fitting the 49-session sample rather than the effect.
MIN_VOLUME_Z = 2.0
MIN_IMPULSE_ATR = 1.5
STRONG_VOLUME_Z = 3.0
STRONG_IMPULSE_ATR = 2.5


@dataclass(frozen=True)
class EntryQuality:
    """Measured entry-timing features for one candidate signal."""

    volume_z: float
    impulse_atr: float          # 6-bar move in the trade direction, ATR units
    extension_vwap_atr: float   # distance from session VWAP, ATR units
    rsi: float
    bars_since_extreme: int

    @property
    def has_conviction(self) -> bool:
        """Is there an actual impulse behind this signal, or is it noise?"""
        return self.volume_z >= MIN_VOLUME_Z and self.impulse_atr >= MIN_IMPULSE_ATR

    @property
    def is_strong(self) -> bool:
        return self.volume_z >= STRONG_VOLUME_Z and self.impulse_atr >= STRONG_IMPULSE_ATR

    def describe(self) -> str:
        return (
            f"vol_z={self.volume_z:+.2f} impulse={self.impulse_atr:+.2f}ATR "
            f"vwap_ext={self.extension_vwap_atr:+.2f}ATR rsi={self.rsi:.0f}"
        )


def compute_entry_quality(df: pd.DataFrame, side: str, *, lookback: int = 6) -> EntryQuality | None:
    """Entry-quality features from an indicator frame ending at the signal bar.

    `df` must be the causal history (last row = the closed signal bar) with the
    columns `add_indicators` produces.
    """
    if df is None or len(df) < lookback + 2:
        return None
    row = df.iloc[-1]
    try:
        atr = float(row["atr_14"])
        close = float(row["close"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(atr) or atr <= 0:
        return None

    sign = 1.0 if side == "LONG" else -1.0
    prior = float(df["close"].iloc[-(lookback + 1)])
    impulse = sign * (close - prior) / atr
    extension = sign * (close - float(row["vwap"])) / atr

    # Bars since the session extreme in the trade direction was printed.
    day = df.index.normalize()
    session = df[day == day[-1]]
    extremes = session["high"] if sign > 0 else session["low"]
    if len(extremes):
        offset = int(np.argmax(extremes.to_numpy())) if sign > 0 else int(np.argmin(extremes.to_numpy()))
        bars_since = len(extremes) - 1 - offset
    else:
        bars_since = 0

    return EntryQuality(
        volume_z=float(row.get("volume_z", 0.0)),
        impulse_atr=float(impulse),
        extension_vwap_atr=float(extension),
        rsi=float(row.get("rsi_14", 50.0)),
        bars_since_extreme=bars_since,
    )


# ── Macro alignment ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MacroAlignment:
    """How the three surviving macro readings line up with a trade's direction.

    Each component is the instrument's recent %-change, signed so that positive
    means "supports this trade".  `score` is their standardised sum; callers
    compare it against a rolling quantile fitted on prior data only.
    """

    nifty: float | None = None      # index momentum, signed by side
    inr: float | None = None        # INR strength, signed by side
    crude: float | None = None      # crude weakness, signed by side
    score: float = 0.0
    coverage: int = 0               # how many components were available

    @property
    def is_complete(self) -> bool:
        return self.coverage == 3

    def describe(self) -> str:
        parts = []
        for label, value in (("NIFTY", self.nifty), ("INR", self.inr), ("crude", self.crude)):
            parts.append(f"{label}={value:+.2f}%" if value is not None else f"{label}=n/a")
        return f"macro {self.score:+.2f} [" + " ".join(parts) + "]"


# Standardisation constants: the mean/sd of each component across the 49-session
# study window.  They exist so a live scan can compute the same score the
# walk-forward validated without refitting on every tick; they are scale
# parameters, not fitted predictions, and the study standardised each week
# against prior weeks only.
_NORM = {
    "nifty": (0.0, 0.18),
    "inr": (0.0, 0.09),
    "crude": (0.0, 0.55),
}


def macro_alignment(
    side: str,
    *,
    nifty_change_pct: float | None,
    usdinr_change_pct: float | None,
    crude_change_pct: float | None,
) -> MacroAlignment:
    """Alignment score for one trade side from the three surviving inputs."""
    if side not in ("LONG", "SHORT"):
        return MacroAlignment()
    sign = 1.0 if side == "LONG" else -1.0

    nifty = None if nifty_change_pct is None else nifty_change_pct * sign
    # A weakening rupee (USDINR up) is an equity headwind; a strengthening one
    # is a tailwind — hence the inversion before signing by side.
    inr = None if usdinr_change_pct is None else -usdinr_change_pct * sign
    # India imports ~85% of its crude, so a crude rally is a macro headwind.
    crude = None if crude_change_pct is None else -crude_change_pct * sign

    score = 0.0
    coverage = 0
    for key, value in (("nifty", nifty), ("inr", inr), ("crude", crude)):
        if value is None:
            continue
        mu, sd = _NORM[key]
        score += float(np.clip((value - mu) / sd, -3, 3))
        coverage += 1
    return MacroAlignment(nifty, inr, crude, score, coverage)


# ── The gate ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateConfig:
    """Tunables for `passes_entry_gate`, all defaulted to the study's values."""

    require_conviction: bool = True
    min_macro_score: float = 0.0
    require_macro_coverage: bool = True
    # Trades whose position value would fall below this cannot clear their own
    # flat-fee costs; see costs.min_position_for_cost_target.
    min_position_value: float = 1_40_000.0


@dataclass(frozen=True)
class GateResult:
    allow: bool
    reason: str
    quality: EntryQuality | None = None
    macro: MacroAlignment | None = None

    @property
    def is_strong(self) -> bool:
        return bool(
            self.allow
            and self.quality is not None
            and self.quality.is_strong
            and self.macro is not None
            and self.macro.score >= 1.0
        )


def passes_entry_gate(
    quality: EntryQuality | None,
    macro: MacroAlignment | None,
    *,
    config: GateConfig | None = None,
) -> GateResult:
    """Should this candidate become a live recommendation?

    Two independent conditions, both measured out-of-sample:
      1. an actual impulse (volume expansion + a move worth trading), and
      2. macro alignment at or above the neutral line.
    """
    config = config or GateConfig()
    if quality is None:
        return GateResult(False, "no entry-quality features available")
    if config.require_conviction and not quality.has_conviction:
        return GateResult(
            False,
            f"no conviction: {quality.describe()} "
            f"(need vol_z>={MIN_VOLUME_Z} and impulse>={MIN_IMPULSE_ATR}ATR)",
            quality, macro,
        )
    if macro is None or (config.require_macro_coverage and not macro.is_complete):
        return GateResult(False, "macro panel incomplete", quality, macro)
    if macro.score < config.min_macro_score:
        return GateResult(
            False, f"macro against the trade: {macro.describe()}", quality, macro
        )
    return GateResult(True, f"{quality.describe()} | {macro.describe()}", quality, macro)
