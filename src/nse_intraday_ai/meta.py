"""The book's self-correcting layer: online expert weights and a drift alarm.

Two things a trading model cannot be trusted to notice about itself:

* **That the simple rule it replaced is now doing better.**  Every session,
  each expert's own top-k book (the gap rule's and the ranker's) is scored on
  the day's real outcome and appended to a paired track record.  The book
  trades the ranker unless, over the trailing 60 sessions, it has been worse
  than the rule with t < -2 — then it falls back to the rule until the
  evidence clears.  Measured 2019-2025, that guard never fired (the ranker's
  rolling 60-day edge over the rule was below zero in <1% of windows), so it
  costs nothing and exists for the breakdown nobody has seen yet.
  Hedge (multiplicative) weights are kept alongside as a diagnostic.  They are
  *not* used to blend: blending the rule into the ranker's ranking measured
  -8 bps/trade, because every unit of weight on the rule dilutes better picks.
* **That the whole book has stopped working.**  A one-sided CUSUM on the live
  daily return against the backtest's expectation raises an alarm once the
  shortfall is too persistent to be bad luck.

State is one small JSON file, so the weights survive restarts and can be read
by the app.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data" / "models" / "meta_state.json"


@dataclass
class MetaState:
    weights: dict[str, float] = field(default_factory=lambda: {"rule": 0.5, "ranker": 0.5})
    eta: float = 0.5                 # learning rate on returns in % (bps / 100)
    floor: float = 0.05              # no expert's weight falls below floor / n
    updated_through: str | None = None
    history: list[dict] = field(default_factory=list)
    # CUSUM drift monitor on the live book's net bps/trade
    expected_bps: float = 20.0       # what the backtest says a session should average
    slack_bps: float = 10.0          # tolerated shortfall before evidence accumulates
    threshold_bps: float = 400.0     # alarm when the accumulated shortfall exceeds this
    cusum: float = 0.0
    alarm: bool = False
    # paired daily record of each expert's own book (net bps/trade)
    track: list[dict] = field(default_factory=list)
    guard_window: int = 60
    guard_t: float = -2.0

    # ── persistence ────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: Path | None = None) -> "MetaState":
        path = STATE if path is None else path      # resolved at call time, not import time
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path | None = None) -> None:
        path = STATE if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    # ── the two mechanisms ─────────────────────────────────────────────────
    def normalised(self, experts: list[str]) -> dict[str, float]:
        w = {e: max(self.weights.get(e, 1.0 / len(experts)), 1e-12) for e in experts}
        total = sum(w.values())
        return {e: v / total for e, v in w.items()}

    def update(self, session: str, expert_returns_bps: dict[str, float],
               book_return_bps: float | None = None) -> None:
        """Fold one session's realised outcomes in (idempotent per session)."""
        if self.updated_through is not None and session <= self.updated_through:
            return
        experts = sorted(set(self.weights) | set(expert_returns_bps))
        w = self.normalised(experts)
        for e in experts:
            r = expert_returns_bps.get(e)
            if r is not None and math.isfinite(r):
                w[e] *= math.exp(self.eta * r / 100.0)
        total = sum(w.values())
        # Mix with uniform: every expert keeps at least floor/n and the weights
        # still sum to one (clipping then renormalising would undercut the floor).
        n = len(experts)
        self.weights = {e: self.floor / n + (1 - self.floor) * v / total for e, v in w.items()}
        if book_return_bps is not None and math.isfinite(book_return_bps):
            self.cusum = max(0.0, self.cusum + (self.expected_bps - self.slack_bps) - book_return_bps)
            self.alarm = self.cusum > self.threshold_bps
        self.updated_through = session
        self.history.append({"session": session, "weights": dict(self.weights),
                             "returns": expert_returns_bps, "book": book_return_bps,
                             "cusum": round(self.cusum, 2)})
        self.history = self.history[-500:]
        self.track.append({"session": session, **{k: float(v) for k, v in expert_returns_bps.items()
                                                   if v is not None and math.isfinite(v)}})
        self.track = self.track[-500:]

    def guard(self, primary: str = "ranker", fallback: str = "rule") -> tuple[str, float | None]:
        """Which expert to trade: `primary` unless it is significantly worse lately.

        Returns (expert, t-stat of primary - fallback over the trailing window;
        None while the record is too short to judge).
        """
        rows = [r for r in self.track if primary in r and fallback in r][-self.guard_window:]
        if len(rows) < self.guard_window:
            return primary, None
        diff = np.array([r[primary] - r[fallback] for r in rows])
        sd = diff.std(ddof=1)
        t = float(diff.mean() / sd * math.sqrt(len(diff))) if sd > 0 else 0.0
        return (fallback if t < self.guard_t else primary), round(t, 2)


def blend(scores: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Weighted sum of cross-sectional percentile ranks (per session)."""
    parts = []
    for name, s in scores.items():
        w = weights.get(name, 0.0)
        if w <= 0 or s is None:
            continue
        sess = s.index.get_level_values("session")
        parts.append(w * s.groupby(sess).rank(pct=True).fillna(0.0))
    if not parts:
        raise ValueError("no expert with a positive weight")
    return sum(parts)


def hedge_replay(scores: dict[str, pd.Series], outcome: pd.Series, *, k: int = 8,
                 universe_mask: pd.Series | None = None, cost_bps: float = 13.0,
                 eta: float = 0.5, floor: float = 0.05) -> tuple[pd.Series, pd.DataFrame]:
    """Backtest the Hedge blend causally: day t is ranked with weights from days < t."""
    names = list(scores)
    frame = pd.DataFrame(scores)
    if universe_mask is not None:
        frame = frame[universe_mask.reindex(frame.index).fillna(False).to_numpy()]
    frame = frame.dropna(how="all")
    sess = frame.index.get_level_values("session")
    ranks = frame.groupby(sess).rank(pct=True)
    y = outcome.reindex(frame.index)
    state = MetaState(weights={n: 1.0 / len(names) for n in names}, eta=eta, floor=floor)
    out, hist = {}, []
    for day, idx in ranks.groupby(level="session").groups.items():
        r = ranks.loc[idx]
        w = state.normalised(names)
        combined = sum(w[n] * r[n].fillna(0.0) for n in names)
        top = combined.nlargest(k).index
        out[day] = float(y.loc[top].mean()) - cost_bps
        own = {}
        for n in names:
            pick = r[n].nlargest(k).index
            own[n] = float(y.loc[pick].mean()) - cost_bps
        hist.append({"session": day, **w})
        state.update(str(day), own)
    return pd.Series(out).sort_index(), pd.DataFrame(hist).set_index("session")
