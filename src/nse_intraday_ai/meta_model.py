"""Meta-label model: a secondary classifier over the ensemble's signals.

The primary system decides *what* to trade (strategy votes + gates); this
model learns *which* of those signals to trust, from replayed candidate
events (López de Prado meta-labeling).  Scores are hit probabilities against
round-trip costs; the veto drops signals scoring below a threshold learned
on the training distribution of gate-passing events.

Validated Jul 2026 on 9 out-of-sample weeks of commodity events: the veto
turned the strict-gate portfolio from −554 to +681 with win rate 32→38% and
max drawdown −27%.  The same veto could NOT rescue NSE equities (≈0 gross
edge − 18 bps costs stays negative at any abstention level), so it defaults
off there — see scan_config.DEFAULTS and README.

Feature extraction is shared between training (CandidateEvent) and live
scans (TradePlan) so offline validation describes exactly what runs live.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from nse_intraday_ai.models import Side, TradePlan

STRATEGIES = [
    "trend_continuation", "vwap_mean_reversion", "opening_range_breakout",
    "volatility_compression_breakout", "ema_scalp", "vwap_bounce_scalp",
    "momentum_burst_scalp", "supertrend", "bb_kc_squeeze", "rsi_divergence",
    "fair_value_gap",
]
REGIMES = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]
FEATURE_NAMES = ["conf", "rr", "votes", "share", "side", "sin_hour", "cos_hour", "dow",
                 "stop_bps", "tgt_bps", "sess_mult", *REGIMES, *STRATEGIES]

_REGIME_RE = re.compile(r"Market regime: (\w+)")
_SESS_RE = re.compile(r"Session multiplier: ([\d.]+)x")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data"


def model_path(universe: str, base_dir: Path | str | None = None) -> Path:
    return Path(base_dir or DEFAULT_MODEL_DIR) / f"meta_model_{universe}.json"


def _regime_session_from_reasons(reasons: tuple[str, ...]) -> tuple[str, float]:
    regime, sess = "RANGING", 1.0
    for reason in reasons:
        match = _REGIME_RE.search(reason)
        if match:
            regime = match.group(1)
        match = _SESS_RE.search(reason)
        if match:
            sess = float(match.group(1))
    return regime, sess


def extract_features(
    *,
    confidence: float,
    reward_risk: float,
    agreeing_count: int,
    vote_share: float,
    side: Side,
    timestamp: datetime,
    entry: float,
    stop_loss: float,
    target: float,
    reasons: tuple[str, ...],
    agreeing_strategies: tuple[str, ...],
) -> list[float]:
    """Entry-time-only features. Must stay in lockstep with FEATURE_NAMES."""
    regime, sess = _regime_session_from_reasons(reasons)
    hour = timestamp.hour + timestamp.minute / 60
    stop_bps = abs(entry - stop_loss) / entry * 1e4
    tgt_bps = abs(target - entry) / entry * 1e4
    row = [
        confidence / 100,
        min(reward_risk, 5) / 5,
        agreeing_count / 4,
        vote_share,
        1.0 if side == Side.LONG else -1.0,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        timestamp.weekday() / 4,
        min(stop_bps, 300) / 300,
        min(tgt_bps, 500) / 500,
        sess,
    ]
    row += [1.0 if regime == r else 0.0 for r in REGIMES]
    row += [1.0 if s in agreeing_strategies else 0.0 for s in STRATEGIES]
    return row


def plan_vote_stats(plan: TradePlan) -> tuple[int, float, tuple[str, ...]]:
    """Agreeing-vote stats for a live plan.

    Mirrors backtest._candidate_vote_stats exactly (kept verifiably in sync
    by tests/test_meta_model.py) so live features match training features.
    """
    trade_votes = [vote for vote in plan.strategy_votes if vote.is_trade and vote.confidence >= 55]
    agreeing = [vote for vote in trade_votes if vote.side == plan.side]
    total_weight = sum(vote.weight for vote in trade_votes)
    agreeing_weight = sum(vote.weight for vote in agreeing)
    vote_share = agreeing_weight / total_weight if total_weight else 0.0
    return len(agreeing), vote_share, tuple(vote.strategy for vote in agreeing)


def features_from_event(event) -> list[float]:
    """Features for a backtest CandidateEvent."""
    return extract_features(
        confidence=event.confidence,
        reward_risk=event.reward_risk,
        agreeing_count=event.agreeing_count,
        vote_share=event.vote_share,
        side=event.side,
        timestamp=event.timestamp,
        entry=event.entry,
        stop_loss=event.stop_loss,
        target=event.target,
        reasons=event.reasons,
        agreeing_strategies=event.agreeing_strategies,
    )


def features_from_plan(plan: TradePlan) -> list[float] | None:
    """Features for a live TradePlan; None if the plan lacks levels."""
    if plan.entry is None or plan.stop_loss is None or plan.target is None:
        return None
    agreeing_count, vote_share, agreeing_strategies = plan_vote_stats(plan)
    ts = datetime.fromisoformat(plan.timestamp)
    return extract_features(
        confidence=plan.confidence,
        reward_risk=plan.reward_risk,
        agreeing_count=agreeing_count,
        vote_share=vote_share,
        side=plan.side,
        timestamp=ts,
        entry=float(plan.entry),
        stop_loss=float(plan.stop_loss),
        target=float(plan.target),
        reasons=plan.reasons,
        agreeing_strategies=agreeing_strategies,
    )


@dataclass
class MetaModel:
    universe: str
    weights: list[float]            # intercept first, then FEATURE_NAMES order
    mu: list[float]
    sd: list[float]
    taus: dict[str, float]          # veto fraction ("0.5") -> score threshold
    cost_bps: float
    trained_at: str
    n_events: int
    feature_names: list[float] = field(default_factory=lambda: list(FEATURE_NAMES))
    validation: dict = field(default_factory=dict)

    def score(self, features: list[float]) -> float:
        x = (np.asarray(features) - np.asarray(self.mu)) / np.asarray(self.sd)
        z = float(self.weights[0] + np.dot(self.weights[1:], x))
        return 1 / (1 + math.exp(-max(-30, min(30, z))))

    def tau(self, veto_fraction: float) -> float:
        key = f"{veto_fraction:.1f}"
        if key not in self.taus:
            raise KeyError(f"veto fraction {key} not in trained taus {sorted(self.taus)}")
        return self.taus[key]

    def passes(self, features: list[float], veto_fraction: float) -> bool:
        return self.score(features) >= self.tau(veto_fraction)

    def save(self, path: Path | str) -> None:
        payload = {
            "universe": self.universe, "weights": self.weights, "mu": self.mu,
            "sd": self.sd, "taus": self.taus, "cost_bps": self.cost_bps,
            "trained_at": self.trained_at, "n_events": self.n_events,
            "feature_names": self.feature_names, "validation": self.validation,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "MetaModel":
        payload = json.loads(Path(path).read_text())
        model = cls(**payload)
        if list(model.feature_names) != list(FEATURE_NAMES):
            raise ValueError(
                f"meta model {path} was trained on different features "
                f"({len(model.feature_names)} vs {len(FEATURE_NAMES)}); retrain it"
            )
        return model


def load_if_available(universe: str, base_dir: Path | str | None = None) -> MetaModel | None:
    path = model_path(universe, base_dir)
    if not path.exists():
        return None
    try:
        return MetaModel.load(path)
    except Exception:
        return None


def train_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                   iters: int = 800, lr: float = 0.5) -> np.ndarray:
    weights = np.zeros(X.shape[1] + 1)
    Xb = np.c_[np.ones(len(X)), X]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ weights, -30, 30)))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[0, weights[1:]] / len(y)
        weights -= lr * grad
    return weights


def train_meta_model(
    events: list,
    *,
    universe: str,
    cost_bps: float,
    gates: tuple[float, float, int],
    veto_fractions: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> MetaModel:
    """Train on ALL given events; taus come from gate-passing training scores."""
    X = np.array([features_from_event(e) for e in events])
    bps = np.array([e.per_share_pnl / e.entry * 1e4 for e in events])
    y = (bps > cost_bps).astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    weights = train_logistic((X - mu) / sd, y)

    z = np.clip(np.c_[np.ones(len(X)), (X - mu) / sd] @ weights, -30, 30)
    prob = 1 / (1 + np.exp(-z))
    conf_g, rr_g, votes_g = gates
    gate_mask = np.array([
        e.confidence >= conf_g and e.reward_risk >= rr_g
        and e.agreeing_count >= votes_g and e.vote_share >= 0.5
        for e in events
    ])
    base = prob[gate_mask] if gate_mask.any() else prob
    taus = {f"{frac:.1f}": float(np.quantile(base, frac)) for frac in veto_fractions}

    return MetaModel(
        universe=universe,
        weights=[float(w) for w in weights],
        mu=[float(v) for v in mu],
        sd=[float(v) for v in sd],
        taus=taus,
        cost_bps=cost_bps,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        n_events=len(events),
    )
