from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    last_price: float
    timestamp: str
    data_source: str


@dataclass(frozen=True)
class StrategySignal:
    strategy: str
    side: Side
    confidence: float
    entry: float | None
    stop_loss: float | None
    target: float | None
    reason: str
    weight: float = 1.0

    @property
    def is_trade(self) -> bool:
        return self.side in {Side.LONG, Side.SHORT} and self.entry is not None


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    side: Side
    confidence: float
    entry: float | None
    stop_loss: float | None
    target: float | None
    quantity: int
    risk_amount: float
    reward_amount: float
    reward_risk: float
    decision: str
    reasons: tuple[str, ...]
    strategy_votes: tuple[StrategySignal, ...]
    timestamp: str

    @property
    def is_actionable(self) -> bool:
        return self.side in {Side.LONG, Side.SHORT} and self.decision == "ACTIONABLE"
