from __future__ import annotations

from dataclasses import dataclass

from nse_intraday_ai.models import Side, StrategySignal, TradePlan


@dataclass(frozen=True)
class RiskConfig:
    capital: float = 100_000
    risk_per_trade_pct: float = 0.5
    max_position_pct: float = 25.0
    min_reward_risk: float = 1.5
    min_confidence: float = 78.0
    estimated_cost_bps: float = 15.0
    slippage_bps: float = 3.0

    @property
    def risk_budget(self) -> float:
        return self.capital * self.risk_per_trade_pct / 100

    @property
    def max_position_value(self) -> float:
        return self.capital * self.max_position_pct / 100


def _reward_risk(side: Side, entry: float, stop: float, target: float) -> tuple[float, float, float]:
    if side == Side.LONG:
        risk_per_share = entry - stop
        reward_per_share = target - entry
    elif side == Side.SHORT:
        risk_per_share = stop - entry
        reward_per_share = entry - target
    else:
        return 0, 0, 0

    if risk_per_share <= 0 or reward_per_share <= 0:
        return risk_per_share, reward_per_share, 0
    return risk_per_share, reward_per_share, reward_per_share / risk_per_share


def build_trade_plan(
    *,
    symbol: str,
    side: Side,
    confidence: float,
    entry: float | None,
    stop_loss: float | None,
    target: float | None,
    reasons: list[str],
    strategy_votes: list[StrategySignal],
    timestamp: str,
    config: RiskConfig,
) -> TradePlan:
    reject_reasons: list[str] = []

    if side == Side.WAIT or entry is None or stop_loss is None or target is None:
        return TradePlan(
            symbol=symbol,
            side=Side.WAIT,
            confidence=confidence,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            quantity=0,
            risk_amount=0,
            reward_amount=0,
            reward_risk=0,
            decision="WAIT",
            reasons=tuple(reasons or ["No high-conviction setup."]),
            strategy_votes=tuple(strategy_votes),
            timestamp=timestamp,
        )

    risk_per_share, reward_per_share, rr = _reward_risk(side, entry, stop_loss, target)
    estimated_roundtrip_cost = entry * (config.estimated_cost_bps + config.slippage_bps) / 10_000
    net_reward_per_share = reward_per_share - estimated_roundtrip_cost
    net_rr = net_reward_per_share / risk_per_share if risk_per_share > 0 else 0

    if confidence < config.min_confidence:
        reject_reasons.append(f"Confidence {confidence:.1f}% is below {config.min_confidence:.1f}%.")
    if net_rr < config.min_reward_risk:
        reject_reasons.append(
            f"Net reward/risk {net_rr:.2f} is below {config.min_reward_risk:.2f} after costs."
        )
    if risk_per_share <= 0:
        reject_reasons.append("Invalid stop-loss distance.")

    quantity_by_risk = int(config.risk_budget // max(risk_per_share, 0.01))
    quantity_by_position = int(config.max_position_value // entry)
    quantity = max(0, min(quantity_by_risk, quantity_by_position))
    if quantity <= 0:
        reject_reasons.append("Capital/risk settings produce zero tradable quantity.")

    decision = "ACTIONABLE" if not reject_reasons else "WAIT"
    risk_amount = quantity * risk_per_share
    reward_amount = quantity * net_reward_per_share
    return TradePlan(
        symbol=symbol,
        side=side if decision == "ACTIONABLE" else Side.WAIT,
        confidence=confidence,
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        quantity=quantity if decision == "ACTIONABLE" else 0,
        risk_amount=max(0, risk_amount),
        reward_amount=max(0, reward_amount),
        reward_risk=max(0, net_rr),
        decision=decision,
        reasons=tuple([*reasons, *reject_reasons]),
        strategy_votes=tuple(strategy_votes),
        timestamp=timestamp,
    )
