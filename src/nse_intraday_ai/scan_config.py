"""Shared scan configuration — single source of truth for app, daemon and monitor.

The Streamlit app writes its universe's section whenever settings change.
The scanner daemon and monitor loop read the *nse* section before each scan.

Sections are per universe ("nse", "commodity") because their validated gates
are different — the July-2026 studies landed on conf 70 / rr 1.5 / 2 votes /
15 bps for NSE equities but conf 85 / rr 1.5 / 1 vote / share 0.70 / 5 bps
for commodity futures.  A single flat file let whichever scanner tab was open
last clobber the other universe's gates (2026-07-07: the NSE daemon silently
inherited conf>=85 + share>=0.70 from a commodity session and produced zero
signals all morning) — hence the split.
"""
from __future__ import annotations

import json
from pathlib import Path

from nse_intraday_ai.risk import RiskConfig
from nse_intraday_ai.strategies import EnsembleConfig

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "scan_config.json"

_COMMON = {
    "capital": 100_000,
    "risk_per_trade_pct": 0.5,
    "max_position_pct": 25.0,
    "n_symbols": 30,
    # 5m is the validated signal timeframe (1m edges were smaller than costs).
    # At 5m the strategies need 35–70 bars of history to vote, so period must
    # span multiple sessions — "1d" starves them until mid-afternoon.
    "interval": "5m",
    "period": "5d",
    # Meta-label veto (scripts/train_meta_model.py): applied wherever a trained
    # model file exists for the universe; the trainer refuses to write one
    # unless the veto improved the walk-forward portfolio (commodity passed,
    # NSE refused — Jul 2026).
    "meta_veto": True,
    "meta_veto_fraction": 0.5,
    # Entry-quality gate (entry_quality.py), from the Aug-2026 280K-signal
    # study: require an actual impulse behind the signal (volume expansion +
    # a move worth trading) instead of trusting the confidence score, which
    # measured corr +0.009 with forward return and selected *below*-average
    # signals out of sample at the shipped conf>=70 gate.
    "entry_quality_gate": True,
    "min_volume_z": 2.0,
    "min_impulse_atr": 1.5,
    # Macro alignment gate: NIFTY momentum + USDINR + crude, the only three
    # macro readings whose sign survived an out-of-sample split.  Foreign
    # equity indices (ES, Nikkei, DAX, HSI) did not and are not gated on.
    "macro_gate": True,
    "min_macro_score": 0.0,
    # News risk gate (news_feed.py): abstain on fresh high-impact stock news,
    # size down on a risk-off macro tape.  Off by default — it needs a live
    # network fetch per cycle and has no historical archive to validate on yet.
    "news_gate": False,
}

# Gate defaults per universe, from the July-2026 walk-forward studies.
DEFAULTS_BY_UNIVERSE = {
    "nse": {
        **_COMMON,
        "min_confidence": 70.0,
        "min_reward_risk": 1.5,
        "estimated_cost_bps": 15.0,
        "slippage_bps": 3.0,
        "min_agreeing_votes": 2,
        "min_vote_share": 0.50,
    },
    "commodity": {
        **_COMMON,
        # Lowered 85 -> 70 on 2026-08-17.  The 85 gate was calibrated in July on
        # data pooled with an April *backfill*; on live data only (Jul 1 - Aug
        # 17) it does not hold.  On the three contracts that carry the futures
        # edge — RB=F, HO=F, SB=F — conf>=85 cuts 420 candidate signals to 11
        # and turns +43.49 bps into -1.68, while conf>=70 keeps 268 of them at
        # +54.70 bps.  Across the whole 15-symbol universe the gate produced 29
        # signals in seven weeks, which is why the scanner logged nothing at all
        # between 08-13 and 08-17.
        "min_confidence": 70.0,
        "min_reward_risk": 1.5,
        "estimated_cost_bps": 5.0,
        "slippage_bps": 3.0,
        "min_agreeing_votes": 1,
        "min_vote_share": 0.70,
    },
}

# Backward-compat alias (pre-split callers imported DEFAULTS directly).
DEFAULTS = DEFAULTS_BY_UNIVERSE["nse"]


def _read_file(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def load(universe: str = "nse", path: Path | str | None = None) -> dict:
    if universe not in DEFAULTS_BY_UNIVERSE:
        raise ValueError(f"unknown universe {universe!r}; expected one of {sorted(DEFAULTS_BY_UNIVERSE)}")
    p = Path(path) if path else _DEFAULT_PATH
    saved = _read_file(p)
    if universe in saved and isinstance(saved[universe], dict):
        section = saved[universe]
    elif "min_confidence" in saved and not any(k in saved for k in DEFAULTS_BY_UNIVERSE):
        # Legacy flat file (pre-split): ambiguous about which universe wrote
        # it last, so it only seeds the nse section; commodity starts from
        # its own defaults rather than inheriting foreign gates.
        section = saved if universe == "nse" else {}
    else:
        section = {}
    return {**DEFAULTS_BY_UNIVERSE[universe], **section}


def save(cfg: dict, universe: str = "nse", path: Path | str | None = None) -> None:
    if universe not in DEFAULTS_BY_UNIVERSE:
        raise ValueError(f"unknown universe {universe!r}; expected one of {sorted(DEFAULTS_BY_UNIVERSE)}")
    p = Path(path) if path else _DEFAULT_PATH
    existing = _read_file(p)
    if existing and not any(k in existing for k in DEFAULTS_BY_UNIVERSE):
        existing = {"nse": existing}  # migrate legacy flat file in place
    existing[universe] = cfg
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(existing, indent=2))


def to_risk_config(cfg: dict) -> RiskConfig:
    return RiskConfig(
        capital=float(cfg["capital"]),
        risk_per_trade_pct=float(cfg["risk_per_trade_pct"]),
        max_position_pct=float(cfg["max_position_pct"]),
        min_confidence=float(cfg["min_confidence"]),
        min_reward_risk=float(cfg["min_reward_risk"]),
        estimated_cost_bps=float(cfg["estimated_cost_bps"]),
        slippage_bps=float(cfg["slippage_bps"]),
    )


def to_ensemble_config(cfg: dict) -> EnsembleConfig:
    return EnsembleConfig(
        min_agreeing_votes=int(cfg["min_agreeing_votes"]),
        min_vote_share=float(cfg["min_vote_share"]),
        min_weighted_confidence=float(cfg["min_confidence"]),
    )
