"""Shared signal log — single append-only JSONL file written by both the
Streamlit app and the scanner daemon whenever an actionable signal fires.
This makes it easy to compare what the app showed vs what triggered a
desktop notification.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "signals_log.jsonl"


def record(
    *,
    symbol: str,
    side: str,
    confidence: float,
    entry: float,
    stop_loss: float,
    target: float,
    reward_risk: float,
    regime: str | None,
    strategies: str,
    source: str,           # "app" | "daemon" | "monitor"
    path: Path | str | None = None,
) -> None:
    p = Path(path) if path else _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "side": side,
        "confidence": round(confidence, 1),
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "reward_risk": round(reward_risk, 2),
        "regime": regime,
        "strategies": strategies,
        "source": source,
    }
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")


def tail(n: int = 30, path: Path | str | None = None) -> list[dict]:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return []
    lines = p.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-n:]]
