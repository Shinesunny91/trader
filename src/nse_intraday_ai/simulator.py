from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nse_intraday_ai.models import Side, TradePlan

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


@dataclass(frozen=True)
class AccountSummary:
    capital: float
    realized_pnl: float
    open_risk: float
    equity: float
    open_trades: int
    closed_trades: int
    win_rate: float


class PaperTradingStore:
    def __init__(self, db_path: Path | str, starting_capital: float = 100_000) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.starting_capital = float(starting_capital)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    exit_price REAL,
                    pnl REAL,
                    status TEXT NOT NULL,
                    close_reason TEXT,
                    reasons_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_strategies (
                    trade_id INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    side TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    weight REAL NOT NULL,
                    agreed INTEGER NOT NULL,
                    FOREIGN KEY(trade_id) REFERENCES trades(id)
                )
                """
            )

    def open_trade(self, plan: TradePlan) -> int | None:
        if not plan.is_actionable or plan.entry is None or plan.stop_loss is None or plan.target is None:
            return None

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM trades WHERE status = 'OPEN' AND symbol = ?", (plan.symbol,)
            ).fetchone()
            if existing:
                return int(existing["id"])

            cursor = conn.execute(
                """
                INSERT INTO trades (
                    symbol, side, entry, stop_loss, target, quantity, confidence,
                    opened_at, status, reasons_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                """,
                (
                    plan.symbol,
                    plan.side.value,
                    plan.entry,
                    plan.stop_loss,
                    plan.target,
                    plan.quantity,
                    plan.confidence,
                    now_ist(),
                    json.dumps(list(plan.reasons)),
                ),
            )
            trade_id = int(cursor.lastrowid)
            for vote in plan.strategy_votes:
                agreed = int(vote.side == plan.side)
                conn.execute(
                    """
                    INSERT INTO trade_strategies
                    (trade_id, strategy, side, confidence, weight, agreed)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id,
                        vote.strategy,
                        vote.side.value,
                        vote.confidence,
                        vote.weight,
                        agreed,
                    ),
                )
            return trade_id

    def update_open_trades(self, latest_df: pd.DataFrame, symbol: str | None = None) -> list[dict]:
        if latest_df.empty:
            return []

        row = latest_df.iloc[-1]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        closed: list[dict] = []

        with self._connect() as conn:
            if symbol:
                open_trades = conn.execute(
                    "SELECT * FROM trades WHERE status = 'OPEN' AND symbol = ?", (symbol,)
                ).fetchall()
            else:
                open_trades = conn.execute("SELECT * FROM trades WHERE status = 'OPEN'").fetchall()
            for trade in open_trades:
                side = Side(trade["side"])
                exit_price = None
                close_reason = None

                if side == Side.LONG:
                    # Conservative if both are touched in one candle: assume stop first.
                    if low <= trade["stop_loss"]:
                        exit_price = float(trade["stop_loss"])
                        close_reason = "STOP_LOSS"
                    elif high >= trade["target"]:
                        exit_price = float(trade["target"])
                        close_reason = "TARGET"
                    pnl = (exit_price - trade["entry"]) * trade["quantity"] if exit_price else None
                else:
                    if high >= trade["stop_loss"]:
                        exit_price = float(trade["stop_loss"])
                        close_reason = "STOP_LOSS"
                    elif low <= trade["target"]:
                        exit_price = float(trade["target"])
                        close_reason = "TARGET"
                    pnl = (trade["entry"] - exit_price) * trade["quantity"] if exit_price else None

                if exit_price is None:
                    continue

                conn.execute(
                    """
                    UPDATE trades
                    SET status = 'CLOSED', closed_at = ?, exit_price = ?, pnl = ?, close_reason = ?
                    WHERE id = ?
                    """,
                    (now_ist(), exit_price, pnl, close_reason, trade["id"]),
                )
                closed.append(
                    {
                        "id": trade["id"],
                        "symbol": trade["symbol"],
                        "side": trade["side"],
                        "entry": trade["entry"],
                        "exit_price": exit_price,
                        "quantity": trade["quantity"],
                        "pnl": pnl,
                        "close_reason": close_reason,
                    }
                )

        return closed

    def update_open_trades_from_bar(
        self, *, symbol: str, high: float, low: float, close: float | None = None
    ) -> list[dict]:
        closed: list[dict] = []

        with self._connect() as conn:
            open_trades = conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' AND symbol = ?", (symbol,)
            ).fetchall()
            for trade in open_trades:
                side = Side(trade["side"])
                exit_price = None
                close_reason = None

                if side == Side.LONG:
                    if low <= trade["stop_loss"]:
                        exit_price = float(trade["stop_loss"])
                        close_reason = "STOP_LOSS"
                    elif high >= trade["target"]:
                        exit_price = float(trade["target"])
                        close_reason = "TARGET"
                    pnl = (exit_price - trade["entry"]) * trade["quantity"] if exit_price else None
                else:
                    if high >= trade["stop_loss"]:
                        exit_price = float(trade["stop_loss"])
                        close_reason = "STOP_LOSS"
                    elif low <= trade["target"]:
                        exit_price = float(trade["target"])
                        close_reason = "TARGET"
                    pnl = (trade["entry"] - exit_price) * trade["quantity"] if exit_price else None

                if exit_price is None:
                    continue

                conn.execute(
                    """
                    UPDATE trades
                    SET status = 'CLOSED', closed_at = ?, exit_price = ?, pnl = ?, close_reason = ?
                    WHERE id = ?
                    """,
                    (now_ist(), exit_price, pnl, close_reason, trade["id"]),
                )
                closed.append(
                    {
                        "id": trade["id"],
                        "symbol": trade["symbol"],
                        "side": trade["side"],
                        "entry": trade["entry"],
                        "exit_price": exit_price,
                        "quantity": trade["quantity"],
                        "pnl": pnl,
                        "close_reason": close_reason,
                    }
                )

        return closed

    def mark_to_market(self, latest_price_by_symbol: dict[str, float]) -> float:
        with self._connect() as conn:
            open_trades = conn.execute("SELECT * FROM trades WHERE status = 'OPEN'").fetchall()
        unrealized = 0.0
        for trade in open_trades:
            price = latest_price_by_symbol.get(trade["symbol"])
            if price is None:
                continue
            if trade["side"] == Side.LONG.value:
                unrealized += (price - trade["entry"]) * trade["quantity"]
            else:
                unrealized += (trade["entry"] - price) * trade["quantity"]
        return unrealized

    def summary(self, latest_price_by_symbol: dict[str, float] | None = None) -> AccountSummary:
        latest_price_by_symbol = latest_price_by_symbol or {}
        with self._connect() as conn:
            realized = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS pnl FROM trades WHERE status = 'CLOSED'"
            ).fetchone()["pnl"]
            open_rows = conn.execute("SELECT * FROM trades WHERE status = 'OPEN'").fetchall()
            closed_rows = conn.execute("SELECT pnl FROM trades WHERE status = 'CLOSED'").fetchall()

        open_risk = sum(abs(float(row["entry"]) - float(row["stop_loss"])) * row["quantity"] for row in open_rows)
        wins = sum(1 for row in closed_rows if float(row["pnl"]) > 0)
        closed_count = len(closed_rows)
        win_rate = (wins / closed_count * 100) if closed_count else 0.0
        equity = self.starting_capital + float(realized) + self.mark_to_market(latest_price_by_symbol)
        return AccountSummary(
            capital=self.starting_capital,
            realized_pnl=float(realized),
            open_risk=float(open_risk),
            equity=float(equity),
            open_trades=len(open_rows),
            closed_trades=closed_count,
            win_rate=win_rate,
        )

    def trades_frame(self, limit: int = 100) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, side, entry, stop_loss, target, quantity, confidence,
                       opened_at, closed_at, exit_price, pnl, status, close_reason
                FROM trades
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def strategy_weights(self) -> dict[str, float]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts.strategy,
                       COUNT(*) AS n,
                       AVG(CASE WHEN t.pnl > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                       AVG(t.pnl) AS avg_pnl
                FROM trade_strategies ts
                JOIN trades t ON t.id = ts.trade_id
                WHERE t.status = 'CLOSED' AND ts.agreed = 1
                GROUP BY ts.strategy
                """
            ).fetchall()

        weights: dict[str, float] = {}
        for row in rows:
            n = int(row["n"])
            if n < 5:
                weights[row["strategy"]] = 1.0
                continue
            win_rate = float(row["win_rate"] or 0)
            avg_pnl = float(row["avg_pnl"] or 0)
            # Conservative bounded online adaptation from paper-trade outcomes.
            score = 1.0 + (win_rate - 0.5) * 0.8 + max(-0.25, min(0.25, avg_pnl / 1000))
            weights[row["strategy"]] = max(0.55, min(1.6, score))
        return weights

    def strategy_stats_frame(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts.strategy,
                       COUNT(*) AS closed_samples,
                       ROUND(AVG(CASE WHEN t.pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) AS win_rate,
                       ROUND(AVG(t.pnl), 2) AS avg_pnl,
                       ROUND(SUM(t.pnl), 2) AS total_pnl
                FROM trade_strategies ts
                JOIN trades t ON t.id = ts.trade_id
                WHERE t.status = 'CLOSED' AND ts.agreed = 1
                GROUP BY ts.strategy
                ORDER BY total_pnl DESC
                """
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM trade_strategies")
            conn.execute("DELETE FROM trades")
