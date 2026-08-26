"""Intra-week (swing) book: hold one name for a few sessions, not a few bars.

Why this exists as a *separate* engine rather than a longer intraday hold.

The intraday book's problem is arithmetic, not intelligence: the measured gross
edge on a 5-minute signal is a few basis points and the round trip is ~8-13, so
almost nothing clears.  An intra-week hold changes both sides of that ratio.
The hurdle roughly triples — overnight positions settle as DELIVERY, where STT
is 0.1% on *both* legs instead of 0.025% on the sell, taking a ₹2.5L round trip
from ~10 bps to ~30 — but the move being chased goes from single-digit bps to
whole percent.  A median NSE 500 name moves 3-4% over five sessions.  30 bps
against 300 is a fundamentally more forgiving problem than 10 against 5.

What it does *not* change is the hard part, which is predicting direction.  So
this module is built to be falsified: features are computed only from bars that
closed before the decision, entries fill at the next session's open, exits pay
their own leg, and the harness in `scripts/swing_backtest.py` runs walk-forward
splits, a permutation test and a benchmark comparison before any rule is
believed.

Conventions
-----------
* One row per (symbol, date).  All features use data up to and including that
  date's close; the position opens at the NEXT session's open.
* `hold_days` counts sessions, not calendar days.
* Stops and targets are checked against daily high/low, which means a bar that
  touches both is resolved pessimistically (stop first).  Overnight gaps fill
  at the open, which is how a real gap-down through a stop actually executes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nse_intraday_ai.costs import Segment, segment_round_trip_cost

# ── Feature construction ───────────────────────────────────────────────────

MIN_HISTORY = 220          # need 200-day SMA plus warmup


def build_features(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Per-symbol daily feature panel. `frame` is OHLCV indexed by date."""
    if frame.empty or len(frame) < MIN_HISTORY:
        return pd.DataFrame()

    d = frame.sort_index().copy()
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]

    out = pd.DataFrame(index=d.index)
    out["symbol"] = symbol
    out["close"] = c
    out["open_next"] = d["open"].shift(-1)      # the fill price, never a feature

    # Momentum over several horizons (the classic cross-sectional factor).
    for n, label in ((5, "1w"), (20, "4w"), (60, "12w"), (125, "26w")):
        out[f"ret_{label}"] = c / c.shift(n) - 1.0
    # 12-week momentum skipping the most recent week — the standard
    # construction, because the last week is where short-term reversal lives.
    out["ret_12w_skip1w"] = c.shift(5) / c.shift(60) - 1.0

    # Trend location
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    out["above_sma50"] = (c > sma50).astype(float)
    out["above_sma200"] = (c > sma200).astype(float)
    out["dist_sma50"] = c / sma50 - 1.0
    out["dist_sma200"] = c / sma200 - 1.0
    out["sma50_slope"] = sma50 / sma50.shift(20) - 1.0

    # Range position
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    out["pct_of_20d_range"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    out["dist_20d_high"] = c / hi20 - 1.0
    out["dist_52w_high"] = c / h.rolling(250).max() - 1.0

    # Short-term reversal / RSI
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Volatility (ATR as a fraction of price — the sizing input)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr"] = atr
    out["atr_pct"] = atr / c
    out["vol_20d"] = c.pct_change().rolling(20).std()

    # Liquidity and participation
    turnover = c * v
    out["turnover_20d"] = turnover.rolling(20).mean()
    out["volume_ratio"] = v.rolling(5).mean() / v.rolling(60).mean().replace(0, np.nan)

    out["weekday"] = out.index.weekday
    return out.dropna(subset=["ret_26w", "atr", "open_next"])


def build_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-symbol features and add cross-sectional ranks per date."""
    parts = [f for f in (build_features(fr, s) for s, fr in frames.items()) if not f.empty]
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts).reset_index().rename(columns={"index": "date", "ts": "date"})
    panel["date"] = pd.to_datetime(panel["date"])

    # Cross-sectional percentile ranks — a stock is only "strong" relative to
    # what else was available to buy that day.
    for col in ("ret_1w", "ret_4w", "ret_12w", "ret_12w_skip1w", "ret_26w",
                "rsi_14", "atr_pct", "volume_ratio", "dist_52w_high",
                "pct_of_20d_range", "turnover_20d"):
        panel[f"r_{col}"] = panel.groupby("date")[col].rank(pct=True)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


# ── Backtest ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SwingConfig:
    capital: float = 10_00_000.0
    positions: int = 1              # how many names held at once
    hold_days: int = 5              # sessions; 5 = enter Mon, exit Fri
    stop_atr: float = 2.5
    target_atr: float = 0.0         # 0 = no target, exit on time
    # Size by RISK, not by notional.  Sizing a single slot at ~95% of capital
    # produced max drawdowns above 100% in the first sweep — the book was
    # levered into one name and a 2.5 ATR stop on a volatile future is a
    # double-digit hit to equity.  Quantity is set so that a stop-out costs
    # `risk_per_trade_pct` of capital, then capped by `max_position_pct` so a
    # very low-volatility name cannot silently become a leveraged bet.
    risk_per_trade_pct: float = 2.0
    max_position_pct: float = 40.0
    segment: Segment = Segment.EQUITY_DELIVERY
    slippage_bps_per_leg: float = 5.0   # wider than intraday: daily-bar fills
    entry_weekday: int | None = 0       # 0=Mon; None = any day a slot is free
    min_turnover: float = 5e7           # ₹5 crore/day 20d average
    allow_short: bool = False           # delivery cannot short; futures can


@dataclass
class SwingResult:
    trades: pd.DataFrame
    equity: pd.Series
    config: SwingConfig = field(repr=False, default_factory=SwingConfig)

    @property
    def n(self) -> int:
        return len(self.trades)

    def summary(self) -> dict:
        t = self.trades
        if t.empty:
            return {"trades": 0, "net_pct": 0.0}
        wins = t[t.net_pnl > 0]
        losses = t[t.net_pnl < 0]
        gross_win = wins.net_pnl.sum()
        gross_loss = -losses.net_pnl.sum()
        eq = self.equity
        peak = eq.cummax()
        return {
            "trades": len(t),
            "net_pnl": t.net_pnl.sum(),
            "net_pct": t.net_pnl.sum() / self.config.capital * 100,
            "win_pct": len(wins) / len(t) * 100,
            "avg_bps": t.net_bps.mean(),
            "median_bps": t.net_bps.median(),
            "profit_factor": gross_win / gross_loss if gross_loss else float("inf"),
            "costs": t.costs.sum(),
            "max_dd_pct": ((peak - eq) / peak.replace(0, np.nan)).max() * 100 if len(eq) else 0.0,
        }


def _price_lookup(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {s: f.sort_index() for s, f in frames.items() if not f.empty}


def backtest(
    panel: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    score: "callable",
    config: SwingConfig = SwingConfig(),
) -> SwingResult:
    """Run the swing book.

    `score(rows) -> Series` ranks the symbols available on one date; the top
    `positions` are bought at the next session's open.  A higher score means
    more attractive.  Returning all-NaN for a date means "stay flat".
    """
    prices = _price_lookup(frames)
    if panel.empty or not prices:
        return SwingResult(pd.DataFrame(), pd.Series(dtype=float), config)

    eligible = panel[panel["turnover_20d"] >= config.min_turnover]
    if config.entry_weekday is not None:
        eligible = eligible[eligible["weekday"] == config.entry_weekday]

    trades: list[dict] = []
    busy_until: dict[int, pd.Timestamp] = {}      # slot -> date it frees up

    for date, rows in eligible.groupby("date", sort=True):
        free = [i for i in range(config.positions)
                if busy_until.get(i) is None or busy_until[i] <= date]
        if not free:
            continue
        scores = score(rows)
        if scores is None:
            continue
        ranked = rows.assign(_score=scores).dropna(subset=["_score"])
        ranked = ranked.sort_values("_score", ascending=False)
        for slot, (_, pick) in zip(free, ranked.iterrows()):
            trade = _simulate(pick, prices, config)
            if trade is None:
                continue
            trades.append(trade)
            busy_until[slot] = trade["exit_date"]

    if not trades:
        return SwingResult(pd.DataFrame(), pd.Series(dtype=float), config)

    frame = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    equity = config.capital + frame.set_index("exit_date")["net_pnl"].cumsum()
    return SwingResult(frame, equity, config)


def _position_size(entry: float, atr: float, config: SwingConfig) -> int:
    """Risk-based quantity, capped by notional.

    A stop-out should cost `risk_per_trade_pct` of capital regardless of how
    volatile the name is; the notional cap stops a very quiet symbol from
    turning that into leverage.
    """
    risk_rupees = config.capital * config.risk_per_trade_pct / 100
    per_share_risk = config.stop_atr * atr if config.stop_atr else atr * 2.0
    if per_share_risk <= 0 or entry <= 0:
        return 0
    qty = int(risk_rupees // per_share_risk)
    cap_qty = int(config.capital * config.max_position_pct / 100 // entry)
    return max(0, min(qty, cap_qty))


def _simulate(pick, prices, config: SwingConfig) -> dict | None:
    """Fill at the next open, then walk forward bar by bar to the exit."""
    symbol = pick["symbol"]
    bars = prices.get(symbol)
    if bars is None:
        return None
    after = bars.loc[bars.index > pick["date"]]
    if len(after) < 2:
        return None

    entry = float(after.iloc[0]["open"])
    if not np.isfinite(entry) or entry <= 0:
        return None
    side = 1
    atr = float(pick["atr"])
    qty = _position_size(entry, atr, config)
    if qty <= 0:
        return None
    stop = entry - config.stop_atr * atr if config.stop_atr else None
    target = entry + config.target_atr * atr if config.target_atr else None

    window = after.iloc[: config.hold_days]
    exit_price = float(window.iloc[-1]["close"])
    exit_date = window.index[-1]
    reason = "TIME"
    for ts, bar in window.iterrows():
        # Gap through the level executes at the open, not the level.
        if stop is not None and float(bar["low"]) <= stop:
            exit_price = min(float(bar["open"]), stop)
            exit_date, reason = ts, "STOP"
            break
        if target is not None and float(bar["high"]) >= target:
            exit_price = max(float(bar["open"]), target)
            exit_date, reason = ts, "TARGET"
            break

    gross = (exit_price - entry) * qty * side
    cost = segment_round_trip_cost(
        entry, exit_price, qty,
        segment=config.segment,
        slippage_bps_per_leg=config.slippage_bps_per_leg,
        symbol=symbol,
    ).total
    turnover = entry * qty
    return {
        "entry_date": window.index[0], "exit_date": exit_date, "symbol": symbol,
        "entry": entry, "exit": exit_price, "quantity": qty,
        "held_days": len(window.loc[:exit_date]),
        "gross_pnl": gross, "costs": cost, "net_pnl": gross - cost,
        "net_bps": (gross - cost) / turnover * 1e4,
        "gross_bps": gross / turnover * 1e4,
        "cost_bps": cost / turnover * 1e4,
        "exit_reason": reason, "score": float(pick.get("_score", np.nan)),
    }


# ── Candidate scores ───────────────────────────────────────────────────────
# Deliberately simple and nameable.  Each takes the rows available on one date
# and returns a rank score.  These are hypotheses, not recommendations — see
# scripts/swing_backtest.py for which survive.

def s_momentum_12w(rows):        return rows["r_ret_12w"]
def s_momentum_12w_skip(rows):   return rows["r_ret_12w_skip1w"]
def s_momentum_4w(rows):         return rows["r_ret_4w"]
def s_reversal_1w(rows):         return -rows["r_ret_1w"]
def s_rsi_oversold(rows):        return -rows["r_rsi_14"]
def s_breakout(rows):            return rows["r_pct_of_20d_range"]
def s_random(rows):              return pd.Series(np.random.rand(len(rows)), index=rows.index)


def s_trend_pullback(rows):
    """Uptrend by the slow measures, pulled back by the fast one."""
    trend = rows["above_sma200"] * rows["r_ret_12w_skip1w"]
    return trend * (1 - rows["r_ret_1w"])


def s_momentum_lowvol(rows):
    """12-week momentum, penalised for volatility — the classic quality tilt."""
    return rows["r_ret_12w_skip1w"] * (1 - rows["r_atr_pct"])


SCORES = {
    "momentum_12w": s_momentum_12w,
    "momentum_12w_skip1w": s_momentum_12w_skip,
    "momentum_4w": s_momentum_4w,
    "reversal_1w": s_reversal_1w,
    "rsi_oversold": s_rsi_oversold,
    "breakout_20d": s_breakout,
    "trend_pullback": s_trend_pullback,
    "momentum_lowvol": s_momentum_lowvol,
    "random": s_random,
}
