import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_intraday_ai.portfolio_sim import IntradayPortfolioSimulator, SimConfig

IST = "Asia/Kolkata"


def frame(prices, start="2026-08-11 09:15", freq="5min"):
    index = pd.date_range(start, periods=len(prices), freq=freq, tz=IST)
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {"open": prices, "high": prices * 1.001, "low": prices * 0.999,
         "close": prices, "volume": 10_000},
        index=index,
    )


def signal(ts, symbol="AAA.NS", side="LONG", atr=1.0, rank=1.0):
    return pd.DataFrame([{"ts": pd.Timestamp(ts, tz=IST), "symbol": symbol,
                          "side": side, "atr": atr, "rank": rank, "note": ""}])


BASE = SimConfig(
    starting_capital=10_00_000, max_concurrent_positions=3, scale_out_fraction=0.0,
    slippage_bps_per_leg=0.0, cost_target_bps=999.0, trail_atr=0.0,
    breakeven_after_atr=0.0,
)


def test_entry_fills_at_the_next_bars_open_not_the_signal_close():
    """The signal bar's close is not an attainable price."""
    prices = [100.0, 105.0] + [105.0] * 20
    frames = {"AAA.NS": frame(prices)}
    result = IntradayPortfolioSimulator(BASE).run(signal("2026-08-11 09:15"), frames)
    assert not result.trades.empty
    assert result.trades.iloc[0]["entry"] == 105.0   # open of 09:20, not 100.0


def test_target_and_stop_resolve_at_their_levels():
    up = [100.0, 100.0] + list(np.linspace(100, 130, 20))
    result = IntradayPortfolioSimulator(BASE).run(
        signal("2026-08-11 09:15", atr=5.0), {"AAA.NS": frame(up)}
    )
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "TARGET"
    assert trade["gross_pnl"] > 0


def test_stop_is_resolved_before_target_within_a_bar():
    """Intra-bar ordering is unknowable, so the pessimistic branch must win."""
    prices = [100.0, 100.0, 100.0]
    frames = {"AAA.NS": frame(prices)}
    # A bar spanning both levels: high above target, low below stop.
    frames["AAA.NS"].loc[frames["AAA.NS"].index[2], "high"] = 120.0
    frames["AAA.NS"].loc[frames["AAA.NS"].index[2], "low"] = 80.0
    result = IntradayPortfolioSimulator(BASE).run(
        signal("2026-08-11 09:15", atr=5.0), frames
    )
    assert result.trades.iloc[0]["exit_reason"] == "STOP"


def test_positions_are_squared_off_before_the_close():
    """Intraday means intraday — nothing may survive the session."""
    prices = [100.0] * 80        # 09:15 -> ~15:50, never hits stop or target
    config = SimConfig(**{**BASE.__dict__, "max_hold_bars": 999})
    result = IntradayPortfolioSimulator(config).run(
        signal("2026-08-11 09:15", atr=50.0), {"AAA.NS": frame(prices)}
    )
    assert not result.trades.empty
    exit_time = result.trades.iloc[0]["exit_time"]
    assert exit_time.time() >= pd.Timestamp(config.square_off_at).time()
    assert result.trades.iloc[0]["exit_reason"] == "SQUARE_OFF"


def test_max_hold_forces_a_time_exit():
    prices = [100.0] * 40
    config = SimConfig(**{**BASE.__dict__, "max_hold_bars": 6})
    result = IntradayPortfolioSimulator(config).run(
        signal("2026-08-11 09:15", atr=50.0), {"AAA.NS": frame(prices)}
    )
    assert result.trades.iloc[0]["exit_reason"] == "TIME_EXIT"
    assert result.trades.iloc[0]["bars_held"] == 6


def test_concurrency_cap_is_respected():
    prices = [100.0] * 40
    frames = {f"S{i}.NS": frame(prices) for i in range(6)}
    signals = pd.concat([
        signal("2026-08-11 09:15", symbol=f"S{i}.NS", atr=1.0, rank=float(i))
        for i in range(6)
    ])
    config = SimConfig(**{**BASE.__dict__, "max_concurrent_positions": 2})
    result = IntradayPortfolioSimulator(config).run(signals, frames)
    entries = result.trades.drop_duplicates(["symbol", "entry_time"])
    assert len(entries) == 2


def test_higher_rank_wins_the_contested_slot():
    prices = [100.0] * 40
    frames = {"LOW.NS": frame(prices), "HIGH.NS": frame(prices)}
    signals = pd.concat([
        signal("2026-08-11 09:15", symbol="LOW.NS", rank=0.1),
        signal("2026-08-11 09:15", symbol="HIGH.NS", rank=9.9),
    ])
    config = SimConfig(**{**BASE.__dict__, "max_concurrent_positions": 1})
    result = IntradayPortfolioSimulator(config).run(signals, frames)
    assert result.trades.iloc[0]["symbol"] == "HIGH.NS"


def test_no_new_entries_after_the_cutoff():
    prices = [100.0] * 80
    result = IntradayPortfolioSimulator(BASE).run(
        signal("2026-08-11 15:00"), {"AAA.NS": frame(prices)}
    )
    assert result.trades.empty


def test_undersized_positions_are_refused_rather_than_taken_small():
    """Below the cost floor a trade cannot clear its own brokerage."""
    prices = [100.0] * 40
    config = SimConfig(**{**BASE.__dict__, "cost_target_bps": 12.0,
                          "risk_per_trade_pct": 0.001})
    result = IntradayPortfolioSimulator(config).run(
        signal("2026-08-11 09:15"), {"AAA.NS": frame(prices)}
    )
    assert result.trades.empty


def test_shorts_profit_when_price_falls():
    down = [100.0, 100.0] + list(np.linspace(100, 70, 20))
    result = IntradayPortfolioSimulator(BASE).run(
        signal("2026-08-11 09:15", side="SHORT", atr=5.0), {"AAA.NS": frame(down)}
    )
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "TARGET" and trade["gross_pnl"] > 0


def test_scale_out_produces_two_fills_and_costs_an_extra_leg():
    up = [100.0, 100.0] + list(np.linspace(100, 130, 20))
    config = SimConfig(**{**BASE.__dict__, "scale_out_fraction": 0.5,
                          "scale_out_atr": 0.9})
    result = IntradayPortfolioSimulator(config).run(
        signal("2026-08-11 09:15", atr=5.0), frames := {"AAA.NS": frame(up)}
    )
    reasons = set(result.trades["exit_reason"])
    assert "SCALE_OUT" in reasons
    assert len(result.trades) >= 2

    single = IntradayPortfolioSimulator(BASE).run(
        signal("2026-08-11 09:15", atr=5.0), frames
    )
    assert result.trades["costs"].sum() > single.trades["costs"].sum()


def test_daily_loss_limit_halts_trading():
    """After a bad morning the book must stop, not keep digging."""
    crash = [100.0, 100.0] + [50.0] * 40
    frames = {f"S{i}.NS": frame(crash) for i in range(4)}
    signals = pd.concat([
        signal("2026-08-11 09:15", symbol="S0.NS", atr=1.0),
        signal("2026-08-11 09:15", symbol="S1.NS", atr=1.0),
        signal("2026-08-11 11:00", symbol="S2.NS", atr=1.0),
        signal("2026-08-11 12:00", symbol="S3.NS", atr=1.0),
    ])
    config = SimConfig(**{**BASE.__dict__, "daily_loss_limit_pct": 0.5,
                          "max_concurrent_positions": 4})
    result = IntradayPortfolioSimulator(config).run(signals, frames)
    later = result.trades[result.trades["entry_time"] >= pd.Timestamp("2026-08-11 11:00", tz=IST)]
    assert later.empty, "trading should be halted after the daily loss limit"


def test_empty_signals_returns_unchanged_capital():
    result = IntradayPortfolioSimulator(BASE).run(pd.DataFrame(), {})
    assert result.ending_capital == BASE.starting_capital
    assert "no trades" in result.summary()


def test_daily_trade_cap_counts_entries_not_exits():
    """A '3 trades a day' book must open 3 positions, not 3 exits' worth.

    The cap used to increment on exit fills, so every position opened before
    the first one closed was free — and a scale-out counted twice.
    """
    prices = [100.0] * 60
    frames = {f"S{i}.NS": frame(prices) for i in range(8)}
    signals = pd.concat([
        signal("2026-08-11 09:15", symbol=f"S{i}.NS", atr=1.0, rank=float(8 - i))
        for i in range(8)
    ])
    # max_position_pct kept small so the cap binds before gross exposure does.
    config = SimConfig(**{**BASE.__dict__, "max_trades_per_day": 3,
                          "max_concurrent_positions": 8, "max_position_pct": 5.0,
                          "max_hold_bars": 999})
    result = IntradayPortfolioSimulator(config).run(signals, frames)
    entries = result.trades.drop_duplicates(["symbol", "entry_time"])
    assert len(entries) == 3, f"opened {len(entries)} positions under a 3/day cap"


def test_recency_split_flags_a_decayed_result():
    """A book that made everything early and bled since must say so."""
    import numpy as np

    from nse_intraday_ai.portfolio_sim import SimResult

    daily = pd.DataFrame(
        {"pnl": [5000.0] * 5 + [-2000.0] * 5, "trades": [1] * 10},
        index=pd.date_range("2026-08-01", periods=10).date,
    )
    result = SimResult(pd.DataFrame(), pd.DataFrame(), daily, BASE, 10_00_000, 10_15_000)
    text = result.recency_split()
    assert "earlier" in text and "recent" in text
    assert "entire gain is in the earlier period" in text


def test_recency_split_stays_quiet_when_the_result_holds_up():
    from nse_intraday_ai.portfolio_sim import SimResult

    daily = pd.DataFrame(
        {"pnl": [1000.0] * 10, "trades": [1] * 10},
        index=pd.date_range("2026-08-01", periods=10).date,
    )
    result = SimResult(pd.DataFrame(), pd.DataFrame(), daily, BASE, 10_00_000, 10_10_000)
    assert "entire gain" not in result.recency_split()


def test_open_positions_are_marked_to_market_when_data_ends():
    """A truncated feed (mid-session) must not silently drop open positions."""
    prices = [100.0, 100.0] + [95.0] * 10     # ends well before square-off
    config = SimConfig(**{**BASE.__dict__, "max_hold_bars": 999})
    result = IntradayPortfolioSimulator(config).run(
        signal("2026-08-11 09:15", atr=50.0), {"AAA.NS": frame(prices)}
    )
    assert not result.trades.empty, "an open position must still be accounted for"
    assert result.trades.iloc[0]["exit_reason"] == "MARK_TO_MARKET"
    assert result.trades.iloc[0]["net_pnl"] < 0, "the open loss must be recognised"


def test_zero_trade_cap_means_unlimited_not_zero():
    """max_trades_per_day == 0 disables the cap; it must not disable the book."""
    prices = [100.0] * 60
    frames = {f"S{i}.NS": frame(prices) for i in range(5)}
    signals = pd.concat([
        signal("2026-08-11 09:15", symbol=f"S{i}.NS", atr=1.0, rank=float(5 - i))
        for i in range(5)
    ])
    config = SimConfig(**{**BASE.__dict__, "max_trades_per_day": 0,
                          "max_concurrent_positions": 5, "max_position_pct": 5.0,
                          "max_hold_bars": 999})
    result = IntradayPortfolioSimulator(config).run(signals, frames)
    entries = result.trades.drop_duplicates(["symbol", "entry_time"])
    assert len(entries) == 5, f"uncapped book opened only {len(entries)} positions"


def test_position_never_survives_into_the_next_session():
    """An MIS position cannot be carried overnight, even if the feed has a gap.

    Regression for 2026-08-13: degraded collection truncated the session, no bar
    existed after the 15:15 square-off, and a clock-only test left INFY and LTM
    open until the next morning's first bar.
    """
    day1 = frame([100.0] * 12, start="2026-08-13 11:45")   # ends 12:40, no 15:15 bar
    day2 = frame([90.0] * 6, start="2026-08-14 09:15")
    combined = pd.concat([day1, day2])
    config = SimConfig(**{**BASE.__dict__, "max_hold_bars": 999, "stop_atr": 99.0,
                          "target_atr": 99.0})
    result = IntradayPortfolioSimulator(config).run(
        signal("2026-08-13 11:45", atr=1.0), {"AAA.NS": combined}
    )
    assert not result.trades.empty
    exit_time = result.trades.iloc[0]["exit_time"]
    entry_time = result.trades.iloc[0]["entry_time"]
    assert exit_time.date() == entry_time.date(), (
        f"position entered {entry_time} exited {exit_time} — carried overnight"
    )


# ── Which positions are still open? (2026-08-18) ────────────────────────────
# `write_tickets` needs this and there is exactly one correct signal for it.
# Getting it wrong is silent in both directions: treat every traded symbol as
# open and the screen never offers another ticket (0 published on 2026-08-17
# against 71 gated signals); treat none as open and it offers a second ticket
# in a name the book already holds.

def test_every_trade_row_carries_an_exit_time_even_when_still_open():
    """So `exit_time.isna()` can never be used to detect an open position."""
    import dataclasses
    flat = [100.0] * 30
    cfg = dataclasses.replace(BASE, max_hold_bars=999, stop_atr=50.0, target_atr=50.0)
    result = IntradayPortfolioSimulator(cfg).run(
        signal("2026-08-11 09:15"), {"AAA.NS": frame(flat)}
    )
    assert not result.trades.empty
    assert result.trades["exit_time"].notna().all(), (
        "exit_time is always stamped — an open position is identified by "
        "exit_reason == MARK_TO_MARKET, not by a null exit_time"
    )
    assert (result.trades["exit_reason"] == "MARK_TO_MARKET").any()


def test_mark_to_market_marks_exactly_the_unclosed_positions():
    """A filled target is closed; a flat position is still open."""
    import dataclasses
    up = [100.0, 100.0] + list(np.linspace(100, 130, 28))
    flat = [100.0] * 30
    frames = {"AAA.NS": frame(up), "BBB.NS": frame(flat)}
    signals = pd.concat([
        signal("2026-08-11 09:15", symbol="AAA.NS", atr=5.0, rank=2.0),
        signal("2026-08-11 09:15", symbol="BBB.NS", atr=5.0, rank=1.0),
    ])
    cfg = dataclasses.replace(BASE, max_hold_bars=999)
    result = IntradayPortfolioSimulator(cfg).run(signals, frames)
    by_symbol = dict(zip(result.trades["symbol"], result.trades["exit_reason"]))
    assert by_symbol.get("AAA.NS") == "TARGET"
    assert by_symbol.get("BBB.NS") == "MARK_TO_MARKET"
    still_open = result.trades[result.trades["exit_reason"] == "MARK_TO_MARKET"]
    assert set(still_open["symbol"]) == {"BBB.NS"}
