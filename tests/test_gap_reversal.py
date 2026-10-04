from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai import gap_reversal as G


def _daily(n_days: int = 40, symbols: tuple[str, ...] = ("A.NS", "B.NS", "C.NS", "D.NS"),
           *, gaps: dict[str, float] | None = None, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Wide daily bars; `gaps` sets each symbol's overnight gap on the LAST day."""
    rng = np.random.default_rng(seed)
    days = list(pd.bdate_range("2026-01-01", periods=n_days).date)
    close = pd.DataFrame(100 + rng.normal(0, 1, (n_days, len(symbols))).cumsum(axis=0),
                         index=days, columns=list(symbols))
    open_ = close.shift(1).fillna(close.iloc[0]) * 1.0
    for sym, g in (gaps or {}).items():
        open_.loc[days[-1], sym] = close.loc[days[-2], sym] * (1 + g)
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = pd.DataFrame(1e6, index=days, columns=list(symbols))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def test_select_ranks_by_yesterdays_gap_using_only_prior_bars():
    daily = _daily(gaps={"A.NS": 0.01, "B.NS": 0.05, "C.NS": -0.02, "D.NS": 0.03})
    last = daily["close"].index[-1]
    session = (pd.Timestamp(last) + pd.offsets.BDay(1)).date()        # not in the data
    picks = G.select(daily, session, G.GapReversalConfig(picks=2, reserves=1, universe_size=10))
    assert [p.symbol for p in picks] == ["B.NS", "D.NS", "A.NS"]
    assert [p.reserve for p in picks] == [False, False, True]
    assert picks[0].gap_prev_pct == pytest.approx(5.0, abs=1e-6)
    assert picks[0].prev_close == pytest.approx(round(daily["close"].loc[last, "B.NS"], 2))

    # The same ranking when the session's own bar exists (a replay): it must
    # not leak into the features.
    replay = {k: v.copy() for k, v in daily.items()}
    for k in replay:
        replay[k].loc[session] = replay[k].iloc[-1] * (5.0 if k != "volume" else 1)
    again = G.select(replay, session, G.GapReversalConfig(picks=2, reserves=1, universe_size=10))
    assert [(p.symbol, p.atr, p.gap_prev_pct) for p in again] == \
           [(p.symbol, p.atr, p.gap_prev_pct) for p in picks]


def test_select_respects_universe_size_and_min_price():
    daily = _daily(gaps={"A.NS": 0.09, "B.NS": 0.05, "C.NS": 0.04, "D.NS": 0.03})
    daily["volume"]["A.NS"] = 10.0                   # illiquid: out of a top-3 universe
    daily["close"]["D.NS"] = daily["close"]["D.NS"] * 0 + 20.0   # below min price
    session = (pd.Timestamp(daily["close"].index[-1]) + pd.offsets.BDay(1)).date()
    # D fails the price filter before the universe is cut, so a top-2 universe
    # by turnover is {B, C} and the illiquid biggest gapper A is not eligible.
    picks = G.select(daily, session, G.GapReversalConfig(picks=3, reserves=0, universe_size=2))
    assert [p.symbol for p in picks] == ["B.NS", "C.NS"]


def test_pick_sizes_and_stop_distance():
    daily = _daily(gaps={"A.NS": 0.02})
    session = (pd.Timestamp(daily["close"].index[-1]) + pd.offsets.BDay(1)).date()
    cfg = G.GapReversalConfig(picks=4, reserves=0, capital=400_000, stop_atr=0.75)
    pick = next(p for p in G.select(daily, session, cfg) if p.symbol == "A.NS")
    assert pick.quantity == int(100_000 // pick.prev_close)
    assert pick.stop_distance == pytest.approx(0.75 * pick.atr, abs=0.01)
    assert "SELL SHORT" in pick.ticket()


def _bars(rows: dict[int, tuple[float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=["open", "high", "low", "close"])
    return frame.sort_index()


def _pick(atr: float = 4.0) -> G.Pick:
    return G.Pick(rank=1, symbol="X.NS", side="SHORT", gap_prev_pct=3.0, prev_close=100.0,
                  atr=atr, stop_distance=3.0, quantity=1000, position_value=1e5, turnover_cr=50)


CFG = G.GapReversalConfig(picks=1, capital=100_000, stop_atr=0.75, slippage_bps_per_leg=0.0)


def test_square_off_at_1515_open_when_stop_never_hit():
    rows = {k: (100.0, 101.0, 98.0, 99.0) for k in range(75)}
    rows[G.SQUARE_OFF_BAR] = (95.0, 96.0, 94.0, 95.0)
    trade = G.simulate_pick(_pick(), _bars(rows), date(2026, 9, 1), CFG)
    assert trade.exit_reason == "SQUARE_OFF" and trade.exit == 95.0 and trade.exit_time == "15:15"
    assert trade.gross == pytest.approx((100.0 - 95.0) * trade.quantity)


def test_stop_on_first_bar_fills_at_stop():
    rows = {k: (100.0, 101.0, 98.0, 99.0) for k in range(75)}
    rows[0] = (100.0, 104.0, 99.0, 103.0)                       # stop = 100 + 3 = 103
    trade = G.simulate_pick(_pick(), _bars(rows), date(2026, 9, 1), CFG)
    assert trade.exit_reason == "STOP" and trade.exit == pytest.approx(103.0)
    assert trade.exit_time == "09:15"


def test_gap_through_stop_fills_at_the_worse_open():
    rows = {k: (100.0, 101.0, 98.0, 99.0) for k in range(75)}
    rows[10] = (106.0, 107.0, 105.0, 106.0)                     # opens above the 103 stop
    trade = G.simulate_pick(_pick(), _bars(rows), date(2026, 9, 1), CFG)
    assert trade.exit_reason == "STOP" and trade.exit == pytest.approx(106.0)


def test_missing_square_off_bar_uses_last_close_before_it():
    rows = {k: (100.0, 101.0, 98.0, 99.0) for k in range(60)}
    rows[59] = (99.0, 100.0, 96.0, 97.0)
    rows[74] = (90.0, 90.0, 90.0, 90.0)                         # after square-off: ignored
    trade = G.simulate_pick(_pick(), _bars(rows), date(2026, 9, 1), CFG)
    assert trade.exit_reason == "LAST_BAR" and trade.exit == pytest.approx(97.0)


def test_no_opening_bar_means_no_trade():
    rows = {k: (100.0, 101.0, 98.0, 99.0) for k in range(1, 75)}
    assert G.simulate_pick(_pick(), _bars(rows), date(2026, 9, 1), CFG) is None


def test_backtest_replaces_a_pick_without_bars_by_the_reserve():
    daily = _daily(gaps={"A.NS": 0.05, "B.NS": 0.04, "C.NS": 0.03, "D.NS": 0.02})
    session = (pd.Timestamp(daily["close"].index[-1]) + pd.offsets.BDay(1)).date()
    flat = _bars({k: (100.0, 100.5, 99.5, 100.0) for k in range(75)})

    def loader(symbol, _session):
        return pd.DataFrame(columns=["open", "high", "low", "close"]) if symbol == "A.NS" else flat

    cfg = G.GapReversalConfig(picks=2, reserves=2, universe_size=10, capital=1_000_000)
    result = G.backtest([session], cfg, daily=daily, bar_loader=loader)
    assert [t.symbol for t in result.trades] == ["B.NS", "C.NS"]
    summary = result.summary()
    assert summary["trades"] == 2 and summary["sessions"] == 1
    assert summary["net_rupees"] < 0                            # flat prices: costs only


@pytest.mark.parametrize("now, expected", [
    ("2026-09-28 08:45", date(2026, 9, 28)),     # Monday morning -> today
    ("2026-09-28 15:40", date(2026, 9, 29)),     # after the close -> tomorrow
    ("2026-10-02 18:00", date(2026, 10, 5)),     # Friday evening -> Monday
    ("2026-10-03 10:00", date(2026, 10, 5)),     # Saturday -> Monday
])
def test_next_session(now, expected):
    assert G.next_session(pd.Timestamp(now, tz=G.IST)) == expected


def test_load_daily_drops_holiday_placeholder_bars(tmp_path):
    db = tmp_path / "candles.sqlite3"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE candles (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL,"
                " low REAL, close REAL, volume REAL)")
    rows = []
    for sym in ("A.NS", "B.NS", "C.NS"):
        for day, vol in (("2026-09-11", 1e5), ("2026-09-14", 0.0), ("2026-09-15", 1e5)):
            rows.append((sym, "1d", f"{day}T10:00:00+00:00", 100, 101, 99, 100, vol))
    con.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    daily = G.load_daily(since="2026-09-01", db_path=db)
    assert list(daily["close"].index) == [date(2026, 9, 11), date(2026, 9, 15)]


def test_save_and_load_picks_round_trip(tmp_path):
    path = tmp_path / "picks.json"
    pick = _pick()
    pick.entry, pick.stop_price = 101.0, 104.0
    G.save_picks(date(2026, 9, 29), [pick], path=path, based_on="2026-09-28")
    loaded = G.load_picks(date(2026, 9, 29), path=path)
    assert loaded == [pick]
    assert G.load_picks(date(2026, 9, 30), path=path) is None


def test_backtest_daily_applies_the_stop_from_the_open():
    daily = _daily(n_days=40, gaps={"A.NS": 0.05})
    cfg = G.GapReversalConfig(picks=1, universe_size=10, stop_atr=0.75)
    out = G.backtest_daily(daily, cfg, cost_bps=0.0)
    assert not out.empty
    # A stopped short loses exactly its stop distance, so no day can lose more
    # than the widest stop in the universe.
    feats = G.features(daily, cfg)
    worst = -(cfg.stop_atr * feats["atr"] / daily["open"] * 1e4).max(axis=1)
    assert (out["net_bps"] >= worst.reindex(out.index) - 1e-9).all()


def _daily_db(tmp_path, days: dict[str, dict[str, float]]):
    """days: {"2026-09-28": {"A.NS": volume, ...}, ...} -> candles.sqlite3"""
    db = tmp_path / "candles.sqlite3"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE candles (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL,"
                " low REAL, close REAL, volume REAL)")
    con.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)", [
        (sym, "1d", f"{day}T10:00:00+00:00", 100, 101, 99, 100, vol)
        for day, vols in days.items() for sym, vol in vols.items()])
    con.commit()
    con.close()
    return db


SYMS = [f"S{i}.NS" for i in range(10)]


def test_check_fresh_accepts_a_complete_previous_session(tmp_path):
    db = _daily_db(tmp_path, {"2026-09-28": {s: 1e5 for s in SYMS}})
    assert G.check_fresh(SYMS, date(2026, 9, 29), db_path=db) == date(2026, 9, 28)


def test_check_fresh_refuses_a_partially_downloaded_session(tmp_path):
    # The 2026-09-29 failure: yesterday arrived for 4 of 10 names only.
    db = _daily_db(tmp_path, {"2026-09-28": {s: 1e5 for s in SYMS},
                              "2026-09-29": {s: 1e5 for s in SYMS[:4]}})
    with pytest.raises(G.StaleDataError, match="incomplete"):
        G.check_fresh(SYMS, date(2026, 9, 30), db_path=db)


def test_check_fresh_steps_over_a_holiday_of_placeholder_bars(tmp_path, monkeypatch):
    from nse_intraday_ai import nse_calendar
    monkeypatch.setattr(nse_calendar, "is_trading_day", lambda d: d != date(2026, 9, 14))
    db = _daily_db(tmp_path, {"2026-09-11": {s: 1e5 for s in SYMS},
                              "2026-09-14": {s: 0.0 for s in SYMS}})
    assert G.check_fresh(SYMS, date(2026, 9, 15), db_path=db) == date(2026, 9, 11)


def test_check_fresh_refuses_a_trading_day_that_was_never_downloaded(tmp_path, monkeypatch):
    # 2026-10-04 dry run: Yahoo bars stopped at 09-29, Sep 30 / Oct 1 had no
    # rows at all, and the fallback ranked Monday on 09-29 closes.
    from nse_intraday_ai import nse_calendar
    monkeypatch.setattr(nse_calendar, "is_trading_day", lambda d: d != date(2026, 10, 2))
    db = _daily_db(tmp_path, {"2026-09-29": {s: 1e5 for s in SYMS}})
    with pytest.raises(G.StaleDataError, match="trading day 2026-09-30"):
        G.check_fresh(SYMS, date(2026, 10, 5), db_path=db)


def test_check_fresh_refuses_when_nothing_recent_exists(tmp_path):
    db = _daily_db(tmp_path, {"2026-08-01": {s: 1e5 for s in SYMS}})
    with pytest.raises(G.StaleDataError):
        G.check_fresh(SYMS, date(2026, 9, 30), db_path=db)


# ── auction entry limit ─────────────────────────────────────────────────────

def test_tick_size_and_limit_rounding():
    from nse_intraday_ai import gap_reversal as G
    assert G.tick_size(171.37) == 0.01 and G.tick_size(386.6) == 0.05
    assert G.tick_size(1767.5) == 0.10 and G.tick_size(6768) == 0.50 and G.tick_size(21500) == 5.0
    lim = G.entry_limit(1000.0, -0.0075)            # 992.5 -> tick 0.05
    assert lim == 992.5
    lim = G.entry_limit(1388.83, -0.0075)           # 1378.41... -> rounded UP to 0.10
    assert abs(lim - 1378.5) < 1e-9 and lim >= 1388.83 * (1 - 0.0075)


def test_no_fill_when_open_below_limit():
    import pandas as pd
    from datetime import date
    from nse_intraday_ai import gap_reversal as G
    pick = G.Pick(rank=1, symbol="X.NS", side="SHORT", gap_prev_pct=3.0, prev_close=100.0, atr=3.0,
                  stop_distance=2.25, quantity=100, position_value=10_000, turnover_cr=50.0)
    G.with_entry_limits([pick])
    assert pick.limit_price == 99.25
    bars = pd.DataFrame({"open": [99.0] + [98.0] * 72, "high": [99.5] * 73, "low": [97.0] * 73,
                         "close": [98.0] * 73})
    t = G.simulate_pick(pick, bars, date(2026, 10, 5))
    assert t.exit_reason == "NO_FILL" and t.quantity == 0 and t.net == 0
    bars.loc[0, "open"] = 99.30                      # opens at/above the limit -> trades
    t = G.simulate_pick(pick, bars, date(2026, 10, 5))
    assert t.exit_reason != "NO_FILL" and t.quantity > 0


def test_ranker_lists_enter_at_the_open_without_a_limit():
    # The A/B found the limit helps the rule's picks but not the ranker's.
    import pandas as pd
    from datetime import date
    from nse_intraday_ai import gap_reversal as G
    pick = G.Pick(rank=1, symbol="X.NS", side="SHORT", gap_prev_pct=3.0, prev_close=100.0, atr=3.0,
                  stop_distance=2.25, quantity=100, position_value=10_000, turnover_cr=50.0,
                  ranked_by="ranker")
    G.with_entry_limits([pick])
    assert pick.limit_price is None
    bars = pd.DataFrame({"open": [99.0] + [98.0] * 72, "high": [99.5] * 73, "low": [97.0] * 73,
                         "close": [98.0] * 73})
    t = G.simulate_pick(pick, bars, date(2026, 10, 5))
    assert t.exit_reason != "NO_FILL" and t.quantity > 0
    assert G.limit_applies("rule") and not G.limit_applies("ranker")
