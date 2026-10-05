"""Intraday monitor: snapshot, paper strategies, grading, shadow book, gap-book alerts."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from nse_intraday_ai import intraday_monitor as M

DAY = date(2026, 10, 5)


def _panel(n_sym: int = 3, nbar: int = M.NBARS, price: float = 100.0) -> M.DayPanel:
    shape = (n_sym, M.NBARS)
    o = np.full(shape, price)
    return M.DayPanel(DAY, [f"S{i}" for i in range(n_sym)], o.copy(), o + 0.2, o - 0.2, o.copy(),
                      np.full(shape, 1000.0), nbar)


def _ctx(n_sym: int = 3, prev: float = 100.0, atr: float = 2.0, base_per_bar: float = 1000.0,
         band: float = 20.0) -> M.Context:
    cum = np.tile(np.arange(1, M.NBARS + 1) * base_per_bar, (n_sym, 1))
    return M.Context(np.full(n_sym, prev), np.full(n_sym, atr), cum, np.full(n_sym, band))


def test_completed_bars():
    ist = M.IST
    assert M.completed_bars(datetime(2026, 10, 5, 9, 14, tzinfo=ist)) == 0
    assert M.completed_bars(datetime(2026, 10, 5, 9, 20, 1, tzinfo=ist)) == 1     # 09:15 bar done
    assert M.completed_bars(datetime(2026, 10, 5, 10, 2, tzinfo=ist)) == 9
    assert M.completed_bars(datetime(2026, 10, 5, 16, 0, tzinfo=ist)) == M.NBARS


def test_snapshot_relative_volume_and_moves():
    p, ctx = _panel(nbar=10), _ctx()
    p.v[0, :10] = 3000.0                                  # 3x normal
    p.c[1, 9] = 105.0                                     # +5% vs prev close
    p.h[1, 9] = 105.5
    snap = M.snapshot(p, ctx).set_index("symbol")
    assert round(snap.at["S0", "rel_volume"], 6) == 3.0
    assert round(snap.at["S1", "chg_pct"], 6) == 5.0 and bool(snap.at["S1", "new_high"])
    assert round(snap.at["S1", "to_upper_pct"], 2) == round((120 / 105 - 1) * 100, 2)


def test_gap_fade_signal_and_target_exit():
    p, ctx = _panel(), _ctx()
    p.o[0, :] = 104.0                                     # gapped up 2 ATR
    p.h[0, :], p.lo[0, :], p.c[0, :] = 104.2, 103.8, 104.0
    p.lo[0, 5] = 99.5                                     # falls back to the previous close at 09:40
    sigs = M.gap_fade(p, ctx)
    assert [(s.symbol, s.side, s.bar) for s in sigs] == [("S0", -1, 1)]
    r = M.grade(p, sigs[0])
    assert r["status"] == "TARGET" and r["exit"] == 100.0
    assert r["net_bps"] == round((104 - 100) / 104 * 1e4 - M.COST_BPS, 1)


def test_orb60_needs_first_hour_and_volume():
    p, ctx = _panel(nbar=M.FIRST_HOUR), _ctx()
    assert M.orb60(p, ctx) == []                          # first hour not complete yet
    p = _panel()
    p.v[0, :M.FIRST_HOUR] = 5000.0                        # 5x volume in the first hour
    p.c[0, M.FIRST_HOUR - 1] = 100.1                      # up candle
    p.h[0, 20] = 101.0                                    # breaks the 100.2 range high at 11:55
    sigs = M.orb60(p, ctx)
    assert len(sigs) == 1 and sigs[0].side == 1 and sigs[0].bar == 20 and sigs[0].time == "10:55"
    assert sigs[0].entry == 100.2 and sigs[0].stop == 99.8


def test_vol_breakout_one_signal_per_name():
    p, ctx = _panel(), _ctx()
    p.v[0, :] = 2000.0
    p.h[0, 15] = 101.0
    p.h[0, 30] = 102.0                                    # second breakout ignored
    sigs = M.vol_breakout(p, ctx)
    assert [(s.symbol, s.bar, s.side) for s in sigs] == [("S0", 15, 1)]


def test_grade_stop_and_open_and_close():
    p = _panel()
    sig = M.Signal("x", "S0", 1, 10, 100.0, 99.0)
    p.lo[0, 30] = 98.0
    p.o[0, 30] = 99.5
    assert M.grade(p, sig)["status"] == "STOP" and M.grade(p, sig)["exit"] == 99.0
    p2 = _panel(nbar=20)
    assert M.grade(p2, M.Signal("x", "S0", -1, 10, 100.0, 101.0))["status"] == "OPEN"
    assert M.grade(_panel(), M.Signal("x", "S0", -1, 10, 100.0, 101.0))["status"] == "CLOSED"


def test_record_day_replaces_and_board_stats(tmp_path):
    path = tmp_path / "book.csv"
    rows = [{"strategy": "gap_fade", "symbol": "A", "side": 1, "bar": 1, "entry": 1, "stop": 0.9, "target": None,
             "reason": "", "time": "09:20", "status": "STOP", "exit": 0.9, "exit_bar": 3, "gross_bps": -1000.0,
             "net_bps": -1013.3},
            {**{"strategy": "orb60", "symbol": "B", "status": "OPEN", "net_bps": 5.0}}]
    assert M.record_day(DAY, rows, path) == 1             # OPEN rows are not final
    assert M.record_day(DAY, rows, path) == 1             # idempotent
    book = pd.read_csv(path)
    assert len(book) == 1
    stats = M.board_stats(book).set_index("strategy")
    assert stats.at["gap_fade", "sessions"] == 1 and stats.at["orb60", "trades"] == 0
    assert all(s.startswith("paper") for s in stats["status"])


def test_gap_book_watch_and_alerts():
    p = _panel(nbar=20)
    payload = {"session": DAY.isoformat(), "picks": [
        {"symbol": "S0.NS", "stop_price": 100.5, "entry": 100.0, "reserve": False},
        {"symbol": "S1.NS", "stop_price": 100.1, "entry": 100.0, "reserve": False},
        {"symbol": "S2.NS", "stop_price": 110.0, "entry": 100.0, "reserve": True}]}
    rows = {r["symbol"]: r for r in M.gap_book_watch(p, payload)}
    assert set(rows) == {"S0", "S1"}
    assert not rows["S0"]["stop_hit"] and rows["S1"]["stop_hit"]
    alerts = M.gap_alerts(list(rows.values()), set())
    assert {k for k, _ in alerts} == {"near:S0", "hit:S1"}
    assert M.gap_alerts(list(rows.values()), {"near:S0", "hit:S1"}) == []
    assert M.gap_book_watch(p, {**payload, "session": "2026-10-04"}) == []


def test_gap_book_watch_caps_stop_under_upper_circuit():
    p = _panel(n_sym=1, nbar=5, price=97.0)               # LALITHAA-like: stop above the 5% band
    payload = {"session": DAY.isoformat(), "picks": [
        {"symbol": "S0.NS", "stop_price": 105.5, "entry": 97.0, "prev_close": 100.0, "reserve": False}]}
    row = M.gap_book_watch(p, payload, np.array([5.0]))[0]
    assert row["stop_capped"] and row["stop"] == 104.99 and not row["stop_hit"]
    row = M.gap_book_watch(p, payload, np.array([20.0]))[0]
    assert not row["stop_capped"] and row["stop"] == 105.5


def test_build_live_is_json_clean():
    import json
    p, ctx = _panel(nbar=20), _ctx()
    ctx.band[:] = np.nan
    out = M.build_live(p, ctx, pd.DataFrame(columns=["symbol", "ts", "desc", "text"]), None,
                       datetime(2026, 10, 5, 11, 0, tzinfo=M.IST))
    json.dumps(out, allow_nan=False)
    assert out["last_bar"] == "10:50"


def test_gap_alerts_handles_none_to_stop():
    rows = [{"symbol": "S0", "stop_hit": False, "to_stop_pct": None, "last": None, "stop": 100.0}]
    # Should not raise TypeError: '<=' not supported between instances of 'int' and 'NoneType'
    assert M.gap_alerts(rows, set()) == []


def test_snapshot_handles_lagging_nan_bars():
    p, ctx = _panel(nbar=10), _ctx()
    p.c[0, 9] = np.nan  # bar 9 missing / no trade, but bar 8 had 102.0
    p.c[0, 8] = 102.0
    snap = M.snapshot(p, ctx).set_index("symbol")
    assert "S0" in snap.index
    assert snap.at["S0", "last"] == 102.0
