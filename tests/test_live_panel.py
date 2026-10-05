"""Live-panel helpers: the running short P&L path and the circuit-capped stop."""
from __future__ import annotations

import pandas as pd

from nse_intraday_ai import gap_reversal as G
from nse_intraday_ai import nse_bands


def _bars(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-10-05 {t}", tz="Asia/Kolkata") for t, *_ in rows])
    return pd.DataFrame([r[1:] for r in rows], index=idx, columns=["open", "high", "close"])


def test_path_open_position_tracks_close():
    path, status, price = G.live_short_path(
        _bars([("09:15", 100, 101, 99), ("09:16", 99, 100, 98)]), entry=100, stop=105)
    assert status == "OPEN" and price == 98
    assert list(path.round(6)) == [1.0, 2.0]


def test_path_stop_freezes_pnl_at_stop_or_worse_gap():
    bars = _bars([("09:15", 100, 101, 100), ("09:16", 103, 106, 104), ("09:17", 104, 104, 90)])
    path, status, price = G.live_short_path(bars, entry=100, stop=105)
    assert status == "STOPPED" and price == 105
    assert list(path.round(6)) == [0.0, -5.0, -5.0]               # no credit for the later fall
    gap = _bars([("09:15", 100, 101, 100), ("09:16", 107, 108, 107)])
    assert G.live_short_path(gap, entry=100, stop=105)[2] == 107  # gapped through: fills at the open


def test_path_covers_at_square_off_open():
    bars = _bars([("09:15", 100, 101, 99), ("15:14", 97, 98, 97), ("15:15", 96, 120, 96)])
    path, status, price = G.live_short_path(bars, entry=100, stop=105)
    assert status == "COVERED" and price == 96                     # 15:15 high is after the cover
    assert round(path.iloc[-1], 6) == 4.0


def _pick(symbol: str, prev_close: float, stop_price: float) -> G.Pick:
    return G.Pick(rank=1, symbol=f"{symbol}.NS", side="SHORT", gap_prev_pct=1.0, prev_close=prev_close,
                  atr=20.0, stop_distance=17.05, quantity=10, position_value=1.0, turnover_cr=1.0,
                  stop_price=stop_price)


def test_circuit_capped_stop():
    table = nse_bands.parse("Symbol,Series,Security Name,Band,Remarks\n"
                            "LALITHAA,EQ,L,5,-\nMIDCO,EQ,M,20,-\nBIG,EQ,B,No Band,-\n")
    stop, upper = G.circuit_capped_stop(_pick("LALITHAA", 387.85, 407.70), table)
    assert upper == 407.24 and stop == 407.15 and stop < upper     # one tick under, on the tick grid
    assert G.circuit_capped_stop(_pick("LALITHAA", 387.85, 400.00), table) is None
    assert G.circuit_capped_stop(_pick("MIDCO", 387.85, 407.70), table) is None
    assert G.circuit_capped_stop(_pick("BIG", 387.85, 999.0), table) is None
