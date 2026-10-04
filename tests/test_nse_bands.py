"""NSE price bands/series: parsing, point-in-time archive lookup, and the untradeable filter."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from nse_intraday_ai import gap_reversal as G
from nse_intraday_ai import nse_bands

SAMPLE = (
    "Symbol,Series,Security Name,Band,Remarks\n"
    "AUGMONT,EQ,Augmont Enterprises Ltd,5,-\n"
    "RELIANCE,EQ,Reliance Industries Ltd,No Band,-\n"
    "SMALLCO,BE,Small Co Ltd,2,-\n"
    "BONDX,N1,Some Bond,20,-\n"
    "DUAL,BE,Dual Ltd,10,-\n"
    "DUAL,EQ,Dual Ltd,20,-\n"
    "MIDCO,EQ,Mid Co Ltd,20,-\n"
)


def test_parse_reads_band_and_series_and_prefers_eq_rows():
    t = nse_bands.parse(SAMPLE.encode())
    assert t.at["AUGMONT", "band"] == 5 and t.at["MIDCO", "band"] == 20
    assert pd.isna(t.at["RELIANCE", "band"])
    assert t.at["SMALLCO", "series"] == "BE"
    assert t.at["DUAL", "series"] == "EQ" and t.at["DUAL", "band"] == 20
    assert t.index.is_unique


def test_circuit_risk_respects_margin_and_unknown_names():
    t = nse_bands.parse(SAMPLE)
    assert nse_bands.circuit_risk("AUGMONT.NS", 5.2, t)       # stop beyond the 5% circuit
    assert nse_bands.circuit_risk("AUGMONT", 4.6, t)          # inside the 0.5 pt margin
    assert not nse_bands.circuit_risk("AUGMONT", 4.4, t)
    assert not nse_bands.circuit_risk("RELIANCE", 9.0, t)     # F&O: no band
    assert not nse_bands.circuit_risk("UNKNOWN", 9.0, t)


def test_untradeable_reason():
    t = nse_bands.parse(SAMPLE)
    assert nse_bands.untradeable_reason("MIDCO.NS", 4.0, t) is None
    assert "circuit" in nse_bands.untradeable_reason("AUGMONT.NS", 5.2, t)
    assert "BE series" in nse_bands.untradeable_reason("SMALLCO.NS", 1.0, t)
    assert nse_bands.untradeable_reason("GONE.NS", 1.0, t) == "not listed today"
    assert nse_bands.untradeable_reason("GONE.NS", 1.0, t.iloc[:0]) is None   # no data: never block


def test_load_is_point_in_time_and_works_offline(tmp_path):
    old = date.today() - timedelta(days=10)
    (tmp_path / f"sec_list_{old:%Y%m%d}.csv").write_text(SAMPLE)
    logs: list[str] = []
    assert nse_bands.load(date.today(), fetch=False, cache=tmp_path, log=logs.append).at["AUGMONT", "band"] == 5
    assert logs and "using" in logs[0]                        # stale file is announced
    assert nse_bands.load(old - timedelta(days=1), fetch=False, cache=tmp_path).empty
    assert nse_bands.load(fetch=False, cache=tmp_path / "missing").empty


def _pick(rank: int, symbol: str, stop_pct: float) -> G.Pick:
    return G.Pick(rank=rank, symbol=f"{symbol}.NS", side="SHORT", gap_prev_pct=3.0, prev_close=100.0,
                  atr=stop_pct / 0.75, stop_distance=stop_pct, quantity=10, position_value=1000.0,
                  turnover_cr=5.0, reserve=rank > 8)


def test_drop_untradeable_promotes_reserves_and_renumbers():
    table = nse_bands.parse(SAMPLE)
    names = ["MIDCO", "AUGMONT", "SMALLCO", "RELIANCE", "DUAL", "MIDCO", "MIDCO", "MIDCO",
             "R1", "R2", "R3", "R4"]
    table = pd.concat([table, pd.DataFrame({"series": "EQ", "band": 20.0},
                                           index=pd.Index(["R1", "R2", "R3", "R4"], name="symbol"))])
    picks = [_pick(i, s, 5.2 if s == "AUGMONT" else 3.0) for i, s in enumerate(names, 1)]
    kept, dropped = G.drop_untradeable(picks, table)
    assert [d.split()[0] for d in dropped] == ["AUGMONT", "SMALLCO"]
    assert [p.rank for p in kept] == list(range(1, 11))
    main = [p.symbol.removesuffix(".NS") for p in kept if not p.reserve]
    assert main == ["MIDCO", "RELIANCE", "DUAL", "MIDCO", "MIDCO", "MIDCO", "R1", "R2"]
    assert sum(p.reserve for p in kept) == 2


def test_drop_untradeable_without_data_is_a_no_op():
    picks = [_pick(1, "AUGMONT", 9.0)]
    assert G.drop_untradeable(picks, pd.DataFrame(columns=["series", "band"])) == (picks, [])
