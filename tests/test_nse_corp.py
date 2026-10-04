"""NSE corporate events: parsing, point-in-time windows, cache, and the feature hook.

No network: the JSON below is trimmed from real API responses (2017, 2025, 2026).
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai import features as FT
from nse_intraday_ai import nse_corp as C

BM_JSON = [
    {"bm_symbol": "ABREL", "bm_date": "31-Jan-2017", "bm_purpose": "Results",
     "bm_desc": "inter alia to consider and take on record the Unaudited Financial Results of the Company for "
                "the Third Quarter (October to December) ended on December 31, 2016.",
     "sm_indusrty": "-", "bm_timestamp": "17-Jan-2017 16:29:00", "sm_name": "Aditya Birla Real Estate Limited",
     "sm_isin": "INE055A01016"},
    {"bm_symbol": "NET4", "bm_date": "02-Jan-2017", "bm_purpose": "Board Meeting Adjourned",
     "bm_desc": "NET4 : 02-Jan-2017 : The Company had informed the Exchange regarding a meeting of the Board "
                "... to consider the Un- Audited Financial Results", "bm_timestamp": "02-Jan-2017 10:00:00"},
    {"bm_symbol": "DLINKINDIA", "bm_date": "31-Oct-2026", "bm_purpose": "Board Meeting Intimation",
     "bm_desc": "D-LINK (INDIA) LIMITED has informed the Exchange about Board Meeting to be held on 31-Oct-2026 "
                "to consider and approve the Quarterly Unaudited Financial results of the Company",
     "bm_timestamp": "01-Oct-2026 15:24:55"},
    {"bm_symbol": "XYZ", "bm_date": "15-Oct-2026", "bm_purpose": "Buyback", "bm_desc": "Buyback of shares",
     "bm_timestamp": "-"},
]
RES_JSON = [
    {"audited": "Un-Audited", "broadCastDate": "28-Feb-2017 19:08:37", "consolidated": "Non-Consolidated",
     "exchdisstime": "28-Feb-2017 19:12:04", "filingDate": "28-Feb-2017 19:08", "relatingTo": "Third Quarter",
     "symbol": "TIJARIA", "period": "Quarterly"},
    {"audited": "Un-Audited", "broadCastDate": "-", "consolidated": "Non-Consolidated",
     "exchdisstime": "01-Feb-2017 10:12:05", "filingDate": "01-Feb-2017 10:09", "relatingTo": "Third Quarter",
     "symbol": "NAVA", "period": "Quarterly"},
]
INTEGRATED_JSON = {"data": [
    {"broadcast_Date": "13-Nov-2025 23:59:47", "creation_Date": "13-Nov-2025 23:59:47", "consolidated": "Standalone",
     "qe_Date": "30-SEP-2025", "symbol": "SWELECTES", "type": "Integrated Filing- Financials", "type_Sub": "Original"},
    {"broadcast_Date": None, "creation_Date": "28-Feb-2025 19:17:34", "qe_Date": "31-DEC-2024",
     "symbol": "VIPIND", "type": "Integrated Filing- Governance", "type_Sub": "Revision"},
    {"broadcast_Date": "14-Nov-2025 10:00:00", "symbol": "ABC", "type": "Integrated Filing- Financials",
     "type_Sub": "Revision"},
], "size": 5000, "page": -1, "totalCount": 3}
ANN_JSON = [
    {"an_dt": "05-Jan-2016 20:11:00", "attchmntText": "Wipro Limited has informed the Exchange that the Company "
     "has fixed Record Date as January 27, 2016 for the purpose of Interime dividend", "desc": "Record Date",
     "seq_id": "100354482", "smIndustry": "Computers - Software", "symbol": "WIPRO"},
    {"an_dt": "13-Nov-2025 23:43:31", "attchmntText": "Trigyn Technologies Limited has submitted to the Exchange, "
     "the financial results for the period ended September 30, 2025.", "desc": "Outcome of Board Meeting",
     "seq_id": "2", "smIndustry": None, "symbol": "TRIGYN"},
    {"an_dt": "13-Nov-2025 11:00:00", "attchmntText": "Significant increase in volume has been observed in "
     "Thyrocare Technologies Limited.", "desc": "Spurt in Volume", "seq_id": "3", "symbol": "THYROCARE"},
    {"an_dt": "13-Nov-2025 12:00:00", "attchmntText": "x", "desc": "Reasons for Delayed/Non-submission of "
     "Financial Results", "seq_id": "4", "symbol": "LATECO"},
]


# ── parsing ─────────────────────────────────────────────────────────────────

def test_parse_board_meetings_flags_results_meetings():
    df = C.parse_board_meetings(BM_JSON).set_index("symbol")
    assert df.loc["ABREL", "bm_date"] == pd.Timestamp("2017-01-31")
    assert df.loc["ABREL", "known_at"] == pd.Timestamp("2017-01-17 16:29:00")
    assert bool(df.loc["ABREL", "is_results"])
    assert not bool(df.loc["NET4", "is_results"])                   # adjourned
    assert bool(df.loc["DLINKINDIA", "is_results"])                 # intimation that mentions results
    assert not bool(df.loc["XYZ", "is_results"]) and pd.isna(df.loc["XYZ", "known_at"])


def test_parse_results_and_integrated():
    r = C.parse_results(RES_JSON).set_index("symbol")
    assert r.loc["TIJARIA", "ts"] == pd.Timestamp("2017-02-28 19:08:37")
    assert r.loc["NAVA", "ts"] == pd.Timestamp("2017-02-01 10:12:05")   # broadcast missing -> dissemination
    i = C.parse_integrated(INTEGRATED_JSON)
    assert list(i["symbol"]) == ["SWELECTES"]                       # governance and revisions dropped
    assert i["ts"].iloc[0] == pd.Timestamp("2025-11-13 23:59:47")
    assert C.parse_results([]).empty and C.parse_integrated({"data": []}).empty


def test_parse_announcements():
    a = C.parse_announcements(ANN_JSON)
    assert len(a) == 4 and a["ts"].min() == pd.Timestamp("2016-01-05 20:11:00")
    assert set(a.columns) >= {"symbol", "ts", "desc", "text"}
    res = C._results_from_announcements(a, C.parse_board_meetings([]))
    assert list(res["symbol"]) == ["TRIGYN"]                        # not the delayed-results notice


def test_symbol_changes_follow_renames_back_in_time():
    raw = ("Aditya Birla Real Estate Limited,CENTURYTEX,ABREL,10-OCT-2024\n"
           "Foo Ltd,AAA,BBB,01-JAN-2018\nFoo Ltd,BBB,CCC,01-JAN-2022\n")
    ch = C.parse_symbol_changes(raw)
    sym = pd.Series(["ABREL", "ABREL", "CCC", "CCC", "CCC", "OTHER"])
    when = pd.Series(pd.to_datetime(["2017-01-31", "2025-01-01", "2017-06-01", "2020-06-01", "2023-01-01",
                                     "2017-01-01"]))
    assert list(C.symbol_as_of(sym, when, ch)) == ["CENTURYTEX", "ABREL", "AAA", "BBB", "CCC", "OTHER"]


def test_cluster_results_keeps_first_public_time():
    df = pd.DataFrame({"symbol": ["A", "A", "A", "B"],
                       "ts": pd.to_datetime(["2025-11-13 15:00", "2025-11-13 23:59", "2026-02-10 18:00",
                                             "2025-11-14 09:00"])})
    out = C.cluster_results(df)
    assert list(zip(out["symbol"], out["ts"].dt.strftime("%m-%d %H:%M"))) == [
        ("A", "11-13 15:00"), ("A", "02-10 18:00"), ("B", "11-14 09:00")]


# ── point-in-time windows ───────────────────────────────────────────────────

SESS = list(pd.bdate_range("2026-03-02", periods=15).date)     # Mon 2 Mar .. Fri 20 Mar


def _events(results=(), ann=(), bm=()) -> C.CorpEvents:
    cov_days = np.arange(np.datetime64("2025-10-01"), np.datetime64("2026-04-30"), dtype="datetime64[D]")
    res = pd.DataFrame(list(results), columns=["symbol", "ts"]).assign(source="t")
    res["ts"] = pd.to_datetime(res["ts"])
    a = pd.DataFrame(list(ann), columns=["symbol", "ts", "desc"])
    a["ts"] = pd.to_datetime(a["ts"])
    b = pd.DataFrame(list(bm), columns=["symbol", "bm_date", "known_at"])
    b["bm_date"], b["known_at"] = pd.to_datetime(b["bm_date"]), pd.to_datetime(b["known_at"])
    b = b.assign(purpose="Results", desc="", is_results=True)
    cov = {k: cov_days for k in ("board_meetings", "results", "integrated", "announcements", "results_any")}
    return C.CorpEvents(board_meetings=b, results=res, announcements=a, coverage=cov)


def test_gap1_window_evening_t_minus_2_counts_morning_t_minus_1_does_not():
    t = SESS[5]                                          # Mon 9 Mar; t-1 = Fri 6, t-2 = Thu 5
    ev = _events(results=[("A", "2026-03-05 18:45:00"),  # evening of t-2 -> produced gap1 of t
                          ("B", "2026-03-06 10:00:00"),  # intraday t-1 -> not in gap1
                          ("C", "2026-03-06 09:10:00"),  # pre-open t-1 -> in gap1
                          ("D", "2026-03-05 15:00:00")])  # intraday t-2 -> not in gap1
    f = C.corp_features(ev, SESS, ["A", "B", "C", "D", "E"])
    g = f["res_in_gap1"].loc[t]
    assert g.to_dict() == {"A": 1.0, "B": 0.0, "C": 1.0, "D": 0.0, "E": 0.0}
    assert f["res_in_gap1"].loc[SESS[4], "A"] == 0.0     # it is gap0 (not gap1) for t-1
    assert f["res_in_gap1"].loc[SESS[6], "A"] == 0.0
    r3 = f["res_recent3"].loc[t]
    assert r3["A"] == 1 and r3["B"] == 1 and r3["D"] == 1 and r3["E"] == 0
    assert f["days_since_results"].loc[t, "A"] == 4.0    # Thu 5 -> Mon 9 (calendar days)
    assert f["res_in_gap1"].dtypes.eq(np.float32).all()


def test_nothing_after_the_0845_cutoff_is_visible():
    t = SESS[10]                                         # Mon 16 Mar
    ev = _events(results=[("A", "2026-03-16 08:30:00"), ("B", "2026-03-16 08:50:00")],
                 ann=[("A", "2026-03-16 08:30:00", "Spurt in Volume"), ("B", "2026-03-16 08:50:00", "Spurt in Volume")])
    f = C.corp_features(ev, SESS, ["A", "B"])
    assert f["res_night0"].loc[t, "A"] == 1 and f["res_night0"].loc[t, "B"] == 0
    assert f["clar_gap1"].loc[t, "A"] == 1 and f["clar_gap1"].loc[t, "B"] == 0
    assert f["res_in_gap1"].loc[SESS[11], "A"] == 1       # Monday pre-open news produced Tuesday's gap1
    assert f["ann_n_5d"].loc[t, "B"] == 0 and f["ann_n_5d"].loc[SESS[11], "B"] == 1
    assert np.isnan(f["ann_n_5d"].loc[SESS[5], "A"])      # needs 6 prior sessions


def test_announcement_counts_and_clarification_window():
    t = SESS[10]                                         # Mon 16 Mar
    ev = _events(ann=[("A", "2026-03-12 20:00:00", "Press Release"),
                      ("A", "2026-03-13 07:00:00", "Updates"),
                      ("A", "2026-03-13 12:00:00", "Updates"),                 # intraday t-1
                      ("B", "2026-03-11 16:00:00", "Price movement")])          # after close of t-3
    f = C.corp_features(ev, SESS, ["A", "B"])
    assert f["ann_n_gap1"].loc[t, "A"] == 2
    assert f["ann_n_5d"].loc[t, "A"] == 3
    assert f["clar_gap1"].loc[t, "B"] == 1
    assert f["clar_gap1"].loc[SESS[11], "B"] == 0          # out of the 2-session window a day later


def test_board_meeting_schedule_is_known_in_advance_only_once_announced():
    ev = _events(bm=[("A", "2026-03-12", "2026-03-04 17:00:00"),    # announced Wed 4 for Thu 12
                     ("B", "2026-03-12", "2026-03-12 09:00:00")])   # announced the same morning (too late)
    f = C.corp_features(ev, SESS, ["A", "B", "Z"])
    t = date(2026, 3, 12)
    assert f["bm_results_today"].loc[t, "A"] == 1 and f["bm_results_today"].loc[t, "B"] == 0
    assert f["days_to_bm"].loc[date(2026, 3, 9), "A"] == 3
    assert f["days_to_bm"].loc[date(2026, 3, 4), "A"] == 30   # announced after 08:45 that day
    assert f["days_to_bm"].loc[date(2026, 3, 5), "A"] == 7
    assert f["days_to_bm"].loc[t, "Z"] == 30


def test_uncovered_windows_are_nan_not_zero():
    ev = _events(results=[("A", "2026-03-05 18:45:00")])
    ev.coverage = {k: v[v < np.datetime64("2026-03-06")] for k, v in ev.coverage.items()}
    f = C.corp_features(ev, SESS, ["A"])
    assert f["res_in_gap1"].loc[SESS[3], "A"] == 0         # Thu 5: its window is covered
    assert np.isnan(f["res_in_gap1"].loc[SESS[6], "A"])    # t-1 = Mon 9 not covered
    assert np.isnan(f["res_in_gap1"].loc[SESS[0], "A"])    # first session: no t-2
    none = C.corp_features(None, SESS, ["A"])
    assert set(none) == set(C.FEATURES) and all(v.isna().all().all() for v in none.values())


# ── cache ───────────────────────────────────────────────────────────────────

def test_cache_round_trip_coverage_and_completeness(tmp_path):
    m = date(2017, 1, 1)
    C._store("board_meetings", m, C.parse_board_meetings(BM_JSON[:2]), date(2017, 1, 31), tmp_path, append=False)
    C._store("results", date(2017, 2, 1), C.parse_results(RES_JSON), date(2017, 2, 28), tmp_path, append=False)
    C._store("announcements", m, C.parse_announcements(ANN_JSON[:1]), date(2017, 1, 7), tmp_path, append=True)
    assert C.month_complete("board_meetings", m, tmp_path)          # fetched (now) after month end
    assert not C.month_complete("announcements", m, tmp_path)       # only through the 7th
    ev = C.load(date(2017, 2, 1), date(2017, 2, 28), cache=tmp_path, map_symbols=False)
    assert set(ev.results["symbol"]) == {"TIJARIA", "NAVA"}
    assert ev.covered("announcements", ["2017-01-07", "2017-01-08"]).tolist() == [True, False]
    assert ev.covered("results_any", ["2017-02-10"]).tolist() == [True]
    assert C.results_day_names(["A.NS"], date(2017, 2, 1), cache=tmp_path / "nothing") == {}


def test_months_and_chunks():
    assert C.months(date(2016, 11, 15), date(2017, 2, 1)) == [date(2016, 11, 1), date(2016, 12, 1),
                                                              date(2017, 1, 1), date(2017, 2, 1)]
    ch = C._chunks(date(2025, 2, 1), date(2025, 2, 28), 7)
    assert ch[0] == (date(2025, 2, 1), date(2025, 2, 7)) and ch[-1][1] == date(2025, 2, 28) and len(ch) == 4


# ── features.build integration ─────────────────────────────────────────────

def _bhav(n_days: int = 90, n_syms: int = 8, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = list(pd.bdate_range("2026-01-01", periods=n_days).date)
    rows = []
    for j in range(n_syms):
        close = 100.0 + 10 * j
        for d in days:
            prev = close
            o = prev * (1 + rng.normal(0, 0.01))
            close = o * (1 + rng.normal(0, 0.015))
            hi, lo = max(o, close) * 1.004, min(o, close) * 0.996
            vol = float(rng.integers(1e5, 1e6)) * (1 + j)
            rows.append({"session": d, "symbol": f"S{j}", "series": "EQ", "open": o, "high": hi, "low": lo,
                         "close": close, "prev_close": prev, "volume": vol, "value": vol * close,
                         "trades": vol / 50, "deliv_qty": vol * 0.4, "deliv_pct": 40.0})
    return pd.DataFrame(rows)


def test_features_build_without_corp_cache_has_nan_corp_columns():
    inputs = FT.Inputs(bhav=_bhav(), fo=pd.DataFrame(), series={}, participant=pd.DataFrame())
    panel = FT.build(inputs, universe=8, min_price=1.0)
    for col in FT.CORP_FEATURES:
        assert col in panel.columns and panel[col].dtype == np.float32
    assert panel[list(C.FEATURES)].isna().all().all()
    assert panel["gap1_z"].notna().mean() > 0.5
    exp = panel["gap1"] / panel["gap1_z"]                 # denominator is a positive std
    assert (exp.dropna() > 0).all()


def test_features_build_with_corp_events_live_equals_research():
    bhav = _bhav()
    sessions = sorted(bhav["session"].unique())
    target = sessions[70]
    t2 = sessions[68]
    ev = _events(results=[("S1", datetime.combine(t2, datetime.min.time()) + pd.Timedelta(hours=18))],
                 bm=[("S2", str(target), "2026-01-02 10:00:00")])
    ev.coverage = {k: np.arange(np.datetime64("2025-06-01"), np.datetime64("2026-12-31"), dtype="datetime64[D]")
                   for k in ev.coverage}
    full = FT.build(FT.Inputs(bhav=bhav, fo=pd.DataFrame(), series={}, participant=pd.DataFrame(), corp=ev),
                    universe=8, min_price=1.0)
    live = FT.build(FT.Inputs(bhav=bhav[bhav["session"] < target], fo=pd.DataFrame(), series={},
                              participant=pd.DataFrame(), corp=ev),
                    universe=8, min_price=1.0, live_session=target)
    a = full.xs(target, level="session").sort_index()
    b = live.xs(target, level="session").sort_index()
    cols = list(FT.CORP_FEATURES)
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert a.loc["S1", "res_in_gap1"] == 1 and a.loc["S0", "res_in_gap1"] == 0
    assert a.loc["S2", "bm_results_today"] == 1 and a.loc["S2", "days_to_bm"] == 0


if __name__ == "__main__":                               # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
