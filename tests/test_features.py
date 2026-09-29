"""The feature builder must never look ahead, and the NSE parsers must not be fooled."""
from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from nse_intraday_ai import features as FT
from nse_intraday_ai import nse_bhav, nse_fo


def _bhav(n_days: int = 90, n_syms: int = 12, seed: int = 0, split: tuple | None = None) -> pd.DataFrame:
    """Synthetic EQ bhavcopy rows with a consistent PREV_CLOSE chain."""
    rng = np.random.default_rng(seed)
    days = list(pd.bdate_range("2026-01-01", periods=n_days).date)
    rows = []
    for j in range(n_syms):
        sym = f"S{j:02d}"
        close = 100.0 + 10 * j
        for i, d in enumerate(days):
            prev = close
            if split and split[0] == sym and split[1] == i:
                prev_unadjusted, prev = prev, prev          # NSE sometimes leaves it unadjusted
                close = close / 10
                o = close * (1 + rng.normal(0, 0.005))
                prev_close = prev_unadjusted                 # -> a fake -90% "gap"
            else:
                o = prev * (1 + rng.normal(0, 0.01))
                close = o * (1 + rng.normal(0, 0.015))
                prev_close = prev
            hi = max(o, close) * (1 + abs(rng.normal(0, 0.005)))
            lo = min(o, close) * (1 - abs(rng.normal(0, 0.005)))
            vol = float(rng.integers(1e5, 1e6)) * (1 + j)
            rows.append({"session": d, "symbol": sym, "series": "EQ", "open": o, "high": hi,
                         "low": lo, "close": close, "prev_close": prev_close, "volume": vol,
                         "value": vol * close, "trades": vol / 50, "deliv_qty": vol * 0.4,
                         "deliv_pct": 40.0 + rng.normal(0, 5)})
    return pd.DataFrame(rows)


def _inputs(bhav: pd.DataFrame) -> FT.Inputs:
    return FT.Inputs(bhav=bhav, fo=pd.DataFrame(), series={}, participant=pd.DataFrame())


def test_live_row_equals_research_row_built_from_the_future():
    """As-of equivalence: the row for session S must not change when data after S exists."""
    bhav = _bhav()
    sessions = sorted(bhav["session"].unique())
    target = sessions[70]
    full = FT.build(_inputs(bhav), universe=10, min_price=1.0)
    live = FT.build(_inputs(bhav[bhav["session"] < target]), universe=10, min_price=1.0,
                    live_session=target)
    a = full.xs(target, level="session").sort_index()
    b = live.xs(target, level="session").sort_index()
    assert list(a.index) == list(b.index) and len(a) == 10
    cols = [c for c in b.columns if c not in ("sector",) and c in a.columns]
    np.testing.assert_allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), rtol=1e-5,
                               atol=1e-6, equal_nan=True)
    assert not set(FT.TARGETS) & set(live.columns), "a live row must not carry labels"


def test_targets_use_the_session_itself_and_features_do_not():
    bhav = _bhav()
    full = FT.build(_inputs(bhav), universe=10, min_price=1.0)
    s = sorted(bhav["session"].unique())[60]
    row = full.loc[(s, "S03")]
    today = bhav[(bhav["session"] == s) & (bhav["symbol"] == "S03")].iloc[0]
    yday = bhav[(bhav["session"] < s) & (bhav["symbol"] == "S03")].iloc[-1]
    assert row["gap1"] == pytest.approx(yday["open"] / yday["prev_close"] - 1, rel=1e-5)
    raw_short = (today["open"] - today["close"]) / today["open"] * 1e4
    assert row["short_bps"] == pytest.approx(raw_short, rel=1e-4) or row["short_bps"] < 0


def test_unadjusted_split_is_masked_not_read_as_a_crash():
    bhav = _bhav(split=("S04", 50))
    full = FT.build(_inputs(bhav), universe=12, min_price=1.0)
    s = sorted(bhav["session"].unique())
    after = full.xs("S04", level="symbol")
    # The split day's "gap" is unknown, so the next session has no signal and
    # no row at all — never a -90% gap that ranks as a huge reversal candidate.
    assert s[51] not in after.index
    assert s[52] in after.index
    assert after["gap1"].min() > -0.3
    assert after["ret20"].dropna().abs().max() < 0.5            # returns chain across the split


def test_live_session_must_be_after_the_data():
    bhav = _bhav(n_days=40)
    last = max(bhav["session"])
    with pytest.raises(ValueError):
        FT.build(_inputs(bhav), live_session=last)


def test_prior_value_is_strictly_before_and_not_stale():
    s = pd.Series([1.0, 2.0, 3.0], index=[date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)])
    sess = pd.Index([date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 30)])
    out = FT.prior_value(s, sess)
    assert list(out.iloc[:3]) == [1.0, 2.0, 3.0]
    assert np.isnan(out.iloc[3])                                # > 7 days old


# ── NSE parsers ─────────────────────────────────────────────────────────────

FULL_CSV = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, "
    "CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "ABC, EQ, 29-Sep-2026, 100.00, 101.00, 103.00, 99.00, 102.00, 102.50, 101.9, 1000, 1.02, 50, 400, 40.00\n"
    "XYZ, BE, 29-Sep-2026, 50.00, 50.00, 51.00, 49.00, 50.50, 50.50, 50.1, 200, 0.10, 5, -, -\n"
    "BND, GS, 29-Sep-2026, 99.00, 99.00, 99.00, 99.00, 99.00, 99.00, 99.0, 10, 0.01, 1, -, -\n"
)


def test_parse_full_bhavcopy():
    f = nse_bhav._parse_full(FULL_CSV.encode())
    assert f.attrs["file_date"] == date(2026, 9, 29)
    assert set(f["series"]) == {"EQ", "BE"}                     # other series dropped
    abc = f[f["symbol"] == "ABC"].iloc[0]
    assert abc["open"] == 101 and abc["prev_close"] == 100 and abc["value"] == pytest.approx(1.02e5)
    assert abc["deliv_pct"] == 40 and np.isnan(f[f["symbol"] == "XYZ"].iloc[0]["deliv_pct"])


def test_holiday_copy_of_previous_file_is_rejected(monkeypatch):
    # NSE serves 2026-09-11's rows under the 2026-09-14 (holiday) file name.
    stale = FULL_CSV.replace("29-Sep-2026", "11-Sep-2026").encode()
    monkeypatch.setattr(nse_bhav, "_get", lambda s, url, retries=3: stale if "full" in url else None)
    assert nse_bhav.fetch_day(date(2026, 9, 14)) is None
    fresh = FULL_CSV.replace("29-Sep-2026", "14-Sep-2026").encode()
    monkeypatch.setattr(nse_bhav, "_get", lambda s, url, retries=3: fresh if "full" in url else None)
    assert len(nse_bhav.fetch_day(date(2026, 9, 14))) == 2


def test_corrupt_full_file_falls_back_to_cm_and_mto(monkeypatch):
    cm = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
          "ABC,EQ,10,11,9,10.5,10.5,10,100,1050,08-AUG-2022,7,INE0,\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("cm08AUG2022bhav.csv", cm)
    mto = ("Security Wise Delivery Position\n10,MTO,08082022\nTrade Date <08-AUG-2022>\nRecord Type\n"
           "20,1,ABC,EQ,100,60,60.00\n")

    def fake(_s, url, retries=3):
        if "sec_bhavdata_full" in url:
            return b"PK\x03\x04 this is an xlsx, not a csv" + b"\x00" * 600
        if url.endswith(".zip"):
            return buf.getvalue()
        return mto.encode()

    monkeypatch.setattr(nse_bhav, "_get", fake)
    f = nse_bhav.fetch_day(date(2022, 8, 8))
    assert len(f) == 1 and f.iloc[0]["deliv_pct"] == 60 and f.iloc[0]["prev_close"] == 10


def test_fo_aggregate_uses_all_expiries_and_near_month_price():
    rows = pd.DataFrame({
        "symbol": ["ABC"] * 4, "kind": ["FUT", "FUT", "OPT", "OPT"], "index": [False] * 4,
        "expiry": pd.to_datetime(["2026-10-27", "2026-11-24", "2026-10-27", "2026-10-27"]),
        "opt": ["XX", "XX", "CE", "PE"], "close": [101.0, 102.0, 3.0, 2.0],
        "oi": [1000.0, 200.0, 500.0, 800.0], "chg_oi": [100.0, 50.0, -20.0, 40.0],
        "value": [1e6, 2e5, 1e4, 2e4]})
    a = nse_fo.aggregate(rows).set_index("symbol").loc["ABC"]
    assert a["fut_close"] == 101.0 and a["fut_oi_all"] == 1200 and a["fut_chg_oi_all"] == 150
    assert a["call_oi"] == 500 and a["put_oi"] == 800 and a["fno"] == 1.0 and a["kind"] == "STOCK"
