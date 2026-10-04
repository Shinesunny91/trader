from __future__ import annotations

from datetime import date

import pytest

from nse_intraday_ai import nse_preopen as PO


def _payload(stamp: str, n: int = 10):
    data = []
    for i in range(n):
        data.append({"metadata": {"symbol": f"S{i}", "series": "EQ", "previousClose": 100.0 + i},
                     "detail": {"preOpenMarket": {"IEP": 101.0 + i, "finalQuantity": 1000 * (i + 1),
                                                  "totalBuyQuantity": 500, "totalSellQuantity": 300,
                                                  "atoBuyQty": 10, "atoSellQty": 5,
                                                  "lastUpdateTime": stamp}}})
    data.append({"metadata": {"symbol": "BEX", "series": "BE", "previousClose": 50.0},
                 "detail": {"preOpenMarket": {"IEP": 52.0, "lastUpdateTime": stamp}}})
    return {"data": data}


def test_parse_and_final_for_today():
    frame = PO.parse(_payload("30-Sep-2026 09:11:40"))
    final = PO.final_for(date(2026, 9, 30), frame)
    assert set(final["series"]) == {"EQ"} and len(final) == 10
    opens = PO.opens(final)
    assert opens.loc["S3", "open"] == 104.0 and opens.loc["S3", "prev_close"] == 103.0


def test_refuses_yesterdays_snapshot_and_unfinished_auction():
    with pytest.raises(PO.PreOpenNotReady, match="not 2026-09-30"):
        PO.final_for(date(2026, 9, 30), PO.parse(_payload("29-Sep-2026 09:09:12")))
    with pytest.raises(PO.PreOpenNotReady, match="too early"):
        PO.final_for(date(2026, 9, 30), PO.parse(_payload("30-Sep-2026 09:05:00")))


def test_auction_cutoff_follows_the_2026_09_07_restructure():
    # Before: entry closed at exactly 09:08, so a 09:08:30 stamp is final.
    assert len(PO.final_for(date(2026, 9, 4), PO.parse(_payload("04-Sep-2026 09:08:30")))) == 10
    # After: entry closes at a random moment in 09:08-09:10; 09:09 may be indicative.
    with pytest.raises(PO.PreOpenNotReady, match="too early"):
        PO.final_for(date(2026, 9, 30), PO.parse(_payload("30-Sep-2026 09:09:12")))
