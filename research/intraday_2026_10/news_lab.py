"""News-drift lab: intraday reaction to NSE corporate announcements (hourly bars, ~2y).

Chan (2003, JFE): moves WITH news drift, moves without news reverse.  For each
announcement time-stamped inside market hours (09:15-14:15) on a NIFTY-500 name:
reaction = move from the last hourly close before the announcement to the open of
the next bar, in ATR units.  If |reaction| >= k, trade in the reaction's direction
(follow) or against it (fade) from that open to 15:15, stop stop_atr x ATR.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from hourly_lab import SQ, load, sim  # noqa: E402
from intraday_lab import COST_BPS, ROOT  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
NOISE = {"ESOP/ESOS/ESPS", "Trading Window", "Certificate under SEBI (Depositories and Participants) Regulations, 2018",
         "Copy of Newspaper Publication", "Loss of Share Certificates", "Duplicate Share Certificate",
         "Shareholders meeting", "Analysts/Institutional Investor Meet/Con. Call Updates", "Updates",
         "Change in Directorate", "Record Date", "Book Closure", "Investor Presentation", "Press Release",
         "Spurt in Volume", "Price movement", "Clarification - Financial Results", "General Updates"}


def main():
    days, symbols, arr, atr, _ = load()
    di = {d: i for i, d in enumerate(days)}
    si = {s.removesuffix(".NS"): i for i, s in enumerate(symbols)}
    files = sorted(glob.glob(str(ROOT / "data/nse_corp/announcements/*.parquet")))
    ann = pd.concat(pd.read_parquet(f, columns=["symbol", "ts", "desc"]) for f in files if Path(f).stem >= days[0][:7])
    ann["ts"] = pd.to_datetime(ann.ts, errors="coerce")
    ann = ann.dropna(subset=["ts"]).reset_index(drop=True)
    ann["day"] = ann.ts.dt.date.astype(str)
    mins = ann.ts.dt.hour * 60 + ann.ts.dt.minute
    ann["bar_in"] = (mins - 555) // 60
    ann = ann[(mins >= 555) & (mins < 855) & ann.day.isin(di) & ann.symbol.isin(si)]
    ann = ann.copy()  #          # hourly bar containing the announcement
    ann["material"] = ~ann.desc.isin(NOISE)
    ann = ann.sort_values("ts").drop_duplicates(["symbol", "day"])   # first announcement per name/day
    print(f"{len(ann)} in-session announcements on {ann.day.nunique()} days; material {int(ann.material.sum())}", flush=True)
    print(ann.desc.value_counts().head(25).to_string(), flush=True)
    cut = days[int(len(days) * 0.6)]
    res = {}
    for only_material in (False, True):
        for k in (0.0, 0.3, 0.6, 1.0):
            for follow in (True, False):
                rows = []
                for r in ann[ann.material | (not only_material)].itertuples():
                    d, s, b = di[r.day], si[r.symbol], int(r.bar_in) + 1
                    if b >= SQ or np.isnan(atr[d, s]) or atr[d, s] <= 0:
                        continue
                    before = arr["close"][d, s, b - 2] if b >= 2 else arr["open"][d, s, 0]
                    entry = arr["open"][d, s, b]
                    if np.isnan(before) or np.isnan(entry):
                        continue
                    react = (entry - before) / atr[d, s]
                    if abs(react) < k or react == 0:
                        continue
                    sd = int(np.sign(react)) * (1 if follow else -1)
                    g, why = sim(arr, d, s, b, sd, entry, entry - sd * 0.5 * atr[d, s])
                    rows.append((r.day, g, why, r.desc))
                if not rows:
                    continue
                t = pd.DataFrame(rows, columns=["day", "gross", "why", "desc"])
                t["net"] = t.gross - COST_BPS
                key = f"{'material' if only_material else 'all'}_k{k}_{'follow' if follow else 'fade'}"
                out = {}
                for p, m in (("IS", t.day < cut), ("OOS", t.day >= cut)):
                    x = t[m]
                    dm = x.groupby("day").net.mean()
                    out[p] = {"n": int(len(x)), "net": round(float(x.net.mean()), 2), "gross": round(float(x.gross.mean()), 2),
                              "t": round(float(dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm)))), 2) if len(dm) > 2 else 0.0,
                              "hit": round(float((x.net > 0).mean()), 3)}
                res[key] = out
                print(f"{key:<26} " + "  ".join(f"{p}: net {v['net']:+6.1f} (gross {v['gross']:+6.1f}) t={v['t']:+5.2f} n={v['n']}"
                                                for p, v in out.items()), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
