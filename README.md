# NSE Gap-Reversal Book

An NSE intraday short book that has been validated walk-forward. Every trading morning it shorts the
liquid stocks most likely to give back yesterday's overnight gap-up. Picks are ranked by a
gradient-boosted model that learns daily and is re-validated weekly. A paper book records
every session from official NSE data.

Developer: **Shine** · Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
Experiments and decisions: [docs/research-log.md](docs/research-log.md)

## The trade

| | |
|---|---|
| Universe | The 300 most liquid NSE EQ stocks by 20-session traded value, point-in-time. Price ≥ ₹50; names that were trade-for-trade in the last 20 sessions are excluded. |
| Ranking | HistGradientBoosting ranker over ~190 point-in-time features. It falls back to the gap rule (yesterday's overnight gap) whenever its live paired record says so. |
| Size | Top 8, equal weight (₹1.25 L each on ₹10 L), MIS, no leverage. |
| Entry | **Pre-open SELL LIMIT at prev close × (1 − 0.75%)**, rounded up to the tick. Place it 09:00–09:08; **cancel anything unfilled at 09:15**. |
| Stop | BUY SL-M at fill + 0.75 × 14-day ATR. |
| Exit | Cover at 15:15. |

### Evidence (net of 13 bps round-trip, walk-forward, quarterly refits)

| | bps / trade | t | Sharpe |
|---|---|---|---|
| Gap rule, 10 years (2016–2026) | +31 | 10.5 | 3.3 |
| Gap rule + pre-open limit (adopted 2026-10) | +4.8 bps/day on capital over the rule | paired t 4.8 | 3.3 → 4.3 |
| Ranker vs rule, 2024-10 → 2026-10 (first promotion) | +50.1 vs +30.0 | paired t 3.7 | — |

Live paper record since 2026-09-29 is in the app (Performance page) and in
`data/gap_reversal/paper_book.csv`.

## Data sources (all free)

- NSE equity bhavcopy (prices, delivery %, trades), F&O bhavcopy (OI, basis, PCR), MWPL / ban / short-sales, and participant-wise OI.
- **NSE corporate filings**: board-meeting calendar, financial results, Integrated Filing and all announcements (2016+). These are catalyst features: results in the gap window, exchange price-movement queries, announcement counts. The features are built, but they are gated until their A/B passes (`features.UNVALIDATED_FEATURES`).
- NSE pre-open auction snapshots, archived daily from 2026-10.
- Yahoo: global indices, VIX, FX, commodities, US 10y, NSE sector indices, and 5 ADRs.
- NSE holiday master, used for the trading calendar.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
deploy/install.sh                      # systemd user timers + dashboard
PYTHONPATH=src .venv/bin/python -m pytest -q -W error      # 90+ tests, zero warnings
```

Day to day, everything runs on timers (see ARCHITECTURE §4). Manual commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/gap_reversal.py picks --push      # today's list
PYTHONPATH=src .venv/bin/python scripts/gap_reversal.py record            # paper-record a session
PYTHONPATH=src .venv/bin/python scripts/learn.py                          # retrain + walk-forward review
PYTHONPATH=src .venv/bin/python scripts/health_check.py                   # system health
PYTHONPATH=src .venv/bin/python scripts/daily_monitor.py --push           # live vs expected
PYTHONPATH=src .venv/bin/python scripts/compare_rankers.py morning morning_corp   # research A/B
./run_app.sh                                                              # dashboard on :8501
```

Phone alerts use ntfy: `PYTHONPATH=src .venv/bin/python -m nse_intraday_ai.alerts --set-ntfy-topic <random-topic>`.

## Safety notes

- This is research and paper trading. Verify costs against a contract note before trading real money.
- The dashboard (:8501) and `ticket_api` (:8502) have no login, so keep them on localhost or a trusted LAN.
- An ntfy topic is effectively a password, so use a long random one.
- Retired components (the 5-minute voting scanner and the swing book) are in git tag `pre-cleanup`, and their state is in `data/archive/`.
