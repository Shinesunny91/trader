# NSE Intraday Signal Lab

A local research and paper-trading workspace for NSE intraday equities: a
Streamlit app, scheduled jobs that publish each day's trade list to your phone,
and the backtests behind it.

> Research tool, not investment advice. Every number below is a backtest or a
> paper trade on free Yahoo data; past edge can decay. Validate with your own
> broker's fills before risking money.

## The validated trade: the gap-reversal book

Indian stocks earn their return overnight and give part of it back during the
session, and the give-back concentrates in names that gapped up the day before.
The book shorts those names for one session:

| | |
|---|---|
| Universe | the 300 most liquid NIFTY 500 names by 20-session rupee turnover, price ≥ ₹50 |
| Signal | **yesterday's** overnight gap, `open(t-1) / close(t-2) − 1` — known the evening before |
| Trade | **short the top 8**, equal weight (₹1.25L each on ₹10L, no leverage), MIS |
| Entry | at the **open** — a market order in the 09:00–09:07 pre-open auction |
| Stop | buy-stop (SL-M) at entry + 0.75 × the 14-day daily ATR |
| Exit | buy to cover at 15:15 |

Costs are Groww's MIS schedule at the actual position size plus 3 bps of
slippage per leg. The configuration was fixed on the first two rows below
before the held-out window was run once:

| window | result |
|---|---|
| 10 years of daily bars (2,486 sessions) | +24.7 bps/trade net, t = 9.2, **every year 2016–2026 positive**, max drawdown 14.4% |
| development, 49 sessions of 5m bars (07-01 → 09-07) | +11.6%, profit factor 1.34, both halves ≈ +5.8% |
| **held out, last 14 sessions (09-08 → 09-28)** | **+10.6% (₹1,06,068 on ₹10L), 10 of 14 days up, profit factor 2.63** |
| same 14 sessions, the model-ranked book it replaced | −0.3% |

Random shorts from the same universe with the same stop made +1.4% ± 1.9%
over those 14 sessions, so the stock selection is ~4 sd above chance. That
window was also unusually kind (NIFTY fell intraday most days). **Plan on the
decade figure — roughly +0.2% of capital a day on average, with losing weeks
and months — not on 10% a fortnight.** Details and everything that did not
work: [`docs/research-log.md`](docs/research-log.md).

### A trading day

| IST | what happens |
|---|---|
| 08:45 | `nse-gap-picks` ranks the universe and pushes the 8 shorts + 4 reserves to your phone |
| 09:00–09:07 | you place MIS market SELL orders in the pre-open auction; skip a name your broker won't let you short (ASM/T2T) and take the next reserve |
| 09:18 | `nse-gap-levels` pushes each name's exact SL-M trigger from the opening print |
| 15:15 | you cover everything still open |
| 15:40 | `nse-gap-record` paper-trades the session and pushes the result |

Entering late is not a small detail: at a 09:20 entry the development window
was roughly break-even after costs.

## Quick start

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -e '.[dev]'
./run_tests.sh -q                      # 237 tests
./run_app.sh                           # app at http://127.0.0.1:8501
./deploy/install.sh                    # systemd user units: app + all timers
```

```bash
python scripts/gap_reversal.py picks            # next session's short list
python scripts/gap_reversal.py levels           # after 09:15: SL-M prices
python scripts/gap_reversal.py record           # after the close (catches up missed days)
python scripts/gap_reversal.py backtest --decade --save   # re-run the evidence
python scripts/health_check.py                  # is everything actually working?
```

Phone alerts use [ntfy](https://ntfy.sh): install the app and subscribe to the
topic shown in the app's sidebar (`NTFY_TOPIC` overrides it).

## Layout

```text
app.py                      Streamlit entry point (also used by Streamlit Cloud)
src/nse_intraday_ai/
  gap_reversal.py           the validated strategy: selection, execution sim, backtests
  gap_reversal_ui.py        its app page (default mode)
  app.py                    the Streamlit app and the research modes
  scan_service.py, strategies.py, scanner.py, ...   voting-engine scanner (research)
  costs.py                  Groww / NSE charge schedule
  candle_cache.py, data.py  SQLite candle cache and Yahoo providers
scripts/
  gap_reversal.py           picks | levels | record | backtest
  scanner_daemon.py         one voting-engine scan cycle (desktop alerts only)
  catchup_fetch.py          refill 5m bars after downtime (Yahoo keeps 60 days)
  fetch_daily.py            10 years of daily bars
  backfill_context.py       macro/context symbols for the scanner
  health_check.py           units, app, data freshness, today's book
  swing_backtest.py, swing_today.py   intra-week book (daily bars)
  ticket_api.py             JSON API for the Android app
  start_terminal.sh, stop_terminal.sh
deploy/                     systemd units + install.sh
android/                    WebView companion app (build_apk.sh, install_apk.sh, run_server.sh)
tests/                      pytest suite
docs/research-log.md        every study, including the ones that failed
data/                       local only: candle cache, pick lists, paper book, state
```

`data/gap_reversal/` holds `picks.json` (today's list), `paper_book.csv` (the
forward record, one row per trade) and `backtest.json` (what the app charts).

## The other modes

The sidebar keeps the older tools as research views. None of them has passed
an out-of-sample test, and none pushes to the phone:

- **NIFTY 50/100/500 scanner** — the 17-strategy voting engine. Its 5-minute
  signals have ~+0.8 bps of gross edge against a ~10 bps round trip; every
  gate and ranking model built on them lost money out of sample.
- **Recommendation workbench**, **Intra-week book** — daily-bar horizons with
  their ten-year numbers beside every pick (NSE intra-week lost money; the
  commodity book is the only positive one).
- **Backtest**, **Single symbol** — the voting engine on demand.

The scanner daemon (`nse-scanner.timer`) still runs during NSE hours to keep
the candle cache and shadow learner current. `NSE_DAEMON_UNIVERSES=nse,commodity`
and `NSE_PUSH_SCANNER_ALERTS=1` restore commodity scanning and phone pushes.

## Maintenance

- After the machine has been off: `python scripts/catchup_fetch.py` (5m bars)
  and `python scripts/fetch_daily.py --years 1 --only nse` (daily bars);
  `gap_reversal.py record` backfills the paper book by itself.
- The workbench's NSE-flows panel reads parquet; under pandas 3, tz-aware
  parquet needs `pyarrow >= 25`. Nothing in the gap-reversal path uses parquet.
- Yahoo is not a broker feed: it lags, rate-limits and writes placeholder bars
  on holidays (filtered). Check picks against your broker before trading.
