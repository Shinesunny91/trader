# NSE Intraday Signal Lab

A local research and paper-trading workspace for NSE intraday equities: a
Streamlit app, scheduled jobs that publish each day's trade list to your phone,
and the backtests behind it.

> Research tool, not investment advice. Every number below is a backtest or a
> paper trade on free Yahoo data; past edge can decay. Validate with your own
> broker's fills before risking money.

## The trade: gap-reversal shorts, ranked by a learned model

Indian stocks earn their return overnight and give part of it back during the
session, and the give-back concentrates in names that gapped up the day before.
The book shorts such names for one session — intraday (MIS) in ordinary
cash-market stocks, decided before the open:

| | |
|---|---|
| Universe | the 300 most liquid NSE **EQ** stocks by 20-session traded value *as of the previous day* (point-in-time: delisted and demoted names included), price ≥ ₹50, none that were trade-for-trade in the last 20 sessions |
| Ranking | a gradient-boosted **ranker** over ~150 point-in-time features (below), trained walk-forward on the day's relative short return; falls back to the plain gap rule (yesterday's overnight gap) whenever it is not proven |
| Trade | **short the top 8**, equal weight (₹1.25L each on ₹10L, no leverage), MIS |
| Entry | at the **open** — a market order in the 09:00–09:07 pre-open auction |
| Stop | buy-stop (SL-M) at entry + 0.75 × the 14-day ATR |
| Exit | buy to cover at 15:15 |

### What it learns from

Everything is dated strictly before the session it ranks (`features.py`):

- **NSE equity bhavcopy** — every EQ stock every day since 2016: gaps, returns,
  ranges, **delivery %** (bought to hold vs day-traded), trade size, series.
- **NSE F&O bhavcopy** — futures OI build-up or covering, futures basis,
  options put/call OI, F&O membership; NIFTY / BANKNIFTY positioning.
- **NSE crowding and shorting** — OI against the market-wide limit, the next
  day's F&O ban list, institutional short sales.
- **NSE participant OI** — FII / DII / Pro / Client net index and stock futures.
- **Global and macro** — US/EU/Asian index closes, VIX, USDINR, DXY, crude,
  gold, copper, US 10y, NSE sector and midcap indices, and the New York ADRs
  of INFY, WIPRO, HDFCBANK, ICICIBANK, DRREDDY.
- **Cross-sectional anomalies from the literature** — multi-horizon overnight
  vs intraday "tug of war", idiosyncratic volatility, skewness, the MAX effect,
  Amihud illiquidity, market- and sector-residual reversal, gap streaks.
- **Today's opening prices** (09:09 re-rank only) — NSE's pre-open call
  auction fixes every stock's open at 09:08; the final list uses today's gap
  and its cross-sectional context.  Each day's full auction snapshot (order
  imbalance, auction volume) is archived for the learner to test once enough
  history exists.

### Evidence

Walk-forward throughout: every score comes from a model trained only on earlier
sessions. Choices were made on 2019-01 → 2025-08; the last 13 months were held
out and evaluated once. Net of a 13 bps round trip:

| | development 2019-01 → 2025-08 (1,644 sessions) | held out 2025-09 → 2026-09 (267 sessions) |
|---|---|---|
| gap rule | +29.9 bps/trade, t 8.0, Sharpe 3.1 | +31.5, t 3.5, Sharpe 3.4 |
| **ranker** | **+68.1, t 16.6, Sharpe 6.5, max DD 8.1%** | **+45.2, t 4.4, Sharpe 4.3, max DD 8.0%** |
| ranker − rule, paired by day | +38.2 bps, t 13.0 | +13.7 bps, t 1.8 (better in 10 of 13 months) |

The ranker's *lead* over the rule has shrunk recently (about +14 bps a day in
2025 and in the held-out year, against about +38 before) while its own edge
held at about +45 bps/trade — which is why the guard, the drift alarm and the
weekly re-validation exist.

Why it is believable, in one paragraph: the ranker's picks are as liquid as the
rule's; its median trade beats its mean (not a few lucky days); it keeps its
lead within the 100 most liquid names, at 30 bps costs, and under a
deliberately pessimistic per-stock spread model under which the rule breaks
even; the same pipeline trained on shuffled labels scores +9.8 bps; and live
feature rows rebuilt from data truncated before each session match the
research panel exactly. Blending the rule back in, confidence gating,
volatility-weighted sizing and a long leg were all tested and all made it
worse or no better — see [`docs/research-log.md`](docs/research-log.md).

**Plan on the long-run figure, not on the best fortnight.** Part of any short
book's return is the market's intraday drift; a strong up-day hurts all eight
positions at once.

### How it improves itself

| when | what |
|---|---|
| every morning | pull yesterday's official NSE files → score **yesterday** for both the ranker and the gap rule on what actually happened → extend their paired record, update the drift alarm → rank today with the same code that built the training data |
| every morning | **guard**: if the ranker's last 60 sessions were worse than the rule's with t < −2, trade the rule until the evidence clears (it never fired 2019-2025) |
| every morning | **drift alarm**: a CUSUM on the live book's daily return against its backtest; alerts on the phone when the shortfall stops looking like luck |
| Saturdays, and any evening the drift alarm is up | rebuild the history, walk-forward the last two years for both models, fit challengers on everything, **promote only if** the 08:45 model beat the rule and the 09:09 model beat the 08:45 model (net of its later entry) with t ≥ 2 |

### A trading day

| IST | what happens |
|---|---|
| 08:45 | `nse-gap-picks` refreshes the data, learns from yesterday, ranks today and pushes a **preliminary** list (8 shorts + 4 reserves) |
| 09:00–09:07 | *either* place MIS market SELL orders for the 08:45 list in the pre-open auction … |
| 09:09 | … *or* wait for `nse-gap-final`: the list re-ranked on today's opening prices (if that model is promoted), pushed as **FINAL** — then SELL at market at 09:15:00. Never both. |
| 09:18 | `nse-gap-levels` pushes each name's exact SL-M trigger from the opening print |
| 15:15 | you cover everything still open |
| 15:40 | `nse-gap-record` paper-trades the session on 5-minute bars and pushes the result |
| 19:00 | `nse-gap-learn --if-needed`: Saturdays (or on a drift alarm) retrains and re-validates both models; pushes the verdict |

Entering late is not a small detail: at a 09:20 entry the earlier development
window was roughly break-even after costs.

## Quick start

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -e '.[dev]'
./run_tests.sh -q                      # 265+ tests
python scripts/backfill_nse.py         # first time: 10 years of NSE archives (~45 min, resumable)
python scripts/learn.py                # first model: walk-forward, fit, promote on evidence
./run_app.sh                           # app at http://127.0.0.1:8501
./deploy/install.sh                    # systemd user units: app + all timers
```

```bash
python scripts/gap_reversal.py picks            # next session's short list
python scripts/gap_reversal.py levels           # after 09:15: SL-M prices
python scripts/gap_reversal.py record           # after the close (catches up missed days)
python scripts/gap_reversal.py backtest --decade --save   # the gap rule on 5m bars + decade
python scripts/learn.py --push                  # weekly retrain / re-validate / promote
python scripts/health_check.py                  # is everything actually working?
```

Phone alerts use [ntfy](https://ntfy.sh): install the app and subscribe to the
topic shown in the app's sidebar (`NTFY_TOPIC` overrides it).

## Layout

```text
app.py                      Streamlit entry point (also used by Streamlit Cloud)
src/nse_intraday_ai/
  nse_bhav.py, nse_fo.py, nse_extra.py   NSE archives: equity, F&O, crowding/shorts
  features.py               point-in-time features (research = training = live)
  ranker.py                 walk-forward gradient-boosted ranker, artifacts, champion
  meta.py                   guard, paired track record, drift alarm (Hedge as diagnostic)
  pipeline.py               morning (learn + rank) and weekly (retrain + promote) loops
  gap_reversal.py           book mechanics: picks, 5m execution sim, backtests
  gap_reversal_ui.py        its app page (default mode)
  app.py                    the Streamlit app and the research modes
  scan_service.py, strategies.py, scanner.py, ...   voting-engine scanner (research)
  costs.py                  Groww / NSE charge schedule
  candle_cache.py, data.py  SQLite candle cache and Yahoo providers
scripts/
  gap_reversal.py           picks | levels | record | backtest
  learn.py                  weekly retrain, walk-forward review, promotion
  backfill_nse.py           one-shot / repair download of every NSE source
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
forward record, one row per trade) and `backtest.json`; `data/models/` holds
versioned ranker artifacts, `champion.json`, `meta_state.json` (guard record,
drift alarm) and `weekly_report.json`.

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

- After the machine has been off: the morning job refetches the last 20 days
  of NSE files by itself; for longer gaps run `python scripts/backfill_nse.py
  --since <date>`. `gap_reversal.py record` backfills the paper book by itself.
- If the NSE path fails (archive down, data incomplete), `picks` falls back to
  the gap rule on Yahoo data with its own freshness guard, and says so.
- The workbench's NSE-flows panel reads parquet; under pandas 3, tz-aware
  parquet needs `pyarrow >= 25`. Nothing in the gap-reversal path uses parquet.
- Yahoo is not a broker feed: it lags, rate-limits and writes placeholder bars
  on holidays (filtered). Check picks against your broker before trading.
