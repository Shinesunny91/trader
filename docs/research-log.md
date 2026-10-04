# Research log

The history of what this workspace tested, kept because the negative results are
as useful as the positive one. **Read it as a record, not as instructions**:
the 2026-09-28 cleanup removed the ~45 one-off study scripts, the model-ranked
ticket book (`sim_today.py` + `build_dataset.py` + `train_signal_model.py`) and
the candidate paper track, so many commands below no longer run. Every removed
file is in git history:

```bash
git show fa1b3e2:scripts/<name>.py        # e.g. overnight_drift_study.py
git checkout fa1b3e2 -- scripts/<name>.py # restore one
```

Newest first.

## 2026-10-04 — cleanup, catalyst data, entry limit, and an honest A/B harness

**Cleanup.** The voting scanner (17-strategy 5-minute ensemble, proven
unprofitable after costs) and the swing book were retired. That removed 28 modules plus their
scripts, tests and units (tag `pre-cleanup`; state in `data/archive/`). The
package is now 18 modules around one validated book. See `docs/ARCHITECTURE.md`.

**Oct-1 post-mortem (−₹3,149).** The rule ranks by *yesterday's* gap. Its four
losers gapped up *again* on Oct 1 and kept rising (MOLBIO +15% open-to-close
on 2.8× volume). Every stop filled beyond its level, so the stops limited the loss.
On 10 years of data, picks with today's gap0 < −0.5% lose 30.6 bps (t −5.1),
while picks with gap0 > +1% are the best trades (+70.9). Rejected, all worse than the baseline:
skipping gap0 > 0 names, ranking by gap1/ATR, excluding high-ATR names,
inverse-ATR sizing, and a market-gap day filter.

**Pre-open SELL LIMIT at prev_close × (1 − 0.75%)** (NSE tick, rounded up).
On the rule book (10 y): +4.8 bps/day on capital, paired t 4.84, Sharpe 3.31 → 4.27.
The effect holds across limits from −0.25% to −1.5%.
On the ranker's picks (walk-forward 2019-01 → 2026-10): **−0.4 bps, t −0.3** — no
effect. Adopted for rule-ranked lists only (`GapReversalConfig.entry_limit_for`);
ranker lists enter at the open.

**NSE corporate events (new source).** `nse_corp` pulls board meetings,
results, the Integrated Filing and all announcements (1.3 M since 2016), strictly point-in-time.
Rule-book diagnostic, 2017+, net bps/trade:

| flag | with | without | Welch t |
|---|---|---|---|
| results in the gap1 window | +14.1 (n 2,000) | +33.2 | −3.1 |
| results in last 3 sessions | +17.7 | +34.2 | −3.1 |
| exchange price-movement query (2 d) | +72.5 (n 381) | +30.4 | +1.8 |
| gap1_z top vs bottom tercile | +25.6 | +41.6 | −3.0 |
| ≥ 3 announcements in gap window | +17.4 | +35.0 (none) | — |

This matches the literature: news-driven gaps drift, unexplained ones revert (Chan 2003; Savor 2012).
Hard skips with refill did not help (−0.2 to −1.5 bps/trade).
**Walk-forward A/B** (`compare_rankers.py`, 2019-01 → 2026-10, 1,913 sessions, live
book: 300 names, T2T filter, limit): morning ranker +65.4 bps/slot (t 19.1,
Sharpe 6.9); + catalyst features +67.6 (Sharpe 7.3); paired **+2.2 bps, t 1.41,
Holm p 0.08 → not adopted**. The features stay built but gated
(`features.UNVALIDATED_FEATURES`). Re-test once the 2025+ Integrated-Filing
period has more data.

**Ranker variants (open-feature panel, superseded harness).** LightGBM
regression +1.5 bps (t 0.96) and 3-seed HGB +2.2 (t 1.86): not significant.
LambdaRank (NDCG, integer grades) −19.2 (t −7.1): worse. HGB stays.
*Caveat*: that harness included today's opening gap (an open-model setup) and inflated every
variant to ~80 bps. It was rewritten to mirror the 08:45 book.

**Bugs found by the cleanup.**
- A SyntaxError in the 08:45 message would have crashed Monday's job.
- Champion models failed to load after a config refactor.
- The fallback's freshness guard accepted a 3-day-stale "yesterday".
- `walk_forward` recency weighting crashed (TimedeltaIndex has no `.dt`).
- ±inf features (pcr1, amihud20, pcr_chg1, fo_banknifty_oi_chg) went into tree binning.

All are fixed, with tests. A script-level smoke suite now exists.

**Multiple-testing tally.** About 30 variants have been tried on this panel so far (DSR `TRIALS_SO_FAR`).

## 2026-09-29 (round 2) — point-in-time data, a learned ranker, and a self-correcting loop

**Question.** Can the gap-reversal book be improved by ranking on more
information than yesterday's gap — and does any improvement survive the
tests that sank every earlier model in this repo?

### New information, all point-in-time, all from public archives

| source | what it adds | first date |
|---|---|---|
| NSE equity bhavcopy (+ MTO) | **every** EQ/BE stock each day — delisted and demoted ones included; NSE's prev close; delivery %; trade count; series (EQ = MIS-shortable, BE = trade-for-trade) | 2016-06 |
| NSE F&O bhavcopy | futures OI across expiries, basis vs spot, options put/call OI, point-in-time F&O membership | 2016-06 |
| NSE combineoi / ban / short-selling | crowding vs the market-wide position limit, next-day F&O ban, institutional short sales | 2016-06 |
| NSE participant OI | FII / DII / Pro / Client net positioning in index and stock futures | 2016-08 |
| Yahoo daily | S&P 500, Nasdaq, Russell, VIX, DAX, FTSE, Nikkei, Hang Seng, KOSPI, Shanghai, EEM/INDA, DXY, USDINR, crude, gold, copper, US 10y, NSE sector and midcap indices, 5 Indian ADRs | 2016-09 |

Timing rule, enforced in one place (`features.py`): session t sees only data
dated before t.  Asian closes come after the NSE open, so only their t−1 close
is used; the F&O ban for t is read from "No Fresh Positions" at the end of t−1.

### Three data traps found on the way

* **Survivorship.** Yahoo is queried for *today's* NIFTY 500, so its decade of
  history holds only names that became winners.  The point-in-time universe
  passes through 1,685 distinct stocks over 2016-2026, not ~500.  The gap rule
  does **better** on it (+29.9 vs +21.3 bps/trade, 2019-2025): the missing
  names are mostly losers, which a short book wants.
* **Unadjusted splits.** NSE's PREV_CLOSE is not always adjusted on ex-dates
  (BAJFINANCE 2025-06-17 "opens −90%": a 1:10 split).  No stock can open 30%
  down or 35% up on a normal day, so those days (1,276) are masked.
* **Stale archive copies.** On holidays the archive can serve the previous
  session's file under the holiday's name (2026-09-14 returns 2026-09-11
  verbatim); the date printed inside each file now decides.  One file
  (2022-08-08) is an .xlsx behind a .csv name; the older bhavcopy + MTO pair
  covers it.

### Protocol

Walk-forward only: every score comes from a model fit on sessions strictly
before it (yearly refits for research; production retrains weekly).  Model and
book choices were made on **DEV = 2019-01 .. 2025-08**; **HOLD = 2025-09 ..
2026-09** was evaluated once, at the end.  A day's book: top 8 of the 300 most
liquid EQ stocks (as of t−1), excluding names that were trade-for-trade in
the last 20 sessions, equal weight, 13 bps round trip.

### DEV results (2019-01 .. 2025-08, 1,644 sessions)

| book | net bps/trade | t | Sharpe | max DD | worst year |
|---|---|---|---|---|---|
| gap rule (live until now) | +29.9 | 8.0 | 3.1 | 13.9% | +19.6 (2022) |
| random 8 shorts | +2.2 | 0.7 | 0.3 | 48% | — |
| ridge, all features | +40.4 | 9.7 | 3.8 | 14.6% | |
| boosted ranker, gap-family features only | +57.4 | 14.4 | | | |
| boosted ranker, no F&O features | +65.1 | 15.4 | 6.0 | 9.3% | +44.8 (2025) |
| **boosted ranker, all features** | **+68.1** | **16.6** | **6.5** | **8.1%** | +44.8 (2025) |
| same, recency-weighted (half-life 2y) | +66.2 | 16.0 | 6.2 | 10.7% | +53.4 (2025) |

Ranker minus rule, paired by day: **+38.2 bps, t = 13.0**, better on 63% of days.

**Is it real?**
* *Not illiquidity:* the ranker's picks trade ₹95 cr/day at the median (rule:
  ₹92 cr); within the 100 most liquid names it still makes +39.3 vs +18.6.
* *Not a few lucky days:* median trade +86.6 bps (above the mean); the top 1%
  of trades are 15% of the P&L (rule: 28%); fewer stops (14% vs 19%).
* *Costs:* at a 30 bps round trip +51.0 (rule +12.9).  With each stock charged
  fees plus its own Abdi-Ranaldo spread estimate — deliberately pessimistic,
  the estimator overstates spreads for volatile names — ranker +39.8, rule
  +1.0: only the ranker clears it.
* *Noise floor:* trained on labels shuffled within each day, the same
  pipeline scores +9.8 bps (t 2.8) — its selection drifts to extreme names that
  carry some of the gap effect — against +68.1 for the real model.
* *No look-ahead:* live feature rows rebuilt from data truncated before the
  session equal the history panel on every feature for 10 sessions audited
  across 2019-2026 (one EWM warm-up difference on one day).

**What did not help, and is not used:**
* *Blending the rule back in* (static 25-75%) or *Hedge weights* (+60.1):
  every unit of weight on the rule dilutes better picks.
* *Confidence gating* (trade only predicted edge > X): no gain.
* *A long leg* (ranker trained on long returns): +5.2 bps (t 1.5) before
  realistic costs — Indian stocks drift down intraday; rejected.
* *F&O-only universe* (always shortable): ranker +30.9, rule +9.0 — the edge
  lives mostly in ordinary cash-market names, so the book is not restricted.
* *Fewer names:* K=3 earns more per trade (+88.7) with more risk; K=8 has the
  best t-stat and lowest drawdown and stays.

### HOLD (2025-09-01 .. 2026-09-29, 267 sessions) — evaluated once

| book | net bps/trade | t | Sharpe | max DD |
|---|---|---|---|---|
| gap rule | +31.5 | 3.5 | 3.4 | 9.3% |
| **ranker** | **+45.2** | **4.4** | **4.3** | **8.0%** |
| Hedge blend (for comparison) | +39.8 | 4.1 | 3.9 | 6.7% |

Ranker minus rule: **+13.7 bps/day, t = 1.8**; the ranker beat the rule in 10 of
13 months.  Under the pessimistic per-stock spread costs: ranker +13.3, rule +1.2.

The honest reading: the ranker still wins out of sample, but its lead over the
rule has **decayed** — about +38 bps/day in 2019-2024, about +14 in 2025 (inside
DEV) and +14 in HOLD.  The absolute edge (+45 bps/trade) is intact; the part
the model adds on top of the rule is a third of what it was.  That is exactly
what the self-correcting loop below is for, and why the forward paper record,
not any backtest, is now the arbiter.

### The self-correcting loop (production)

* Morning: pull NSE/F&O/global data → score **yesterday** for rule and ranker
  on official bars → update the paired record, the guard and the drift CUSUM
  → build today's features through the same code → rank → publish.
* Guard: trade the ranker unless its last 60 sessions were worse than the
  rule's with t < −2 (never triggered 2019-2025: the rolling 60-day edge was
  negative in <1% of windows).
* Weekly: rebuild the history, walk-forward the last two years, fit a
  challenger on everything, promote only if its out-of-sample book beat the
  rule with t ≥ 2.

### Round 3 — literature anomalies, today's opening gap, adaptive memory

Asked for more features, faster prediction at the right moment and better
self-learning, three ideas were tested walk-forward on DEV (HOLD shown as a
second look, since round 2 had already opened it):

| model | DEV net bps/trade | DEV vs round-2 | HOLD net bps/trade | HOLD vs round-2 |
|---|---|---|---|---|
| round-2 ranker | +67.4 | — | +41.0 | — |
| **+ anomaly features** (tug-of-war 5/60d, idiosyncratic vol, skew, MAX, Amihud, residual reversal, gap streak, days since last session) | **+68.8** | +1.4 (t 0.8) | **+45.4** | +4.4 (t 1.0) |
| + today's opening gap (09:09 model, charged +5 bps) | +80.7 | +13.3 (t 5.0) | +39.5 | −1.5 (t −0.2) |
| + today's gap, recency-weighted (half-life 1y) | +76.4 | +9.1 | +37.3 | −3.7 |
| expanding + recency ensemble / Hedge over them | +82.5 | +15.1 | +38.0 | −3.0 |

* **Anomaly features: adopted** for the 08:45 model — small but in the same
  direction in both periods.
* **Today's opening gap: built, gated.**  NSE's pre-open call auction fixes
  every stock's open at 09:08 and publishes it (`nse_preopen.py`); a 09:09
  re-rank turns it into a final list for a 09:15 market entry.  On 1-minute
  bars, the day's biggest gap-ups trade +6 bps *higher* at 09:16 than in the
  auction (noisy, ±40 bps per trade), so the later entry costs nothing on
  average; it is still charged +5 bps.  The model was the best result of the
  whole study on 2019-2025 (every year better, +63 vs +45 in 2025) and did
  not beat the morning model in the held-out year.  It therefore runs only
  when the weekly walk-forward over the last two years shows it beating the
  morning model with t ≥ 2 — the loop turns it on if and when the evidence
  comes back, not a judgment made on one noisy year.
* **Adaptive memory (recency weighting, memory ensembles): rejected** — no gain
  in either period.
* The pre-open feed also carries what NSE does not archive (unmatched buy/sell
  quantities, auction volume).  Every snapshot is archived from now on, so
  auction imbalance can be tested once a few hundred sessions exist.

**Also measured and not adopted:** volatility-scaled sizing (Sharpe 6.55 vs
6.44, mean lower), confidence gating, stops other than 0.75 ATR for the
ranker's picks (0.5-1.0 ATR is a plateau; no stop doubles the drawdown).

**Event-driven learning.**  The learner now checks every evening: it retrains
on Saturdays, and on any evening the drift alarm is up, instead of waiting for
the weekend.

---

## 2026-09-28 — the gap-reversal book replaces the model-ranked book

**Why.** The live intraday tool had stopped making money: the model-ranked book
(`sim_today.py`, one trade a day from ~4,500 voting-engine signals) lost 0.3%
over the 14 sessions 2026-09-08..09-28 (4 up / 10 down), its own walk-forward
validation read −2.17% over 14 sessions (profit factor 0.18), and the paper
record had lost money since mid-August. The audit below (2026-08-19) had
already concluded that the 5-minute signal population has no edge to rank:
+0.8 bps gross against a ~10 bps round trip. No ranking fixes that.

**What changed in approach.** Stop chasing 5-minute moves of a few bps and trade
something whose typical move dwarfs the ~13 bps round trip: a *day-level*
position decided before the open. The overnight-drift study (August 2026) had
found Indian stocks earn their return overnight and give part of it back
intraday, concentrated in names that gapped up recently — but it never reached
the live tool.

**Protocol.** 5-minute data split into a development window (2026-05-21..09-07,
56 sessions incl. resampled 1m history) and a held-out test window (the last 14
sessions, 2026-09-08..09-28). Hypotheses were screened on 10 years of daily bars
(2,500 sessions) and on the dev window; the configuration was written down
before the test window was run once.

**Findings, dev window and decade:**

| hypothesis | result |
|---|---|
| same-day gap fade (short today's gap-ups at 09:20) | **artifact** — gap-ups kept rising after 09:20 (−35 bps gross/trade on the top 3). The 10-year daily "fade" happens inside the first five minutes, from the opening print, and cannot be captured. |
| random 5 shorts / longs, open→15:15 | +12 / −12 bps gross — the intraday drift is negative, so any short book starts with a tailwind |
| **short yesterday's biggest gap-ups** (`gap1`) | +19 bps gross at 09:20 entry on the dev window, both halves positive; 10 years: negative intraday return in the top decile in every year |
| yesterday's return, ATR%, 5-day return, first-bar direction/volume | weaker or unstable across halves |
| entry at 09:20 vs the open | at 09:20 the book is ~break-even after costs; at the open it is clearly positive. The signal is known the evening before, so enter in the pre-open auction. |
| protective buy-stop 0.5–1.0 × daily ATR | improves t-stat, worst trade and drawdown over 10 years; plateau, not a spike |
| universe size | stronger in less liquid names (top-300/500 by turnover) than top-100/200 |

**Locked configuration:** top-300 NIFTY 500 names by 20-session turnover, short
the 8 largest *yesterday's* gaps at the open, buy-stop at entry + 0.75 × daily
ATR14, cover at 15:15, equal weight on ₹10L at 1×, Groww MIS costs + 3 bps/leg.

**Results** (`python scripts/gap_reversal.py backtest --decade`):

| window | result |
|---|---|
| 10 years, daily bars, 2,486 sessions | +24.7 bps/trade net, t = 9.2, every calendar year 2016–2026 positive (weakest 2022, +6.4), max drawdown 14.4% |
| dev window, 49 native-5m sessions | +11.6%, profit factor 1.34, +5.9% / +5.6% in the two halves, max drawdown 3.2% |
| **held-out 14 sessions** | **+10.6% (₹1,06,068 on ₹10L), profit factor 2.63, 10 of 14 days up, max drawdown 1.2%** |
| same 14 sessions, random 8 shorts (same universe, stop, costs) | +1.4% ± 1.9% (200 draws) — the selection is ~4 sd above random |
| same 14 sessions, the model-ranked book it replaces | −0.3% |

The held-out window was unusually favourable (NIFTY fell 17.5 bps/day intraday
across it). The decade is the expectation: roughly +0.2% of capital a day on
average, with losing months.

**Data bug found on the way:** Yahoo writes placeholder daily bars on exchange
holidays (volume 0, O=H=L=C; 480 of them on the 2026-09-14 holiday — confirmed
in round 2 from NSE's own files, whose 09-15 PREV_CLOSE equals 09-11's close). Left in, the holiday
becomes "yesterday" and its zero gap replaces the real one.
`gap_reversal.load_daily` drops them.

**Also fixed:** the app had crashed on every page load since 2026-09-09
(`NameError: IST` in the market-phase banner added that day); the health
check's HTTP probe could not see it because Streamlit serves its shell page
regardless. The scanner daemon now scans NSE only and no longer pushes
voting-engine signals to the phone (a post-reboot burst of commodity catch-up
alerts had the ntfy topic rate-limited).

---

Local Python workspace for NSE intraday signal research, multi-strategy voting, and paper trading.

This is a research and simulation tool. It does not guarantee profit and it is not personal financial advice. For any real-money workflow, use an authorized broker or licensed NSE data feed, validate the strategy out of sample, and keep manual approval until the system has proven itself in shadow trading.

## Why Python

Python is the right first language for this project because the main risks are strategy validity, market data quality, transaction costs, slippage, and overfitting. Python has the strongest ecosystem for research, dashboards, ML, and broker integrations. C++ or Rust can be added later only for a narrow execution component if latency actually becomes the bottleneck.

## What The App Does

- Pulls intraday NSE-style symbols and major commodity futures through Yahoo Finance by default, or uses a synthetic demo stream.
- Computes EMA, RSI, ATR, VWAP, volume expansion, range, and momentum features.
- Runs multiple independent intraday strategies:
  - trend continuation
  - VWAP mean reversion
  - opening range breakout
  - volatility compression breakout
- Uses weighted voting before any signal becomes actionable.
- Builds a full trade plan: side, entry, target, stop loss, quantity, expected risk, expected reward, and confidence.
- Applies estimated round-trip commission and slippage to reward/risk and backtest P&L. The default round-trip commission is 0.5%.
- Paper trades in simulation mode with virtual capital.
- Tracks closed paper-trade outcomes and adjusts strategy weights conservatively.
- Scans the NIFTY 50 universe in parallel and only surfaces very high-confidence actionable plans.
- Scans a built-in commodity futures universe in parallel for metals, energy, and agricultural futures.
- Shows closest blocked candidates when no trade passes the strict filters.
- Runs a shadow learner that records candidate outcomes for future RL/contextual-bandit training.
- Replays NIFTY 50 candles with configurable backtest duration, target trade count, and loss controls.

## Multi-Strategy Voting

Multi-strategy voting is better than one strategy for this use case when the goal is fewer, higher-quality trades. Intraday signals are noisy; a single setup often fires during bad regimes. The ensemble only acts when:

- enough strategies agree on the same side,
- weighted vote share is high enough,
- confidence is above the configured threshold,
- reward/risk is still acceptable after estimated costs and slippage,
- the risk engine can size the trade within capital limits.

This lowers trade frequency, but that is intentional. The app is designed to wait unless the setup is unusually clean.

## NIFTY 50 Scanner

The scanner fetches the current NIFTY 50 constituent CSV from NSE/Nifty Indices infrastructure and caches it locally at `data/nifty50_symbols.csv`. It then scans the universe concurrently with a configurable worker count. With Yahoo Finance this is still best treated as a research scan, because free data can lag or rate-limit.

For the user's current constraint of no broker WebSocket, the app uses a best-effort free-data path:

- Yahoo Finance batch download for NIFTY 50 1-minute candles.
- Per-symbol Yahoo retry for symbols missing from the batch response.
- One NSE public NIFTY 50 index snapshot request for current LTP/freshness overlay.
- Configurable max quote age, defaulting to 120 seconds.

This is intentionally limited to NIFTY 50. It is not a replacement for a licensed market-data feed.

The scanner defaults were recalibrated by the July-2026 two-week event study
(5m bars, market context + event windows on):

- minimum confidence: 70% (positive in both study weeks; 75% flipped sign between weeks)
- minimum net reward/risk: 1.5
- minimum agreeing strategies: 2
- interval: 5m (1m signals were too small relative to costs — roughly +0.7 bps
  gross edge per candidate against an 18 bps round trip)

If no stock passes, the app shows no recommendation.

## Commodity Scanner

The commodity scanner uses Yahoo Finance futures symbols such as `GC=F`, `SI=F`, `CL=F`, `NG=F`, `HG=F`, `ZC=F`, and `ZW=F`. It is a research proxy for commodity futures, not a live MCX execution feed. The NSE quote freshness overlay is disabled for commodities because those symbols are not NSE equities; the scanner uses the latest Yahoo candle close instead.

For Indian MCX real-money workflows, add a broker or licensed commodity market-data provider before considering execution.

The signal engine treats commodities differently from NSE equities (added Jul 2026, validated on a 2-week event study):

- The opening-range breakout strategy does not vote on 24-hour futures — there is no meaningful session open, so its "opening range" was an arbitrary midnight slice.
- A commodity-specific time-of-day multiplier replaces the NSE session table (US-overlap evening liquidity is boosted; the thin 05:00–09:00 IST Globex stretch is suppressed).
- Fast scalp strategies that were negative in both study weeks on futures (`ema_scalp`, `momentum_burst_scalp`, `trend_continuation`) get reduced default vote weights.
- Commission defaults to 0.05% round-trip instead of 0.5% — futures costs are flat per lot (≈2–6 bps of notional), not equity-style percentage brokerage.
- Backtest exits are not clipped at the (IST) day boundary for commodities.
- Macro context feeds the score: DXY momentum (inverse for metals), S&P futures / Nikkei risk appetite (energy and industrial metals follow it, precious metals fade it).

Commodity scan defaults after the study:

- minimum confidence: 85% (this is where positive expectancy concentrated)
- minimum net reward/risk after costs: 1.5
- minimum agreeing strategies: 1 (with ORB removed, a 2-vote gate starved the scanner)
- minimum weighted vote share: 0.70
- require trained positive policy: enabled by default

## Unified Scan Pipeline

The Streamlit app and the headless scanner daemon run **the same code path** —
`nse_intraday_ai.scan_service.run_scan_cycle`. Before this, each wired its own
pipeline around `scan_universe_batch` and they drifted: different gates,
different vetoes, and a shared flat config file they clobbered for each other
(a commodity session once poisoned the NSE daemon's gates and it produced zero
signals for a morning). Now:

- `scan_service` owns the whole cycle: cache-first fetch → closed-bar
  evaluation → policy veto → meta-label veto → learner recording → signal
  logging → ranked output. The daemon is a thin wrapper that only decides
  *when* to scan (per-universe market hours) and *how* to notify (desktop
  notifications with per-day dedup); the app just renders the returned
  `ScanCycle`. What you see in the browser and what fires a notification can no
  longer diverge.
- Per-universe config sections in `scan_config.json` (`nse` / `commodity`), so
  one scanner tab can never overwrite the other's validated gates.

**Sampling model — "don't miss anything, as fast as possible":**

- Signals are computed on **closed bars only** (the timeframe the studies
  validated); a forming candle is never scored.
- A per-universe **bar cursor** (`data/scan_state.json`) records the last
  evaluated bar. Each cycle evaluates *every* newly closed bar since the
  cursor, so a late or skipped cycle still catches up — each closed bar is
  evaluated exactly once (no duplicates at fast cadence, no gaps at slow).
  Bars too old to enter are logged/learned and surfaced as `stale`, never
  silently dropped.
- Because signals only change on bar close, the expensive 500-symbol network
  fetch is **gated on a new bar actually being due**. On a 1-minute timer /
  5-minute timeframe, ~4 of every 5 ticks are cheap cache reads (~15 s, no
  network) and the tick after a bar closes does one fetch (~30–40 s) and
  catches it within a minute. The old code re-downloaded all 500 symbols every
  minute — ~200 s of wasted fetching on 4 of 5 ticks. A manual "Scan now"
  passes `force_fetch` to bypass the gate.

The systemd timer `nse-scanner.timer` fires every minute (`*:*:15`, just after
each bar boundary); the cursor makes overlapping or missed ticks harmless.

## What The August-2026 Study Changed

A 280,562-signal study over 49 sessions and 505 symbols
(`scripts/entry_timing_study.py` → `analyze_entry_timing.py` →
`entry_quality_search.py` → `macro_feature_study.py`) re-tested the engine's
core assumptions on held-out data. Four of them did not survive.

**Two silent bugs, both found by the study rather than by monitoring:**

- `scan_service` called `fetch_index_vix_context(for_commodities=...)`, a
  keyword that function never accepted. The `TypeError` was swallowed by a bare
  `except Exception`, so **every live scan from 2026-07-07 onward ran with
  `market_context = None`** — no NIFTY regime, no VIX, no sector indices, no
  global cues, no USDINR, no VWAP breadth — while this README described a fully
  wired context panel. The candle cache records the outage exactly: every
  equity context symbol stops on that date. Fixed, and `MarketContext` now
  carries a `fetch_error` field so a context outage can never be silent again.
- `event_risk.py` de-rated the **last Thursday** as NSE F&O expiry. NSE moved
  expiry to **Tuesday** on 2025-09-02 (BSE Sensex took Thursday). The engine
  spent a year penalising an ordinary session and leaving the real expiry
  unguarded. `macro_context.nse_expiry_weekday()` now owns the rule and returns
  Thursday for dates before the change, so old backtests stay correct.

**The engine's own gates were selecting below-average signals.**
corr(confidence, forward 6-bar return) = **+0.009**, and the ≥85% confidence
bucket was the *worst* of all (−3.58 bps). Out of sample, the shipped
`conf ≥ 70` gate picked signals averaging −1.20 bps against a −1.03 bps
all-signal baseline; requiring ≥2 agreeing strategies was worse (−1.73 bps).
Both gates cut volume by ~90% while making average quality slightly worse.

**"Signals arrive late and then reverse" is half true.** P(the first bar after
entry is adverse) is 52.8% — but it is ~52% in *every* extension bucket, so
there is no systematic post-signal reversal; that number is the cost of
entering at a bar close. The genuinely extended entries were the *best*
performers at 30 minutes (6-bar run-up > 3 ATR: +4.13 bps; volume z > 3:
+2.19 bps; RSI < 30: +3.30 bps). The real problem is the opposite of late:
**166K of 280K signals fired on below-average volume with no impulse at all**,
and those averaged −1.20 bps. The engine is not late, it is indiscriminate.

**Foreign equity markets do not predict NSE intraday moves.** S&P futures,
Nikkei, DAX and Hang Seng each showed a large in-sample correlation (+0.09 to
+0.10) and essentially zero out of sample (−0.003 to +0.0004); DAX, Nikkei and
gold flipped sign across the split. A session-phase-weighted blend of them was
also zero. They price the overnight gap, and the gap is already in the tape by
09:15. Three macro readings *did* survive: **NIFTY's own 1-hour momentum**
(out-of-sample decile spread +10.9 bps), **USDINR** inverted (+8.7 bps) and
**crude** inverted (−4.0 bps — India imports ~85% of its oil). Their combined
alignment score beat its own bottom decile in 8 of 10 rolling walk-forward
weeks. Calendar effects — day of week, expiry, month-end, season — were all
unstable across the split and are *not* scored.

Consequently `entry_quality.py` gates on volume expansion + impulse size and
macro alignment instead of on confidence, and `strategies.py` no longer boosts
the 10:00–11:00 window (negative in both halves; the table had it at 1.06x).

## What The 2026-08-17 Audit Changed

Four changes, each measured before shipping. Three helped; the fourth is a
negative result worth keeping.

**The shadow policy pooled two asset classes.** The state key was
`SIDE|conf|rr|votes|time-of-day` with no universe component, and both scanners
wrote to one table — so 121,023 NSE observations averaging −1.8 bps set the
verdict for 3,365 commodity observations averaging +4.7 bps. A flat
`ASSUMED_COST_BPS = 18.0` was charged to futures whose round trip is ~8 bps, and
`purge_and_migrate` applied the *NSE trading clock* to 24-hour futures, marking
9,385 legitimate observations DISCARDED. Commodity keys now carry a `C|` prefix,
costs are per-universe, futures get their own five-phase global clock, and 2,038
wrongly-discarded rows with outcomes were restored.

| policy verdict on commodity signals | before | after |
|---|---|---|
| passed | 1.7% (n=57) at +21.06 bps | **22.2% (n=1,201) at +19.50 bps** |
| blocked | 94.6% at **+4.98 bps** | 54.0% at **−4.46 bps** |

The veto now removes losers instead of winners. NSE was checked for collateral
damage and is unchanged (pass rate 1.2% → 1.0%, quality +11.36 → +13.29 bps).

**The commodity confidence gate was calibrated on backfilled data.** `85` was
picked in July from a pool that mixed live signals with an April
`historical_commodities_5d_5m` backfill. On live data only, it inverts the edge
on the three contracts that carry it — RB=F, HO=F, SB=F return +43.49 bps
ungated and **−1.68 bps at conf≥85** (n falls 420 → 11). Across all fifteen
symbols the gate produced 29 signals in seven weeks, which is why the scanner
logged nothing at all between 08-13 and 08-17. Now `70`, where those same
contracts keep 268 signals at +54.70 bps.

**The daily cap was tested tighter than 3 for the first time.** Measured on 34
held-out walk-forward sessions in `data/model_predictions.parquet`, net bps per
trade after costs:

| book | net bps/trade | sessions up | vs random-pick |
|---|---|---|---|
| **top-1** | **+27.62** | 20/34 | **+3.5 sd** |
| top-2 | +0.28 | 17/34 | +1.1 sd |
| top-3 | +1.58 | 17/34 | +1.5 sd |
| every candidate | −9.20 | — | — |

top-1 is the only setting positive in both halves of a calibrate/validate time
split (+565 then +374 bps) and the only one still positive with its best session
removed. Its bootstrap 95% CI on per-session edge is **[−13.1, +69.4] and still
spans zero** — 34 sessions is a small sample, so this is strong evidence, not
proof. `MAX_TRADES_PER_DAY = 1`.

These figures are computed with `argsort` over a whole session at once, which a
live book cannot do; the causal equivalent measures **+29.96 bps** and is
described in the 2026-08-25 audit below.

**No simple "if X then trade" rule survives** (`scripts/rule_search.py`). Every
rule is scored on a calibration window and re-scored untouched on a later one:

- **NSE: 0 of 117 rules survived.** The best calibration rule (09:15–09:45,
  NIFTY down, SHORT, +5.0 bps net) returned **−11.5 bps** out of sample. At an
  18 bps round trip there is nothing left to find at this granularity.
- **Commodity: 11 of 53 survived, against ~13 expected by chance.** Measured as
  *lift over the same window's unconditional baseline*, only 7 of 49 conditions
  held — again below the ~12 chance rate. The conditional rules are noise.
- The specific hypothesis "09:20 + Asian market positive" was tested directly:
  on NSE the Nikkei-up open window returns −14.0 bps out of sample; on
  commodities `asia up (N225 & HSI) LONG` fails to hold across windows.
- What *does* show up is not a rule but a **drift**: commodity LONG beat SHORT
  in every window, and the basket rose **+3.95% (11 of 15 symbols)** over the
  validation period. That is beta, not alpha, and it reverses when commodities
  fall.

## What The 2026-08-25 Audit Changed

The workspace had been idle for a week — no timers running on this machine, the
candle cache a week stale. Catching it up and replaying the missing sessions
turned up four defects, three of them in the path between a signal and an order
ticket. That path is where the bugs live, because every study reads the candle
cache directly and never exercises it.

**The live ticket path was structurally dead.** `build_dataset.build_symbol` is
shared between training and live scoring, which is the right design and had one
consequence nobody traced: it is a *labelling* loop. It needs one bar after the
signal (the fill) and a window after that (the barriers), so it stops two bars
short of the data. The freshest signal a live scan could produce was therefore
10–15 minutes old — and `sim_today.write_tickets` discards anything older than
10 minutes. The screen said "No signal in the last 10 minutes clears the gate"
on every cycle, in every market, and it read exactly like a quiet tape.
`build_symbol` now takes an optional `now`: it truncates the frame to bars
whose window has actually closed (a forming candle is partial data and must
never be scored) and emits the newest closed bars too, with the signal bar's
own close standing in for the unknown fill and the forward-looking label
columns absent, so nothing can train on them. Checked against live data at
12:41 IST: newest signal 12:35, 6.8 minutes old, inside the window.
`tests/test_live_signal_tail.py` pins both halves.

**The live book filtered before ranking; the validated book does not.**
`train_signal_model.py` validates the ranking over *every* engine signal, and
the section above already concluded that the gate is throwing away signal — but
`sim_today` still ran `passes_entry_gate` first and ranked the ~3% that
survived. The book that ran was neither the book that was validated nor any of
the gated books measured at −0.17% / −0.55%. `scripts/entry_policy_study.py`
measures the difference on the same 34 held-out sessions, through the real
portfolio simulator with the live exit design:

| book | net | win rate | profit factor | sessions up | 95% CI on per-session ₹ |
|---|---|---|---|---|---|
| **rank only, no gate** | **+6.09%** | 52.9% | 3.00 | 18/34 | **[+231, +3,514]** |
| gate, then rank (what ran) | +3.74% | 35.3% | 2.15 | 12/34 | [−312, +2,698] |
| ORACLE: session's best (not causal) | +4.04% | 52.9% | 1.99 | 18/34 | [−391, +2,855] |

That is the first interval in this repo that excludes zero, and it deserves its
caveats stated in the same breath: the exit width was chosen by a sweep over
these same 34 sessions, and the *paired* per-session difference between the two
books is +11.8 bps with a bootstrap interval of [−29.8, +53.9], better in 14 of
34 sessions and worse in 14. The direction replicates; the magnitude is not
established. The gate's verdict is still computed and still displayed — it no
longer decides anything.

Replayed forward on the six sessions the outage left unrecorded (2026-08-18 …
08-25, genuinely out of sample for a model trained to 08-17), the same
direction on a sample far too small to be evidence by itself: **+₹6,374
rank-only against +₹4,740 gate-then-rank**.

**Removing the gate removed the macro-outage alarm with it.** On 2026-08-13/14
the context symbols went stale and the gate refused all 1,552 signals; the
empty screen was the only symptom, and that is how the outage was eventually
found. A gate that decides nothing cannot raise that alarm — and the failure
gets *worse* without it, because a stale panel no longer empties the book, it
feeds the model three zero-filled features and produces a confident ticket
ranked on degraded information. `build_window` now checks macro coverage
directly and says so.

**"top-1 = +27.62 bps" was measured non-causally.** It is the whole
justification for `MAX_TRADES_PER_DAY = 1`, and `model_stress.run_fold`
computes it as `argsort(-pred)[:1]` over a whole session at once — the day's
best-predicted signal, chosen with knowledge of signals that had not printed
yet. A live book cannot do that. The causal version — rank only what has
printed, take the best on the first bar that offers one — measures **+29.96
bps, 19/34 sessions up**, slightly *better* than the oracle. So the cap
survives and its stated mechanism does not. What the book actually does is take
**the best-ranked signal on the session's first bar**: the simulator's cap-1
book and the explicit first-bar policy produce byte-identical results, all 34
sessions have signals on the opening bar, and every trade in the six replayed
sessions entered at 09:20. The ranking of everything after 09:20 never changes
an entry.

**The dataset was built over a universe the book never trades.** The
cross-sectional features (`xs_n_signals`, `xs_impulse_rank`, `xs_with_crowd`, …)
are defined relative to everything trading at the same instant, so the pool
they are computed over is part of the feature definition. `build_dataset.py`
built over all 505 cached names while `sim_today` builds only the ~146 liquid
ones it can trade, so `xs_n_signals` had a training median of 116 and a live
value roughly a quarter of that. The model was scoring a feature family whose
distribution it had never seen — the exact train/serve skew `signal_model.py`
guards against for *missing* columns, arriving instead through a column that
was present and meant something different. `--liquid N` (default 150, matching
`sim_today --universe` and `train_signal_model --universe`) restricts the build
to one pool.

**And one thing that was simply out of date:** `train_signal_model.BOOK` was a
hard-coded copy of the 2026-08-11 defaults — 3 trades/day, 1.5 ATR stop, 3.0
ATR target. When the cap moved to 1 and the exits to 2.0/5.0, the evidence gate
went on validating the old book, so `validation` in `signal_model.json`, and
the expectancy line printed on every order ticket, described a configuration
that no longer existed. It is sourced from `execution_plan` now, exactly as
`sim_today.LIVE_CONFIG` is.

### Catching up a cache that has been sitting idle

`scripts/catchup_fetch.py` is the gap-filler the workspace lacked.
`backfill_context.py` refills the macro symbols and the scanner daemon fetches
about a day per cycle, so a workspace switched off for a week had no way to
repair its own history: the daemon's `period="1d"` request cannot reach back
past yesterday, and every study downstream silently replays a stale cache.

```bash
python scripts/catchup_fetch.py                  # nifty500 + context, 10d of 5m bars
python scripts/catchup_fetch.py --only context   # just the macro panel
```

`sim_today.py` also gained `--since`, which replays a whole range in **one**
symbol pass instead of repeating the expensive per-symbol load once per
session. `build_symbol` reads and indicates the entire warmup frame regardless
of how much of it is kept, so a day-at-a-time replay of a missed week did that
work six times over:

```bash
python scripts/sim_today.py --since 2026-08-18   # the six missed sessions
python scripts/entry_policy_study.py             # gate vs rank, causal vs oracle
```

## Intra-Week (Swing) Book

`src/nse_intraday_ai/swing.py` holds a single NSE 500 name or commodity future
for a few sessions instead of a few bars, tested on **10 years of daily bars**
(`scripts/fetch_daily.py` → 1.06M bars over 521 symbols) rather than the four
months of 5-minute data the intraday book runs on. Sixteen weeks is not a sample
for a weekly strategy; ~520 is.

The economics look better on paper and worse in practice. Overnight positions
settle as **delivery, not MIS**, where STT is 0.1% on *both* legs instead of
0.025% on the sell — a ₹2.5L round trip goes from ~10 bps to **~30**. Against
that, a weekly move is whole percent rather than single-digit bps.

**Groww's exact rate card** is now in `costs.py` (`Segment`,
`segment_round_trip_cost`), because the segment decides everything:

| position | MIS intraday | CNC delivery | MCX futures |
|---|---|---|---|
| ₹50,000 | 18.0 bps | 41.4 bps | 16.3 bps |
| ₹1,00,000 | 13.3 bps | 34.3 bps | 11.6 bps |
| ₹5,00,000 | 9.5 bps | 28.7 bps | 7.8 bps |

Of a ₹343 delivery round trip on ₹1L, **₹200 is STT**. Agri futures are
CTT-exempt and priced accordingly.

### What ten years say

`scripts/swing_backtest.py` scores eight classic factors against a random-pick
baseline (same dates, sizing and costs), a NIFTY buy-and-hold benchmark, and a
first-half/second-half split. Entries fill at the next session's open, gaps
through a stop fill at the open, and sizing is risk-based.

**NSE equities at a 5-session hold: every factor lost money.**

| | net % (10y) | vs random | both halves + |
|---|---|---|---|
| trend_pullback | −14.8% | z = +1.16 | no |
| reversal_1w | −14.7% | z = +1.16 | no |
| rsi_oversold | −20.4% | z = +0.92 | no |
| momentum_12w | −33.9% | z = +0.34 | no |
| **NIFTY buy & hold** | **+176.2%** | — | — |

The mechanism is frequency, not stock-picking. At ~41 trades a year and ~37 bps
a round trip, the book pays **~15% of capital per year in charges** before it is
right about anything. Stretching the hold fixes the arithmetic and stops being
intra-week:

| hold | trades/yr | net % (trend_pullback) | gross bps | cost bps |
|---|---|---|---|---|
| 5 sessions | 41.4 | −14.8% | 7.6 | 37.2 |
| 10 | 22.5 | +17.6% | 86.4 | 37.4 |
| 20 | 12.9 | +42.2% | 240.7 | 37.9 |
| **40** | 7.4 | **+77.3%** | 793.3 | 38.7 |

Even the best cell loses to holding the index. **Commodities are the better
side**: `rsi_oversold` returned +55.2% over ten years, **8/10 years positive**,
max drawdown 17.6%, positive in both halves — futures cost ~12 bps rather than
~37, and commodities mean-revert where equities do not (every momentum factor
lost on futures; every mean-reversion factor made money). Its bootstrap CI on
bps/trade still spans zero.

```bash
python scripts/fetch_daily.py                 # 10y daily bars, once
python scripts/swing_backtest.py nse --sweep
python scripts/swing_backtest.py commodity
python scripts/swing_today.py commodity       # today's candidates + expectancy
```

`swing_today.py` prints the measured expectancy beside every pick, including
"do not trade this for profit" for the NSE 5-session case.

## Real Transaction Costs

`costs.py` replaces the flat 18 bps assumption with the actual NSE intraday
schedule. The dominant term for a discount broker is **flat ₹20 per order**,
not a percentage, so cost per rupee falls sharply with size:

| position | round-trip cost |
|---|---|
| ₹50,000 | 15.6 bps |
| ₹1,00,000 | 13.2 bps |
| ₹2,50,000 | 10.4 bps |
| ₹5,00,000 | 9.5 bps |

Since the measured gross edge on the best signal subset is only a few bps, this
is not a rounding detail — it decides whether a trade is viable, and it argues
for **fewer, larger positions**. Below ~₹1.4L a position cannot clear its own
fees at a 12 bps target, so the sizer refuses it rather than trading small.

## Intraday Portfolio Simulator

`portfolio_sim.py` models what `backtest.py` could not: several positions open
at once competing for shared capital and risk budget, partial exits that each
pay their own brokerage leg, a real cost model that interacts with sizing,
per-day loss limits, and **mandatory square-off at 15:15** — these are MIS
positions the broker will auto-square anyway. Entries fill at the **next bar's
open**; the old harness filled at the signal bar's close, which quietly awards
the last tick of the impulse to the strategy.

Results on ₹10,00,000 over 49 sessions (2026-05-13…08-12), top-150 liquid
names, 3 trades/day, stop 1.5 ATR / target 3.0 ATR (the defaults at the time; now 2.0/5.0 — see below), no scale-out:

| gate | full window | first half | second half |
|---|---|---|---|
| engine gates (conf≥70, rr≥1.5, ≥2 votes) | −1.12% | — | — |
| quality gate (volume + impulse) | **+2.79%** (PF 1.15) | −1.83% | +4.73% |
| quality + macro alignment | −0.42% (PF 0.98) | −2.83% | +2.49% |

**No configuration is stable across the two halves**, and both halves had
near-identical NIFTY drift (+2.0% vs +1.9%), so the difference is noise rather
than regime. Gross P&L is positive in every variant and transaction costs
consume it. The full-window row is the honest one; a single-half result is not
evidence. This is the same conclusion the meta-label study reached, arrived at
from a different direction.

Design choices that *did* replicate: scale-out hurts (an extra order costs an
extra flat fee, and it caps winners while losers still run full size — mean
gross 1.9 bps with, 5.0 bps without); wider stops and targets beat tight ones;
2–3 top-ranked signals a day beat 6–12.

### Two simulator bugs that inflated the first published numbers

Both were found by tests written after the fact, and both mattered:

- **The daily trade cap was checked once per bar, not per candidate.** Signals
  arrive in clusters on the same bar, so a book configured for 3 trades a day
  routinely opened 6–8. The quality gate's headline "+2.79% over 49 sessions"
  was an artefact of this; corrected, it is **−0.55%**.
- **Positions still open when the data ran out were dropped, not closed.** On a
  full session the square-off catches everything, so this was invisible in
  backtests — but mid-session (the live case) it silently discarded open P&L,
  which flatters a losing book. Open positions are now marked to market with
  an explicit `MARK_TO_MARKET` exit.

Corrected results, 49 sessions, top-150 liquid names, 3 trades/day, stop 1.5
ATR / target 3.0 ATR, entry at the next bar's open, 1.5 bps/leg slippage:

| ranking | net | win rate | profit factor |
|---|---|---|---|
| engine gates (conf≥70, rr≥1.5, ≥2 votes) | −0.17% | 33.6% | 0.98 |
| conviction gate (volume + impulse) | −0.55% | 28.6% | 0.96 |
| conviction + macro alignment | −2.82% | 26.2% | 0.80 |

**No rule-based gate is profitable.** Gross P&L is positive in all three
(+₹9K to +₹33K) and transaction costs consume it.

### Learned ranking

`build_dataset.py` produces 269,303 engine signals with **triple-barrier
labels** — for each (stop, target, time) design, which barrier was touched
first and the resulting net bps after real costs. A fixed-horizon return is
not what the book earns; whichever barrier hits first is. Features cover
timing, bar-level microstructure (close-location value, wick ratios, an
order-flow proxy), session structure, liquidity, cross-sectional rank among
simultaneous signals, and the three surviving macro readings.

`train_model.py` and `train_signal_model.py` fit a forest per session on
strictly prior sessions and use it only to *rank* — which three of ~50 gated
signals the book takes, which is the entire decision. `train_signal_model.py`
refuses to write `data/signal_model.json` unless the model-ranked walk-forward
book beat the composite-ranked one, so the file existing is the evidence.

Walk-forward book comparison over 34 held-out sessions, same portfolio rules,
the only difference being which three signals a session's book takes:

| ranking | net | win rate | profit factor | sessions up |
|---|---|---|---|---|
| composite (macro + volume + impulse) | +2.56% | 32.4% | 1.30 | 17/34 |
| **model** | **+7.36%** | 35.3% | 1.93 | 22/34 |

and the model book's recency split is positive in both halves (+4.19% earlier,
+3.18% recent), which the rule-based gates never managed.

Note what the comparison implies: both books rank over *all* engine signals,
not just gate-passing ones, and both beat every gated variant. **The gate is
throwing away signal.** Selecting by ranking and taking the top three is
better than filtering first. The live path did not act on that conclusion until
2026-08-25 — see the audit section above, where the cost of filtering first is
measured at 2.35 points of return over the same 34 sessions.

`model_stress.py` exists because that result needs adversarial testing rather
than celebration: it sweeps barriers and hyperparameters, sweeps top-K, drops
the best session, bootstraps a confidence interval on the per-session edge, and
runs a **permutation test** — the same pipeline on shuffled labels, which
measures the noise floor of the procedure itself. Read its output before
believing any ranking result. Its verdict on the current model:

| test | result |
|---|---|
| permutation (5 label shuffles) | real **+11.56 bps** vs shuffled mean **−11.56** (sd 5.14) → **4.5 sd above the noise floor** |
| hyperparameter grid (5 settings) | positive in all five (+7.2 to +35.9 bps) |
| drop the best session | still positive in all five settings |
| recency split | positive in both halves (+3.22% earlier, +1.84% recent) |
| bootstrap CI on per-session edge | **still spans zero** — 34 sessions is a small sample |
| top-K monotonicity | not clean: top-1 is not reliably better than top-3 |

The shuffled runs land near the population mean (−9.4 bps), which is exactly
what a ranking with no information should earn — so the pipeline itself is not
manufacturing the result. That is real evidence. It is not yet proof, and the
last two rows are the reason.

**Hardware note.** The GPU (NVIDIA T400 4GB) was benchmarked and is *slower*
than the CPU for this workload: xgboost 40.6 s on CUDA vs 27.1 s on 32 CPU
threads, and sklearn's forest beats both at 13.2 s. 384 CUDA cores against a
50,000 × 46 tabular problem means transfer and launch overhead dominate. More
compute is not the binding constraint here — **49 sessions of data is**. Extra
CPU time is better spent on more permutation trials than on a bigger model.

### Exit width, and why the daily cap stays

Two sweeps on the model-ranked book (`sim_exit_sweep.py`, `sim_nocap.py`),
both prompted by a losing live session on 2026-08-12 where the *direction was
right* — the book shorted CARTRADE and KPITTECH, both fell ~1.8%, and both were
stopped on the entry bar before the move arrived.

**Stop width.** Averaged across target multiples: 1.5 ATR +4.39%, **2.0 ATR
+4.86%**, 2.5 ATR +4.73%, 3.0 ATR +2.67% — and the wider stop *lowered* max
drawdown (2.18% → 1.66%), because fewer correct trades were shaken out. The
default moved to a 2.0 ATR stop / 5.0 ATR target. This is a sweep over the same
34 sessions, so the direction of the effect is the finding, not the exact cell.

**Skipping the opening bar** was tested and rejected: refusing 09:15/09:20
entries took the book from +4.16% to **−5.17%** and doubled drawdown. Noisy as
it is, that bar is where the best-ranked signals are.

**Removing the daily cap** was tested and rejected, decisively:

| book | positions | net | gross bps | costs |
|---|---|---|---|---|
| 3/day, 3 open @33% | 102 | **+5.22%** | 23.0 | ₹27k |
| no cap, 3 open @33% | 897 | −13.53% | 3.3 | ₹229k |
| no cap, 8 open @12% | 2,625 | −23.06% | 2.8 | ₹306k |

The extra trades are the lower-ranked ones — gross edge collapses from 23 to 3
bps while every trade still pays its full ~8 bps round trip. The cap is the
mechanism by which a thin edge survives, not caution for its own sake. It lives
in `execution_plan.MAX_TRADES_PER_DAY`; set it to `0` for no cap.

### Trading the screen

`sim_today.py` is the single producer of order tickets. It writes
`data/today_tickets.json`, and the app renders that file rather than building
its own ticket — one code path, so the screen and the measured book cannot
disagree (the failure mode behind the 2026-07-07 config incident).

Each ticket is an instruction, not a table row: it quotes the stop and target
**relative to the fill**, because the fill is the next bar's open and is not
known yet. That property is what makes a *live tail* signal publishable at all:
the newest closed bar has no fill yet, and until 2026-08-25 the feature builder
refused to emit it, which left every ticket 10–15 minutes stale and therefore
discarded. Quoting a fixed stop price off the signal bar's stale close is how a
1.5 ATR stop silently becomes a 0.4 ATR one. Every ticket carries the measured
expectancy, including the intervals.

### Daily paper book

`scripts/sim_today.py` runs the same engine, gate and portfolio rules over a
single session and appends the result to `data/daily_sim.csv`. The systemd
timer `nse-paper-book.timer` fires weekdays at 15:45 IST, after the close.

```bash
python scripts/sim_today.py                 # today (partial if mid-session)
python scripts/sim_today.py --date 2026-08-11
python scripts/sim_today.py --history       # the accumulated record
```

The record exists because **a single session's P&L is noise**. Session-to-session
standard deviation in the 49-session study was several times the mean, so the
last row of that CSV carries almost no information about whether the system
works; only the cumulative column does, and only after dozens of sessions.

**Yahoo snapshot rows** (`candle_cache.drop_synthetic_bars`): Yahoo's intraday
endpoints append a live-quote row that sits off the interval grid (e.g.
09:15:17 on a 5m series), carries zero volume, and has O=H=L=C at the last
traded price. It is not a candle. Left in, a backtest that fills "at the next
bar's open" fills at that stale quote instead of the next genuine open — on
2026-08-12 that alone inflated the paper book's day from +₹2,729 to +₹6,826.
The filter is applied on every cache read, so the live scanner no longer scores
a flat doji as its newest closed bar either.

## News Ingestion

`news_feed.py` pulls Google News RSS (India edition), Moneycontrol, Economic
Times, Business Standard, Yahoo Finance per-ticker RSS, and the NSE corporate
announcements API — all free, no keys. Headlines are tagged to symbols by
**full registered company name**, not ticker root: "Reliance Chemotex
re-appoints MD" and "Reliance (NYSE:RS) reaches 12-month high" both contain
"Reliance" and neither is RELIANCE.NS, and a false-positive tag would veto a
good signal.

It is deliberately **not** a direction signal. A free RSS feed reaches you after
the move — the same "arrives late" problem the price engine has. What a slow
feed is genuinely good for is knowing the price model's assumptions have
broken: an ATR-derived stop assumes the last hour describes the next one, and a
results announcement or regulatory action invalidates that outright. So the
gate abstains on fresh high-impact stock news, sizes down on a risk-off macro
tape, and records direction without acting on it. Every item is archived to
`data/news.sqlite3` with its fetch time, so the direction question becomes
answerable from our own archive after a few weeks — the same evidence gate the
meta-model uses. Off by default (`news_gate` in `scan_config.json`).

## Market Context Inputs

Every scan (and now every backtest candle) is scored against market-wide context, fetched in one batch:

- NIFTY 50 regime (trend/range/high-vol) — aligns stock signals with the index tape
- India VIX level — shifts weight between mean-reversion and breakout strategies
- NSE sector indices (`^NSEBANK`, `^CNXIT`, `^CNXAUTO`, `^CNXPHARMA`, `^CNXFMCG`, `^CNXMETAL`, `^CNXENERGY`) — a stock aligned with its own sector index trend gets a boost, against it a penalty
- S&P 500 futures (`ES=F`), Nikkei (`^N225`), Hang Seng (`^HSI`) and DAX (`^GDAXI`) — a global risk appetite score in [-1, +1], now weighted by `macro_context.region_weight()` so each region's momentum reading decays once its session closes (a Nikkei "1-hour change" read at 14:00 IST describes a market that shut 2.5 hours earlier). **Scored into commodity signals and displayed for equities, but deliberately NOT gated on for equities** — see the August-2026 study above, where every foreign index measured zero out of sample
- Crude (`CL=F`) — added to the equity panel: India imports ~85% of its oil, so crude is a first-order driver of Indian equities and was one of only three macro readings to survive the out-of-sample split
- Dollar index (`DX-Y.NYB`) — inverse driver for metals
- USDINR — INR weakness is a mild equity headwind, but a direct **conversion tailwind** for INR-denominated (MCX) commodity contracts, which price as USD rate × USDINR
- Hang Seng momentum — China demand proxy applied to industrial metals (copper)
- US VIX (`^VIX`) — global fear gauge covering the hours India VIX cannot see; above 25 it penalises both sides of equity trades
- US 10-year yield (`^TNX`) — fetched and displayed, but deliberately **not scored**: the July-2026 ablation showed 1-hour yield momentum is pure noise (it alone flipped the profitable commodity subset negative)

Two further factors need no data feed at all and are monitored live in the sidebar:

- **Scheduled macro event windows** (`event_risk.py`, pure IST clock): EIA crude inventories (Wed 20:00), US jobless claims (Thu 18:00), US non-farm payrolls (first Friday 18:00) de-rate commodity entries from 15 minutes before to 30 minutes after the release — energy hardest for EIA, precious hardest for NFP; the NSE F&O expiry afternoon (from 13:00) de-rates equity entries — **Tuesday** since 2025-09-02, weekly −4 and monthly (last Tuesday) −6 confidence points. Because this is clock-based, backtests exercise the identical logic. July-2026 validation: skipping release-window trades lifted the commodity strict-gate subset from +456 to +642 and its week-2-only result from −27 to +144 (67% win rate).
- **Live VWAP breadth**: the fraction of the scanned universe trading above session VWAP, computed from the batch frames *before* signals are evaluated (same cycle, no lag). Above 65% boosts longs / penalises shorts, below 35% the mirror. On the study window the NSE effect was within noise, kept for its mechanistic soundness and monitored in the sidebar.

Backtests replay this context causally through `context_series.py`: at each historical candle the engine sees only context data available at that moment, with a 45-minute staleness guard on momentum readings (12 hours for slow-moving levels like VIX). Set `NSE_AI_ABLATE=eu,inr,hsi` (comma-separated) to disable individual terms when re-running attribution studies.

## Meta-Label Veto

A secondary logistic classifier (López de Prado meta-labeling) scores every
actionable signal on entry-time features only — confidence, reward/risk, vote
stats, side, time of day, stop/target width, session multiplier, market
regime, and which strategies agreed — and vetoes signals scoring below a
threshold learned on the training distribution of gate-passing events.

Trained and validated by `scripts/train_meta_model.py` with rolling weekly
walk-forward (every validation week scored by a model trained only on prior
weeks). The script **refuses to write a model unless the veto demonstrably
improves the out-of-sample portfolio**, so a `data/meta_model_<universe>.json`
file existing *is* the evidence gate — the app and scanner daemon apply the
veto whenever a model file is present (`meta_veto` in `scan_config.json`,
veto fraction 0.5 by default).

July-2026 measurement (8–10 weeks of cached 5m candles, ~110K candidate events):

- **Commodities — PASS, model shipped.** Top-vs-bottom quintile hit-rate lift
  positive in 9/9 OOS weeks (median +0.16). Strict-gate OOS portfolio:
  −554 unfiltered → **+681 at veto-50%** (win rate 32.4% → 37.9%, max
  drawdown −27%); veto-70% keeps +681 with max drawdown −72% (253) on fewer
  trades.
- **NSE equities — REFUSED, no model written.** Ranking works there too
  (median quintile lift +0.19), but no abstention level flips the portfolio
  positive: ~0 gross edge minus an 18 bps round trip is structurally negative
  (veto-85% still −1733 vs −3864 unfiltered). A filter can only cut the bleed,
  not manufacture edge — NSE signals should be treated as research output, not
  tradeable recommendations, until either costs fall or new alpha is added.

Retrain as more weeks accumulate:

```bash
python scripts/two_week_backtest.py commodity --since <10-weeks-ago>
python scripts/train_meta_model.py commodity
python scripts/two_week_backtest.py nse --since <8-weeks-ago>
python scripts/train_meta_model.py nse        # keeps refusing until it earns a pass
```

## Reinforcement Learning

RL is not used as the first trading brain. In real markets, RL often overfits a simulator and fails when spread, slippage, partial fills, market impact, and regime changes appear. The app now runs a runtime shadow learner in parallel:

- each scan records candidate state/action samples,
- later scans evaluate delayed reward after the configured horizon,
- evaluated samples update a contextual-bandit policy table,
- the UI shows samples, win rate, average reward in basis points, and UCB score,
- optional policy assist can rank candidates using learned confidence bonuses once enough samples exist.

Policy assist does not bypass the trade filters or risk engine. It only changes ranking; a setup still needs multi-strategy agreement, confidence, quote freshness, and reward/risk to become actionable.

The better path is:

1. Build a robust simulator and paper-trade record.
2. Validate rule/ML signals with walk-forward testing.
3. Add RL later for position sizing, exit timing, or execution optimization.
4. Keep RL constrained by the same risk engine.

## Run

The app runs as a systemd user service and starts with the machine:

```text
http://127.0.0.1:8501          on this box
http://<lan-ip>:8501           from a phone on the same Wi-Fi
```

```bash
python scripts/health_check.py          # is everything actually working?
systemctl --user status nse-signal-lab  # the app
systemctl --user list-timers 'nse-*'    # everything scheduled
./run_app.sh                            # foreground, for development
```

### The units

| unit | schedule | job |
|---|---|---|
| `nse-signal-lab.service` | always on | the Streamlit app |
| `nse-scanner.timer` | every minute | scan cycle, fills the candle cache |
| `nse-context.timer` | every 10 min, 08:00–16:00 | macro symbols into the cache |
| `nse-paper-book.timer` | every 5 min, market hours | `sim_today.py` — publishes tickets, records the session |
| `nse-learn.timer` | weekdays 17:00 | shadow-policy retrain |
| `nse-retrain.timer` | Saturday 08:00 | rebuild dataset, retrain the ranking model |
| `nse-health.timer` | every 30 min | `health_check.py --quiet` → `data/health.log` |
| `nse-logrotate.timer` | daily | rotate `data/*.log` |

`loginctl enable-linger` is set, so these survive logout and reboot.

### Three silent failures, and the guards added for them

Found on 2026-08-14, all of which had been running unnoticed:

- **The app leaked to 15.2 GB over 35 days**, stopped answering HTTP, and filled
  swap. Guards: `MemoryMax=5G`, `RuntimeMaxSec=86400` (nightly recycle), and
  `--server.fileWatcherType none` — the watcher was walking `.venv` and a 1.7 GB
  `data/` dir, exhausting the per-user inotify limit and starving other units of
  watches.
- **A whole session recorded nothing.** With swap full, the paper book's
  60-second run took over 5 minutes, so each 5-minute tick killed the previous
  run mid-flight. Guard: `TimeoutStartSec=240`, and `sim_today.py` now records
  flat days instead of returning early — a missing row used to be
  indistinguishable from a quiet one.
- **The macro gate refused every signal for two sessions.** Nothing kept
  `^NSEI` / `^INDIAVIX` / `USDINR` flowing into the candle cache — the scanner
  fetched them into a live object but never persisted them, and the weekly
  retrain was the only writer. Once stale, the gate rejected all 1,552 signals a
  day and the screen showed nothing, which looks exactly like a quiet market.
  Guards: `nse-context.timer`, plus `sim_today.py` now prints the dominant
  refusal reason and says explicitly when the context is stale.

`health_check.py` tests for all three, plus disk, memory, unit state, candle
freshness and whether today's session was recorded.

## Android APK

An Android WebView companion app is available in `android-app/`. It installs as `NSE Signal Lab` and loads the running Streamlit app from a configurable server URL.

### Standalone mode (no laptop)

`PicksActivity` runs the intra-week book entirely on the phone. It fetches daily
bars straight from Yahoo's chart endpoint over the phone's own connection,
computes the indicators on-device and sizes the trade — the workspace does not
need to be running, or even switched on.

Three books, each showing its measured ten-year number rather than hiding it:

| book | universe | measured over 10y |
|---|---|---|
| **Commodities, 5-session** | 15 futures | **+55.2%**, 8/10 years positive — the only one marked ✓ |
| NSE, 5-session | 120 liquid names | −14.8% — shown for completeness, not for trading |
| NSE, 40-session | 120 liquid names | +77.3%, but a two-month hold, and still below NIFTY buy & hold |

`SwingEngine`'s RSI-14, ATR-14 and cost model were cross-checked against the
Python that produced the backtest and agree **to six decimal places** on GC=F,
CL=F, SB=F and ZW=F, so the phone computes what was actually tested rather than
an approximation.

Two deliberate limits. NSE uses the **120 most liquid names, not all 500**: a
full refresh is ~5 MB of JSON where 500 would be ~22 MB, which is not a thing to
ask of a phone on mobile data. And the **5-minute intraday scanner is not ported**
— it needs a 500-symbol intraday fetch every bar plus the trained forest and the
shadow-learner database, and its measured edge is negative anyway. The WebView
tab still reaches it when the workspace is on the network.

Results are cached to the SD card, so the last pick survives going fully offline.

Verified on a Realme RMX1971, Android 11 (API 30), laptop services stopped and
no adb tunnels: 15/15 commodity symbols and 120/120 NSE symbols fetched over the
phone's own connection, matching the JVM reference run.

### Desktop / Android feature parity

Parity is structural, not maintained by hand: the phone renders **the same
Streamlit app** the desktop browser does, so every scanner, the backtest, the
single-symbol view and the intra-week book appear on both by construction. A
feature added to `app.py` is on the phone the moment the page reloads — there is
no second UI to keep in step, which is the only reason parity survives.

The two platform-specific pieces have equivalents on both sides:

| capability | desktop | Android |
|---|---|---|
| all scanners, backtest, single symbol, **intra-week book** | Streamlit | same page in the WebView |
| order tickets | `today_tickets.json` rendered in-app | same, plus `/tickets` for polling |
| alerts | daemon desktop notifications, sidebar toggle | JobScheduler poll → system notification, in-app toggle |
| local data | `data/` on disk | SD card (default) via `Storage.java` |
| picking the server | n/a — runs locally | server URL field |

The gap this closed: the intra-week book existed only as a CLI script, so
*neither* platform had it. It is now a sidebar mode in `app.py`, which put it on
both at once.

**Android version support.** The APK declares `minSdk 23` / `targetSdk 35`, so it
installs on Android 6.0 and every release above it — **Android 11 (API 30)
included**. Note that "make it work on Android 11" does *not* mean raising
`minSdk` to 30: that would drop Android 6–10 while changing nothing about
Android 11. Two things do matter on API 30 and are handled:

- **Cleartext HTTP.** The app talks to `http://<lan-ip>:8501`, and Android has
  blocked cleartext by default since API 28. `usesCleartextTraffic="true"` plus
  `res/xml/network_security_config.xml` keep the LAN connection working.
- **Connectivity detection.** Android 11 returns a hardcoded, always-"connected"
  `NetworkInfo` to apps targeting API 29+, so the legacy check could not tell
  that the phone had dropped off Wi-Fi — you got a blank WebView instead of the
  "connect to the same Wi-Fi" hint. `MainActivity` now uses
  `NetworkCapabilities` on API 23+ and keeps the old path below that.

**SD-card storage** (`Storage.java`, default ON). The app keeps its offline copy
of the book on a removable card when one is mounted, falling back to internal
storage otherwise. It does **not** request `WRITE_EXTERNAL_STORAGE`: Android 11
enforces scoped storage, so that permission is ignored on every version this app
targets and asking for it would be theatre. `Context.getExternalFilesDirs()`
returns one directory per volume — index 0 internal, later entries physical
cards — and needs no runtime grant on any supported release. The trade-off worth
knowing: app-specific directories are deleted on uninstall.

**Notifications** (`TicketJobService.java`, default ON). Streamlit serves a UI,
not data, so the phone had nothing to poll. `scripts/ticket_api.py` exposes
`/tickets`, `/record` and `/health` as JSON beside the app on port 8502, and
`run_android_server.sh` now starts both. The app polls with the framework
`JobScheduler` — not WorkManager, because this APK is built with aapt2 and ecj
alone and has no dependency graph — every 15 minutes, which is Android's floor
for periodic jobs and roughly the rate at which tickets actually appear.
Already-seen ticket ids are remembered, so a ticket live across several polls
notifies once. `POST_NOTIFICATIONS` is declared for API 33+ and requested at
runtime; on Android 11 it does not exist and notifications are granted by
default.

Build the debug APK:

```bash
./build_android_apk.sh
```

The script bootstraps everything it needs into `android-app/` — Android
command-line tools, SDK platform 35, build-tools 35.0.0, the Eclipse compiler,
and (since 2026-08-17) a private Temurin 17 JDK if no system `java` is present.
Note the compiler runs against `android.jar` alone, so **lambdas do not
compile** (`java.lang.invoke.LambdaMetafactory` is absent) — use anonymous
classes, as the rest of the app does.

The APK is written to:

```text
nse-signal-lab-debug.apk
```

Install and launch it on a connected Android device or emulator:

```bash
./install_android_apk.sh
```

Run the backend so a phone can reach it:

```bash
./run_android_server.sh
```

Use `http://10.0.2.2:8501` in the emulator, or the printed LAN URL on a physical phone connected to the same Wi-Fi.

## Two-Week Backtest Harness

`scripts/two_week_backtest.py` replays the last two weeks of cached candles as
an event study: every candidate signal is generated with causal market
context, exits are simulated, then a grid of confidence / reward-risk / vote
gates is swept — including a week1-calibrate → week2-validate walk-forward so
gate picks are kept honest.

```bash
python scripts/two_week_backtest.py nse
python scripts/two_week_backtest.py commodity
python scripts/two_week_backtest.py nse --no-context   # ablation
```

Findings from the 2026-06-20 → 2026-07-03 window are baked into the defaults:
strict gates plus market context turned the commodity scanner's selected
subset positive (conf≥85: +381 over the window at futures costs vs −538 for
the old engine at the same gates), and moved NSE from a steady bleed to
roughly breakeven. Pullback limit entries were tested and **rejected**
(adverse selection: the fills you get are the trades moving against you).

## Modern Schemes: Tested, Adopted or Rejected

Every technique gets the same treatment — implement, backtest with a
week1→week2 walk-forward, keep only what survives:

- **Meta-labeling** (secondary classifier deciding which primary signals to
  trust — `scripts/meta_label_study.py`): shows genuine out-of-sample
  hit-rate ranking (commodities 26%→37%, NSE 24%→39% across predicted-prob
  quintiles) but mixed magnitude effects on small strict-gate samples.
  Status: research tool; re-run as cached weeks accumulate, wire in only
  after the portfolio effect validates repeatedly.
- **Chandelier trailing exits** (`BacktestConfig.trailing_atr_mult`, with
  optional `trailing_replaces_target`): rejected as default. Commodities
  gained raw P&L without a target (+518 vs +456) but with 50% more drawdown;
  NSE was decisively worse (−847 vs +51). Fixed target + 0.8R breakeven
  stays.
- **Pullback limit entries** (`entry_limit_offset_atr`): rejected — adverse
  selection (the fills you get are the trades moving against you).
- **Foreign equity indices as intraday predictors** (S&P futures, Nikkei, DAX,
  Hang Seng): rejected Aug 2026 — large in-sample correlation, zero out of
  sample, three of four flipped sign. Kept for commodities and for display.
- **Calendar/seasonality terms** (day of week, expiry proximity, month-end,
  monsoon, results season): rejected Aug 2026 — no bucket held its sign across
  the split. The one clock effect kept is the expiry-afternoon de-rate, which
  is mechanistic rather than fitted.
- **Session-phase-weighted global risk** (`macro_context.phase_weighted_global_risk`):
  mechanically correct — it stops a closed market's stale momentum from voting
  — but measured zero for equities, so it is wired in and *not* gated on.
- **News/sentiment** (`news_feed.py`): now implemented, as an abstention gate
  rather than a direction signal, and off by default until its own archive can
  validate it. The earlier "no free reliable feed" note was half right: the
  feeds exist and are reliable enough, they are just too slow to predict.
- Still not attempted, and why: deep nets on a few weeks of 5m data would
  overfit before they generalise; order-flow/microstructure features need paid
  tick data.

## Manual Commands

```bash
cd /home/hp/Documents/hobby/trading-workspace
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
streamlit run src/nse_intraday_ai/app.py
```

**One version constraint is load-bearing.** On pandas 3.x, `read_parquet`
raises `AttributeError: 'NoneType' object has no attribute 'timezone'` for any
frame with a tz-aware column unless **pyarrow >= 25**. Every `ts` column here
is tz-aware, so on pyarrow 21–24 `dataset.parquet` and
`model_predictions.parquet` are simply unreadable and retraining cannot run at
all — which is a silent, total block on the one part of the system that has
measurable edge. Streamlit's metadata pins `pyarrow<25`; the pin is
conservative, and 25.0.1 serialises dataframes correctly (checked against
`streamlit.dataframe_util.convert_pandas_df_to_arrow_bytes`).

## Real-Time Data Notes

Yahoo Finance is convenient for development, but it is not a broker-grade NSE realtime feed. For live trading research, plug in a provider such as a broker WebSocket or licensed NSE data vendor. The `YFinanceProvider` and `DemoProvider` are intentionally isolated in `src/nse_intraday_ai/data.py` so a broker provider can be added without rewriting the signal engine.

## Safety Defaults

- Simulation mode starts enabled.
- Auto paper-trading is disabled by default.
- Minimum confidence starts high.
- Strategy weights stay neutral until there are at least 5 closed paper-trade samples for that strategy.
- Risk per trade defaults to 0.5% of capital.
- Round-trip commission defaults to 0.5%, with additional slippage applied in basis points.
- Backtest mode includes late-entry cutoff, same-symbol cooldown, stop-loss lockout, max trades per day, and daily loss limit controls.
- Auto-calibration defaults to a risk-adjusted objective; use Maximum P&L only for research, not as a live-trading default.
