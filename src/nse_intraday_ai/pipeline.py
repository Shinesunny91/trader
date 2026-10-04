"""The book's daily and weekly loops: data in, experts scored, weights learned, picks out.

08:45 (before the pre-open auction):
  1. refresh   NSE equity + F&O bhavcopies, crowding files, participant OI,
               overnight global closes
  2. learn     grade the *previous* session for every expert (gap rule, morning
               ranker, open ranker) on official NSE bars; extend their paired
               record (the guard) and the drift alarm
  3. rank      build today's features through the live path (the same code that
               built the training data) and publish the preliminary list

09:09 (after the auction, before the continuous open):
  archive the auction snapshot; if the open model is promoted and its guard
  allows, re-rank on today's opening prices and publish the final list

Weekly (and on a drift alarm):
  rebuild the full point-in-time history, walk-forward the last two years,
  fit challengers, promote each model only on out-of-sample evidence.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from nse_intraday_ai import features as FT
from nse_intraday_ai import gap_reversal as G
from nse_intraday_ai import meta as M
from nse_intraday_ai import nse_bhav, nse_extra, nse_fo, nse_preopen
from nse_intraday_ai import ranker as R

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "data" / "models"
BOOK_UNIVERSE = 300
COST_BPS = 13.0
# The 09:09 list is traded with a market order at 09:15, not in the auction.
# Measured on 1-minute bars the first minute costs nothing on average for the
# day's biggest gap-ups (+6 bps in the short's favour, noisy); 5 bps are charged
# anyway so the open model has to beat the morning one net of a real penalty.
OPEN_ENTRY_EXTRA_BPS = 5.0
LIVE_INPUTS = REPORTS / "live_inputs.pkl"


class DataNotReady(RuntimeError):
    """The official files needed for this session are not all there."""


# ── data ────────────────────────────────────────────────────────────────────

def refresh(through: date, *, days: int = 20, log=print) -> None:
    """Pull the last `days` of NSE files and the global closes up to `through`."""
    start = through - timedelta(days=days)
    ok, hol = nse_bhav.backfill(start, through, workers=2, log=log)
    log(f"  NSE equity: {ok} new sessions, {hol} holidays")
    ok, hol = nse_fo.backfill(start, through, workers=2, log=log)
    log(f"  NSE F&O:    {ok} new sessions, {hol} holidays")
    ok, hol = nse_extra.backfill(start, through, workers=2, log=log)
    log(f"  NSE crowding/short-sales: {ok} new sessions, {hol} holidays")
    try:
        from nse_intraday_ai.nse_flows import load_history
        load_history(start, through, ROOT / "data" / "nse_flows", what="oi")
    except Exception as exc:                                  # noqa: BLE001
        log(f"  participant OI refresh failed: {exc}")
    arrived = G.refresh(list(FT.MARKET_SERIES) + list(FT.ADRS), interval="1d", period="1mo", log=log)
    log(f"  global/market series: {len(arrived)}/{len(FT.MARKET_SERIES) + len(FT.ADRS)}")
    # Corporate events through this morning (filings up to 08:45 count for
    # today's list) and the board-meeting schedule ahead.  Optional: without
    # them the catalyst features are NaN, never wrong.
    try:
        from nse_intraday_ai import nse_corp
        stats = nse_corp.refresh(through - timedelta(days=10), max(through, date.today()),
                                 log=lambda m: log(f"  {m}"))
        log(f"  NSE corporate events: {stats}")
    except Exception as exc:                                  # noqa: BLE001
        log(f"  corporate events refresh failed: {exc}")


def previous_session(session: date) -> date:
    """The last NSE session before `session`, from the files themselves.

    Walks back over weekdays: a day with a bhavcopy is a session, a day with a
    holiday marker is not; a day with neither means the data is not ready.
    """
    day = session - timedelta(days=1)
    for _ in range(15):
        if day.weekday() < 5:
            if nse_bhav._path(day, nse_bhav.CACHE).exists():
                return day
            if not nse_bhav._holiday_marker(day, nse_bhav.CACHE).exists():
                raise DataNotReady(f"no NSE bhavcopy (and no holiday marker) for {day}")
        day -= timedelta(days=1)
    raise DataNotReady(f"no NSE session in the 15 days before {session}")


# ── experts ─────────────────────────────────────────────────────────────────

def expert_scores(rows: pd.DataFrame, model: R.Ranker | None,
                  open_model: R.Ranker | None = None) -> dict[str, pd.Series]:
    """Every expert's score for the candidate rows (higher = better short)."""
    scores = {"rule": rows["gap1"].astype(float)}
    if model is not None:
        scores["ranker"] = pd.Series(model.score(rows), index=rows.index)
    if open_model is not None and "gap0" in rows.columns:
        scores["ranker_open"] = pd.Series(open_model.score(rows), index=rows.index)
    return scores


def top_book(outcome: pd.Series, score: pd.Series, mask: pd.Series, k: int = 8,
             cost: float = COST_BPS, gap0: pd.Series | None = None,
             entry_limit: float | None = None) -> float | None:
    """Per-slot net bps of the top-k; with an entry limit, picks that opened
    below prev_close*(1+limit) were never filled and contribute 0 (as in R.book)."""
    s = score.where(mask).dropna()
    if len(s) < k:
        return None
    top = s.nlargest(k).index
    net = outcome.reindex(top) - cost
    if entry_limit is not None and gap0 is not None:
        net = net.where(gap0.reindex(top) >= entry_limit, 0.0)
    return float(net.mean())


def learn_from(session: date, state: M.MetaState, model: R.Ranker | None, *,
               open_model: R.Ranker | None = None, log=print) -> bool:
    """Grade every expert on `session`'s real outcome and fold it in (idempotent)."""
    if state.updated_through is not None and str(session) <= state.updated_through:
        return False
    inputs = FT.load_inputs(session - timedelta(days=420), session)
    panel = FT.build(inputs, with_open=True)
    if session not in set(panel.index.get_level_values("session")):
        log(f"  no labelled rows for {session}; weights unchanged")
        return False
    rows = panel.xs(session, level="session", drop_level=False)
    rows = rows[rows["was_be_20"].fillna(0) == 0]
    mask = rows["turn_rank"] <= BOOK_UNIVERSE
    # A session the model was trained on is not evidence about it (Saturday's
    # retrain includes Friday; Monday morning must not grade it on Friday).
    if model is not None and model.trained_through and str(session) <= model.trained_through:
        model = None
    if open_model is not None and open_model.trained_through and str(session) <= open_model.trained_through:
        open_model = None
    scores = expert_scores(rows, model, open_model)
    limit = G.GapReversalConfig().entry_limit_pct    # 08:45 lists enter via the auction limit
    returns = {name: (top_book(rows["short_bps"], s, mask, cost=COST_BPS + OPEN_ENTRY_EXTRA_BPS)
                      if name == "ranker_open" else
                      top_book(rows["short_bps"], s, mask, gap0=rows.get("gap0"), entry_limit=limit))
               for name, s in scores.items()}
    traded = traded_expert(state, set(scores))
    book = returns.get(traded)
    state.update(str(session), {k: v for k, v in returns.items() if v is not None}, book)
    log(f"  learned {session}: " + ", ".join(f"{k} {v:+.1f}bps" for k, v in returns.items() if v is not None)
        + f" | book {book:+.1f} | weights " + ", ".join(f"{k} {v:.2f}" for k, v in state.weights.items()))
    return True


def traded_expert(state: M.MetaState, available: set[str]) -> str:
    """Which expert's list the book trades: the most refined one its record allows."""
    if "ranker" not in available:
        return "rule"
    morning_choice, _ = state.guard("ranker", "rule")
    if morning_choice == "ranker" and "ranker_open" in available:
        open_choice, _ = state.guard("ranker_open", "ranker")
        return open_choice
    return morning_choice


# ── the morning run ─────────────────────────────────────────────────────────

def morning(session: date, config: G.GapReversalConfig = G.GapReversalConfig(), *,
            fetch: bool = True, log=print) -> tuple[date, list[G.Pick], dict]:
    """Refresh, learn from the previous session, rank `session`; returns (based_on, picks, info)."""
    if fetch:
        refresh(session - timedelta(days=1), log=log)
    prev = previous_session(session)
    model = R.load_champion()
    state = M.MetaState.load()
    learn_from(prev, state, model, open_model=R.load_champion("ranker_open"), log=log)
    state.save()

    inputs = FT.load_inputs(session - timedelta(days=420), session - timedelta(days=1))
    rows = FT.build(inputs, live_session=session)
    rows = rows.xs(session, level="session", drop_level=False)
    if rows.empty:
        raise DataNotReady(f"no candidates could be built for {session}")
    # Keep the loaded inputs for the 09:09 re-rank: it then only has to add
    # today's opening prices instead of re-reading 14 months of files.
    import pickle
    LIVE_INPUTS.parent.mkdir(parents=True, exist_ok=True)
    LIVE_INPUTS.write_bytes(pickle.dumps({"session": session, "inputs": inputs}, protocol=5))
    # Shortability: names moved to trade-for-trade in the last 20 sessions are
    # the ones brokers refuse to short intraday (measured cost of excluding
    # them: -0.7 bps/trade).
    rows = rows[rows["was_be_20"].fillna(0) == 0]
    scores = expert_scores(rows, model)
    expert, guard_t = state.guard() if "ranker" in scores else ("rule", None)
    picks = G.picks_from_rows(rows.droplevel("session"), scores[expert].droplevel("session"), config,
                              universe=BOOK_UNIVERSE, ranked_by=expert)
    info = {
        "ranked_by": expert, "guard_t": guard_t,
        "weights": state.normalised(list(scores)),
        "experts": list(scores),
        "model": (None if model is None else
                  {"trained_through": model.trained_through, "n_train": model.n_train,
                   "evidence": model.meta.get("evidence")}),
        "drift_alarm": state.alarm, "cusum": round(state.cusum, 1),
        "cautious_mode": state.cautious_mode, "size_multiplier": state.position_size_multiplier,
        "learned_through": state.updated_through,
        "candidates": int((rows["turn_rank"] <= BOOK_UNIVERSE).sum()),
    }
    return prev, picks, info


# ── the 09:09 re-rank on today's opening prices ─────────────────────────────

def open_rerank(session: date, config: G.GapReversalConfig = G.GapReversalConfig(), *,
                preopen: pd.DataFrame | None = None, log=print) -> tuple[list[G.Pick], dict] | None:
    """Re-rank with the auction's opening prices; None = keep the 08:45 list.

    Returns None when no open model is promoted, or when its paired record says
    it has recently been worse than the morning ranker.
    """
    import pickle

    open_model = R.load_champion("ranker_open")
    state = M.MetaState.load()
    # Fetch and archive the auction first, every day, whether or not a model is
    # promoted: the archive is what the auction-microstructure features learn
    # from, and it only grows if it is written on the days nobody trades on it.
    try:
        final = preopen if preopen is not None else nse_preopen.wait_and_fetch(session, log=log)
    except nse_preopen.PreOpenNotReady:
        if open_model is None:
            log("  pre-open feed unavailable; no open model promoted anyway")
            return None
        raise
    try:
        nse_preopen.archive(session, final)
    except Exception as exc:                                  # noqa: BLE001
        log(f"  pre-open archive failed: {exc}")
    if open_model is None:
        log("  no promoted open model: the 08:45 list stands (auction archived)")
        return None
    if traded_expert(state, {"rule", "ranker", "ranker_open"}) != "ranker_open":
        log("  the open model's guard is off (recent record not good enough): 08:45 list stands")
        return None
    cached = pickle.loads(LIVE_INPUTS.read_bytes()) if LIVE_INPUTS.exists() else None
    inputs = (cached["inputs"] if cached and cached["session"] == session
              else FT.load_inputs(session - timedelta(days=420), session - timedelta(days=1)))
    snapshot = final.assign(session=session)
    prior = inputs.preopen if inputs.preopen is not None else pd.DataFrame()
    rows = FT.build(inputs, live_session=session, with_open=True, opens=nse_preopen.opens(final),
                    preopen=pd.concat([prior, snapshot], ignore_index=True))
    rows = rows.xs(session, level="session", drop_level=False)
    rows = rows[(rows["was_be_20"].fillna(0) == 0) & rows["gap0"].notna()]
    score = pd.Series(open_model.score(rows), index=rows.index)
    picks = G.picks_from_rows(rows.droplevel("session"), score.droplevel("session"), config,
                              universe=BOOK_UNIVERSE, ranked_by="ranker_open")
    _, t = state.guard("ranker_open", "ranker")
    info = {"ranked_by": "ranker_open", "guard_t": t, "auctions": int(len(final)),
            "open_model": {"trained_through": open_model.trained_through,
                           "evidence": open_model.meta.get("evidence")}}
    return picks, info


# ── the weekly learning job ─────────────────────────────────────────────────

def weekly(*, eval_years: float = 2.0, min_t: float = 2.0, log=print) -> dict:
    """Retrain both models, re-validate walk-forward, promote each only on evidence.

    ranker       (08:45, no knowledge of the open)   must beat the gap rule
    ranker_open  (09:09, knows today's opening gap)  must beat the ranker, net of
                                                     the extra cost of a 09:15 entry
    """
    morning_cfg = R.RankerConfig(exclude=R.DEFAULT_EXCLUDE + FT.OPEN_FEATURES + FT.UNVALIDATED_FEATURES)
    # Auction-microstructure features join the open model only once the pre-open
    # archive is long enough to learn from; until then they are excluded.
    archived = nse_preopen.archived_sessions()
    open_cfg = R.RankerConfig(exclude=R.DEFAULT_EXCLUDE + FT.UNVALIDATED_FEATURES + (
        () if archived >= FT.MIN_PREOPEN_SESSIONS else FT.PREOPEN_FEATURES))
    panel = FT.history(with_open=True)
    last = max(panel.index.get_level_values("session"))
    start = (pd.Timestamp(last) - pd.DateOffset(years=int(eval_years))).strftime("%Y-%m-%d")
    clean = panel["was_be_20"].fillna(0) == 0          # the live book's shortability filter
    log(f"walk-forward {start}..{last} on {len(panel):,} rows")
    oos = R.walk_forward(panel, morning_cfg, start=start, refit="QS", log=log)
    oos_open = R.walk_forward(panel, open_cfg, start=start, refit="QS", log=log)
    limit = G.GapReversalConfig().entry_limit_pct    # the 08:45 list enters via the auction
    rule_book = R.book(panel, panel["gap1"].where(clean), universe=BOOK_UNIVERSE, entry_limit=limit)
    model_book = R.book(panel, oos.where(clean), universe=BOOK_UNIVERSE, entry_limit=limit)
    open_book = R.book(panel, oos_open.where(clean), universe=BOOK_UNIVERSE,
                       cost_bps=COST_BPS + OPEN_ENTRY_EXTRA_BPS)
    evidence = {
        "window": [start, str(last)],
        "preopen_archive_sessions": archived,
        "rule": R.summarize(rule_book, start),
        "ranker": R.summarize(model_book, start),
        "ranker_open": R.summarize(open_book, start),
        "vs_rule": R.paired(model_book, rule_book, start),
        "open_vs_ranker": R.paired(open_book, model_book, start),
        "ranker_liquidity_cost": R.summarize(
            R.book(panel, oos.where(clean), universe=BOOK_UNIVERSE, cost_bps=None), start),
    }
    verdicts = {}
    for name, cfg, gate, own in (
            ("ranker", morning_cfg, evidence["vs_rule"], evidence["ranker"]),
            ("ranker_open", open_cfg, evidence["open_vs_ranker"], evidence["ranker_open"])):
        challenger = R.Ranker.fit(panel, cfg)
        challenger.meta["evidence"] = evidence
        directory = challenger.save(R.MODELS / f"{name}_{challenger.trained_through}")
        passed = gate["t_stat"] >= min_t and gate["mean_diff_bps"] > 0 and own["net_bps_per_trade"] > 0
        if passed:
            R.promote(directory, evidence, name)
        verdicts[name] = {"promoted": passed, "challenger": directory.name}
        log(f"  {name}: {'PROMOTED' if passed else 'kept previous'} ({directory.name})")

    state = M.MetaState.load()
    # Seed the guard's paired record with the out-of-sample window so a fresh
    # install can judge both models from day one (live sessions extend it).
    if not any("ranker_open" in r for r in state.track):
        j = pd.concat([model_book, rule_book, open_book], axis=1,
                      keys=["ranker", "rule", "ranker_open"]).dropna().tail(250)
        seeded = [{"session": str(d), **{k: float(v) for k, v in row.items()}} for d, row in j.iterrows()]
        live = [r for r in state.track if seeded and r["session"] > seeded[-1]["session"]]
        state.track = (seeded + live)[-500:]
    if verdicts["ranker_open"]["promoted"]:
        ref = evidence["ranker_open"]
    elif verdicts["ranker"]["promoted"]:
        ref = evidence["ranker"]
    else:
        ref = evidence["rule"]
    state.expected_bps = float(ref.get("net_bps_per_trade", state.expected_bps))
    state.save()
    report = {"at": datetime.now().isoformat(timespec="seconds"), "verdicts": verdicts,
              "promoted": verdicts["ranker"]["promoted"], "challenger": verdicts["ranker"]["challenger"],
              **evidence}
    REPORTS.mkdir(parents=True, exist_ok=True)
    from nse_intraday_ai.atomic_io import atomic_write_json
    atomic_write_json(REPORTS / "weekly_report.json", report)
    return report
