"""The book's daily and weekly loops: data in, experts scored, weights learned, picks out.

Morning (before the pre-open):
  1. refresh   NSE equity + F&O bhavcopies, participant OI, overnight global closes
  2. learn     score the *previous* session for every expert against what actually
               happened (official NSE bars) and fold it into the Hedge weights and
               the drift alarm — the book adapts every day without retraining
  3. rank      build today's features through the live path (the same code that
               built the training data), score each expert, blend, publish picks

Weekly:
  rebuild the full point-in-time history, re-run the walk-forward over the last
  two years, fit a challenger on everything, and promote it only if its
  out-of-sample book beat the gap rule — otherwise the champion stays, or the
  book runs on the rule alone.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nse_intraday_ai import features as FT
from nse_intraday_ai import gap_reversal as G
from nse_intraday_ai import meta as M
from nse_intraday_ai import nse_bhav, nse_extra, nse_fo
from nse_intraday_ai import ranker as R

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "data" / "models"
BOOK_UNIVERSE = 300
COST_BPS = 13.0


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

def expert_scores(rows: pd.DataFrame, model: R.Ranker | None) -> dict[str, pd.Series]:
    """Every expert's score for the candidate rows (higher = better short)."""
    scores = {"rule": rows["gap1"].astype(float)}
    if model is not None:
        scores["ranker"] = pd.Series(model.score(rows), index=rows.index)
    return scores


def blended(scores: dict[str, pd.Series], state: M.MetaState, mask: pd.Series) -> pd.Series:
    live = {k: v.where(mask) for k, v in scores.items()}
    weights = state.normalised(list(live))
    return M.blend(live, weights)


def top_book(outcome: pd.Series, score: pd.Series, mask: pd.Series, k: int = 8,
             cost: float = COST_BPS) -> float | None:
    s = score.where(mask).dropna()
    if len(s) < k:
        return None
    return float(outcome.reindex(s.nlargest(k).index).mean()) - cost


def learn_from(session: date, state: M.MetaState, model: R.Ranker | None, *, log=print) -> bool:
    """Fold the realised outcome of `session` into the Hedge weights (idempotent)."""
    if state.updated_through is not None and str(session) <= state.updated_through:
        return False
    inputs = FT.load_inputs(session - timedelta(days=420), session)
    panel = FT.build(inputs)
    if session not in set(panel.index.get_level_values("session")):
        log(f"  no labelled rows for {session}; weights unchanged")
        return False
    rows = panel.xs(session, level="session", drop_level=False)
    rows = rows[rows["was_be_20"].fillna(0) == 0]
    mask = rows["turn_rank"] <= BOOK_UNIVERSE
    scores = expert_scores(rows, model)
    returns = {name: top_book(rows["short_bps"], s, mask) for name, s in scores.items()}
    traded, _ = state.guard() if "ranker" in scores else ("rule", None)
    book = returns.get(traded)
    state.update(str(session), {k: v for k, v in returns.items() if v is not None}, book)
    log(f"  learned {session}: " + ", ".join(f"{k} {v:+.1f}bps" for k, v in returns.items() if v is not None)
        + f" | book {book:+.1f} | weights " + ", ".join(f"{k} {v:.2f}" for k, v in state.weights.items()))
    return True


# ── the morning run ─────────────────────────────────────────────────────────

def morning(session: date, config: G.GapReversalConfig = G.GapReversalConfig(), *,
            fetch: bool = True, log=print) -> tuple[date, list[G.Pick], dict]:
    """Refresh, learn from the previous session, rank `session`; returns (based_on, picks, info)."""
    if fetch:
        refresh(session - timedelta(days=1), log=log)
    prev = previous_session(session)
    model = R.load_champion()
    state = M.MetaState.load()
    learn_from(prev, state, model, log=log)
    state.save()

    rows = FT.live(session)
    rows = rows.xs(session, level="session", drop_level=False)
    if rows.empty:
        raise DataNotReady(f"no candidates could be built for {session}")
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
        "learned_through": state.updated_through,
        "candidates": int((rows["turn_rank"] <= BOOK_UNIVERSE).sum()),
    }
    return prev, picks, info


# ── the weekly learning job ─────────────────────────────────────────────────

def weekly(*, config: R.RankerConfig | None = None, eval_years: float = 2.0,
           min_t: float = 2.0, log=print) -> dict:
    """Retrain, re-validate, and promote only on out-of-sample evidence."""
    config = config or R.RankerConfig()
    panel = FT.history()
    last = max(panel.index.get_level_values("session"))
    start = (pd.Timestamp(last) - pd.DateOffset(years=int(eval_years))).strftime("%Y-%m-%d")
    log(f"walk-forward {start}..{last} on {len(panel):,} rows")
    oos = R.walk_forward(panel, config, start=start, refit="QS", log=log)
    clean = panel["was_be_20"].fillna(0) == 0          # the live book's shortability filter
    rule_book = R.book(panel, panel["gap1"].where(clean), universe=BOOK_UNIVERSE)
    model_book = R.book(panel, oos.where(clean), universe=BOOK_UNIVERSE)
    evidence = {
        "window": [start, str(last)],
        "ranker": R.summarize(model_book, start),
        "rule": R.summarize(rule_book, start),
        "vs_rule": R.paired(model_book, rule_book, start),
        "ranker_liquidity_cost": R.summarize(
            R.book(panel, oos.where(clean), universe=BOOK_UNIVERSE, cost_bps=None), start),
    }
    challenger = R.Ranker.fit(panel, config)
    challenger.meta["evidence"] = evidence
    directory = challenger.save()
    passed = (evidence["vs_rule"]["t_stat"] >= min_t and evidence["vs_rule"]["mean_diff_bps"] > 0
              and evidence["ranker"]["net_bps_per_trade"] > 0)
    if passed:
        R.promote(directory, evidence)
    # Replay the daily Hedge blend over the same out-of-sample window: its mean is
    # what the live book (the blend, not the pure ranker) should earn, which is
    # the drift alarm's reference; and a fresh install starts from its final
    # weights instead of 50/50 (weights the live book has learned are kept).
    state = M.MetaState.load()
    mask = (panel["turn_rank"] <= BOOK_UNIVERSE) & oos.notna()
    blend_daily, w_hist = M.hedge_replay({"rule": panel["gap1"], "ranker": oos}, panel["short_bps"],
                                         universe_mask=mask, cost_bps=COST_BPS, eta=state.eta,
                                         floor=state.floor)
    evidence["hedge_blend_diagnostic"] = R.summarize(blend_daily, start)
    if state.updated_through is None and not w_hist.empty:
        state.weights = {k: float(v) for k, v in w_hist.iloc[-1].items()}
    # Seed the guard's paired record with the out-of-sample window, so a fresh
    # install can judge the ranker from day one (live sessions extend it).
    if not any("ranker" in r for r in state.track):
        j = pd.concat([model_book, rule_book], axis=1, keys=["ranker", "rule"]).dropna().tail(250)
        seeded = [{"session": str(d), "ranker": float(a), "rule": float(b)} for d, (a, b) in j.iterrows()]
        live = [r for r in state.track if r["session"] > seeded[-1]["session"]] if seeded else state.track
        state.track = (seeded + live)[-500:]
    ref = evidence["ranker"] if passed else evidence["rule"]
    state.expected_bps = float(ref.get("net_bps_per_trade", state.expected_bps))
    state.save()
    report = {"at": datetime.now().isoformat(timespec="seconds"), "promoted": passed,
              "challenger": directory.name, **evidence}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "weekly_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
