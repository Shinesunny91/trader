"""Validation machinery for overlapping financial labels.

The existing walk-forward is honest for the intraday book, where a label closes
within the hour and one session's labels do not touch the next.  It stops being
honest the moment the horizon lengthens.  A 21-session label spans the following
twenty sessions, so a model trained on data "before" a test date has, in fact,
already seen twenty labels that resolve *inside* the test window.  The result is
a backtest that looks excellent and a live book that does not.

Three corrections from López de Prado's *Advances in Financial Machine
Learning*, implemented here because they are the difference between a measurable
result and a flattering one:

**Purging.**  Drop training samples whose label window overlaps the test window
at all.  Without it, the model is partly fitted on the answer.

**Embargo.**  Also drop training samples immediately *after* the test window.
Serial correlation in features means the bars just after a test period carry
information about it; a few sessions of embargo removes that path.

**Uniqueness weighting.**  Overlapping labels are not independent observations.
Twenty overlapping 21-day trades carry roughly the information of one, and
unweighted fitting silently over-counts crowded periods.  Each sample is
weighted by the reciprocal of how many other labels are live at the same time.

Also here: a conformal-style abstention threshold, which converts raw model
scores into a calibrated "trade or stand aside" decision using only the
empirical distribution of past validation scores — no distributional assumption,
which matters because return distributions are not normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Sample uniqueness ──────────────────────────────────────────────────────

def concurrency(start: pd.Series, end: pd.Series) -> np.ndarray:
    """How many labels are live at each sample's start.

    `start` and `end` are the first and last bar index each label spans, as
    integer positions on a common timeline.
    """
    s = np.asarray(start, dtype=np.int64)
    e = np.asarray(end, dtype=np.int64)
    if len(s) == 0:
        return np.array([])
    horizon = int(max(e.max(), s.max())) + 2
    # Difference array: +1 when a label opens, -1 just after it closes.
    delta = np.zeros(horizon + 1, dtype=np.int64)
    np.add.at(delta, s, 1)
    np.add.at(delta, np.minimum(e + 1, horizon), -1)
    live = np.cumsum(delta)
    return live[s]


def uniqueness_weights(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Weight each sample by 1 / (labels live at the same time).

    Normalised to mean 1 so it does not change the effective sample size, only
    its distribution across time.
    """
    live = concurrency(start, end)
    if len(live) == 0:
        return np.array([])
    w = 1.0 / np.maximum(live, 1)
    mean = w.mean()
    return w / mean if mean > 0 else w


# ── Purged, embargoed walk-forward ─────────────────────────────────────────

@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray
    test_start: int
    test_end: int


def purged_walk_forward(
    start: pd.Series,
    end: pd.Series,
    *,
    n_splits: int = 6,
    embargo: int = 0,
    min_train: int = 200,
) -> list[Split]:
    """Expanding-window splits that purge overlap and embargo the aftermath.

    Expanding rather than rolling because the alternative — training only on a
    recent window — throws away the little data a long horizon has.
    """
    s = np.asarray(start, dtype=np.int64)
    e = np.asarray(end, dtype=np.int64)
    n = len(s)
    if n == 0:
        return []

    order = np.argsort(s, kind="stable")
    bounds = np.array_split(order, n_splits + 1)[1:]     # first block is seed training

    splits: list[Split] = []
    for block in bounds:
        if len(block) == 0:
            continue
        t0, t1 = int(s[block].min()), int(e[block].max())
        # Train = everything that both starts and ENDS before the test window,
        # minus an embargo tail after it.
        candidate = (e < t0) | (s > t1 + embargo)
        # Never train on anything starting after the test window in a
        # walk-forward — that is future data.
        candidate &= s < t0
        train = np.where(candidate)[0]
        if len(train) < min_train:
            continue
        splits.append(Split(train=train, test=np.asarray(block),
                            test_start=t0, test_end=t1))
    return splits


def walk_forward_predict(
    features: pd.DataFrame,
    target: pd.Series,
    start: pd.Series,
    end: pd.Series,
    *,
    model_factory=None,
    n_splits: int = 6,
    embargo: int = 5,
    weight_by_uniqueness: bool = True,
) -> pd.Series:
    """Out-of-sample predictions with purging, embargo and uniqueness weights.

    Returns a Series aligned to `features.index`, NaN where a sample was never
    in a test fold.
    """
    if model_factory is None:
        model_factory = default_model

    x = features.to_numpy(dtype=float)
    y = np.asarray(target, dtype=float)
    weights = uniqueness_weights(start, end) if weight_by_uniqueness else np.ones(len(y))

    out = np.full(len(y), np.nan)
    for split in purged_walk_forward(start, end, n_splits=n_splits, embargo=embargo):
        model = model_factory()
        xt, yt, wt = x[split.train], y[split.train], weights[split.train]
        ok = np.isfinite(yt) & np.isfinite(xt).all(axis=1)
        if ok.sum() < 50:
            continue
        try:
            model.fit(xt[ok], yt[ok], sample_weight=wt[ok])
        except TypeError:                       # model without sample_weight
            model.fit(xt[ok], yt[ok])
        test_ok = np.isfinite(x[split.test]).all(axis=1)
        idx = split.test[test_ok]
        if len(idx):
            out[idx] = model.predict(x[idx])
    return pd.Series(out, index=features.index)


def default_model():
    """Histogram gradient boosting — strong on tabular data, no extra deps.

    Shallow and heavily regularised on purpose: the binding constraint here is
    sample size, not model capacity. A decade of monthly holds is ~120
    independent observations, and a deep model will memorise them.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


# ── Calibrated abstention ──────────────────────────────────────────────────

def conformal_threshold(scores: np.ndarray, outcomes: np.ndarray, target_rate: float) -> float:
    """Score cut-off that historically kept `target_rate` of the winners.

    Distribution-free: it reads the empirical quantile of scores among past
    *profitable* outcomes rather than assuming normality. Trading only above it
    is a calibrated abstention rule — the model says "I have seen scores like
    this work" instead of "this is 0.6 probable".
    """
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    ok = np.isfinite(scores) & np.isfinite(outcomes)
    winners = scores[ok & (outcomes > 0)]
    if len(winners) < 20:
        return float("-inf")
    return float(np.quantile(winners, 1.0 - target_rate))


def deflated_sharpe(returns: np.ndarray, n_trials: int) -> float:
    """Sharpe adjusted for how many variants were tried to find it.

    Searching a grid of strategies and reporting the best one's Sharpe is
    selection bias with extra steps. This shrinks the estimate by the expected
    maximum of `n_trials` draws from a null distribution, which is the honest
    number when a sweep produced the result.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 8 or r.std(ddof=1) == 0:
        return float("nan")
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252)
    if n_trials <= 1:
        return float(sharpe)
    # Expected max of n_trials standard normals (Bailey & López de Prado).
    euler = 0.5772156649
    z = ((1 - euler) * _norm_ppf(1 - 1.0 / n_trials)
         + euler * _norm_ppf(1 - 1.0 / (n_trials * np.e)))
    return float(sharpe - z * r.std(ddof=1) / r.std(ddof=1) / np.sqrt(len(r)) * np.sqrt(252))


def _norm_ppf(p: float) -> float:
    from math import sqrt
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except ImportError:
        # Acklam-style rational approximation, adequate for this use.
        if p <= 0 or p >= 1:
            return 0.0
        q = p - 0.5
        if abs(q) <= 0.425:
            r = 0.180625 - q * q
            return q * (((2509.08 * r + 33430.6) * r + 67265.8) * r + 45921.95) / \
                   (((5226.5 * r + 28729.1) * r + 39307.9) * r + 21213.79)
        r = p if q < 0 else 1 - p
        r = sqrt(-np.log(r))
        val = (((2.938163 + r * 4.374664) * r + 1.0) / ((1.0 + r * 2.445134) * r + 1.0)) + r
        return -val if q < 0 else val
