"""Intra-stock dependency: what the book is *actually* betting on.

Two separate uses, and they are worth keeping distinct because only one of them
reliably works.

**Risk (works).**  A book holding three names is holding three bets only if the
names are independent.  Three PSU banks, or three Adani entities, are one bet at
triple size — and the day it goes wrong it goes wrong three times.  Effective
bet count (`effective_positions`) turns the correlation matrix into a single
honest number: how many independent positions the book really has.  This needs
no forecast to be useful, which is why it is the part to trust.

**Signal (mostly does not work).**  Lead-lag — "RELIANCE moved and its peers
have not, so the peers will follow" — is the classic intraday pairs intuition.
It is also heavily arbitraged, and `scripts/correlation_study.py` measures how
much of it survives costs on this universe rather than assuming.

Correlations are computed on **returns, not prices** (price correlation is
mostly a shared upward drift), over a rolling window, and are always lagged: the
matrix used to judge a signal at time t is estimated from data ending before t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def return_matrix(frames: dict[str, pd.DataFrame], column: str = "close") -> pd.DataFrame:
    """Aligned daily (or bar) returns, one column per symbol."""
    series = {}
    for symbol, frame in frames.items():
        if frame is None or frame.empty or column not in frame:
            continue
        s = frame[column].astype(float)
        s = s[~s.index.duplicated(keep="last")]
        series[symbol] = s.pct_change()
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def rolling_corr(returns: pd.DataFrame, window: int = 120, min_periods: int = 60) -> pd.DataFrame:
    """Correlation matrix over the trailing window, ending at the last row."""
    if returns.empty:
        return pd.DataFrame()
    tail = returns.tail(window)
    valid = tail.columns[tail.notna().sum() >= min_periods]
    if len(valid) < 2:
        return pd.DataFrame()
    return tail[valid].corr()


def effective_positions(corr: pd.DataFrame, symbols: list[str]) -> float:
    """How many *independent* bets a set of positions really represents.

    Equal-weighted, so it is a property of the correlation structure alone:
    n identical names give 1.0, n perfectly independent names give n. A book
    that thinks it holds 3 positions and scores 1.2 here is concentrated,
    whatever the position sizes say.
    """
    present = [s for s in symbols if s in corr.columns]
    if len(present) < 2:
        return float(len(present))
    sub = corr.loc[present, present].to_numpy(dtype=float)
    sub = np.nan_to_num(sub, nan=0.0)
    np.fill_diagonal(sub, 1.0)
    n = len(present)
    w = np.full(n, 1.0 / n)
    variance = float(w @ sub @ w)
    return float(1.0 / variance) if variance > 0 else float(n)


def most_correlated(corr: pd.DataFrame, symbol: str, top: int = 5) -> pd.Series:
    """The names a position in `symbol` is quietly also a position in."""
    if corr.empty or symbol not in corr.columns:
        return pd.Series(dtype=float)
    row = corr[symbol].drop(labels=[symbol], errors="ignore")
    return row.reindex(row.abs().sort_values(ascending=False).index).head(top)


@dataclass(frozen=True)
class DiversifiedPick:
    symbol: str
    score: float
    max_corr_to_held: float


def pick_diversified(
    ranked: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    n: int,
    max_corr: float = 0.7,
    symbol_col: str = "symbol",
    score_col: str = "_score",
) -> list[DiversifiedPick]:
    """Take the top-scoring names, skipping any too correlated with one held.

    Greedy on purpose. The alternative — optimising the whole basket — fits the
    correlation matrix, and a correlation matrix estimated from a few months of
    daily returns is noisy enough that optimising against it mostly fits noise.
    A hard "not more than `max_corr` with something already held" is crude and
    robust, which is the right trade at this sample size.
    """
    chosen: list[DiversifiedPick] = []
    for _, row in ranked.iterrows():
        symbol = row[symbol_col]
        worst = 0.0
        for held in chosen:
            if symbol in corr.columns and held.symbol in corr.columns:
                c = corr.at[held.symbol, symbol]
                if pd.notna(c):
                    worst = max(worst, abs(float(c)))
        if worst <= max_corr:
            chosen.append(DiversifiedPick(symbol, float(row[score_col]), worst))
        if len(chosen) >= n:
            break
    return chosen


def cluster_labels(corr: pd.DataFrame, threshold: float = 0.7) -> dict[str, int]:
    """Group names into blocks that move together (single-linkage, no deps).

    Useful for showing *why* two candidates are the same bet without pulling in
    a clustering library.
    """
    if corr.empty:
        return {}
    symbols = list(corr.columns)
    parent = {s: s for s in symbols}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            c = corr.at[a, b]
            if pd.notna(c) and abs(float(c)) >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    roots: dict[str, int] = {}
    out: dict[str, int] = {}
    for s in symbols:
        r = find(s)
        if r not in roots:
            roots[r] = len(roots)
        out[s] = roots[r]
    return out


def breadth(returns: pd.DataFrame, window: int = 20) -> dict:
    """Market-wide co-movement — how much of the universe is one trade today.

    When average pairwise correlation spikes, stock selection stops mattering
    and everything becomes a bet on the index. That is exactly when a
    "diversified" three-name book is most misleading.
    """
    if returns.empty or len(returns) < window:
        return {}
    tail = returns.tail(window)
    valid = tail.columns[tail.notna().sum() >= window // 2]
    if len(valid) < 3:
        return {}
    c = tail[valid].corr().to_numpy(dtype=float)
    iu = np.triu_indices_from(c, k=1)
    pairs = c[iu]
    pairs = pairs[np.isfinite(pairs)]
    if pairs.size == 0:
        return {}
    return {
        "avg_pairwise_corr": float(pairs.mean()),
        "pct_above_0_7": float((pairs > 0.7).mean() * 100),
        "symbols": int(len(valid)),
    }
