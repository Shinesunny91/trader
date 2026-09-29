"""Cross-sectional ranking model for the gap-reversal book.

The gap rule ranks the universe by one number, yesterday's overnight gap.  The
ranker ranks it by a gradient-boosted estimate of each stock's *relative*
short-side return for the session (the day's cross-sectional mean removed, so
the model learns which names to short, not whether the market will fall),
using every point-in-time feature in `features.py`.

Everything here is walk-forward: a model only ever scores sessions after the
last one it was trained on, and `walk_forward` refits on a schedule exactly as
the weekly job does live.  Artifacts are versioned directories under
data/models/, with `champion.json` pointing at the one in use.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nse_intraday_ai.features import META_COLUMNS, TARGETS

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"
NON_FEATURES = set(META_COLUMNS) | set(TARGETS)


@dataclass(frozen=True)
class RankerConfig:
    target: str = "short_bps"
    target_kind: str = "demean"        # demean | rank | raw
    clip_bps: float = 800.0
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 400
    l2_regularization: float = 1.0
    max_features: float = 0.6
    seeds: tuple[int, ...] = (0,)
    exclude: tuple[str, ...] = ()      # features to leave out


def feature_columns(panel: pd.DataFrame, config: RankerConfig = RankerConfig()) -> list[str]:
    return [c for c in panel.columns
            if c not in NON_FEATURES and c not in config.exclude and panel[c].dtype.kind == "f"]


def make_target(panel: pd.DataFrame, config: RankerConfig) -> pd.Series:
    y = panel[config.target].clip(-config.clip_bps, config.clip_bps)
    sess = panel.index.get_level_values("session")
    if config.target_kind == "demean":
        return y - y.groupby(sess).transform("mean")
    if config.target_kind == "rank":
        return y.groupby(sess).rank(pct=True) - 0.5
    return y


@dataclass
class Ranker:
    config: RankerConfig
    features: list[str]
    models: list = field(default_factory=list, repr=False)
    trained_through: str | None = None
    n_train: int = 0
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, panel: pd.DataFrame, config: RankerConfig = RankerConfig(),
            features: list[str] | None = None, sample_weight: np.ndarray | None = None) -> "Ranker":
        from sklearn.ensemble import HistGradientBoostingRegressor

        feats = features or feature_columns(panel, config)
        X = panel[feats].to_numpy(np.float32)
        y = make_target(panel, config).to_numpy()
        models = []
        for seed in config.seeds:
            m = HistGradientBoostingRegressor(
                max_iter=config.max_iter, learning_rate=config.learning_rate,
                max_leaf_nodes=config.max_leaf_nodes, min_samples_leaf=config.min_samples_leaf,
                l2_regularization=config.l2_regularization, max_features=config.max_features,
                early_stopping=False, random_state=seed)
            m.fit(X, y, sample_weight=sample_weight)
            models.append(m)
        last = max(panel.index.get_level_values("session"))
        return cls(config=config, features=feats, models=models, trained_through=str(last),
                   n_train=len(panel))

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise KeyError(f"frame lacks {len(missing)} model features, e.g. {missing[:5]}")
        X = frame[self.features].to_numpy(np.float32)
        return np.mean([m.predict(X) for m in self.models], axis=0)

    # ── persistence ────────────────────────────────────────────────────────
    def save(self, directory: Path | None = None) -> Path:
        directory = directory or MODELS / f"ranker_{self.trained_through}_{datetime.now():%H%M%S}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "models.pkl").write_bytes(pickle.dumps(self.models, protocol=5))
        (directory / "meta.json").write_text(json.dumps({
            "config": asdict(self.config), "features": self.features,
            "trained_through": self.trained_through, "n_train": self.n_train,
            "saved_at": datetime.now().isoformat(timespec="seconds"), **self.meta,
        }, indent=2, default=str))
        return directory

    @classmethod
    def load(cls, directory: Path) -> "Ranker":
        meta = json.loads((directory / "meta.json").read_text())
        cfg = meta.pop("config")
        cfg["seeds"] = tuple(cfg.get("seeds", (0,)))
        cfg["exclude"] = tuple(cfg.get("exclude", ()))
        return cls(config=RankerConfig(**cfg), features=meta.pop("features"),
                   models=pickle.loads((directory / "models.pkl").read_bytes()),
                   trained_through=meta.pop("trained_through"), n_train=meta.pop("n_train"),
                   meta=meta)


def champion_path() -> Path:
    return MODELS / "champion.json"


def load_champion() -> Ranker | None:
    """The promoted model, or None (the book then runs on the gap rule)."""
    pointer = champion_path()
    if not pointer.exists():
        return None
    try:
        return Ranker.load(MODELS / json.loads(pointer.read_text())["directory"])
    except (OSError, KeyError, ValueError, pickle.UnpicklingError):
        return None


def promote(directory: Path, evidence: dict) -> None:
    champion_path().parent.mkdir(parents=True, exist_ok=True)
    champion_path().write_text(json.dumps({
        "directory": directory.name, "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "evidence": evidence}, indent=2, default=str))


# ── walk-forward evaluation ─────────────────────────────────────────────────

def walk_forward(panel: pd.DataFrame, config: RankerConfig = RankerConfig(), *,
                 start: str = "2019-01-01", refit: str = "QS", min_train_years: float = 2.0,
                 window_years: float | None = None, halflife_years: float | None = None,
                 log=None) -> pd.Series:
    """Out-of-sample scores: each block is scored by a model fit only on data before it."""
    d = pd.to_datetime(panel.index.get_level_values("session"))
    edges = pd.date_range(start, d.max() + pd.Timedelta(days=1), freq=refit)
    if len(edges) == 0 or edges[0] > pd.Timestamp(start):
        edges = pd.DatetimeIndex([pd.Timestamp(start)]).append(edges)
    edges = edges.append(pd.DatetimeIndex([d.max() + pd.Timedelta(days=1)]))
    out = pd.Series(np.nan, index=panel.index, dtype="float64")
    for a, b in zip(edges[:-1], edges[1:]):
        test = (d >= a) & (d < b)
        if not test.any():
            continue
        lo = d.min() if window_years is None else a - pd.DateOffset(years=int(window_years))
        train = (d < a) & (d >= lo)
        if not train.any() or (a - d[train].min()).days < 365 * min_train_years:
            continue
        weight = None
        if halflife_years:
            age = (a - d[train]).days.to_numpy() / 365.25
            weight = 0.5 ** (age / halflife_years)
        model = Ranker.fit(panel[train], config, sample_weight=weight)
        out[test] = model.score(panel[test])
        if log:
            log(f"  fold {a.date()}..{(b - pd.Timedelta(days=1)).date()} train {int(train.sum()):,}")
    return out


FEES_BPS = 7.0


def book(panel: pd.DataFrame, score: pd.Series, *, k: int = 8, universe: int = 300,
         cost_bps: float | None = 13.0, side: str = "short") -> pd.Series:
    """Net bps per trade of each session's top-k book (score descending).

    cost_bps=None charges each trade fees + its own estimated effective spread
    (Abdi-Ranaldo, floored at 3 bps) instead of a flat cost.
    """
    ok = score.notna() & (panel["turn_rank"] <= universe)
    df = pd.DataFrame({"s": score[ok], "y": panel.loc[ok, f"{side}_bps"]})
    df["c"] = (FEES_BPS + panel.loc[ok, "spread_bps"].fillna(20.0).clip(lower=3.0)
               if cost_bps is None else cost_bps)
    df["r"] = df.groupby(level="session")["s"].rank(ascending=False, method="first")
    top = df[df["r"] <= k]
    return (top["y"] - top["c"]).groupby(level="session").mean()


def summarize(daily: pd.Series, start: str | None = None, end: str | None = None) -> dict:
    x = daily.copy()
    x.index = pd.to_datetime(x.index)
    if start:
        x = x[x.index >= start]
    if end:
        x = x[x.index <= end]
    x = x.dropna()
    if len(x) < 2:
        return {"sessions": len(x)}
    eq = (x / 100).cumsum()
    return {
        "sessions": int(len(x)), "first": str(x.index[0].date()), "last": str(x.index[-1].date()),
        "net_bps_per_trade": round(float(x.mean()), 2),
        "t_stat": round(float(x.mean() / x.std() * np.sqrt(len(x))), 2),
        "sharpe": round(float(x.mean() / x.std() * np.sqrt(250)), 2),
        "up_day_pct": round(float((x > 0).mean() * 100), 1),
        "max_drawdown_pct": round(float((eq.cummax() - eq).max()), 2),
        "by_year": {int(k): round(float(v), 1) for k, v in x.groupby(x.index.year).mean().items()},
    }


def paired(a: pd.Series, b: pd.Series, start: str | None = None, end: str | None = None) -> dict:
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    j.index = pd.to_datetime(j.index)
    if start:
        j = j[j.index >= start]
    if end:
        j = j[j.index <= end]
    diff = j["a"] - j["b"]
    return {"mean_diff_bps": round(float(diff.mean()), 2),
            "t_stat": round(float(diff.mean() / diff.std() * np.sqrt(len(diff))), 2),
            "sessions": int(len(diff)), "a_better_pct": round(float((diff > 0).mean() * 100), 1)}
