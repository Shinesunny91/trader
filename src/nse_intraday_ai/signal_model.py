"""Persisted signal-ranking model, with an evidence gate on shipping it.

The engine emits far more signals than the book can hold — a few hundred a
session against three slots. Which three it picks is therefore the whole
decision, and the Aug-2026 study showed the engine's own confidence score
cannot make it (corr +0.009 with forward return). This module holds a model
trained to predict the **net bps of the triple-barrier trade** and used purely
for ranking.

Two guardrails, both borrowed from `meta_model.py`, which earned them:

* **Feature extraction is shared** between training and live scoring, so an
  offline result describes what actually runs. The feature list is stored in
  the model file and a mismatch refuses to load rather than silently scoring
  garbage.
* **The file existing is the evidence.** `scripts/train_signal_model.py`
  refuses to write a model unless it improved a walk-forward book, so a
  present `data/signal_model.json` means the check passed on the data
  available at that time. It does not mean the edge is established — see
  `validation` in the file, and the caveat in `expectancy_note()`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data"

# Must stay in lockstep with scripts/build_dataset.py, which produces them.
FEATURE_NAMES = [
    "vol_z", "run3", "run6", "run12", "ext_vwap", "ext_ema9", "ext_ema21", "ext_ema50",
    "pos_in_range", "age_extreme", "rsi", "adx",
    "clv", "clv_flow", "body", "wick_with", "wick_against", "bar_size_atr", "streak",
    "or_position", "sess_range_atr", "d_prior_high", "d_prior_low", "gap_atr",
    "minute", "bars_since_open", "atr_bps",
    "turnover_lakh", "turnover_z",
    "xs_impulse_rank", "xs_volume_rank", "xs_turnover_rank", "xs_n_signals",
    "xs_with_crowd",
    "m_nifty", "m_inr", "m_crude", "rel_strength",
    "conf", "rr", "n_strategies",
    # Advanced Microstructure & Order Flow Factors
    "tod_morning", "tod_lunch", "tod_afternoon",
    "clv_impulse", "run_acceleration", "trend_stack",
    "effective_spread_bps", "vwap_dispersion_z", "cvd_momentum", "poc_dist_atr",
]
REGIMES = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL"]
DERIVED = ["is_long", *[f"regime_{r}" for r in REGIMES]]
ALL_FEATURES = [*FEATURE_NAMES, *DERIVED]


def model_path(base_dir: Path | str | None = None) -> Path:
    return Path(base_dir or DEFAULT_MODEL_DIR) / "signal_model.json"


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Feature matrix in the canonical order, from a dataset-shaped frame.

    Derives missing interaction and microstructure factors automatically if
    processing legacy frames, ensuring strict forward-backward compatibility.
    """
    df_copy = frame.copy()
    if "tod_morning" not in df_copy.columns and "minute" in df_copy.columns:
        df_copy["tod_morning"] = ((df_copy["minute"] >= 555) & (df_copy["minute"] <= 630)).astype(float)
    if "tod_lunch" not in df_copy.columns and "minute" in df_copy.columns:
        df_copy["tod_lunch"] = ((df_copy["minute"] >= 690) & (df_copy["minute"] <= 780)).astype(float)
    if "tod_afternoon" not in df_copy.columns and "minute" in df_copy.columns:
        df_copy["tod_afternoon"] = ((df_copy["minute"] >= 810) & (df_copy["minute"] <= 900)).astype(float)
    if "clv_impulse" not in df_copy.columns and "clv" in df_copy.columns and "vol_z" in df_copy.columns:
        df_copy["clv_impulse"] = (df_copy["clv"] * df_copy["vol_z"].clip(0, 5)).astype(float)
    if "run_acceleration" not in df_copy.columns and "run3" in df_copy.columns and "run6" in df_copy.columns:
        df_copy["run_acceleration"] = (df_copy["run3"] - (df_copy["run6"] - df_copy["run3"])).astype(float)
    if "trend_stack" not in df_copy.columns and "ext_ema9" in df_copy.columns:
        df_copy["trend_stack"] = (
            (df_copy["ext_ema9"] > 0).astype(float) +
            (df_copy["ext_ema21"] > 0).astype(float) +
            (df_copy["ext_ema50"] > 0).astype(float) +
            (df_copy["ext_vwap"] > 0).astype(float)
        ) / 4.0
    for optional_col in ["effective_spread_bps", "vwap_dispersion_z", "cvd_momentum", "poc_dist_atr"]:
        if optional_col not in df_copy.columns:
            df_copy[optional_col] = 0.0

    missing = [name for name in FEATURE_NAMES if name not in df_copy.columns]
    if missing:
        raise ValueError(f"feature columns missing from frame: {missing}")
    X = df_copy[FEATURE_NAMES].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X["is_long"] = (df_copy["side"] == "LONG").astype(float)
    for regime in REGIMES:
        X[f"regime_{regime}"] = (df_copy["regime"] == regime).astype(float)
    return X[ALL_FEATURES].to_numpy(float)


@dataclass
class SignalModel:
    """Stacked ensemble model (Random Forest + HistGradientBoosting) with session decay.

    Both model estimators are pickled next to the JSON metadata; the JSON carries
    the normalization stats, validation metrics, and feature contract.
    """

    mu: list[float]
    sd: list[float]
    feature_names: list[str]
    trained_at: str
    n_events: int
    barrier: str
    validation: dict = field(default_factory=dict)
    _forest: object = None
    _hgb: object = None

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Predicted net bps per signal, using stacked ensemble blend."""
        if self._forest is None and self._hgb is None:
            raise RuntimeError("model estimators not loaded")
        X = feature_matrix(frame)
        Z = (X - np.asarray(self.mu)) / (np.asarray(self.sd) + 1e-9)
        preds = []
        if self._forest is not None:
            preds.append(self._forest.predict(Z))
        if self._hgb is not None:
            preds.append(self._hgb.predict(Z))
        if len(preds) == 2:
            return 0.5 * preds[0] + 0.5 * preds[1]
        return preds[0]

    def save(self, path: Path | str) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "mu": self.mu, "sd": self.sd, "feature_names": self.feature_names,
            "trained_at": self.trained_at, "n_events": self.n_events,
            "barrier": self.barrier, "validation": self.validation,
        }, indent=2))
        with path.with_suffix(".forest.pkl").open("wb") as handle:
            pickle.dump(self._forest, handle)
        if self._hgb is not None:
            with path.with_suffix(".hgb.pkl").open("wb") as handle:
                pickle.dump(self._hgb, handle)

    @classmethod
    def load(cls, path: Path | str) -> "SignalModel":
        import pickle

        path = Path(path)
        payload = json.loads(path.read_text())
        if list(payload["feature_names"]) != ALL_FEATURES:
            raise ValueError(
                f"{path} was trained on different features "
                f"({len(payload['feature_names'])} vs {len(ALL_FEATURES)}); retrain it"
            )
        model = cls(**payload)
        forest_path = path.with_suffix(".forest.pkl")
        if forest_path.exists():
            with forest_path.open("rb") as handle:
                model._forest = pickle.load(handle)
        hgb_path = path.with_suffix(".hgb.pkl")
        if hgb_path.exists():
            with hgb_path.open("rb") as handle:
                model._hgb = pickle.load(handle)
        return model


def load_if_available(base_dir: Path | str | None = None) -> SignalModel | None:
    path = model_path(base_dir)
    if not path.exists() or not path.with_suffix(".forest.pkl").exists():
        return None
    try:
        return SignalModel.load(path)
    except Exception:
        return None


def train(
    frame: pd.DataFrame,
    *,
    barrier: str,
    validation: dict | None = None,
    recency_half_life_sessions: float = 30.0,
) -> SignalModel:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    X = feature_matrix(frame)
    y = frame[f"net_bps_{barrier}"].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9

    # Calculate recency importance weights across sessions prioritizing 2026 market regime
    weights = np.ones(len(frame), dtype=float)
    if "ts" in frame.columns:
        try:
            ts = pd.to_datetime(frame["ts"])
            sessions = sorted(ts.dt.normalize().unique())
            sess_map = {sess: idx for idx, sess in enumerate(sessions)}
            sess_indices = ts.dt.normalize().map(sess_map).to_numpy(float)
            max_idx = max(sess_map.values()) if sess_map else 0
            if max_idx > 0:
                decay = np.exp((sess_indices - max_idx) / max(recency_half_life_sessions, 1.0))
                is_2026 = (ts.dt.year >= 2026).to_numpy()
                weights = np.where(is_2026, decay * 1.5, decay)
                weights = weights / (weights.mean() if weights.mean() > 0 else 1.0)
        except Exception:
            weights = np.ones(len(frame), dtype=float)

    Z = (X - mu) / sd
    forest = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=80, n_jobs=-1, random_state=42
    )
    forest.fit(Z, y, sample_weight=weights)

    hgb = HistGradientBoostingRegressor(
        max_iter=150, max_depth=6, min_samples_leaf=80, l2_regularization=1.5, random_state=42
    )
    hgb.fit(Z, y, sample_weight=weights)

    model = SignalModel(
        mu=[float(v) for v in mu], sd=[float(v) for v in sd],
        feature_names=list(ALL_FEATURES),
        trained_at=datetime.now().isoformat(timespec="seconds"),
        n_events=len(frame), barrier=barrier, validation=validation or {},
    )
    model._forest = forest
    model._hgb = hgb
    return model


def extract_live_features_row(
    candidate_or_symbol: object,
    plan: TradePlan | None = None,
    df: pd.DataFrame | None = None,
    market_context=None,
) -> dict:
    """Extract standard features from a closed-bar indicator dataframe for one candidate signal."""
    if hasattr(candidate_or_symbol, "symbol") and hasattr(candidate_or_symbol, "plan"):
        symbol = candidate_or_symbol.symbol
        plan = candidate_or_symbol.plan
        df = getattr(candidate_or_symbol, "frame", None)
    else:
        symbol = str(candidate_or_symbol)

    if df is None or plan is None or len(df) < 15:
        return {}

    idx = df.index
    row = df.iloc[-1]
    side = 1.0 if plan.side.value == "LONG" else -1.0

    o = float(row["open"])
    h = float(row["high"])
    lo = float(row["low"])
    c = float(row["close"])
    v = float(row["volume"])
    a = float(row.get("atr_14", 1.0))
    if not np.isfinite(a) or a <= 0:
        a = 1.0

    vwap_val = float(row.get("vwap", c))
    ema9_val = float(row.get("ema_9", c))
    ema21_val = float(row.get("ema_21", c))
    ema50_val = float(row.get("ema_50", c))
    rsi_val = float(row.get("rsi_14", 50.0))
    adx_val = float(row.get("adx_14", 20.0))
    vz_val = float(row.get("volume_z", 0.0))

    clv_val = float(row.get("clv", 0.0))
    clv_flow_val = float(row.get("clv_flow", 0.0))
    body_val = float(row.get("body_ratio", abs(c - o) / max(h - lo, 1e-9)))
    lower_wick_val = float(row.get("lower_wick_ratio", 0.0))
    upper_wick_val = float(row.get("upper_wick_ratio", 0.0))
    streak_val = float(row.get("streak", 0.0))

    turnover = c * v
    turnover_lakh = turnover / 1e5
    turnover_z = vz_val

    # Session bookkeeping
    if isinstance(idx, pd.DatetimeIndex) and not df.empty:
        day_mask = idx.normalize() == idx[-1].normalize()
        sess_df = df[day_mask]
        sess_hi = float(sess_df["high"].max())
        sess_lo = float(sess_df["low"].min())
        bars_since_open = len(sess_df) - 1
        minutes = int(idx[-1].hour * 60 + idx[-1].minute)
        opening_slice = sess_df.iloc[: min(6, len(sess_df))]
        or_hi = float(opening_slice["high"].max())
        or_lo = float(opening_slice["low"].min())
        extremes = sess_df["high"] if side > 0 else sess_df["low"]
        offset = int(np.argmax(extremes.to_numpy())) if side > 0 else int(np.argmin(extremes.to_numpy()))
        age_extreme = len(sess_df) - 1 - offset
    else:
        sess_hi = float(df["high"].max())
        sess_lo = float(df["low"].min())
        bars_since_open = len(df) - 1
        minutes = 600
        or_hi = sess_hi
        or_lo = sess_lo
        age_extreme = 0

    rng = max(sess_hi - sess_lo, a * 0.1)
    pos_in_range = (c - sess_lo) / rng if side > 0 else (sess_hi - c) / rng
    or_position = (c - or_lo) / max(or_hi - or_lo, a * 0.1) if side > 0 else (or_hi - c) / max(or_hi - or_lo, a * 0.1)

    # Runs
    c_arr = df["close"].to_numpy(float)
    run3 = side * (c - c_arr[-4]) / a if len(c_arr) >= 4 else 0.0
    run6 = side * (c - c_arr[-7]) / a if len(c_arr) >= 7 else 0.0
    run12 = side * (c - c_arr[-13]) / a if len(c_arr) >= 13 else 0.0

    # Macro features
    m_nifty = float(getattr(market_context, "nifty_change_pct", 0.0) or 0.0) * side
    m_inr = -float(getattr(market_context, "usdinr_change_pct", 0.0) or 0.0) * side
    m_crude = -float(getattr(market_context, "crude_change_pct", 0.0) or 0.0) * side
    nifty_ret = float(getattr(market_context, "nifty_change_pct", 0.0) or 0.0)
    stock_ret = float(row.get("return_5", 0.0)) * 100.0
    rel_strength = stock_ret - nifty_ret

    regime_str = str(row.get("regime", "RANGING"))

    # Advanced Microstructure & Order Flow Factors
    tod_morning = 1.0 if (555 <= minutes <= 630) else 0.0
    tod_lunch = 1.0 if (690 <= minutes <= 780) else 0.0
    tod_afternoon = 1.0 if (810 <= minutes <= 900) else 0.0
    clv_impulse = side * clv_val * min(max(vz_val, 0.0), 5.0)
    run_accel = run3 - (run6 - run3)
    trend_stack = (
        ((c > ema9_val) + (c > ema21_val) + (c > ema50_val) + (c > vwap_val)) / 4.0
        if side > 0
        else ((c < ema9_val) + (c < ema21_val) + (c < ema50_val) + (c < vwap_val)) / 4.0
    )
    eff_spread = float(row.get("effective_spread_bps", 0.0))
    vu1 = float(row.get("vwap_u1", vwap_val))
    vwap_disp_z = side * (c - vwap_val) / max(abs(vu1 - vwap_val), 1e-4)
    cvd_cur = float(row.get("cvd", 0.0))
    cvd_lag = float(df["cvd"].iloc[-6] if "cvd" in df.columns and len(df) >= 6 else cvd_cur)
    cvd_mom = side * (cvd_cur - cvd_lag) / max(v, 1.0)
    poc_val = float(row.get("poc", c))
    poc_dist_atr = abs(c - poc_val) / a

    return {
        "symbol": symbol,
        "side": plan.side.value,
        "regime": regime_str,
        "vol_z": vz_val,
        "run3": run3,
        "run6": run6,
        "run12": run12,
        "ext_vwap": side * (c - vwap_val) / a,
        "ext_ema9": side * (c - ema9_val) / a,
        "ext_ema21": side * (c - ema21_val) / a,
        "ext_ema50": side * (c - ema50_val) / a,
        "pos_in_range": pos_in_range,
        "age_extreme": age_extreme,
        "rsi": rsi_val,
        "adx": adx_val,
        "clv": side * clv_val,
        "clv_flow": side * clv_flow_val,
        "body": body_val,
        "wick_with": lower_wick_val if side > 0 else upper_wick_val,
        "wick_against": upper_wick_val if side > 0 else lower_wick_val,
        "bar_size_atr": (h - lo) / a,
        "streak": streak_val if side > 0 else -streak_val,
        "or_position": or_position,
        "sess_range_atr": rng / a,
        "d_prior_high": 0.0,
        "d_prior_low": 0.0,
        "gap_atr": 0.0,
        "minute": minutes,
        "bars_since_open": bars_since_open,
        "atr_bps": a / c * 1e4,
        "turnover_lakh": turnover_lakh,
        "turnover_z": turnover_z,
        "xs_impulse_rank": 0.5,
        "xs_volume_rank": 0.5,
        "xs_turnover_rank": 0.5,
        "xs_n_signals": 1.0,
        "xs_with_crowd": 1.0,
        "m_nifty": m_nifty,
        "m_inr": m_inr,
        "m_crude": m_crude,
        "rel_strength": rel_strength,
        "conf": float(plan.confidence),
        "rr": float(plan.reward_risk),
        "n_strategies": sum(1 for x in plan.strategy_votes if x.side == plan.side and x.is_trade),
        # Enhanced factors
        "tod_morning": tod_morning,
        "tod_lunch": tod_lunch,
        "tod_afternoon": tod_afternoon,
        "clv_impulse": clv_impulse,
        "run_acceleration": run_accel,
        "trend_stack": trend_stack,
        "effective_spread_bps": eff_spread,
        "vwap_dispersion_z": vwap_disp_z,
        "cvd_momentum": cvd_mom,
        "poc_dist_atr": poc_dist_atr,
    }


def score_and_rank_scan_results(
    results: list,
    market_context=None,
    model: SignalModel | None = None,
) -> list:
    """Score all ScanResults using SignalModel (or composite fallback) and rank by predicted net bps."""
    if not results:
        return []

    import dataclasses

    model = model if model is not None else load_if_available()
    rows = []
    valid_indices = []

    for idx, r in enumerate(results):
        if r.frame is not None and not r.frame.empty and r.plan.is_actionable:
            feat = extract_live_features_row(r.symbol, r.plan, r.frame, market_context)
            if feat:
                rows.append(feat)
                valid_indices.append(idx)

    if not rows:
        return results

    df_feats = pd.DataFrame(rows)
    # Cross-sectional ranks across this scan batch
    n_signals = len(df_feats)
    df_feats["xs_impulse_rank"] = df_feats["run6"].rank(pct=True) if n_signals > 1 else 0.5
    df_feats["xs_volume_rank"] = df_feats["vol_z"].rank(pct=True) if n_signals > 1 else 0.5
    df_feats["xs_turnover_rank"] = df_feats["turnover_lakh"].rank(pct=True) if n_signals > 1 else 0.5
    df_feats["xs_n_signals"] = float(n_signals)
    long_share = float((df_feats["side"] == "LONG").mean())
    df_feats["xs_with_crowd"] = np.where(df_feats["side"] == "LONG", long_share, 1.0 - long_share)

    # Compute enhanced alpha scores
    composite_scores = (
        df_feats["vol_z"].clip(0, 5) * 1.5
        + df_feats["run6"].clip(0, 4) * 1.0
        + df_feats["run3"].clip(0, 4) * 0.8
        + df_feats["m_nifty"].clip(-3, 3) * 1.2
        + df_feats["trend_stack"].clip(0, 1) * 2.0
        + df_feats["tod_morning"] * 1.0
        + df_feats["m_inr"].clip(-3, 3) * 0.5
        + df_feats["m_crude"].clip(-3, 3) * 0.5
        - df_feats["effective_spread_bps"].clip(0, 50) * 0.05
    ).to_numpy(float)

    scores = np.zeros(len(df_feats), dtype=float)
    if model is not None:
        try:
            model_preds = model.score(df_feats)
            scores = composite_scores + np.clip(model_preds / 10.0, -2.0, 2.0)
        except Exception:
            scores = composite_scores
    else:
        scores = composite_scores

    # Compute cross-stock return correlations across active candidates
    corr_matrix = pd.DataFrame()
    frames_dict = {}
    for idx, r in enumerate(results):
        if r.frame is not None and not r.frame.empty and len(r.frame) >= 15:
            frames_dict[r.symbol] = r.frame

    if len(frames_dict) >= 2:
        try:
            from nse_intraday_ai.correlation import return_matrix, rolling_corr
            ret_mat = return_matrix(frames_dict)
            if not ret_mat.empty:
                corr_matrix = rolling_corr(ret_mat, window=60, min_periods=10)
        except Exception:
            corr_matrix = pd.DataFrame()

    # Attach preliminary scores
    score_map = {valid_indices[i]: float(scores[i]) for i in range(len(scores))}
    preliminary = []
    for idx, r in enumerate(results):
        sc = score_map.get(idx, -999.0)
        preliminary.append(
            dataclasses.replace(
                r,
                predicted_net_bps=sc if model is not None else None,
                rank_score=sc,
            )
        )

    # Sort preliminary by score descending
    preliminary.sort(key=lambda x: getattr(x, "rank_score", -999.0) or -999.0, reverse=True)

    # Apply cross-stock correlation diversification penalty
    enriched = []
    chosen_symbols: list[str] = []
    for r in preliminary:
        base_score = float(getattr(r, "rank_score", -999.0) or -999.0)
        worst_corr = 0.0
        if not corr_matrix.empty and r.symbol in corr_matrix.columns:
            for held_sym in chosen_symbols:
                if held_sym in corr_matrix.columns:
                    c_val = corr_matrix.at[held_sym, r.symbol]
                    if pd.notna(c_val):
                        worst_corr = max(worst_corr, abs(float(c_val)))

        # If collinear with a higher-ranked pick (corr > 0.65), apply penalty
        corr_penalty = max(0.0, (worst_corr - 0.65) * 4.0) if worst_corr > 0.65 else 0.0
        final_score = base_score - corr_penalty

        chosen_symbols.append(r.symbol)
        enriched.append(
            dataclasses.replace(
                r,
                rank_score=final_score,
                predicted_net_bps=final_score if model is not None else None,
            )
        )

    # Final sort by correlation-adjusted score and assign model_rank
    enriched.sort(key=lambda x: getattr(x, "rank_score", -999.0) or -999.0, reverse=True)
    final_ranked = []
    for rank_idx, r in enumerate(enriched, 1):
        final_ranked.append(dataclasses.replace(r, model_rank=rank_idx))

    return final_ranked


def expectancy_note(model: SignalModel | None) -> str:
    """What the ranking is and is not known to do."""
    if model is None:
        return (
            "No ranking model is installed, so signals are ordered by the "
            "conviction + macro composite. That ordering measured −0.55% over "
            "49 sessions."
        )
    v = model.validation
    return (
        f"Signals ranked by a model trained on {model.n_events:,} labelled signals "
        f"(barrier {model.barrier}, retrained {model.trained_at[:10]}). "
        f"Walk-forward book: {v.get('net_pct', float('nan')):+.2f}% over "
        f"{v.get('sessions', 0)} held-out sessions, profit factor "
        f"{v.get('profit_factor', float('nan')):.2f}. "
        f"That is a point estimate on a short window — the bootstrap interval on the "
        f"per-session edge still spans zero, so it is evidence worth paper-trading, "
        f"not a profit forecast."
    )
