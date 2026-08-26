from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return predictable OHLCV columns from yfinance/broker-style frames."""
    if df.empty:
        return df

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]).lower().replace(" ", "_") for col in out.columns]
    else:
        out.columns = [str(col).lower().replace(" ", "_") for col in out.columns]

    if "close" not in out.columns and "adj_close" in out.columns:
        out = out.rename(columns={"adj_close": "close"})
    keep = [col for col in ["open", "high", "low", "close", "volume"] if col in out.columns]
    out = out[keep].dropna(subset=["open", "high", "low", "close"])
    if "volume" not in out:
        out["volume"] = 0
    return out.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    if isinstance(df.index, pd.DatetimeIndex):
        session = df.index.normalize()
        cumulative_volume = df["volume"].replace(0, np.nan).groupby(session).cumsum()
        cumulative_value = (typical_price * df["volume"]).groupby(session).cumsum()
    else:
        cumulative_volume = df["volume"].replace(0, np.nan).cumsum()
        cumulative_value = (typical_price * df["volume"]).cumsum()
    return (cumulative_value / cumulative_volume).ffill().fillna(df["close"])


def rolling_zscore(series: pd.Series, window: int = 30) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return ((series - mean) / std).fillna(0)


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index — returns (plus_di, minus_di, adx)."""
    up_move = df["high"].diff()
    dn_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)
    atr_vals = atr(df, period).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_vals
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_vals
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx_vals = dx.ewm(alpha=1 / period, adjust=False).mean()
    return plus_di.fillna(0), minus_di.fillna(0), adx_vals.fillna(0)


def supertrend(
    df: pd.DataFrame, period: int = 7, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Supertrend indicator.

    Returns ``(line, direction)`` where direction is +1.0 (bullish) or -1.0
    (bearish).  The line sits below price in an uptrend and above price in a
    downtrend, acting as a dynamic trailing stop.
    """
    atr_vals = atr(df, period)
    hl_mid = (df["high"] + df["low"]) / 2
    basic_upper = (hl_mid + multiplier * atr_vals).values.copy()
    basic_lower = (hl_mid - multiplier * atr_vals).values.copy()
    closes = df["close"].values

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    for i in range(1, len(df)):
        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )

    direction = np.zeros(len(df), dtype=float)
    if len(df) > period:
        direction[period] = 1.0
        for i in range(period + 1, len(df)):
            prev = direction[i - 1]
            if prev == -1.0:
                direction[i] = 1.0 if closes[i] > final_upper[i] else -1.0
            else:
                direction[i] = -1.0 if closes[i] < final_lower[i] else 1.0

    st_line = np.where(
        direction == 1.0, final_lower, np.where(direction == -1.0, final_upper, np.nan)
    )
    return pd.Series(st_line, index=df.index), pd.Series(direction, index=df.index)


def bollinger_bands(
    close: pd.Series, period: int = 20, std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper, mid, lower) Bollinger Bands."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return mid + std_mult * std, mid, mid - std_mult * std


def keltner_channels(
    df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper, mid, lower) Keltner Channels (EMA ± ATR)."""
    mid = ema(df["close"], period)
    atr_vals = atr(df, period)
    return mid + atr_mult * atr_vals, mid, mid - atr_mult * atr_vals


def volume_profile_poc_vah_val(
    df: pd.DataFrame, num_bins: int = 24, value_area_pct: float = 0.70
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute session-anchored Volume Profile Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).

    For each session, distributes bar volume across price bins between session high and low.
    POC is the price with peak volume; VAH/VAL enclose `value_area_pct` (default 70%) of total session volume.
    """
    if df.empty or "volume" not in df or "close" not in df:
        empty = pd.Series(np.nan, index=df.index)
        return empty, empty, empty

    poc = pd.Series(index=df.index, dtype=float)
    vah = pd.Series(index=df.index, dtype=float)
    val = pd.Series(index=df.index, dtype=float)

    session_group = df.index.normalize() if isinstance(df.index, pd.DatetimeIndex) else np.zeros(len(df), dtype=int)
    unique_sessions = np.unique(session_group)

    for sess in unique_sessions:
        mask = session_group == sess
        sess_df = df[mask]
        n_bars = len(sess_df)
        if n_bars == 0:
            continue

        c = sess_df["close"].to_numpy(float)
        h = sess_df["high"].to_numpy(float)
        l = sess_df["low"].to_numpy(float)
        v = sess_df["volume"].to_numpy(float)

        sess_poc = np.zeros(n_bars, dtype=float)
        sess_vah = np.zeros(n_bars, dtype=float)
        sess_val = np.zeros(n_bars, dtype=float)

        cum_v = np.cumsum(v)

        for i in range(n_bars):
            if cum_v[i] <= 0:
                sess_poc[i] = c[i]
                sess_vah[i] = h[i]
                sess_val[i] = l[i]
                continue

            sub_h = h[: i + 1]
            sub_l = l[: i + 1]
            sub_c = c[: i + 1]
            sub_v = v[: i + 1]

            min_p = float(np.min(sub_l))
            max_p = float(np.max(sub_h))

            if max_p <= min_p:
                sess_poc[i] = sub_c[-1]
                sess_vah[i] = max_p
                sess_val[i] = min_p
                continue

            bin_edges = np.linspace(min_p, max_p, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_v = np.zeros(num_bins, dtype=float)

            bin_indices = np.clip(np.digitize(sub_c, bin_edges) - 1, 0, num_bins - 1)
            np.add.at(bin_v, bin_indices, sub_v)

            poc_idx = int(np.argmax(bin_v))
            sess_poc[i] = bin_centers[poc_idx]

            # Enclose value_area_pct starting from POC
            target_vol = cum_v[i] * value_area_pct
            current_vol = bin_v[poc_idx]
            low_idx = poc_idx
            high_idx = poc_idx

            while current_vol < target_vol and (low_idx > 0 or high_idx < num_bins - 1):
                next_low_vol = bin_v[low_idx - 1] if low_idx > 0 else -1
                next_high_vol = bin_v[high_idx + 1] if high_idx < num_bins - 1 else -1

                if next_high_vol >= next_low_vol and next_high_vol >= 0:
                    high_idx += 1
                    current_vol += next_high_vol
                elif low_idx > 0:
                    low_idx -= 1
                    current_vol += next_low_vol
                else:
                    break

            sess_vah[i] = bin_edges[high_idx + 1]
            sess_val[i] = bin_edges[low_idx]

        poc.loc[mask] = sess_poc
        vah.loc[mask] = sess_vah
        val.loc[mask] = sess_val

    return poc.ffill().fillna(df["close"]), vah.ffill().fillna(df["high"]), val.ffill().fillna(df["low"])


def clv(df: pd.DataFrame) -> pd.Series:
    """Close Location Value in [-1, 1]: where within the bar's range price closed."""
    span = (df["high"] - df["low"]).replace(0, np.nan)
    val = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span
    return val.fillna(0.0).clip(-1.0, 1.0)


def clv_flow(df: pd.DataFrame, window: int = 6) -> pd.Series:
    """Volume-weighted CLV over the trailing window — microstructure order flow proxy."""
    c = clv(df)
    v = df["volume"]
    cv_sum = (c * v).rolling(window, min_periods=1).sum()
    v_sum = v.rolling(window, min_periods=1).sum().replace(0, np.nan)
    return (cv_sum / v_sum).fillna(0.0).clip(-1.0, 1.0)


def bar_streak(df: pd.DataFrame) -> pd.Series:
    """Count of consecutive same-signed bars (+N for consecutive up-bars, -N for down-bars)."""
    delta = np.sign((df["close"] - df["open"]).to_numpy())
    streak = np.zeros(len(df), dtype=float)
    cur = 0.0
    for i in range(len(delta)):
        if delta[i] > 0:
            cur = cur + 1.0 if cur >= 0 else 1.0
        elif delta[i] < 0:
            cur = cur - 1.0 if cur <= 0 else -1.0
        else:
            cur = 0.0
        streak[i] = cur
    return pd.Series(streak, index=df.index)


def cpr_pivots(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Central Pivot Range: Pivot (P), Bottom Central (BC), Top Central (TC)."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        p = (df["high"] + df["low"] + df["close"]) / 3
        return p, p, p

    day_codes = pd.factorize(df.index.normalize())[0]
    p_arr = np.full(len(df), np.nan)
    bc_arr = np.full(len(df), np.nan)
    tc_arr = np.full(len(df), np.nan)

    unique_codes = sorted(set(day_codes))
    for a, b in zip(unique_codes, unique_codes[1:]):
        mask_prev = day_codes == a
        mask_now = day_codes == b
        prev_h = df.loc[mask_prev, "high"].max()
        prev_l = df.loc[mask_prev, "low"].min()
        prev_c = df.loc[mask_prev, "close"].iloc[-1]

        p = (prev_h + prev_l + prev_c) / 3.0
        bc = (prev_h + prev_l) / 2.0
        tc = (p - bc) + p
        p_arr[mask_now] = p
        bc_arr[mask_now] = bc
        tc_arr[mask_now] = tc

    p_s = pd.Series(p_arr, index=df.index).ffill().bfill().fillna(df["close"])
    bc_s = pd.Series(bc_arr, index=df.index).ffill().bfill().fillna(df["close"])
    tc_s = pd.Series(tc_arr, index=df.index).ffill().bfill().fillna(df["close"])
    return p_s, bc_s, tc_s


def corwin_schultz_spread(df: pd.DataFrame) -> pd.Series:
    """Corwin-Schultz (2012) High-Low Bid-Ask Spread Estimator."""
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    n = len(df)
    spread = np.zeros(n, dtype=float)
    if n < 2:
        return pd.Series(spread, index=df.index)

    h_2 = np.maximum(h[:-1], h[1:])
    l_2 = np.minimum(l[:-1], l[1:])

    gamma = (np.log(np.maximum(h_2 / np.maximum(l_2, 1e-9), 1.0))) ** 2
    beta = (np.log(np.maximum(h[:-1] / np.maximum(l[:-1], 1e-9), 1.0))) ** 2 + (
        np.log(np.maximum(h[1:] / np.maximum(l[1:], 1e-9), 1.0))
    ) ** 2

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

    exp_alpha = np.exp(alpha)
    s_est = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
    s_est = np.clip(np.nan_to_num(s_est, nan=0.0), 0.0, 0.05)
    spread[1:] = s_est
    return pd.Series(spread, index=df.index)


def vwap_dispersion_bands(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """VWAP Volatility Dispersion Bands (upper 1/2 sigma, lower 1/2 sigma)."""
    vwap_s = df.get("vwap", vwap(df))
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        diff = df["close"] - vwap_s
        std = (diff ** 2).rolling(30, min_periods=1).mean().pow(0.5).fillna(0.0)
        return vwap_s + std, vwap_s - std, vwap_s + 2 * std, vwap_s - 2 * std

    day_codes = pd.factorize(df.index.normalize())[0]
    u1 = np.zeros(len(df), dtype=float)
    l1 = np.zeros(len(df), dtype=float)
    u2 = np.zeros(len(df), dtype=float)
    l2 = np.zeros(len(df), dtype=float)

    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    vw = vwap_s.to_numpy(dtype=float)

    for code in np.unique(day_codes):
        mask = day_codes == code
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        sub_c = c[idx]
        sub_v = v[idx]
        sub_vw = vw[idx]

        sq_diff = (sub_c - sub_vw) ** 2
        cum_sq = np.cumsum(sq_diff * sub_v)
        cum_v = np.cumsum(sub_v)
        var = np.where(cum_v > 0, cum_sq / np.maximum(cum_v, 1e-9), 0.0)
        sd = np.sqrt(np.maximum(var, 0.0))

        u1[idx] = sub_vw + sd
        l1[idx] = sub_vw - sd
        u2[idx] = sub_vw + 2.0 * sd
        l2[idx] = sub_vw - 2.0 * sd

    return (
        pd.Series(u1, index=df.index),
        pd.Series(l1, index=df.index),
        pd.Series(u2, index=df.index),
        pd.Series(l2, index=df.index),
    )


def cvd_flow(df: pd.DataFrame) -> pd.Series:
    """Cumulative Volume Delta (CVD) order-flow proxy per session."""
    c = clv(df).to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    signed_v = c * v
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return pd.Series(np.cumsum(signed_v), index=df.index)

    day_codes = pd.factorize(df.index.normalize())[0]
    out = np.zeros(len(df), dtype=float)
    for code in np.unique(day_codes):
        mask = day_codes == code
        idx = np.where(mask)[0]
        out[idx] = np.cumsum(signed_v[idx])
    return pd.Series(out, index=df.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_ohlcv(df)
    if out.empty:
        return out

    out["ema_5"] = ema(out["close"], 5)
    out["ema_9"] = ema(out["close"], 9)
    out["ema_21"] = ema(out["close"], 21)
    out["ema_50"] = ema(out["close"], 50)
    out["rsi_14"] = rsi(out["close"], 14)
    out["atr_14"] = atr(out, 14).bfill()
    out["vwap"] = vwap(out)
    out["volume_z"] = rolling_zscore(out["volume"], 30)
    out["range_pct"] = ((out["high"] - out["low"]) / out["close"]).replace([np.inf, -np.inf], 0)
    out["return_1"] = out["close"].pct_change().fillna(0)
    out["return_5"] = out["close"].pct_change(5).fillna(0)
    st_line, st_dir = supertrend(out, 7, 3.0)
    out["supertrend"] = st_line
    out["supertrend_dir"] = st_dir
    bb_upper, bb_mid, bb_lower = bollinger_bands(out["close"], 20, 2.0)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    kc_upper, _, kc_lower = keltner_channels(out, 20, 1.5)
    out["kc_upper"] = kc_upper
    out["kc_lower"] = kc_lower

    # Microstructure & Order Flow Proxies
    out["clv"] = clv(out)
    out["clv_flow"] = clv_flow(out, 6)
    out["cvd"] = cvd_flow(out)
    out["effective_spread_bps"] = corwin_schultz_spread(out) * 1e4
    span = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_ratio"] = (out["close"] - out["open"]).abs() / span
    out["upper_wick_ratio"] = (out["high"] - out[["open", "close"]].max(axis=1)) / span
    out["lower_wick_ratio"] = (out[["open", "close"]].min(axis=1) - out["low"]) / span
    out["body_ratio"] = out["body_ratio"].fillna(0.0).clip(0.0, 1.0)
    out["upper_wick_ratio"] = out["upper_wick_ratio"].fillna(0.0).clip(0.0, 1.0)
    out["lower_wick_ratio"] = out["lower_wick_ratio"].fillna(0.0).clip(0.0, 1.0)
    out["streak"] = bar_streak(out)

    # Intraday Volume Profile POC/VAH/VAL
    poc, vah, val = volume_profile_poc_vah_val(out)
    out["poc"] = poc
    out["vah"] = vah
    out["val"] = val

    # VWAP Dispersion Bands
    vu1, vl1, vu2, vl2 = vwap_dispersion_bands(out)
    out["vwap_u1"] = vu1
    out["vwap_l1"] = vl1
    out["vwap_u2"] = vu2
    out["vwap_l2"] = vl2

    # Central Pivot Range
    cpr_p, cpr_bc, cpr_tc = cpr_pivots(out)
    out["cpr_p"] = cpr_p
    out["cpr_bc"] = cpr_bc
    out["cpr_tc"] = cpr_tc

    # Market regime: ADX + ATR percentile
    plus_di, minus_di, adx_vals = adx(out, 14)
    out["adx_14"] = adx_vals
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di
    _win = min(60, max(15, len(out)))
    out["atr_pct"] = out["atr_14"].rolling(_win, min_periods=14).rank(pct=True).fillna(0.5)
    _regime = pd.Series("RANGING", index=out.index, dtype=object)
    _high_vol = out["atr_pct"] >= 0.75
    _trending_up = (adx_vals >= 20) & (plus_di > minus_di)
    _trending_dn = (adx_vals >= 20) & (minus_di >= plus_di)
    _regime = _regime.where(~_high_vol, "HIGH_VOL")
    _regime = _regime.where(~_trending_up, "TRENDING_UP")
    _regime = _regime.where(~_trending_dn, "TRENDING_DOWN")
    out["regime"] = _regime
    return out
