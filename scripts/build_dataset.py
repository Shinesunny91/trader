"""Build the modelling dataset: engine signals + rich features + triple-barrier labels.

Supersedes `entry_timing_study.py`, which (a) was generated before the
Yahoo snapshot-row fix, so a handful of its bars were stale quotes rather than
candles, and (b) labelled outcomes with fixed-horizon returns.  A fixed-horizon
return is not what the book earns: the book earns whichever barrier price
touches first.  So every signal here carries **triple-barrier labels** (López
de Prado): for a set of (stop, target) ATR pairs, which barrier was hit first,
after how many bars, and the resulting net bps after real costs.

Feature families, all computed causally from the signal bar backwards:

  timing        volume z, impulse over 3/6/12 bars, extension from VWAP/EMA,
                position in the day's range, bars since the session extreme
  microstructure close-location value, body and wick ratios, run of same-signed
                bars, bar size vs ATR — a bar-level proxy for order flow, which
                is the closest we get without paid tick data
  session       opening-range position, overnight gap and whether it filled,
                distance to prior-day high/low/close, minutes since the open
  liquidity     rupee turnover and its z-score — this decides real slippage
  cross-section relative strength vs NIFTY, and the signal's rank among all
                simultaneous signals (added in a second pass)

Usage:
    python scripts/build_dataset.py --workers 24 --since 2026-05-01
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from entry_timing_study import load_frame, nse_symbols  # noqa: E402
from nse_intraday_ai.costs import round_trip_bps  # noqa: E402
from nse_intraday_ai.indicators import add_indicators  # noqa: E402
from nse_intraday_ai.models import Side  # noqa: E402
from nse_intraday_ai.risk import RiskConfig  # noqa: E402
from nse_intraday_ai.strategies import EnsembleConfig, VotingSignalEngine  # noqa: E402

OUT = ROOT / "data" / "dataset.parquet"
TAIL = 200
WARMUP = 80

# (stop_atr, target_atr, max_bars) triples the labels are computed for.  The
# first is the configuration the portfolio simulator currently runs.
BARRIERS = [(1.5, 3.0, 12), (1.5, 3.0, 24), (1.0, 2.0, 12), (2.0, 4.0, 36)]
# Assumed round-trip cost for the net-bps label, at the ~₹3L position size the
# sizer actually takes.  Kept explicit so relabelling under a different cost
# assumption is a one-line change.
LABEL_COST_BPS = round_trip_bps(1000.0, 300)


def _clv(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Close-location value in [-1, 1]: where in the bar's range it closed."""
    span = np.maximum(high - low, 1e-9)
    return ((close - low) - (high - close)) / span


def build_symbol(args: tuple) -> list[dict]:
    """Signals + features for one symbol.

    ``args`` is ``(symbol, since)`` for the training build, or
    ``(symbol, since, now)`` for a **live** build.  The difference is the tail:

    * Training needs a bar after the signal (the fill) and a window after that
      (the barriers), so it stops two bars short of the data.  That is correct
      for labelling and wrong for trading — it makes the freshest signal a
      live scan can produce 10-15 minutes old, which `sim_today.write_tickets`
      then discards as stale.  The screen showed "no signal in the last 10
      minutes" essentially every cycle, and the cause was here rather than in
      the market.
    * A live build passes ``now``, which truncates the frame to bars whose
      window has actually closed (a forming candle is partial data and must
      never be scored) and then emits the newest closed bars too, with the
      signal bar's own close standing in for the unknown fill and the
      forward-looking label columns left absent.
    """
    symbol, since, *rest = args
    now = rest[0] if rest else None
    frame = load_frame(symbol, since)
    if now is not None and not frame.empty:
        frame = frame[frame.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]
    if frame.empty or len(frame) < WARMUP + 30:
        return []
    df = add_indicators(frame)
    if df.empty:
        return []

    engine = VotingSignalEngine(
        config=EnsembleConfig(min_agreeing_votes=1, min_vote_share=0.0, min_weighted_confidence=0.0)
    )
    lenient = RiskConfig(capital=1_000_000, min_confidence=0.0, min_reward_risk=0.0)

    idx = df.index
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    atr = df["atr_14"].to_numpy(float)
    vwap = df["vwap"].to_numpy(float)
    ema9 = df["ema_9"].to_numpy(float)
    ema21 = df["ema_21"].to_numpy(float)
    ema50 = df["ema_50"].to_numpy(float)
    rsi = df["rsi_14"].to_numpy(float)
    adx = df["adx_14"].to_numpy(float)
    vz = df["volume_z"].to_numpy(float)
    regime = df["regime"].to_numpy(object)
    minutes = np.asarray(idx.hour * 60 + idx.minute)

    clv = _clv(h, lo, c)
    body = np.abs(c - o) / np.maximum(h - lo, 1e-9)
    upper_wick = (h - np.maximum(o, c)) / np.maximum(h - lo, 1e-9)
    lower_wick = (np.minimum(o, c) - lo) / np.maximum(h - lo, 1e-9)
    turnover = c * v
    turnover_z = pd.Series(turnover).rolling(30).apply(
        lambda s: (s.iloc[-1] - s.mean()) / (s.std() + 1e-9), raw=False
    ).to_numpy()
    signed_bar = np.sign(c - o)
    # Volume-weighted CLV over the last 6 bars: a crude order-flow proxy.
    clv_flow = pd.Series(clv * v).rolling(6).sum().to_numpy() / np.maximum(
        pd.Series(v).rolling(6).sum().to_numpy(), 1e-9
    )

    # session bookkeeping
    day_codes = pd.factorize(idx.normalize())[0]
    session_start = np.zeros(len(df), dtype=int)
    first_of: dict[int, int] = {}
    for i, code in enumerate(day_codes):
        first_of.setdefault(code, i)
        session_start[i] = first_of[code]
    # prior-day levels
    prior_hi = np.full(len(df), np.nan)
    prior_lo = np.full(len(df), np.nan)
    prior_close = np.full(len(df), np.nan)
    unique_codes = sorted(set(day_codes))
    for a, b in zip(unique_codes, unique_codes[1:]):
        mask_prev = day_codes == a
        mask_now = day_codes == b
        prior_hi[mask_now] = h[mask_prev].max()
        prior_lo[mask_now] = lo[mask_prev].min()
        prior_close[mask_now] = c[mask_prev][-1]

    out: list[dict] = []
    # A live build walks to the last closed bar; a training build stops where
    # the labels stop being computable.
    horizon = len(df) if now is not None else len(df) - 2
    for pos in range(WARMUP, horizon):
        if not (555 <= minutes[pos] <= 890):
            continue
        a = atr[pos]
        if not np.isfinite(a) or a <= 0:
            continue
        history = df.iloc[max(0, pos - TAIL + 1) : pos + 1]
        plan = engine.analyze_precomputed(symbol, history, lenient, {}, None)
        if not plan.is_actionable or plan.entry is None:
            continue
        side = 1 if plan.side == Side.LONG else -1

        s0 = session_start[pos]
        end = pos
        while end + 1 < len(df) and day_codes[end + 1] == day_codes[pos]:
            end += 1
        # The book fills at the NEXT bar's open, so features are measured at
        # the signal bar but the label must start from that fill.  On the
        # newest bars neither exists yet: the ticket quotes its stop and target
        # relative to a fill that has not printed, which is exactly why
        # ExecutionPlan.signal_price is documented as reference-only.
        labelled = pos + 1 < len(df) and end > pos + 1
        if not labelled and now is None:
            continue
        fill = o[pos + 1] if pos + 1 < len(df) else c[pos]
        if not np.isfinite(fill) or fill <= 0:
            continue

        sess_hi = h[s0 : pos + 1].max()
        sess_lo = lo[s0 : pos + 1].min()
        rng = max(sess_hi - sess_lo, a * 0.1)
        opening = slice(s0, min(s0 + 6, pos + 1))          # first 30 minutes
        or_hi, or_lo = h[opening].max(), lo[opening].min()
        extremes = h[s0 : pos + 1] if side > 0 else lo[s0 : pos + 1]
        age_extreme = int(pos - s0 - (np.argmax(extremes) if side > 0 else np.argmin(extremes)))
        streak = 0
        while streak < 10 and signed_bar[pos - streak] == side:
            streak += 1

        row: dict = {
            "symbol": symbol, "ts": idx[pos], "side": plan.side.value,
            "fill": float(fill), "atr": float(a),
            "atr_bps": float(a / fill * 1e4),
            "conf": float(plan.confidence), "rr": float(plan.reward_risk),
            "strategies": ",".join(
                x.strategy for x in plan.strategy_votes if x.side == plan.side and x.is_trade
            ),
            "n_strategies": sum(
                1 for x in plan.strategy_votes if x.side == plan.side and x.is_trade
            ),
            "regime": str(regime[pos]), "minute": int(minutes[pos]),
            "bars_since_open": int(pos - s0),
            # timing
            "vol_z": float(vz[pos]),
            "run3": float(side * (c[pos] - c[pos - 3]) / a),
            "run6": float(side * (c[pos] - c[pos - 6]) / a),
            "run12": float(side * (c[pos] - c[pos - 12]) / a),
            "ext_vwap": float(side * (c[pos] - vwap[pos]) / a),
            "ext_ema9": float(side * (c[pos] - ema9[pos]) / a),
            "ext_ema21": float(side * (c[pos] - ema21[pos]) / a),
            "ext_ema50": float(side * (c[pos] - ema50[pos]) / a),
            "pos_in_range": float((c[pos] - sess_lo) / rng if side > 0 else (sess_hi - c[pos]) / rng),
            "age_extreme": age_extreme,
            "rsi": float(rsi[pos]), "adx": float(adx[pos]),
            # microstructure
            "clv": float(side * clv[pos]),
            "clv_flow": float(side * clv_flow[pos]) if np.isfinite(clv_flow[pos]) else 0.0,
            "body": float(body[pos]),
            "wick_with": float(lower_wick[pos] if side > 0 else upper_wick[pos]),
            "wick_against": float(upper_wick[pos] if side > 0 else lower_wick[pos]),
            "bar_size_atr": float((h[pos] - lo[pos]) / a),
            "streak": streak,
            # session structure
            "or_position": float(
                (c[pos] - or_lo) / max(or_hi - or_lo, a * 0.1) if side > 0
                else (or_hi - c[pos]) / max(or_hi - or_lo, a * 0.1)
            ),
            "sess_range_atr": float(rng / a),
            "d_prior_high": float(side * (c[pos] - prior_hi[pos]) / a) if np.isfinite(prior_hi[pos]) else 0.0,
            "d_prior_low": float(side * (c[pos] - prior_lo[pos]) / a) if np.isfinite(prior_lo[pos]) else 0.0,
            "gap_atr": float(side * (o[s0] - prior_close[pos]) / a) if np.isfinite(prior_close[pos]) else 0.0,
            # liquidity
            "turnover_lakh": float(turnover[pos] / 1e5),
            "turnover_z": float(turnover_z[pos]) if np.isfinite(turnover_z[pos]) else 0.0,
            # enhanced microstructure and regime factors
            "tod_morning": float(1.0 if (555 <= minutes[pos] <= 630) else 0.0),
            "tod_lunch": float(1.0 if (690 <= minutes[pos] <= 780) else 0.0),
            "tod_afternoon": float(1.0 if (810 <= minutes[pos] <= 900) else 0.0),
            "clv_impulse": float(side * clv[pos] * np.clip(vz[pos], 0.0, 5.0)),
            "run_acceleration": float(side * (c[pos] - c[pos - 3]) / a - side * (c[pos - 3] - c[pos - 6]) / a),
            "trend_stack": float(
                ((c[pos] > ema9[pos]) + (c[pos] > ema21[pos]) + (c[pos] > ema50[pos]) + (c[pos] > vwap[pos])) / 4.0
                if side > 0
                else ((c[pos] < ema9[pos]) + (c[pos] < ema21[pos]) + (c[pos] < ema50[pos]) + (c[pos] < vwap[pos])) / 4.0
            ),
            "effective_spread_bps": float(df["effective_spread_bps"].iloc[pos] if "effective_spread_bps" in df.columns else 0.0),
            "vwap_dispersion_z": float(side * (c[pos] - vwap[pos]) / max(abs(df["vwap_u1"].iloc[pos] - vwap[pos]) if "vwap_u1" in df.columns else a, 1e-4)),
            "cvd_momentum": float(side * (df["cvd"].iloc[pos] - (df["cvd"].iloc[pos - 6] if pos >= 6 else df["cvd"].iloc[pos])) / max(v[pos], 1.0)) if "cvd" in df.columns else 0.0,
            "poc_dist_atr": float(abs(c[pos] - (df["poc"].iloc[pos] if "poc" in df.columns else c[pos])) / a),
        }

        # ── triple-barrier labels from the fill price ────────────────────────
        # Only where the future exists.  A live tail row carries features and
        # no labels; nothing downstream may train on it.
        if not labelled:
            out.append(row)
            continue
        for stop_atr, tgt_atr, max_bars in BARRIERS:
            key = f"{stop_atr}_{tgt_atr}_{max_bars}"
            stop = fill - side * stop_atr * a
            target = fill + side * tgt_atr * a
            last = min(pos + 1 + max_bars, end)
            hit, bars_to, exit_price = "TIME", last - (pos + 1), c[last]
            for j in range(pos + 1, last + 1):
                # Stop first: intra-bar ordering is unknowable, so assume the worst.
                if (lo[j] <= stop) if side > 0 else (h[j] >= stop):
                    hit, bars_to, exit_price = "STOP", j - (pos + 1), stop
                    break
                if (h[j] >= target) if side > 0 else (lo[j] <= target):
                    hit, bars_to, exit_price = "TARGET", j - (pos + 1), target
                    break
            gross = side * (exit_price - fill) / fill * 1e4
            row[f"hit_{key}"] = hit
            row[f"bars_{key}"] = bars_to
            row[f"gross_bps_{key}"] = float(gross)
            row[f"net_bps_{key}"] = float(gross - LABEL_COST_BPS)

        # Fixed-horizon context, kept for comparability with the earlier study.
        for horizon in (6, 12, 24):
            j = min(pos + 1 + horizon, end)
            row[f"ret{horizon}_bps"] = float(side * (c[j] - fill) / fill * 1e4)
        j_eod = end
        row["ret_eod_bps"] = float(side * (c[j_eod] - fill) / fill * 1e4)
        row["bars_to_eod"] = int(end - pos)
        out.append(row)
    return out


def add_cross_sectional(frame: pd.DataFrame) -> pd.DataFrame:
    """Features that only exist relative to everything else trading right now."""
    frame = frame.sort_values("ts").copy()
    grouped = frame.groupby("ts")
    # How strong is this impulse compared with every other signal on this bar?
    frame["xs_impulse_rank"] = grouped["run6"].rank(pct=True)
    frame["xs_volume_rank"] = grouped["vol_z"].rank(pct=True)
    frame["xs_turnover_rank"] = grouped["turnover_lakh"].rank(pct=True)
    frame["xs_n_signals"] = grouped["symbol"].transform("size")
    # Crowding: what fraction of simultaneous signals point the same way?  A
    # one-sided tape is information the single-symbol view cannot see.
    long_share = grouped["side"].transform(lambda s: (s == "LONG").mean())
    frame["xs_long_share"] = long_share
    frame["xs_with_crowd"] = np.where(
        frame["side"] == "LONG", long_share, 1.0 - long_share
    )
    return frame


def add_macro(frame: pd.DataFrame) -> pd.DataFrame:
    """Relative strength vs NIFTY, and the three surviving macro alignments."""
    from intraday_sim import macro_panel

    since = (frame["ts"].min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    panel = macro_panel(since)
    merged = pd.merge_asof(
        frame.sort_values("ts"), panel, left_on="ts", right_index=True,
        direction="backward", tolerance=pd.Timedelta("45min"),
    )
    sign = np.where(merged["side"] == "LONG", 1.0, -1.0)
    merged["m_nifty"] = merged["nifty"] * sign
    merged["m_inr"] = -merged["usdinr"] * sign
    merged["m_crude"] = -merged["crude"] * sign
    # Relative strength: the stock's own 1-hour move minus the index's.  A stock
    # ripping while the index is flat is a different animal from one simply
    # riding beta, and the single-symbol features cannot tell them apart.
    merged["rel_strength"] = merged["run12"] - merged["m_nifty"].fillna(0.0)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-05-01")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--liquid", type=int, default=150,
                        help="restrict to the N most liquid cached names (0 = all); "
                             "must match the universe the live book builds")
    args = parser.parse_args()

    symbols = nse_symbols(args.since, min_rows=400)
    if args.liquid:
        # The cross-sectional features (xs_n_signals, xs_*_rank) are defined
        # relative to *everything trading at the same instant*, so the pool
        # they are computed over is part of the feature definition.  The live
        # path (sim_today) builds only the liquid names it can trade, so a
        # dataset built over all 500 gives the model an xs_n_signals four times
        # larger than anything it will ever see live.  One pool, or the feature
        # means something different in training than in production.
        from intraday_sim import liquid_symbols

        liquid = liquid_symbols(args.liquid)
        symbols = [s for s in symbols if s in liquid]
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"{len(symbols)} symbols since {args.since} | label cost {LABEL_COST_BPS:.2f} bps")

    started = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_symbol, (s, args.since)) for s in symbols]
        for done, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if done % 50 == 0 or done == len(symbols):
                print(f"  {done}/{len(symbols)} symbols, {len(rows):,} signals "
                      f"({time.time() - started:.0f}s)", flush=True)

    frame = pd.DataFrame(rows)
    print(f"\n{len(frame):,} raw signals; adding cross-sectional + macro features...")
    frame = add_cross_sectional(frame)
    frame = add_macro(frame)
    frame = frame.sort_values("ts").reset_index(drop=True)
    frame.to_parquet(OUT)
    print(f"-> {OUT}  ({len(frame):,} rows x {frame.shape[1]} cols)")
    print(f"window {frame['ts'].min()} .. {frame['ts'].max()} "
          f"({frame['ts'].dt.date.nunique()} sessions)")
    key = "1.5_3.0_12"
    print(f"\nbaseline on the shipped barrier ({key}):")
    print(f"  net bps mean {frame[f'net_bps_{key}'].mean():+.2f}  "
          f"P(target first) {(frame[f'hit_{key}'] == 'TARGET').mean():.1%}  "
          f"P(stop first) {(frame[f'hit_{key}'] == 'STOP').mean():.1%}")


if __name__ == "__main__":
    main()
