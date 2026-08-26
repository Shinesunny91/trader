"""The recommendation workbench: every horizon, every factor, one page.

Kept out of `app.py` because that file is already 1,900 lines, and because this
view is rendered identically by the desktop browser and the Android WebView —
one implementation, so the two can never drift.

Design stance: this page shows *the evidence beside the recommendation*, always.
Two of the four horizons measured negative over ten years, and a screen that
prints a confident "TOP PICK" without that number is not a research tool, it is
a tip sheet. So every horizon carries its measured result, its independent
observation count, and its cost hurdle, and the ones that lost money say so in
red.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nse_intraday_ai import horizons as H
from nse_intraday_ai.correlation import (
    breadth,
    cluster_labels,
    effective_positions,
    most_correlated,
    return_matrix,
    rolling_corr,
)
from nse_intraday_ai.costs import segment_round_trip_bps
from nse_intraday_ai.swing import SCORES, _position_size

ROOT = Path(__file__).resolve().parents[2]

# Measured by scripts/horizon_study.py and scripts/swing_backtest.py on 10 years
# of daily bars with Groww costs. Keyed (universe, horizon).
EVIDENCE: dict[tuple[str, str], tuple[str, str]] = {
    ("commodity", "intraweek"): (
        "good",
        "+55.2% over 10 years · 8/10 years positive · max drawdown 17.6% · "
        "profit factor 1.34 · positive in both halves. Bootstrap CI on bps/trade "
        "still spans zero.",
    ),
    ("commodity", "intramonth"): (
        "warn",
        "Best rules returned +26% to +48% over 10 years but none cleared the "
        "random-pick noise floor by 2 sd, and all trailed NIFTY buy & hold (+176%).",
    ),
    ("nse", "intraweek"): (
        "bad",
        "-14.8% over 10 years. EVERY factor tested was negative at a 5-session "
        "hold; NIFTY buy & hold returned +176% over the same window.",
    ),
    ("nse", "intramonth"): (
        "warn",
        "Improves on the weekly book as costs amortise, but still trailed NIFTY "
        "buy & hold over the same decade.",
    ),
    ("nse", "intraday"): (
        "bad",
        "No rule-based gate was profitable, and the +27.6 bps/trade model-ranked "
        "book once reported here is RETRACTED — it was the best of four model "
        "families picked on their own test sessions, and over 48 sessions all "
        "four lose money on their top pick (rf, the one shipped, -20.5 bps). "
        "The population has nothing to rank: gross edge +0.8 bps against a "
        "10.1 bps round trip, and 0 of 41 features have a best decile that "
        "clears cost out of sample.",
    ),
    ("commodity", "intraday"): (
        "bad",
        "Re-measured 2026-08-19 on a purpose-built dataset (41,478 signals, 91 "
        "sessions, 15 contracts): gross edge is NEGATIVE at every barrier "
        "(-1.02 bps at the shipped one) against an 8.43 bps round trip, and 0 of "
        "41 features have a best decile that clears cost out of sample. Holding "
        "shorts to the close looks profitable (+5.4 bps) only because commodities "
        "fell in the first half of the window — 0 of 15 contracts are positive in "
        "both halves, and longs are the exact mirror image. That is beta, not edge.",
    ),
}
_LONG_HORIZON_NOTE = (
    "A 250-session hold gives ~10 independent observations per decade. No "
    "statistic computed on that is reliable, and this horizon competes directly "
    "with simply holding an index fund."
)



def _styled(frame: pd.DataFrame, style_fn):
    """Apply pandas styling, falling back to the plain frame.

    `DataFrame.style` needs jinja2 at a version pandas accepts, and
    `background_gradient` additionally needs matplotlib. Styler methods are
    *lazy* — they queue and only run when the page renders — so the failure
    would otherwise escape this guard and take the whole tab down. Forcing
    `_compute()` here moves the error inside the try. Colour is decoration; the
    numbers are the product.
    """
    try:
        styler = style_fn(frame.style)
        styler._compute()          # surface lazy failures here, not at render
        return styler
    except (AttributeError, ImportError, ValueError):
        return frame


def _evidence(universe: str, horizon_key: str) -> tuple[str, str]:
    if horizon_key == "intrayear":
        return "warn", _LONG_HORIZON_NOTE
    return EVIDENCE.get((universe, horizon_key), ("warn", "Not yet measured at this horizon."))


@st.cache_data(ttl=1800, show_spinner=False)
def _load(universe: str):
    """Daily frames + feature panel. Cached — it reads ~1M rows."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from swing_backtest import load_frames
    from nse_intraday_ai.swing import build_panel
    frames = load_frames(universe)
    return build_panel(frames), frames


def _momentum_table(panel: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    latest = panel[panel.date == panel.date.max()]
    rows = []
    for s in symbols:
        r = latest[latest.symbol == s]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({
            "Symbol": s.replace(".NS", "").replace("=F", ""),
            "1w %": r.get("ret_1w", np.nan) * 100,
            "4w %": r.get("ret_4w", np.nan) * 100,
            "12w %": r.get("ret_12w", np.nan) * 100,
            "26w %": r.get("ret_26w", np.nan) * 100,
            "RSI": r.get("rsi_14", np.nan),
            "vs 50d %": r.get("dist_sma50", np.nan) * 100,
            "vs 200d %": r.get("dist_sma200", np.nan) * 100,
            "ATR %": r.get("atr_pct", np.nan) * 100,
            "Vol surge": r.get("volume_ratio", np.nan),
        })
    return pd.DataFrame(rows)


def render() -> None:
    st.subheader("Recommendation workbench")
    st.caption(
        "Every horizon measured on the same bench: 10 years of daily bars, Groww "
        "costs by settlement segment, risk-based sizing, scored against a "
        "random-pick baseline and buy & hold."
    )

    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1, 1])
    universe = c1.selectbox("Universe", ["nse", "commodity"],
                            format_func=lambda u: "NSE 500" if u == "nse" else "Commodities")
    horizon_key = c2.selectbox("Horizon", H.ORDER,
                               index=1, format_func=lambda k: H.get(k).label)
    horizon = H.get(horizon_key)
    capital = c3.number_input("Capital ₹", 50_000.0, 5_00_00_000.0, 10_00_000.0, step=50_000.0)
    risk_pct = c4.slider("Risk/trade %", 0.5, 5.0, 2.0, 0.5)

    tone, text = _evidence(universe, horizon_key)
    {"good": st.success, "warn": st.warning, "bad": st.error}[tone](
        f"**Measured over 10 years — {H.get(horizon_key).label}, "
        f"{'commodities' if universe == 'commodity' else 'NSE'}:** {text}"
    )

    hurdle = segment_round_trip_bps(
        1000.0, int(capital * 0.3 / 1000), segment=horizon.segment_for(universe),
        slippage_bps_per_leg=5.0, symbol="GC=F" if universe == "commodity" else "",
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hold", f"{horizon.hold_sessions} session"
                      f"{'s' if horizon.hold_sessions > 1 else ''}")
    m2.metric("Settlement", horizon.segment_for(universe).value.replace("_", " "))
    m3.metric("Cost hurdle", f"{hurdle:.0f} bps", help="Round trip you must clear before profit")
    m4.metric("Independent obs / decade", f"~{horizon.independent_obs()}",
              help="A long hold gives few genuinely independent trades, however many bars exist")

    if horizon.interval != "1d":
        st.info(
            "The intraday book runs on 5-minute bars through the live scanner and "
            "`sim_today.py`, not this daily panel. Use the scanner modes in the "
            "sidebar for it; the measured result is shown above."
        )
        return

    strategy = st.selectbox(
        "Ranking rule",
        [k for k in SCORES if k != "random"],
        index=[k for k in SCORES if k != "random"].index(
            "rsi_oversold" if universe == "commodity" else "trend_pullback"),
        help="Commodities mean-revert (rsi_oversold measured best); equities trend.",
    )
    top_n = st.slider("Candidates to show", 3, 15, 6)
    diversify = st.checkbox(
        "Drop candidates correlated with a higher-ranked pick", value=True,
        help="Three correlated names are one bet at triple size.")
    max_corr = st.slider("Max correlation allowed", 0.3, 0.95, 0.70, 0.05,
                         disabled=not diversify)

    if not st.button("Find candidates", type="primary"):
        st.info(f"Reproduce every number: `python scripts/horizon_study.py "
                f"--universe {universe} --horizon {horizon_key} --model`")
        return

    with st.spinner("Loading 10 years of daily bars..."):
        panel, frames = _load(universe)
    if panel.empty:
        st.warning("No daily data cached. Run `python scripts/fetch_daily.py` first.")
        return

    latest = panel.date.max()
    min_turnover = 0.0 if universe == "commodity" else 5e7
    rows = panel[(panel.date == latest) & (panel.turnover_20d >= min_turnover)]
    if rows.empty:
        st.warning("Nothing liquid enough on the latest bar.")
        return

    ranked = rows.assign(_score=SCORES[strategy](rows)).dropna(subset=["_score"])
    if universe != "commodity" and strategy == "trend_pullback":
        # A zero score means the rule declined (below the 200-day average), not
        # a weak buy — in a downtrend everything scores zero and the sort would
        # be arbitrary.
        ranked = ranked[ranked._score > 0]
    ranked = ranked.sort_values("_score", ascending=False)
    if ranked.empty:
        st.warning("No name currently qualifies under this rule. A flat book is a position.")
        return

    rets = return_matrix(frames)
    corr = rolling_corr(rets, window=120)

    if diversify and not corr.empty:
        from nse_intraday_ai.correlation import pick_diversified
        picks = pick_diversified(ranked, corr, n=top_n, max_corr=max_corr)
        keep = [p.symbol for p in picks]
        dropped = len(ranked.head(top_n * 3)) - len(keep)
        ranked = ranked[ranked.symbol.isin(keep)].set_index("symbol").loc[keep].reset_index()
    else:
        ranked = ranked.head(top_n)

    st.caption(f"Data through **{latest.date()}** · fills at the next session's open · "
               f"stop {horizon.stop_atr} ATR · hold {horizon.hold_sessions} sessions")

    tabs = st.tabs(["Recommendations", "Momentum", "Correlation & risk",
                    "Market context", "Method"])

    # ── Recommendations ───────────────────────────────────────────────────
    with tabs[0]:
        records = []
        for _, r in ranked.iterrows():
            entry, atr = float(r["close"]), float(r["atr"])
            from nse_intraday_ai.swing import SwingConfig
            cfg = SwingConfig(capital=capital, risk_per_trade_pct=risk_pct,
                              stop_atr=horizon.stop_atr,
                              segment=horizon.segment_for(universe))
            qty = _position_size(entry, atr, cfg)
            if qty <= 0:
                continue
            cost = segment_round_trip_bps(entry, qty, segment=cfg.segment,
                                          slippage_bps_per_leg=5.0, symbol=r["symbol"])
            records.append({
                "Symbol": r["symbol"],
                "Ref close": round(entry, 2),
                "Stop": round(entry - horizon.stop_atr * atr, 2),
                "Stop %": round(horizon.stop_atr * atr / entry * 100, 2),
                "Qty": qty,
                "Value ₹": round(qty * entry),
                "Risk ₹": round(qty * horizon.stop_atr * atr),
                "Cost bps": round(cost, 1),
                "Breakeven %": round(cost / 100, 2),
                "Score": round(float(r["_score"]), 4),
            })
        if not records:
            st.warning("No candidate clears the sizing rules at this capital.")
        else:
            st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
            st.caption(
                "**Breakeven %** is how far the trade must move just to pay its own "
                "charges. The stop is quoted off the *fill* — the next session's "
                "open — not off the reference close."
            )
            if diversify and not corr.empty:
                eff = effective_positions(corr, [r["Symbol"] for r in records[:3]])
                st.metric("Effective bets in the top 3", f"{eff:.2f} of 3",
                          help="Correlated names are one bet at multiple size.")

    # ── Momentum ──────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("**Multi-timeframe momentum for the candidates**")
        mom = _momentum_table(panel, list(ranked.symbol))
        if not mom.empty:
            st.dataframe(
                _styled(mom, lambda st_: st_
                        .format({c: "{:.1f}" for c in mom.columns if c != "Symbol"})
                        .background_gradient(cmap="RdYlGn",
                                             subset=["1w %", "4w %", "12w %", "26w %"])),
                width="stretch", hide_index=True)
            st.caption(
                "Read across, not down: a name strong at 12w and weak at 1w is a "
                "pullback in an uptrend, which is what `trend_pullback` selects. "
                "Strong everywhere is late; weak everywhere is a falling knife.")
        st.markdown("**Where the whole universe sits**")
        latest_rows = panel[panel.date == latest]
        u1, u2, u3 = st.columns(3)
        u1.metric("Above 200-day avg", f"{latest_rows.above_sma200.mean() * 100:.0f}%")
        u2.metric("Above 50-day avg", f"{latest_rows.above_sma50.mean() * 100:.0f}%")
        u3.metric("Median RSI(14)", f"{latest_rows.rsi_14.median():.0f}")
        st.caption("Breadth below ~40% above the 200-day average is a market where "
                   "long-only selection rules struggle regardless of which names they pick.")

    # ── Correlation ───────────────────────────────────────────────────────
    with tabs[2]:
        if corr.empty:
            st.info("Not enough overlapping history to estimate correlations.")
        else:
            syms = [s for s in ranked.symbol if s in corr.columns]
            if len(syms) >= 2:
                st.markdown("**Correlation between the candidates**")
                sub = corr.loc[syms, syms].round(2)
                sub.index = [s.replace(".NS", "").replace("=F", "") for s in sub.index]
                sub.columns = sub.index
                st.dataframe(
                    _styled(sub, lambda st_: st_.background_gradient(
                        cmap="RdYlGn_r", vmin=-1, vmax=1)),
                    width="stretch")
                for n in (2, 3, min(5, len(syms))):
                    if n <= len(syms):
                        st.write(f"Top {n} together → **{effective_positions(corr, syms[:n]):.2f} "
                                 f"independent bets** of {n}")
            st.markdown("**What each pick is quietly also a position in**")
            for s in list(ranked.symbol)[:5]:
                mc = most_correlated(corr, s, 4)
                if not mc.empty:
                    st.write(f"`{s}` → " + ", ".join(
                        f"{k.replace('.NS','').replace('=F','')} {v:+.2f}" for k, v in mc.items()))
            b = breadth(rets)
            if b:
                st.markdown("**Market-wide co-movement**")
                bb1, bb2 = st.columns(2)
                bb1.metric("Average pairwise correlation", f"{b['avg_pairwise_corr']:.3f}")
                bb2.metric("Pairs above 0.7", f"{b['pct_above_0_7']:.1f}%")
                st.caption("When this spikes, stock selection stops mattering and every "
                           "position becomes a bet on the index.")

    # ── Market context ────────────────────────────────────────────────────
    with tabs[3]:
        _render_context()

    # ── Method ────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown(f"""
**Horizon.** {horizon.note}

**Costs.** Groww's published rates by settlement segment. Overnight positions
settle as delivery, where STT is 0.1% on *both* legs against 0.025% on the sell
for intraday — a ₹1,00,000 round trip is ~34 bps as CNC against ~13 as MIS, and
₹200 of that ₹343 is STT alone. Commodity futures are ~12 bps.

**Sizing.** Risk-based: a stop-out costs {risk_pct:.1f}% of capital regardless of
how volatile the name is, capped by notional so a quiet name cannot become
leverage.

**Validation.** Entries fill at the next session's open; a gap through a stop
fills at the open, not the stop. Long-horizon labels overlap heavily, so the
learned ranker uses **purged, embargoed walk-forward with uniqueness weighting**
— an ordinary walk-forward leaks a 21-session label into the following twenty
test days and produces a backtest that cannot be reproduced live.

**What is *not* claimed.** No configuration measured here has beaten a NIFTY
index fund over the same decade. The rules that measured positive did so with
bootstrap confidence intervals that still span zero. Treat every number as
evidence worth paper-trading, not a forecast.
""")


def _render_context() -> None:
    """Delivery %, participant OI and news — the non-price inputs."""
    flows_dir = ROOT / "data" / "nse_flows"
    shown = False

    oi_files = sorted(flows_dir.glob("oi_*.parquet")) if flows_dir.exists() else []
    if oi_files:
        try:
            from nse_intraday_ai.nse_flows import participant_features
            latest_oi = pd.read_parquet(oi_files[-1])
            feats = participant_features(latest_oi)
            if feats:
                shown = True
                st.markdown("**FII / DII positioning in index futures** "
                            f"(as of {oi_files[-1].stem.split('_')[-1]})")
                cols = st.columns(4)
                for col, who in zip(cols, ("fii", "dii", "client", "pro")):
                    v = feats.get(f"{who}_idx_fut_ratio")
                    if v is not None and np.isfinite(v):
                        col.metric(who.upper(), f"{v:+.2f}",
                                   help="Net index-future position as a share of that "
                                        "participant's total. +1 all long, -1 all short.")
                st.caption("FIIs run large directional index books; a deeply negative "
                           "reading is the closest thing Indian equities have to a "
                           "published bearish sentiment gauge. It is *lagged one "
                           "session* — published after the close.")
        except Exception as exc:                       # noqa: BLE001
            st.caption(f"Participant OI unavailable: {type(exc).__name__}")

    deliv = sorted(flows_dir.glob("delivery_*.parquet")) if flows_dir.exists() else []
    if deliv:
        try:
            d = pd.read_parquet(deliv[-1])
            shown = True
            st.markdown(f"**Delivery percentage** ({len(d):,} stocks, "
                        f"{deliv[-1].stem.split('_')[-1]})")
            dd1, dd2, dd3 = st.columns(3)
            dd1.metric("Median delivery %", f"{d.deliv_pct.median():.1f}%")
            dd2.metric("Above 70%", f"{(d.deliv_pct > 70).mean() * 100:.0f}%")
            dd3.metric("Below 30%", f"{(d.deliv_pct < 30).mean() * 100:.0f}%")
            st.caption("A move on high delivery was bought by people willing to hold "
                       "overnight; the same move on 20% delivery is speculative flow "
                       "that unwinds. Yahoo's volume field cannot tell them apart.")
        except Exception as exc:                       # noqa: BLE001
            st.caption(f"Delivery data unavailable: {type(exc).__name__}")

    if not shown:
        st.info("No NSE flow data cached yet. Fetch it with:\n\n"
                "```\npython -c \"from datetime import date; from pathlib import Path; "
                "from nse_intraday_ai.nse_flows import load_history; "
                "load_history(date(2025,1,1), date.today(), Path('data/nse_flows'))\"\n```")

    st.divider()
    st.markdown("**News risk gate**")
    st.caption(
        "`news_feed.py` pulls Google News, Moneycontrol, Economic Times, Business "
        "Standard and NSE corporate announcements — free, no keys. It is deliberately "
        "an *abstention* gate rather than a direction signal: a free RSS headline "
        "reaches you after the move, but it still tells you the price model's "
        "assumptions have broken. An ATR-derived stop assumes the last hour describes "
        "the next one, and a results announcement invalidates that outright. Enable it "
        "with `news_gate` in `data/scan_config.json`."
    )
