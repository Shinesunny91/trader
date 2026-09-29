"""The gap-reversal book, rendered for the app — the validated intraday trade.

Reads what `scripts/gap_reversal.py` writes (picks, backtest, paper book) so the
screen, the phone push and the measured book are one code path.
"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nse_intraday_ai import gap_reversal as G

IST = ZoneInfo("Asia/Kolkata")
OUT = G.OUT_DIR


def _read_json(name: str) -> dict | None:
    path = OUT / name
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (OSError, ValueError):
        return None


def _picks_table(payload: dict) -> pd.DataFrame:
    rows = []
    for p in payload.get("picks", []):
        rows.append({
            "#": p["rank"],
            "Symbol": p["symbol"].removesuffix(".NS"),
            "Role": "reserve" if p.get("reserve") else "SHORT",
            "Qty": p["quantity"],
            "Gap yesterday %": p["gap_prev_pct"],
            "Prev close ₹": p["prev_close"],
            "Stop above fill ₹": p["stop_distance"],
            "Stop %": p.get("stop_pct"),
            "09:15 open ₹": p.get("entry"),
            "BUY SL-M at ₹": p.get("stop_price"),
            "Turnover ₹cr/day": p["turnover_cr"],
            "Score": p.get("score"),
        })
    return pd.DataFrame(rows)


def _equity_figure(daily: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=daily["session"], y=daily["net"], name="Session P&L",
                marker_color=["#2e7d32" if v >= 0 else "#c62828" for v in daily["net"]])
    fig.add_scatter(x=daily["session"], y=daily["net"].cumsum(), name="Cumulative",
                    mode="lines+markers", line={"color": "#1565c0", "width": 2}, yaxis="y2")
    fig.update_layout(
        title=title, height=320, margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis={"title": "₹ per session"},
        yaxis2={"title": "₹ cumulative", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": -0.2}, bargap=0.3,
    )
    return fig


def _render_picks() -> None:
    payload = _read_json("picks.json")
    now = datetime.now(IST)
    if payload is None:
        st.info("No pick list yet. It is written every weekday at 08:45 by the "
                "`nse-gap-picks` timer, or run `python scripts/gap_reversal.py picks`.")
    else:
        session = payload["session"]
        levels = " · stop prices set from the 09:15 open" if payload.get("levels_at") else ""
        st.markdown(f"#### Short list for **{session}**  \n"
                    f"<small>ranked on closes of {payload.get('based_on', '?')} · generated "
                    f"{payload['generated_at'][11:16]} IST{levels}</small>", unsafe_allow_html=True)
        if session < now.date().isoformat():
            st.warning("This list is for a past session — the picks timer has not run yet today.")
        if payload.get("drift_alarm"):
            st.error("**Drift alarm.** The live book has run persistently below its backtest "
                     f"(CUSUM {payload.get('cusum')} bps). Consider half size until the weekly review.")
        c = st.columns(3)
        who = payload.get("ranked_by", "rule")
        guard_t = payload.get("guard_t")
        c[0].metric("Ranked by", "trained model" if who == "ranker" else "gap rule",
                    f"guard t {guard_t:+.1f}" if guard_t is not None else None,
                    help="The model ranks unless its last 60 sessions were significantly worse "
                         "than the gap rule's (t < -2); then the rule takes over automatically.")
        model = payload.get("model") or {}
        c[1].metric("Model trained through", model.get("trained_through") or "—")
        c[2].metric("Data", payload.get("source", "yahoo-rule"))
        table = _picks_table(payload)
        st.dataframe(table, hide_index=True, width="stretch")


def _render_workflow(config: G.GapReversalConfig) -> None:
    st.markdown(
        f"""
**The trade — every weekday, decided before the market opens**

| When (IST) | Do this |
|---|---|
| 08:45 | Pick list arrives on your phone (ntfy) and on this page |
| **09:00 – 09:07** | Pre-open: **SELL (MIS) at market** each of the {config.picks} names, ~₹{config.capital / config.picks:,.0f} each. If a name is blocked for shorting (ASM/T2T), take the next reserve. |
| 09:18 | Stop levels arrive: place a **BUY SL-M** at your fill + the stop distance ({config.stop_atr} × daily ATR) |
| {config.square_off} | **Buy to cover** everything still open (the broker auto-squares MIS at ~15:20, usually worse) |
| 15:40 | Paper result arrives and is added to the record below |

Being late costs real money: entering at 09:20 instead of the open roughly halved
the gross edge over the development window. The signal is known the evening
before, so there is no reason to wait for the first candle.
"""
    )


def _render_evidence() -> None:
    backtest = _read_json("backtest.json")
    if backtest is None:
        st.info("No saved backtest. Run `python scripts/gap_reversal.py backtest --decade --save`.")
        return
    s = backtest["window"]["summary"]
    daily = pd.DataFrame(backtest["window"]["daily"])
    c = st.columns(5)
    c[0].metric("Net P&L", f"₹{s['net_rupees']:+,.0f}", f"{s['net_pct']:+.2f}% of capital")
    c[1].metric("Up / down days", f"{s['up_days']} / {s['down_days']}")
    c[2].metric("Win rate", f"{s['win_rate_pct']:.0f}%", f"{s['trades']} trades")
    c[3].metric("Profit factor", f"{s['profit_factor']:.2f}" if s["profit_factor"] else "—")
    c[4].metric("Max drawdown", f"{s['max_drawdown_pct']:.2f}%")
    st.plotly_chart(_equity_figure(daily, f"Last {s['sessions']} sessions "
                                          f"({s['first']} → {s['last']}), ₹10L, costs included"),
                    width="stretch")
    st.caption(f"Average {s['avg_gross_bps']:+.1f} bps gross / {s['avg_net_bps']:+.1f} bps net per "
               f"trade · ₹{s['costs_rupees']:,.0f} in brokerage, taxes and slippage · "
               f"{s['stops_hit']} stops hit · saved {backtest['generated_at'][:16]}")

    decade = backtest.get("decade")
    if decade:
        years = pd.Series(decade["by_year"]).sort_index()
        fig = go.Figure(go.Bar(x=[str(y) for y in years.index], y=years.values,
                               marker_color=["#2e7d32" if v > 0 else "#c62828" for v in years.values]))
        fig.update_layout(height=240, margin={"l": 10, "r": 10, "t": 40, "b": 10},
                          title="Ten-year check: net bps per trade, by year (daily bars, 13 bps cost)",
                          yaxis={"title": "net bps / trade"})
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{decade['sessions']:,} sessions {decade['first']} → {decade['last']}: "
                   f"{decade['net_bps_per_trade']:+.1f} bps/trade net, t = {decade['t_stat']}, "
                   f"max drawdown {decade['max_drawdown_pct']:.1f}% of capital. Every year positive.")


def _render_learning() -> None:
    """What the self-improving layers have concluded, and when."""
    report = _read_json("../models/weekly_report.json")
    state = _read_json("../models/meta_state.json")
    st.markdown("**Weekly review** — the ranker is retrained on all data, re-tested "
                "walk-forward over the last two years, and promoted only if its "
                "out-of-sample book beat the gap rule (t ≥ 2).")
    if report is None:
        st.info("No weekly review yet (`python scripts/learn.py`, or the nse-gap-learn timer).")
    else:
        r, b, v = report["ranker"], report["rule"], report["vs_rule"]
        c = st.columns(4)
        c[0].metric("Verdict", "promoted" if report["promoted"] else "kept previous")
        c[1].metric("Ranker, net/trade", f"{r.get('net_bps_per_trade', 0):+.1f} bps", f"t {r.get('t_stat')}")
        c[2].metric("Gap rule, net/trade", f"{b.get('net_bps_per_trade', 0):+.1f} bps", f"t {b.get('t_stat')}")
        c[3].metric("Ranker − rule", f"{v['mean_diff_bps']:+.1f} bps/day", f"t {v['t_stat']}")
        st.caption(f"Window {report['window'][0]} → {report['window'][1]} · challenger "
                   f"{report['challenger']} · reviewed {report['at'][:16]}")
    st.markdown("**Daily expert weights** (Hedge, learned from each morning's realised results)")
    if not state or not state.get("history"):
        st.info("No sessions learned yet — the first update happens on the next morning run.")
        return
    hist = pd.DataFrame([{"session": h["session"], **h["weights"]} for h in state["history"]])
    fig = go.Figure()
    for col in [c for c in hist.columns if c != "session"]:
        fig.add_scatter(x=hist["session"], y=hist[col], name=col, mode="lines")
    fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      yaxis={"title": "weight", "range": [0, 1]}, legend={"orientation": "h"})
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Learned through {state.get('updated_through')} · drift CUSUM "
               f"{state.get('cusum', 0):.0f} / {state.get('threshold_bps')} bps "
               f"(expected {state.get('expected_bps'):.1f} bps/trade)")


def _render_paper_book() -> None:
    path = OUT / "paper_book.csv"
    if not path.exists():
        st.info("The forward paper record starts with the first session after deployment "
                "(`nse-gap-record` timer, 15:40 on weekdays).")
        return
    book = pd.read_csv(path)
    daily = book.groupby("session", as_index=False)["net"].sum()
    total = daily["net"].sum()
    st.metric("Forward paper P&L", f"₹{total:+,.0f}",
              f"{total / G.GapReversalConfig().capital * 100:+.2f}% over {len(daily)} sessions")
    st.plotly_chart(_equity_figure(daily, "Forward paper book (never seen by the backtest)"),
                    width="stretch")
    with st.expander("Trades"):
        st.dataframe(book.sort_values(["session", "symbol"], ascending=[False, True]),
                     hide_index=True, width="stretch")


def render() -> None:
    config = G.GapReversalConfig()
    st.subheader("Gap-reversal book — validated NSE intraday shorts")
    st.caption(
        f"Short the {config.picks} liquid NSE names that gapped up most **yesterday**, at "
        f"today's open; stop {config.stop_atr}× daily ATR above the fill; cover at "
        f"{config.square_off}. Research tool — not investment advice; results are "
        "backtests and paper trades, and past edge can decay."
    )
    tabs = st.tabs(["Today", "How to trade it", "Backtest", "Paper record", "Learning", "Risks"])
    with tabs[0]:
        _render_picks()
    with tabs[1]:
        _render_workflow(config)
    with tabs[2]:
        _render_evidence()
    with tabs[3]:
        _render_paper_book()
    with tabs[4]:
        _render_learning()
    with tabs[5]:
        st.markdown(
            """
- **Expect about +0.2% of capital a day on average, not +10% a fortnight.** The
  last 14 sessions were unusually good (NIFTY fell intraday most days). Over ten
  years the book had losing months and a 14% peak-to-trough drawdown at 1× leverage.
- **Part of the return is the market.** Indian stocks drift down intraday on
  average; a short book collects that. The stock selection added ~4 standard
  deviations over random shorts in both test windows, but a strong up-day hurts
  every position at once.
- **Shortability.** Brokers block intraday shorts on names under surveillance
  (ASM/GSM) or in trade-for-trade, and big gappers are exactly the names that
  land there. Use the reserves in rank order; never skip to a lower-ranked name
  because it "looks weaker".
- **Stops can slip.** A news spike can blow through an SL-M. The stop is sized
  to cap the typical loss, not the worst case — keep position sizes as shown.
- **No leverage assumed.** Results are at 1× (₹10L exposure on ₹10L). MIS
  leverage multiplies both the return and the drawdown.
"""
        )
