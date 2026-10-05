"""Intraday monitor page: live NSE-500 snapshot and the paper shadow board.

Reads data/intraday/live.json (written every 5 minutes by the nse-intraday-scan
timer) and data/intraday/shadow_book.csv.  Auto-refreshes every 60 s.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nse_intraday_ai import intraday_monitor as M
from nse_intraday_ai.gap_reversal_ui import DOWN, IST, PALETTE, UP, chart, read_csv, read_json, table

NAMES = {"orb60": "First-hour breakout", "gap_fade": "Gap fade", "vol_breakout": "Volume breakout"}
PCT = st.column_config.NumberColumn(format="%+.2f %%")
COLS = {
    "symbol": "Symbol", "last": "Last ₹", "chg_pct": "Day %", "from_open_pct": "From open %",
    "gap_pct": "Gap %", "rel_volume": "Rel. volume ×", "turnover_cr": "Traded ₹ cr",
    "vs_vwap_pct": "vs VWAP %", "to_upper_pct": "To upper circuit %", "to_lower_pct": "To lower circuit %",
    "band": "Band %", "new_high": "New high", "new_low": "New low",
}
FMT = {"Day %": PCT, "From open %": PCT, "Gap %": PCT, "vs VWAP %": PCT,
       "To upper circuit %": st.column_config.NumberColumn(format="%.2f %%"),
       "To lower circuit %": st.column_config.NumberColumn(format="%.2f %%"),
       "Rel. volume ×": st.column_config.NumberColumn(format="%.1f×",
                                                      help="Volume so far ÷ average volume by this time of day "
                                                           "over the last 20 sessions"),
       "Traded ₹ cr": st.column_config.NumberColumn(format="%.1f")}


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.rename(columns=COLS) if not df.empty else df


@st.fragment(run_every=60)
def _live() -> None:
    live = M.read_live()
    now = datetime.now(IST)
    if not live:
        st.info("No scan yet. The `nse-intraday-scan` timer runs every 5 minutes from 09:20 to 15:45 on "
                "trading days (`python scripts/intraday_monitor.py scan`).")
        return
    stale = live.get("session") != now.date().isoformat()
    b = live.get("breadth") or {}
    c = st.columns(5)
    c[0].metric("Session", live.get("session", "—"), "previous session" if stale else None, delta_color="off")
    c[1].metric("Last bar", live.get("last_bar") or "—", f"scanned {str(live.get('generated_at', ''))[11:16]}",
                delta_color="off")
    if b:
        c[2].metric("Advancing / declining", f"{b['advancing']} / {b['declining']}",
                    f"of {b['symbols']} NIFTY 500", delta_color="off")
        c[3].metric("Median move", f"{b['median_chg_pct']:+.2f}%")
    c[4].metric("Paper signals today", len(live.get("signals", [])))

    gb = live.get("gap_book") or []
    if gb:
        st.subheader("🛡 Gap-book stop watch")
        df = pd.DataFrame(gb)
        for r in df.itertuples():
            to_stop = getattr(r, "to_stop_pct", None)
            if r.stop_hit:
                st.error(f"**{r.symbol}** reached its stop ₹{r.stop:,.2f} — confirm your SL order filled.")
            elif to_stop is not None and not pd.isna(to_stop) and 0 <= to_stop <= M.NEAR_STOP_PCT:
                st.warning(f"**{r.symbol}** is {to_stop:.1f}% from its stop ₹{r.stop:,.2f}. Keep the stop.")
        table(df.rename(columns={"symbol": "Symbol", "entry": "Entry ₹", "stop": "Stop ₹", "last": "Last ₹",
                                 "stop_hit": "Stop hit", "stop_capped": "Capped under circuit",
                                 "to_stop_pct": "To stop %", "pnl_pct": "Short P&L %"}),
              "gap_watch", column_config={"To stop %": st.column_config.NumberColumn(format="%.2f %%"),
                                          "Short P&L %": PCT})

    st.subheader("🔥 Market now")
    tabs = st.tabs(["Stocks in play", "Gainers", "Losers", "New highs", "New lows", "Near circuit", "Announcements"])
    for tab, key, help_ in zip(tabs[:6], ["in_play", "gainers", "losers", "highs", "lows", "circuit_watch"],
                               ["Highest volume vs normal for this time of day (≥ ₹1 cr traded).",
                                "Biggest gains vs yesterday's close.", "Biggest falls vs yesterday's close.",
                                "Made a new session high in the last bar, by relative volume.",
                                "Made a new session low in the last bar, by relative volume.",
                                "Within 1.5% of the upper or lower price band — liquidity can vanish there."]):
        with tab:
            st.caption(help_)
            df = _frame(live.get(key) or [])
            if df.empty:
                st.write("—")
            else:
                table(df, f"live_{key}", column_config=FMT, height=min(38 * (len(df) + 1), 520))
    with tabs[6]:
        st.caption("Today's NSE corporate announcements for NIFTY 500 names (routine filings removed). "
                   "'Move since' = price change from the bar before the announcement to now.")
        df = pd.DataFrame(live.get("announcements") or [])
        if df.empty:
            st.write("—")
        else:
            table(df.rename(columns={"time": "Time", "symbol": "Symbol", "category": "Category", "text": "Details",
                                     "move_since_pct": "Move since %"}), "live_announcements",
                  column_config={"Move since %": PCT}, height=min(38 * (len(df) + 1), 520))

    st.subheader("🧪 Shadow board — paper only")
    st.warning("**Not validated — do not trade these.** These three strategies came closest in the 2026-10-05 "
               "research but none beat costs reliably. They run on paper so their live record can prove or "
               "disprove them; nothing here is pushed to your phone.")
    sig = pd.DataFrame(live.get("signals") or [])
    if sig.empty:
        st.write("No paper signals yet today (first-hour breakout and volume breakout start after 10:15).")
    else:
        sig["Strategy"] = sig["strategy"].map(NAMES)
        sig["Side"] = sig["side"].map({1: "LONG", -1: "SHORT"})
        sig["Status"] = sig["status"].map({"OPEN": "🟢 open", "STOP": "🔴 stop", "TARGET": "🎯 target",
                                           "CLOSED": "⚪ closed 15:15"}).fillna(sig["status"])
        view = sig[["Strategy", "symbol", "Side", "time", "entry", "stop", "target", "exit", "Status",
                    "net_bps", "reason"]].rename(columns={"symbol": "Symbol", "time": "Entry time", "entry": "Entry ₹",
                                                          "stop": "Stop ₹", "target": "Target ₹", "exit": "Last/exit ₹",
                                                          "net_bps": "Net bps", "reason": "Why"})
        c = st.columns(len(NAMES))
        for i, (k, name) in enumerate(NAMES.items()):
            s = sig[sig.strategy == k]
            c[i].metric(name, f"{s.net_bps.mean():+.0f} bps" if len(s) else "—",
                        f"{len(s)} signals, after costs" if len(s) else None, delta_color="off")
        table(view, "shadow_today", column_config={
            "Net bps": st.column_config.NumberColumn(format="%+.0f", help="After the 13.3 bps round-trip cost")})


def _board() -> None:
    book = read_csv(str(M.BOOK_PATH))
    stats = M.board_stats(book if book is not None else pd.DataFrame())
    live = M.read_live() or {}
    research = live.get("research") or {k: list(v) for k, v in M.RESEARCH.items()}
    stats["rule"] = [research.get(k, ["", ""])[0] for k in stats["strategy"]]
    stats["backtest"] = [research.get(k, ["", ""])[1] for k in stats["strategy"]]
    stats["strategy"] = stats["strategy"].map(NAMES)
    st.caption(f"Live forward record since the board started. A strategy becomes *eligible for review* only "
               f"after {M.MIN_SESSIONS} sessions with a positive mean and Holm-adjusted p < 0.05.")
    table(stats.rename(columns={"strategy": "Strategy", "sessions": "Sessions", "trades": "Trades",
                                "net_bps": "Net bps/trade", "t": "t-stat", "hit_rate": "Hit rate",
                                "holm_p": "Holm p", "status": "Status", "rule": "Rule", "backtest": "Research result"})
          .drop(columns=["p"]), "shadow_board",
          column_config={"Holm p": st.column_config.NumberColumn(format="%.3f"),
                         "Hit rate": st.column_config.NumberColumn(format="%.0f %%")})
    if book is not None and not book.empty:
        daily = book.groupby(["session", "strategy"])["net_bps"].mean().unstack().sort_index()
        cum = daily.fillna(0).cumsum()
        fig = go.Figure()
        for i, k in enumerate(cum.columns):
            fig.add_trace(go.Scatter(x=pd.to_datetime(cum.index), y=cum[k], name=NAMES.get(k, k), mode="lines+markers",
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                                     hovertemplate="%{y:+.0f} bps"))
        fig.add_hline(y=0, line=dict(color="#8A92A6", width=1, dash="dot"))
        fig.update_layout(title="Cumulative paper result per strategy (sum of daily mean net bps/trade)",
                          xaxis_title="Session", yaxis_title="Cumulative net (bps)")
        chart(fig, cum.reset_index(), "shadow_cum", height=340)


def page_intraday() -> None:
    st.caption("Information, not advice: a live view of the NIFTY 500 refreshed every 5 minutes, plus paper "
               "tracking of unvalidated intraday strategies. The tradeable strategy is the gap-reversal book.")
    _live()
    with st.expander("📊 Shadow-board record (all sessions)", expanded=False):
        _board()
    _ = (UP, DOWN, read_json)                      # shared theme helpers (keep imports explicit)
