"""Pages of the gap-reversal app: Today, Performance, Model.

Reads what the scheduled jobs write (data/gap_reversal/*, data/models/*), every
read cached for 60 s and failure-tolerant.  The only network read is the live
panel on the Today page: 1-minute Yahoo bars for today's names, refreshed every
60 s during market hours (set NSE_UI_LIVE=0 to disable).
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

from nse_intraday_ai import gap_reversal as G
from nse_intraday_ai.atomic_io import atomic_read_json

IST = ZoneInfo("Asia/Kolkata")
OUT = G.OUT_DIR
MODELS = G.ROOT / "data" / "models"
CAPITAL = G.GapReversalConfig().capital

# ── theme (keep in sync with .streamlit/config.toml) ────────────────────────
BG, GRID, TEXT, UP, DOWN = "#0E1117", "#262B36", "#E6E9EF", "#00D084", "#FF5C5C"
PALETTE = [UP, "#4EA8FF", "#FFB547", "#B78CFF", DOWN, "#7FDBDA"]
_axis = dict(gridcolor=GRID, zerolinecolor="#3A4050", linecolor=GRID, showspikes=True,
             spikemode="across", spikethickness=1, spikedash="dot", spikecolor="#8A92A6")
_tpl = go.layout.Template(pio.templates["plotly_dark"])
_tpl.layout.update(paper_bgcolor=BG, plot_bgcolor=BG, colorway=PALETTE, hovermode="x unified",
                   font=dict(color=TEXT, family="sans-serif", size=13),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
                   margin=dict(l=10, r=10, t=56, b=10), xaxis=_axis, yaxis=_axis)
pio.templates["nse_dark"] = _tpl
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": True,
               "toImageButtonOptions": {"format": "png", "scale": 2}}


# ── cached, failure-tolerant reads ──────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def read_json(path: str) -> dict | None:
    try:
        data = atomic_read_json(path, default=None)
        return data if isinstance(data, dict) else None
    except Exception:                                    # noqa: BLE001 — never a traceback
        return None


@st.cache_data(ttl=60, show_spinner=False)
def read_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if Path(path).exists() else None
    except Exception:                                    # noqa: BLE001
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _holidays() -> set[date]:
    """NSE holidays from the local cache + bhavcopy markers (no network)."""
    from nse_intraday_ai import nse_calendar as cal
    days = {date.fromisoformat(d) for d in (read_json(str(cal.CACHE)) or {}).get("holidays", [])}
    for marker in cal.BHAV.glob("*/bhav_*.holiday"):
        try:
            days.add(datetime.strptime(marker.stem.split("_")[1], "%Y%m%d").date())
        except (IndexError, ValueError):
            continue
    return days


def expected_session(now: datetime) -> date:
    """Mirror of gap_reversal.next_session(), offline: today until 15:30, else next trading day."""
    hol = _holidays()
    trading = lambda d: d.weekday() < 5 and d not in hol            # noqa: E731
    day = now.date()
    if trading(day) and (now.hour, now.minute) < (15, 30):
        return day
    day += timedelta(days=1)
    while not trading(day):
        day += timedelta(days=1)
    return day


@st.cache_data(ttl=600, show_spinner="Loading champion model…")
def feature_importance(directory: str, mtime: float) -> tuple[pd.Series, str]:
    """Top-25 importances (% of total) and the method used; empty on failure."""
    try:
        from nse_intraday_ai.ranker import Ranker
        ranker = Ranker.load(Path(directory))
        imp = ranker.feature_importance()
        method = ("total split gain" if ranker.config.ranker_type == "hgb" else "split importance")
        imp = imp / imp.sum() * 100 if imp.sum() > 0 else imp
        return imp.head(25), method
    except Exception:                                    # noqa: BLE001
        return pd.Series(dtype=float), ""


# ── rendering helpers ───────────────────────────────────────────────────────
def download(df: pd.DataFrame, name: str) -> None:
    st.download_button("⬇ CSV", df.to_csv(index=False).encode(), file_name=f"{name}.csv",
                       mime="text/csv", key=f"dl_{name}", on_click="ignore", type="tertiary",
                       help="Download the data behind this view as CSV")


def chart(fig: go.Figure, data: pd.DataFrame, name: str, height: int = 380) -> None:
    fig.update_layout(template="nse_dark", height=height)
    st.plotly_chart(fig, theme=None, width="stretch", config=PLOT_CONFIG, key=f"fig_{name}")
    download(data, name)


def table(df: pd.DataFrame, name: str, **kw) -> None:
    st.dataframe(df, hide_index=True, width="stretch", **kw)
    download(df, name)


def _signed(values) -> list[str]:
    return [UP if v >= 0 else DOWN for v in values]


# ── Today ───────────────────────────────────────────────────────────────────
def _phase(now: datetime) -> str:
    m = now.hour * 60 + now.minute
    if now.weekday() >= 5 or now.date() in _holidays():
        return "🌙 Market closed today — the next list is published at 08:45 IST on the next session."
    for end, msg in ((540, "🕒 Before the open — list at 08:45; enter in the 09:00–09:07 pre-open auction."),
                     (555, "🟢 Pre-open — place the MIS short orders now (the open is the measured entry)."),
                     (570, "🛑 Market open — place a BUY SL-M at each fill + stop distance."),
                     (915, "⏳ Holding — nothing to do until 15:15 unless a stop fills."),
                     (930, "🔴 15:15 — cover all shorts now (broker auto-squares MIS at ~15:20).")):
        if m < end:
            return msg
    return "🌙 Market closed — paper result recorded at 15:40; next list at 08:45."


# ── Live panel (Today) ──────────────────────────────────────────────────────
def _live_enabled() -> bool:
    return os.environ.get("NSE_UI_LIVE", "1") != "0"


@st.cache_data(ttl=50, show_spinner=False)
def live_bars(symbols: tuple[str, ...], day: str) -> dict[str, pd.DataFrame]:
    """1-minute bars for `day` per Yahoo symbol (IST index); {} on any failure."""
    try:
        import yfinance as yf
        raw = yf.download(list(symbols), period="1d", interval="1m", group_by="ticker",
                          progress=False, threads=True, auto_adjust=False)
    except Exception:                                    # noqa: BLE001 — live view is optional
        return {}
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for sym in symbols:
        try:
            frame = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            frame = frame.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
            idx = frame.index.tz_convert(IST) if frame.index.tz else frame.index.tz_localize(IST)
            frame = frame.set_axis(idx)
            frame = frame[(frame.index.date.astype(str) == day)
                          & (frame.index.hour * 60 + frame.index.minute >= 555)]
            if not frame.empty:
                out[sym] = frame
        except Exception:                                # noqa: BLE001
            continue
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _bands(day: str) -> pd.DataFrame:
    try:
        from nse_intraday_ai import nse_bands
        return nse_bands.load(date.fromisoformat(day), fetch=False, log=lambda *_: None)
    except Exception:                                    # noqa: BLE001
        return pd.DataFrame(columns=["series", "band"])


def countdown(now: datetime, square_off: str = "15:15") -> tuple[str, str]:
    """(label, value) for the time left until the square-off."""
    hh, mm = (int(x) for x in square_off.split(":"))
    left = (hh * 60 + mm) * 60 - (now.hour * 3600 + now.minute * 60 + now.second)
    if left <= 0:
        return f"Cover by {square_off}", "COVER NOW" if left > -15 * 60 else "Closed"
    return f"Time to {square_off} cover", f"{left // 3600}h {left % 3600 // 60:02d}m"


def risk_colour(to_stop_pct: float | None) -> str:
    """Red under 1 % from the stop, amber under 2.5 %, else green."""
    if to_stop_pct is None or to_stop_pct != to_stop_pct:
        return "#8A92A6"
    return DOWN if to_stop_pct < 1 else "#FFB547" if to_stop_pct < 2.5 else UP


@st.fragment(run_every=60)
def live_panel(payload: dict, session: str) -> None:
    """Live short P&L of the names you traded; reruns itself every 60 s."""
    from nse_intraday_ai.costs import round_trip_cost
    picks = payload["picks"]
    traded = st.session_state.get("traded") or [str(p["symbol"]).removesuffix(".NS")
                                                 for p in picks if not p.get("reserve")]
    size = st.session_state.get("size_pct", 100) / 100
    chosen = [p for p in picks if str(p["symbol"]).removesuffix(".NS") in traded]
    bars = live_bars(tuple(sorted(str(p["symbol"]) for p in chosen)), session)
    now = datetime.now(IST)
    if not bars:
        st.info("Live prices not available yet (Yahoo 1-minute bars start a few minutes after 09:15).")
        return
    bands = _bands(session)
    square_off = payload.get("config", {}).get("square_off", "15:15")
    rows, paths, levels = [], {}, {}
    for p in chosen:
        sym = str(p["symbol"])
        b = bars.get(sym)
        if b is None:
            continue
        entry = float(p.get("entry") or b["open"].iloc[0])
        stop = float(p.get("stop_price") or entry + float(p["stop_distance"]))
        note = ""
        pick = G.Pick(**{k: v for k, v in p.items() if k in G.Pick.__dataclass_fields__})
        pick.stop_price = stop
        cap = G.circuit_capped_stop(pick, bands)
        if cap is not None:
            note = f"⚠ stop would be above the ₹{cap[1]:,.2f} upper circuit — set BUY SL-M at ₹{cap[0]:,.2f}"
            stop = cap[0]
        path, status, price = G.live_short_path(b, entry, stop, square_off)
        qty = int(int(p["quantity"]) * size)
        pnl_pct = float(path.iloc[-1])
        name = sym.removesuffix(".NS")
        paths[name] = path
        levels[name] = (sym, entry, stop, G.upper_circuit(pick, bands))
        gross = qty * entry * pnl_pct / 100
        cost = round_trip_cost(entry, price, qty).total if qty else 0.0
        rows.append({
            "": {"STOPPED": "🔴", "COVERED": "⚪"}.get(status, "🟢" if pnl_pct >= 0 else "🟠"),
            "Symbol": name, "Status": status, "Qty": qty, "Entry ₹": round(entry, 2),
            "Last ₹": round(price, 2), "P&L %": round(pnl_pct, 2),
            "P&L ₹": round(gross, 0), "Costs ₹": round(cost, 0), "Net ₹": round(gross - cost, 0),
            "BUY SL-M ₹": round(stop, 2),
            "To stop %": round((stop - price) / price * 100, 2) if status == "OPEN" else None,
            "Note": note})
    if not rows:
        st.info("No live bars for the selected names yet.")
        return
    df = pd.DataFrame(rows)
    net, costs = float(df["Net ₹"].sum()), float(df["Costs ₹"].sum())
    deployed = float((df["Qty"] * df["Entry ₹"]).sum())
    last_bar = max(b.index[-1] for b in bars.values())
    wide = pd.DataFrame(paths).ffill()
    weights = df.set_index("Symbol")["Qty"] * df.set_index("Symbol")["Entry ₹"]
    if weights.sum() <= 0:                                     # size 0 → equal weight for the chart
        weights = weights * 0 + 1
    basket = (wide * weights.reindex(wide.columns)).sum(axis=1) / weights.sum()
    label, left = countdown(now, square_off)
    c = st.columns(5)
    c[0].metric("Basket P&L after costs", f"₹{net:+,.0f}",
                f"{net / deployed * 100:+.2f}% of ₹{deployed:,.0f}" if deployed else None,
                help=f"Gross ₹{net + costs:+,.0f} minus ₹{costs:,.0f} brokerage, STT, exchange, "
                     "GST, stamp and slippage (Groww MIS).")
    c[1].metric("Winning / losing", f"{int((df['P&L %'] > 0).sum())} / {int((df['P&L %'] < 0).sum())}",
                f"stops hit {int((df['Status'] == 'STOPPED').sum())}", delta_color="off")
    c[2].metric("Best / worst today", f"{basket.max():+.2f}% / {basket.min():+.2f}%",
                f"at {basket.idxmax():%H:%M} / {basket.idxmin():%H:%M}", delta_color="off",
                help="Highest and lowest basket P&L (before costs) since the open.")
    c[3].metric(label, left)
    c[4].metric("Last price at", f"{last_bar:%H:%M}", f"refreshed {now:%H:%M:%S}", delta_color="off")
    if df["Note"].astype(bool).any():
        for _, r in df[df["Note"].astype(bool)].iterrows():
            st.error(f"**{r['Symbol']}**: {r['Note']}")
    table(df.drop(columns=[] if df["Note"].astype(bool).any() else ["Note"]),
          f"live_{session}", column_config={
              "P&L %": st.column_config.NumberColumn(format="%+.2f %%"),
              "P&L ₹": st.column_config.NumberColumn(format="₹%+.0f", help="Before costs"),
              "Costs ₹": st.column_config.NumberColumn(format="₹%.0f", help="Round-trip charges + slippage"),
              "Net ₹": st.column_config.NumberColumn(format="₹%+.0f", help="After costs"),
              "To stop %": st.column_config.NumberColumn(
                  format="%.2f %%", help="How far the price must rise to hit your stop")})
    left_col, right_col = st.columns([3, 2])
    with left_col:
        fig = go.Figure()
        for i, col in enumerate(wide.columns):
            fig.add_trace(go.Scatter(x=wide.index, y=wide[col], name=col, mode="lines",
                                     line=dict(width=1, color=PALETTE[(i + 1) % len(PALETTE)]),
                                     opacity=0.55, hovertemplate="%{y:+.2f}%"))
        fig.add_trace(go.Scatter(x=basket.index, y=basket, name="Basket", mode="lines",
                                 line=dict(width=3, color=UP if basket.iloc[-1] >= 0 else DOWN),
                                 hovertemplate="%{y:+.2f}%"))
        fig.add_hline(y=0, line=dict(color="#8A92A6", width=1, dash="dot"))
        fig.update_layout(title="Short P&L since the open (% of entry, + = profit)",
                          xaxis_title="Time (IST)", yaxis_title="P&L (%)", yaxis_ticksuffix="%")
        chart(fig, wide.assign(Basket=basket).reset_index(names="time"), f"live_path_{session}", height=380)
    with right_col:
        risk = df[df["Status"] == "OPEN"].sort_values("To stop %", ascending=False)
        fig = go.Figure(go.Bar(
            x=risk["To stop %"], y=risk["Symbol"], orientation="h",
            marker_color=[risk_colour(v) for v in risk["To stop %"]],
            text=[f"{v:.1f}%" for v in risk["To stop %"]], textposition="outside",
            customdata=risk[["Last ₹", "BUY SL-M ₹"]].to_numpy(),
            hovertemplate="%{y}: %{x:.2f}% to stop<br>last ₹%{customdata[0]:,.2f} → "
                          "stop ₹%{customdata[1]:,.2f}<extra></extra>"))
        fig.add_vline(x=1, line=dict(color=DOWN, width=1, dash="dot"))
        fig.update_layout(title="Distance to stop (open positions)", xaxis_title="Rise needed to hit stop (%)",
                          yaxis_title="", hovermode="closest", showlegend=False)
        chart(fig, risk[["Symbol", "Last ₹", "BUY SL-M ₹", "To stop %"]], f"live_risk_{session}", height=380)
    pick_name = st.selectbox("🔍 Chart one position (1-minute candles)", list(levels),
                             key="live_drill", index=0)
    sym, entry, stop, upper = levels[pick_name]
    b = bars[sym]
    fig = go.Figure(go.Candlestick(x=b.index, open=b["open"], high=b["high"], low=b["low"], close=b["close"],
                                   name=pick_name, increasing_line_color=UP, decreasing_line_color=DOWN))
    for y, text, colour, dash in ((entry, "entry", "#4EA8FF", "solid"), (stop, "stop", DOWN, "dash"),
                                  (upper, "upper circuit", "#FFB547", "dot")):
        if y:
            fig.add_hline(y=y, line=dict(color=colour, width=1.5, dash=dash),
                          annotation_text=f"{text} ₹{y:,.2f}", annotation_position="top left",
                          annotation_font_color=colour)
    fig.update_layout(title=f"{pick_name} — short from ₹{entry:,.2f}, stop ₹{stop:,.2f}",
                      xaxis_title="Time (IST)", yaxis_title="Price (₹)", yaxis_tickprefix="₹",
                      xaxis_rangeslider_visible=False, showlegend=False)
    chart(fig, b.reset_index(names="time"), f"live_candles_{session}_{pick_name}", height=360)
    st.caption("Prices: Yahoo Finance 1-minute bars (can lag NSE by 1–2 minutes). Stops use the "
               "official open unless your fill differs — your broker's order book is the truth.")


def page_today() -> None:
    now = datetime.now(IST)
    st.caption(_phase(now))
    payload = read_json(str(G.PICKS_PATH))
    if not payload or "picks" not in payload:
        st.info("No pick list yet. It is written every trading day at 08:45 IST by the "
                "`nse-gap-picks` timer (`python scripts/gap_reversal.py picks`).")
        return
    session, expected = str(payload.get("session", "?")), expected_session(now)
    state = read_json(str(MODELS / "meta_state.json")) or {}
    if session != expected.isoformat():
        st.warning(f"**Stale list.** This list is for **{session}**; the next trading session is "
                   f"**{expected:%a %d %b %Y}**. Its list is published at 08:45 IST that morning — "
                   "do not trade this one.")
    if payload.get("drift_alarm") or state.get("alarm"):
        st.error(f"**Drift alarm.** Live results have run persistently below the backtest "
                 f"(CUSUM {state.get('cusum', payload.get('cusum', 0)):.0f} / "
                 f"{state.get('threshold_bps', 400):.0f} bps).")
    if state.get("cautious_mode"):
        st.warning("**Cautious mode** — trade at 60% of the listed quantities until the "
                   f"{state.get('recovery_window', 10)}-session recovery check passes.")
    who = payload.get("ranked_by", "rule")
    label = {"ranker_open": "Open model (09:09)", "ranker": "ML ranker (08:45)"}.get(who, "Gap rule")
    gen = str(payload.get("generated_at", ""))
    c = st.columns(4)
    c[0].metric("Session", session)
    c[1].metric("Generated (IST)", gen[:16].replace("T", " ") or "—")
    c[2].metric("Ranked by", label, help="The ML ranker ranks unless its trailing 60-session record is "
                "significantly worse than the gap rule (t < −2); then the rule takes over.")
    c[3].metric("Based on closes of", payload.get("based_on", "—"))
    champ = read_json(str(MODELS / "champion.json"))
    if who == "rule" and champ and str(champ.get("promoted_at", "")) > gen:
        st.caption(f"ℹ️ An ML champion was promoted {str(champ['promoted_at'])[:16].replace('T', ' ')}, "
                   "after this list — the next list will be ranked by it.")
    if payload.get("final_at"):
        st.success(f"**Final list** — re-ranked at {str(payload['final_at'])[11:16]} on today's opening prices.")
    live = session == now.date().isoformat() and now.hour * 60 + now.minute >= 555 and _live_enabled()
    if live:
        st.subheader("📡 Live — your positions")
        names = [str(p.get("symbol", "")).removesuffix(".NS") for p in payload["picks"]]
        main = [n for n, p in zip(names, payload["picks"]) if not p.get("reserve")]
        c = st.columns([3, 1])
        c[0].multiselect("Names I traded", names, default=main, key="traded",
                         help="Swap in a reserve if you used one instead of a listed name.")
        c[1].select_slider("My size (% of listed qty)", [25, 50, 75, 100], value=100, key="size_pct")
        live_panel(payload, session)
        st.subheader("🗒 Morning list")
    rows = [{"Rank": p.get("rank"), "Symbol": str(p.get("symbol", "")).removesuffix(".NS"),
             "Role": "Reserve" if p.get("reserve") else "SHORT", "Qty": p.get("quantity"),
             "Yesterday gap %": p.get("gap_prev_pct"), "Prev close ₹": p.get("prev_close"),
             "Stop distance ₹": p.get("stop_distance"), "Stop %": p.get("stop_pct"),
             "Entry ₹": p.get("entry"), "BUY SL-M ₹": p.get("stop_price"),
             "Score": p.get("score"), "Turnover ₹ cr/day": p.get("turnover_cr")}
            for p in payload["picks"]]
    df = pd.DataFrame(rows).dropna(axis=1, how="all")
    table(df, f"picks_{session}", column_config={
        "Yesterday gap %": st.column_config.NumberColumn(format="%.2f %%"),
        "Stop %": st.column_config.NumberColumn(format="%.2f %%"),
        "Score": st.column_config.NumberColumn(format="%.4f")})
    if payload.get("excluded"):
        with st.expander(f"Excluded names ({len(payload['excluded'])})"):
            table(pd.DataFrame(payload["excluded"]), f"excluded_{session}")
    cfg = payload.get("config", {})
    with st.expander("How to trade the list"):
        st.markdown(
            f"1. **09:00–09:07** pre-open: SELL (MIS) at market each of the top {cfg.get('picks', 8)} "
            "SHORT names; if a name is blocked for shorting (ASM/T2T), take the next reserve in rank order.\n"
            f"2. **09:15+**: place a BUY SL-M at your fill + the stop distance ({cfg.get('stop_atr', 0.75)} × daily ATR).\n"
            f"3. **{cfg.get('square_off', '15:15')}**: buy to cover everything still open.\n\n"
            "Entering late (09:20 instead of the open) roughly halved the measured edge.")


# ── Performance ─────────────────────────────────────────────────────────────
def page_performance() -> None:
    book = read_csv(str(OUT / "paper_book.csv"))
    need = {"session", "symbol", "entry", "quantity", "exit_reason", "net"}
    if book is None or book.empty or not need <= set(book.columns):
        st.info("No forward paper record yet — the `nse-gap-record` timer appends each session at 15:40 IST.")
        return
    book = book.assign(exposure=book["entry"] * book["quantity"]).sort_values(["session", "symbol"])
    book["bps"] = book["net"] / book["exposure"] * 1e4
    backtest = read_json(str(OUT / "backtest.json")) or {}
    decade = backtest.get("decade") or {}
    exp_bps = decade.get("net_bps_per_trade")
    daily = book.groupby("session").agg(trades=("net", "size"), net=("net", "sum"),
                                         exposure=("exposure", "sum")).reset_index()
    daily["cumulative"] = daily["net"].cumsum()
    if exp_bps is not None:
        daily["backtest_pace"] = (daily["exposure"] * exp_bps / 1e4).cumsum()
    wins, losses = book[book["net"] > 0], book[book["net"] <= 0]
    c = st.columns(6)
    c[0].metric("Net P&L", f"₹{book['net'].sum():+,.0f}", f"{book['net'].sum() / CAPITAL * 100:+.2f}% of ₹{CAPITAL / 1e5:.0f}L")
    c[1].metric("Sessions / trades", f"{len(daily)} / {len(book)}")
    c[2].metric("Win rate", f"{len(wins) / len(book) * 100:.0f}%")
    c[3].metric("Stop rate", f"{(book['exit_reason'] == 'STOP').mean() * 100:.0f}%")
    c[4].metric("Avg win / loss", f"₹{wins['net'].mean() if len(wins) else 0:,.0f} / ₹{losses['net'].mean() if len(losses) else 0:,.0f}")
    c[5].metric("Net per trade", f"{book['bps'].mean():+.1f} bps",
                f"{book['bps'].mean() - exp_bps:+.1f} vs backtest" if exp_bps is not None else None)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.62, 0.38])
    fig.add_scatter(x=daily["session"], y=daily["cumulative"], name="Paper equity (cumulative net)",
                    mode="lines+markers", line=dict(width=2.5, color=UP), row=1, col=1)
    if "backtest_pace" in daily:
        fig.add_scatter(x=daily["session"], y=daily["backtest_pace"], name=f"Backtest pace ({exp_bps:+.1f} bps/trade)",
                        mode="lines", line=dict(dash="dash", color=PALETTE[1]), row=1, col=1)
    fig.add_bar(x=daily["session"], y=daily["net"], name="Daily net", marker_color=_signed(daily["net"]), row=2, col=1)
    fig.update_yaxes(title_text="Cumulative net (₹)", tickprefix="₹", row=1, col=1)
    fig.update_yaxes(title_text="Daily net (₹)", tickprefix="₹", row=2, col=1)
    fig.update_xaxes(type="category")
    fig.update_xaxes(title_text="Session date", row=2, col=1)
    fig.update_layout(title="Forward paper book — costs included, never seen by the backtest", bargap=0.35)
    chart(fig, daily, "paper_equity_daily", height=480)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**By exit reason**")
        g = book.groupby("exit_reason").agg(Trades=("net", "size"), WinRate=("net", lambda s: (s > 0).mean() * 100),
                                            Net=("net", "sum"), AvgBps=("bps", "mean")).reset_index()
        g.insert(2, "Share %", g["Trades"] / len(book) * 100)
        g = g.rename(columns={"exit_reason": "Exit reason", "WinRate": "Win rate %", "Net": "Net ₹",
                              "AvgBps": "Avg net bps"}).round(1)
        table(g, "paper_by_exit_reason")
    with right:
        years = pd.Series(decade.get("by_year") or {}, dtype=float)
        if years.empty:
            st.info("No saved backtest (`python scripts/gap_reversal.py backtest --decade --save`).")
        else:
            yr = pd.DataFrame({"year": years.index.astype(str), "net_bps_per_trade": years.values})
            fig = go.Figure(go.Bar(x=yr["year"], y=yr["net_bps_per_trade"], name="Backtest (daily bars)",
                                   marker_color=_signed(yr["net_bps_per_trade"])))
            fig.add_hline(y=book["bps"].mean(), line_dash="dash", line_color=PALETTE[2],
                          annotation_text=f"Paper {book['bps'].mean():+.1f} bps", annotation_position="top left")
            fig.update_layout(title=f"Backtest expectation by year ({decade.get('first', '?')} → {decade.get('last', '?')})",
                              yaxis_title="Net per trade (bps)", xaxis_title="Year", showlegend=False)
            chart(fig, yr, "backtest_bps_by_year", height=300)
    s = (backtest.get("window") or {}).get("summary")
    if s:
        st.caption(f"5-minute backtest {s['first']} → {s['last']}: {s['avg_net_bps']:+.1f} bps/trade net, "
                   f"win rate {s['win_rate_pct']:.0f}%, {s['stops_hit']}/{s['trades']} stops. Decade: "
                   f"{decade.get('net_bps_per_trade', 0):+.1f} bps/trade, t = {decade.get('t_stat')}.")
    st.markdown("**Trades**")
    table(book.sort_values(["session", "symbol"], ascending=[False, True]).round(2), "paper_trades")


# ── Model ───────────────────────────────────────────────────────────────────
def page_model() -> None:
    champ, report = read_json(str(MODELS / "champion.json")), read_json(str(MODELS / "weekly_report.json"))
    ev = (champ or {}).get("evidence") or report or {}
    if champ:
        st.markdown(f"**Champion** `{champ.get('directory')}` · promoted "
                    f"{str(champ.get('promoted_at', ''))[:16].replace('T', ' ')} · walk-forward window "
                    f"{' → '.join(ev.get('window', ['?', '?']))}")
    else:
        st.info("No ML champion promoted — the book ranks by the gap rule.")
    names = {"rule": "Gap rule", "ranker": "ML ranker (08:45)", "ranker_open": "Open model (09:09)",
             "ranker_liquidity_cost": "ML ranker, spread-based costs"}
    rows = [{"Model": lbl, "Net bps/trade": ev[k].get("net_bps_per_trade"), "t-stat": ev[k].get("t_stat"),
             "Sharpe (ann.)": ev[k].get("sharpe"), "Up days %": ev[k].get("up_day_pct"),
             "Max DD %": ev[k].get("max_drawdown_pct"), "Sessions": ev[k].get("sessions")}
            for k, lbl in names.items() if isinstance(ev.get(k), dict)]
    if rows:
        c = st.columns(3)
        for col, key, lbl in ((c[0], "vs_rule", "Ranker − rule"), (c[1], "open_vs_ranker", "Open − ranker")):
            if isinstance(ev.get(key), dict):
                col.metric(lbl, f"{ev[key]['mean_diff_bps']:+.1f} bps/day", f"t {ev[key]['t_stat']:+.2f}")
        verdicts = (report or {}).get("verdicts", {})
        c[2].metric("Weekly review", str((report or {}).get("at", "—"))[:16].replace("T", " "),
                    ", ".join(f"{k}: {'promoted' if v.get('promoted') else 'kept'}" for k, v in verdicts.items()) or None,
                    delta_color="off")
        left, right = st.columns([1, 1])
        with left:
            table(pd.DataFrame(rows), "model_evidence")
        with right:
            yr = pd.DataFrame({names[k]: ev[k].get("by_year", {}) for k in ("rule", "ranker", "ranker_open")
                               if isinstance(ev.get(k), dict)})
            yr.index = yr.index.astype(str)
            fig = go.Figure([go.Bar(x=yr.index, y=yr[col], name=col) for col in yr.columns])
            fig.update_layout(title="Walk-forward net bps/trade by year", barmode="group",
                              yaxis_title="Net per trade (bps)", xaxis_title="Year")
            chart(fig, yr.reset_index(names="year"), "model_bps_by_year", height=320)
    state = read_json(str(MODELS / "meta_state.json"))
    st.markdown("**Live monitoring** — paired expert record and CUSUM drift alarm")
    if not state:
        st.info("No meta state yet — written after the first recorded session.")
    else:
        from nse_intraday_ai.meta import MetaState
        ms = MetaState(**{k: v for k, v in state.items() if k in MetaState.__dataclass_fields__})
        expert, guard_t = ms.guard()
        c = st.columns(4)
        c[0].metric("CUSUM / threshold", f"{ms.cusum:.0f} / {ms.threshold_bps:.0f} bps",
                    "ALARM" if ms.alarm else "ok", delta_color="inverse" if ms.alarm else "normal")
        c[1].metric("Expected per trade", f"{ms.expected_bps:.1f} bps", f"slack {ms.slack_bps:.0f} bps", delta_color="off")
        c[2].metric("Guard selects", expert, f"t {guard_t:+.2f}" if guard_t is not None else
                    f"needs {ms.guard_window} sessions", delta_color="off")
        c[3].metric("Position size", f"{ms.position_size_multiplier:.0%}", "cautious" if ms.cautious_mode else "normal",
                    delta_color="off")
        track = pd.DataFrame(ms.track)
        hist = pd.DataFrame([{"session": h.get("session"), "cusum": h.get("cusum")} for h in ms.history])
        if not track.empty:
            track = track.sort_values("session").set_index("session")
            cum = track.fillna(0).cumsum().rename(columns=names)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.65, 0.35])
            for col in cum.columns:
                fig.add_scatter(x=cum.index, y=cum[col], name=col, mode="lines", row=1, col=1)
            if not hist.empty:
                fig.add_scatter(x=hist["session"], y=hist["cusum"], name="CUSUM", mode="lines+markers",
                                line=dict(color=PALETTE[2]), row=2, col=1)
            fig.add_hline(y=ms.threshold_bps, line_dash="dash", line_color=DOWN, row=2, col=1,
                          annotation_text="alarm threshold", annotation_position="top left")
            fig.update_yaxes(title_text="Cumulative net (bps/trade)", row=1, col=1)
            fig.update_yaxes(title_text="CUSUM (bps)", rangemode="tozero", row=2, col=1)
            fig.update_xaxes(title_text="Session date", row=2, col=1)
            fig.update_layout(title="Each expert's own top-8 book, net of 13 bps cost")
            data = cum.join(hist.set_index("session"), how="outer").reset_index(names="session")
            chart(fig, data, "expert_track_cusum", height=460)
    st.markdown("**Champion feature importances (top 25)**")
    directory = MODELS / str((champ or {}).get("directory", ""))
    if not champ or not (directory / "meta.json").exists():
        st.info("No champion model artifacts to inspect.")
        return
    imp, method = feature_importance(str(directory), (directory / "meta.json").stat().st_mtime)
    if imp.empty:
        st.info("Feature importances are unavailable for this model.")
        return
    fi = imp.iloc[::-1].reset_index()
    fi.columns = ["feature", "importance_pct"]
    fig = go.Figure(go.Bar(x=fi["importance_pct"], y=fi["feature"], orientation="h", marker_color=PALETTE[1],
                           hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.update_layout(title=f"Share of {method} (%)", xaxis_title="Importance (% of total)",
                      yaxis_title=None, hovermode="closest", showlegend=False)
    chart(fig, fi.iloc[::-1], "champion_feature_importance", height=620)
