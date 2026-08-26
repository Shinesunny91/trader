"""Market-wide context: index regime, India VIX, sector clustering, breadth.

Fetched once per scan cycle. Passed into the signal engine so individual stock
signals are boosted or suppressed based on what the whole market is doing.
"""
from __future__ import annotations

import os
import warnings
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd


def _ablated(name: str) -> bool:
    """Experiment switch: NSE_AI_ABLATE=tnx,hsi,... disables single context terms.

    Used by the backtest harness to attribute performance to individual
    inputs; unset in normal operation.
    """
    return name in os.environ.get("NSE_AI_ABLATE", "").split(",")

# ── Sector membership map ─────────────────────────────────────────────────────
_SECTOR_MAP: dict[str, list[str]] = {
    "Banking":       ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS",
                      "INDUSINDBK.NS","BANKBARODA.NS","FEDERALBNK.NS","IDFCFIRSTB.NS","PNB.NS"],
    "IT":            ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS",
                      "LTIM.NS","PERSISTENT.NS","MPHASIS.NS","COFORGE.NS","LTTS.NS"],
    "Energy":        ["RELIANCE.NS","ONGC.NS","COALINDIA.NS","BPCL.NS","IOC.NS",
                      "GAIL.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIENT.NS"],
    "Auto":          ["MARUTI.NS","TATAMOTORS.NS","M&M.NS","BAJAJ-AUTO.NS",
                      "EICHERMOT.NS","HEROMOTOCO.NS","TVSMOTOR.NS"],
    "Pharma":        ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS",
                      "APOLLOHOSP.NS","TORNTPHARM.NS"],
    "Finance":       ["BAJFINANCE.NS","BAJAJFINSV.NS","SBILIFE.NS","HDFCLIFE.NS",
                      "ICICIGI.NS","CHOLAFIN.NS"],
    "Metals":        ["TATASTEEL.NS","HINDALCO.NS","JSWSTEEL.NS","VEDL.NS","SAIL.NS","NMDC.NS"],
    "FMCG":          ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS",
                      "DABUR.NS","MARICO.NS","GODREJCP.NS"],
    "Cement":        ["ULTRACEMCO.NS","SHREECEM.NS","AMBUJACEM.NS","ACC.NS","DALMABHARAT.NS"],
    "Power":         ["NTPC.NS","POWERGRID.NS","ADANIENSOL.NS","TATAPOWER.NS"],
    "Capital Goods": ["LT.NS","SIEMENS.NS","ABB.NS","BHEL.NS","HAVELLS.NS"],
    "Telecom":       ["BHARTIARTL.NS"],
}

# symbol → sector (reverse lookup)
SYMBOL_TO_SECTOR: dict[str, str] = {
    sym: sector for sector, syms in _SECTOR_MAP.items() for sym in syms
}

# Sector → tradable NSE sector index on Yahoo.  Regimes computed from the
# actual sector index are a much stronger read than majority-voting the few
# scanned peers (which is kept as a fallback).
SECTOR_TO_INDEX: dict[str, str] = {
    "Banking":       "^NSEBANK",
    "IT":            "^CNXIT",
    "Auto":          "^CNXAUTO",
    "Pharma":        "^CNXPHARMA",
    "FMCG":          "^CNXFMCG",
    "Metals":        "^CNXMETAL",
    "Energy":        "^CNXENERGY",
    "Finance":       "^NSEBANK",
}

# Commodity futures → macro group, used for group-specific context adjustments.
COMMODITY_GROUP: dict[str, str] = {
    "GC=F": "precious", "SI=F": "precious", "PL=F": "precious", "PA=F": "precious",
    "HG=F": "industrial",
    "CL=F": "energy", "BZ=F": "energy", "RB=F": "energy", "HO=F": "energy", "NG=F": "energy",
    "ZC=F": "agri", "ZS=F": "agri", "ZW=F": "agri", "CT=F": "agri", "SB=F": "agri",
}


def is_commodity_symbol(symbol: str) -> bool:
    return "=" in symbol

# Strategies that work best in low-vol (mean reversion) vs high-vol (breakouts)
_MEAN_REV_STRATEGIES  = {"vwap_mean_reversion", "rsi_divergence", "vwap_bounce_scalp"}
_BREAKOUT_STRATEGIES  = {"opening_range_breakout", "volatility_compression_breakout",
                         "bb_kc_squeeze", "momentum_burst_scalp"}


@dataclass
class MarketContext:
    # ── NIFTY 50 index ───────────────────────────────────────────────────────
    index_regime: str = "UNKNOWN"   # TRENDING_UP | TRENDING_DOWN | RANGING | HIGH_VOL
    index_long_adj: float = 0.0     # confidence pts added to LONG signals
    index_short_adj: float = 0.0    # confidence pts added to SHORT signals
    # NIFTY's own recent %-change.  Unlike the regime label this is continuous,
    # and it was the single strongest surviving macro predictor in the Aug-2026
    # study (out-of-sample decile spread +10.9 bps) — the gate needs the number,
    # not the bucket.
    nifty_change_pct: float | None = None

    # ── India VIX ────────────────────────────────────────────────────────────
    vix_value: float | None = None
    vix_level: str = "NORMAL"       # LOW | NORMAL | ELEVATED | HIGH

    # ── Market breadth (computed from scan results) ──────────────────────────
    breadth_up_pct: float = 0.0
    breadth_signal: str = "NEUTRAL" # BULL_DAY | BEAR_DAY | NEUTRAL
    breadth_long_adj: float = 0.0
    breadth_short_adj: float = 0.0

    # ── Sector regimes (computed from scan results) ──────────────────────────
    sector_regimes: dict[str, str] = field(default_factory=dict)

    # ── Sector index regimes (from actual NSE sector indices) ────────────────
    sector_index_regimes: dict[str, str] = field(default_factory=dict)

    # ── Global cues ───────────────────────────────────────────────────────────
    # Risk appetite in [-1, +1]: blend of S&P futures, Asia and Europe sessions.
    global_risk: float = 0.0
    # Recent %-change of the US dollar index (DXY) — inverse driver for metals.
    dxy_change_pct: float | None = None
    # Recent %-change of USDINR — INR weakness pressures Indian equities, but is
    # a direct price tailwind for INR-denominated (MCX) commodity contracts,
    # which track USD price × USDINR.
    usdinr_change_pct: float | None = None
    # Recent %-change of the US 10-year yield (^TNX) — rates up hurts gold.
    us10y_change_pct: float | None = None
    # Recent %-change of Hang Seng — China demand proxy for industrial metals.
    hsi_change_pct: float | None = None
    # US VIX level — global uncertainty gauge (India VIX only sees NSE hours).
    us_vix_value: float | None = None
    # Fraction of the scanned NSE universe trading above session VWAP —
    # a sharper intraday breadth read than regime counting.  Computed from
    # the batch frames before signals are evaluated (same-cycle, no lag).
    vwap_breadth_pct: float | None = None
    # Recent %-change of front-month crude — India imports ~85% of its oil, so
    # a crude spike is an equity headwind (and an OMC/paint/airline shock).
    crude_change_pct: float | None = None
    # Populated when the context fetch failed, so a silent outage is visible
    # instead of degrading every signal without a trace.
    fetch_error: str | None = None

    # ─────────────────────────────────────────────────────────────────────────

    def vix_strategy_adj(self, strategy_name: str) -> float:
        """Per-strategy confidence modifier based on India VIX level."""
        if self.vix_level == "LOW":
            if strategy_name in _MEAN_REV_STRATEGIES:  return +5.0
            if strategy_name in _BREAKOUT_STRATEGIES:  return -5.0
        elif self.vix_level == "ELEVATED":
            if strategy_name in _MEAN_REV_STRATEGIES:  return -8.0
            if strategy_name in _BREAKOUT_STRATEGIES:  return +6.0
        elif self.vix_level == "HIGH":
            if strategy_name in _MEAN_REV_STRATEGIES:  return -15.0
            if strategy_name in _BREAKOUT_STRATEGIES:  return +10.0
        return 0.0

    def sector_adj(self, symbol: str, side_str: str) -> float:
        """Confidence boost from sector clustering (peers trending same way = reinforcement)."""
        sector = SYMBOL_TO_SECTOR.get(symbol)
        if not sector:
            return 0.0
        regime = self.sector_regimes.get(sector, "")
        if side_str == "LONG":
            if regime == "TRENDING_UP":   return +6.0
            if regime == "TRENDING_DOWN": return -6.0
        elif side_str == "SHORT":
            if regime == "TRENDING_DOWN": return +6.0
            if regime == "TRENDING_UP":   return -6.0
        return 0.0

    def index_adj(self, side_str: str) -> float:
        return self.index_long_adj if side_str == "LONG" else self.index_short_adj

    def extra_symbol_adj(self, symbol: str, side_str: str) -> float:
        """Symbol-aware context adjustment applied inside the signal engine.

        Stocks: sector-index alignment + global risk appetite + INR pressure.
        Commodities: group-specific macro drivers (DXY for metals, risk
        appetite for energy/industrial) — the NIFTY index adjustment does not
        apply to them (see engine).
        """
        if side_str not in ("LONG", "SHORT"):
            return 0.0
        sign = 1.0 if side_str == "LONG" else -1.0
        adj = 0.0

        if is_commodity_symbol(symbol):
            group = COMMODITY_GROUP.get(symbol, "")
            if self.dxy_change_pct is not None and group in ("precious", "industrial"):
                # Dollar down → metals up.  ±0.15% recent DXY move ≈ ±5 pts.
                adj += sign * max(-6.0, min(6.0, -self.dxy_change_pct * 33.0))
            if group in ("energy", "industrial"):
                # Risk-on supports growth-linked commodities.
                adj += sign * self.global_risk * 5.0
            if group == "precious":
                # Risk-off supports gold/silver (safe haven), mildly.
                # NOTE: an intraday US-10Y momentum term was tried here and
                # removed — 1-hour yield "momentum" is noise (Jul-2026 ablation:
                # it single-handedly flipped the strict-gate subset from +381
                # to −56).  us10y_change_pct stays available for display only.
                adj += sign * -self.global_risk * 3.0
            if group == "industrial" and self.hsi_change_pct is not None and not _ablated("hsi"):
                # China demand proxy: Hang Seng momentum aligns copper.
                adj += sign * max(-3.0, min(3.0, self.hsi_change_pct / 0.8 * 3.0))
            if self.usdinr_change_pct is not None and not _ablated("inr"):
                # MCX conversion tailwind: INR contracts = USD price × USDINR,
                # so a weakening INR lifts every INR-denominated commodity.
                adj += sign * max(-3.0, min(3.0, self.usdinr_change_pct * 20.0))
            return adj

        # ── Stocks ──
        sector = SYMBOL_TO_SECTOR.get(symbol)
        if sector:
            regime = self.sector_index_regimes.get(sector) or self.sector_regimes.get(sector, "")
            if regime == "TRENDING_UP":
                adj += sign * 5.0
            elif regime == "TRENDING_DOWN":
                adj += sign * -5.0
        adj += sign * self.global_risk * 4.0
        if self.usdinr_change_pct is not None:
            # INR weakening (USDINR up) is a mild equity headwind (IT gains excluded
            # for simplicity — kept conservative at ±2 pts).
            adj += sign * max(-2.0, min(2.0, -self.usdinr_change_pct * 10.0))
        if self.us_vix_value is not None and self.us_vix_value >= 25:
            # Elevated global fear: penalise both sides — whipsaw risk, and
            # India VIX only covers NSE hours so it can miss overnight stress.
            adj -= 2.0
        if self.vwap_breadth_pct is not None:
            # One-sided tape: penalty-only, applied to trades *fighting* the
            # tape.  Boosting with-tape trades was tried and rejected — the
            # boost pushed marginal signals over the confidence gate (the
            # same failure mode as every additive boost tested).
            if self.vwap_breadth_pct >= 0.65 and side_str == "SHORT":
                adj -= 3.0
            elif self.vwap_breadth_pct <= 0.35 and side_str == "LONG":
                adj -= 3.0
        return adj

    def total_postscan_adj(self, symbol: str, side_str: str) -> float:
        """Total post-scan adjustment (breadth + sector) for ranking / near-miss display."""
        if side_str == "LONG":
            return self.breadth_long_adj + self.sector_adj(symbol, "LONG")
        if side_str == "SHORT":
            return self.breadth_short_adj + self.sector_adj(symbol, "SHORT")
        return 0.0

    def summary(self) -> str:
        vix_str = f"VIX={self.vix_value:.1f}({self.vix_level})" if self.vix_value else "VIX=N/A"
        dxy_str = f"DXY={self.dxy_change_pct:+.2f}%" if self.dxy_change_pct is not None else "DXY=N/A"
        return (
            f"Index={self.index_regime}({self.index_long_adj:+.0f}L/{self.index_short_adj:+.0f}S)  "
            f"{vix_str}  Breadth={self.breadth_signal}({self.breadth_up_pct:.0%})  "
            f"Risk={self.global_risk:+.2f}  {dxy_str}  "
            f"Sectors={sum(1 for r in self.sector_index_regimes.values() if r)}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_vix(vix: float) -> str:
    if vix < 12:  return "LOW"
    if vix < 18:  return "NORMAL"
    if vix < 25:  return "ELEVATED"
    return "HIGH"


def _index_adjs(regime: str) -> tuple[float, float]:
    """Returns (long_adj, short_adj) for index regime."""
    return {
        "TRENDING_UP":   (+6.0, -8.0),
        "TRENDING_DOWN": (-8.0, +6.0),
        "RANGING":       ( 0.0,  0.0),
        "HIGH_VOL":      (-4.0, -4.0),
    }.get(regime, (0.0, 0.0))


def _compute_breadth(scan_results: list) -> tuple[float, str, float, float]:
    """Returns (up_pct, signal, long_adj, short_adj)."""
    up   = sum(1 for r in scan_results if getattr(r, "regime", None) == "TRENDING_UP")
    down = sum(1 for r in scan_results if getattr(r, "regime", None) == "TRENDING_DOWN")
    n    = len(scan_results)
    if n == 0:
        return 0.0, "NEUTRAL", 0.0, 0.0
    up_pct   = up / n
    down_pct = down / n
    if up_pct > 0.60:
        return up_pct, "BULL_DAY", +5.0, -5.0
    if down_pct > 0.60:
        return up_pct, "BEAR_DAY", -5.0, +5.0
    return up_pct, "NEUTRAL", 0.0, 0.0


def _compute_sector_regimes(scan_results: list) -> dict[str, str]:
    """Majority-vote regime per sector from scan results (need ≥2 stocks, >50% agreement)."""
    votes: dict[str, Counter] = {}
    for r in scan_results:
        sector = SYMBOL_TO_SECTOR.get(r.symbol)
        regime = getattr(r, "regime", None)
        if sector and regime:
            votes.setdefault(sector, Counter())[regime] += 1
    result: dict[str, str] = {}
    for sector, counts in votes.items():
        dominant, cnt = counts.most_common(1)[0]
        total = sum(counts.values())
        if total >= 2 and cnt / total > 0.50:
            result[sector] = dominant
    return result


def _last_close(frame) -> float | None:
    try:
        df = frame.dropna()
        if df.empty:
            return None
        close_col = "Close" if "Close" in df.columns else "close"
        return float(df[close_col].iloc[-1])
    except Exception:
        return None


def _recent_pct_change(frame, bars: int = 12) -> float | None:
    """Percent change of close over the last `bars` rows (5m bars → ~1h)."""
    try:
        df = frame.dropna()
        close_col = "Close" if "Close" in df.columns else "close"
        closes = df[close_col]
        if len(closes) < 2:
            return None
        ref = closes.iloc[-min(bars, len(closes))]
        if not ref:
            return None
        return float((closes.iloc[-1] / ref - 1.0) * 100)
    except Exception:
        return None


def _regime_of(frame) -> str | None:
    """Compute the indicator-based regime of a raw yfinance OHLCV frame."""
    try:
        df = frame.dropna()
        df.columns = [str(c).lower() for c in df.columns]
        if len(df) < 20:
            return None
        from nse_intraday_ai.indicators import add_indicators
        return str(add_indicators(df.copy()).iloc[-1].get("regime", "RANGING"))
    except Exception:
        return None


def global_risk_score(
    es_change_pct: float | None,
    asia_change_pct: float | None = None,
    europe_change_pct: float | None = None,
) -> float:
    """Blend global cues into a risk-appetite score in [-1, +1].

    ±0.5% on S&P futures saturates its component; Asia (Nikkei/Hang Seng
    blend) and Europe (DAX) carry smaller weights and only contribute while
    their sessions are fresh (staleness is handled by the caller).
    """
    if _ablated("eu"):
        europe_change_pct = None
    es_weight, asia_weight = (0.7, 0.3) if europe_change_pct is None else (0.55, 0.25)
    score = 0.0
    if es_change_pct is not None:
        score += max(-1.0, min(1.0, es_change_pct / 0.5)) * es_weight
    if asia_change_pct is not None:
        score += max(-1.0, min(1.0, asia_change_pct / 0.8)) * asia_weight
    if europe_change_pct is not None:
        score += max(-1.0, min(1.0, europe_change_pct / 0.6)) * 0.20
    return max(-1.0, min(1.0, score))


# ── Public API ────────────────────────────────────────────────────────────────

# Global cues: US futures, dollar, INR, Asia, Europe, US rates, fear gauge and
# crude (India imports ~85% of its oil, so crude is an equity driver too).
_GLOBAL_TICKERS = [
    "ES=F", "DX-Y.NYB", "USDINR=X", "^N225", "^HSI", "^GDAXI", "^TNX", "^VIX", "CL=F",
]
_NSE_TICKERS = [
    "^NSEI", "^INDIAVIX",
    # NSE sector indices
    "^NSEBANK", "^CNXIT", "^CNXAUTO", "^CNXPHARMA", "^CNXFMCG", "^CNXMETAL", "^CNXENERGY",
]
_CONTEXT_TICKERS = [*_NSE_TICKERS, *_GLOBAL_TICKERS]

_INDEX_TO_SECTOR = {index: sector for sector, index in SECTOR_TO_INDEX.items()}


def fetch_index_vix_context(*, for_commodities: bool = False) -> MarketContext:
    """Fetch NIFTY 50, India VIX, sector indices and global cues.

    Call this BEFORE the batch scan so the adjustments affect individual
    signal decisions.  Breadth and peer-sector regimes are filled later by
    apply_postscan_context() once all scan results are in.

    `for_commodities` skips the NSE-only tickers, which say nothing about gold
    or crude and only slow the fetch down.  The parameter is not optional
    sugar: `scan_service` has always passed it, and because it did not exist
    the resulting TypeError was swallowed by a bare `except Exception` — every
    live scan from 2026-07-07 onward ran with **no market context at all**
    while the README described a fully wired context panel.  The candle cache
    shows the outage precisely: every equity context symbol stops on that date.
    """
    import yfinance as yf
    ctx = MarketContext()
    tickers = _GLOBAL_TICKERS if for_commodities else _CONTEXT_TICKERS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers,
                period="2d", interval="5m",
                group_by="ticker", auto_adjust=True, progress=False,
            )

        def frame_of(ticker):
            try:
                return raw[ticker]
            except Exception:
                return None

        # Index regime
        nsei = frame_of("^NSEI")
        if nsei is not None:
            regime = _regime_of(nsei)
            if regime:
                ctx.index_regime = regime
                ctx.index_long_adj, ctx.index_short_adj = _index_adjs(regime)
            ctx.nifty_change_pct = _recent_pct_change(nsei)

        # India VIX
        vix_df = frame_of("^INDIAVIX")
        if vix_df is not None:
            vix = _last_close(vix_df)
            if vix is not None:
                ctx.vix_value = vix
                ctx.vix_level = _classify_vix(vix)

        # Sector index regimes
        for index_symbol, sector in _INDEX_TO_SECTOR.items():
            fr = frame_of(index_symbol)
            if fr is None:
                continue
            regime = _regime_of(fr)
            if regime:
                ctx.sector_index_regimes[sector] = regime

        # Global cues
        es_chg = _recent_pct_change(frame_of("ES=F"))
        n225_chg = _recent_pct_change(frame_of("^N225"))
        hsi_chg = _recent_pct_change(frame_of("^HSI"))
        europe_chg = _recent_pct_change(frame_of("^GDAXI"))
        # Session-phase weighting: a Nikkei "1-hour change" read at 14:00 IST
        # describes a market that shut 2.5 hours earlier.  Weight each region
        # by whether it is actually trading right now.
        from nse_intraday_ai.macro_context import phase_weighted_global_risk

        ctx.global_risk = phase_weighted_global_risk(
            pd.Timestamp.now(tz="Asia/Kolkata"),
            es_change_pct=es_chg,
            japan_change_pct=n225_chg,
            china_change_pct=hsi_chg,
            europe_change_pct=europe_chg,
        )
        ctx.hsi_change_pct = hsi_chg
        ctx.dxy_change_pct = _recent_pct_change(frame_of("DX-Y.NYB"))
        ctx.usdinr_change_pct = _recent_pct_change(frame_of("USDINR=X"))
        ctx.us10y_change_pct = _recent_pct_change(frame_of("^TNX"))
        ctx.crude_change_pct = _recent_pct_change(frame_of("CL=F"))
        us_vix = frame_of("^VIX")
        if us_vix is not None:
            ctx.us_vix_value = _last_close(us_vix)
    except Exception as exc:
        # Never let a context failure silently disable context scoring again —
        # that is exactly how the 2026-07-07 outage went unnoticed for weeks.
        ctx.fetch_error = f"{type(exc).__name__}: {exc}"
    return ctx


def apply_postscan_context(ctx: MarketContext, scan_results: list) -> MarketContext:
    """Fill breadth + sector fields from completed scan results.
    Returns the same object (mutated in place) for convenience.
    """
    up_pct, signal, l_adj, s_adj = _compute_breadth(scan_results)
    ctx.breadth_up_pct    = up_pct
    ctx.breadth_signal    = signal
    ctx.breadth_long_adj  = l_adj
    ctx.breadth_short_adj = s_adj
    ctx.sector_regimes    = _compute_sector_regimes(scan_results)
    return ctx
