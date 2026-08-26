from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from nse_intraday_ai.indicators import normalize_ohlcv

IST = ZoneInfo("Asia/Kolkata")
NIFTY500_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
NIFTY100_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
NIFTY50_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
NSE_QUOTE_URL = "https://www.nseindia.com/api/quote-equity"
NSE_INDEX_URL = "https://www.nseindia.com/api/equity-stockIndices"
GOOGLE_FINANCE_QUOTE_URL = "https://www.google.com/finance/quote/{symbol}:NSE"


DEFAULT_SYMBOLS = [
    "NIFTYBEES.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
    "LT.NS",
    "AXISBANK.NS",
    "ITC.NS",
]


DEFAULT_COMMODITY_SYMBOLS = [
    "GC=F",
    "SI=F",
    "CL=F",
    "BZ=F",
    "RB=F",
    "HO=F",
    "NG=F",
    "HG=F",
    "PL=F",
    "PA=F",
    "ZC=F",
    "ZS=F",
    "ZW=F",
    "CT=F",
    "SB=F",
]


def to_yahoo_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    if (
        clean.startswith("^")
        or "=" in clean
        or clean.endswith(".NS")
        or clean.endswith(".BO")
    ):
        return clean
    return f"{clean}.NS"


def _unique_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for symbol in symbols:
        yahoo_symbol = to_yahoo_symbol(symbol)
        if yahoo_symbol not in seen:
            seen.add(yahoo_symbol)
            unique.append(yahoo_symbol)
    return unique


@dataclass(frozen=True)
class DataResult:
    symbol: str
    frame: pd.DataFrame
    source: str
    warning: str | None = None


@dataclass(frozen=True)
class UniverseResult:
    symbols: list[str]
    source: str
    warning: str | None = None


@dataclass(frozen=True)
class PublicQuoteResult:
    symbol: str
    last_price: float | None
    last_update_time: str | None
    age_seconds: float | None
    source: str
    warning: str | None = None


def _parse_nse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)


def _parse_google_time(value: str | None) -> datetime | None:
    if not value:
        return None
    clean = unescape(value).replace("\u202f", " ").replace("\xa0", " ")
    match = re.search(
        r"([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{1,2}:\d{2}:\d{2})\s+(AM|PM)",
        clean,
    )
    if not match:
        return None
    year = datetime.now(IST).year
    parsed = datetime.strptime(
        f"{year} {match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}",
        "%Y %b %d %I:%M:%S %p",
    )
    return parsed.replace(tzinfo=IST)


def _parse_index_csv(raw_csv: str, *, min_count: int, index_name: str) -> list[str]:
    frame = pd.read_csv(StringIO(raw_csv))
    if "Symbol" not in frame.columns:
        raise ValueError(f"{index_name} CSV does not contain a Symbol column.")
    symbols = _unique_symbols(frame["Symbol"].dropna().astype(str).str.strip().tolist())
    if len(symbols) < min_count:
        raise ValueError(f"{index_name} source returned only {len(symbols)} symbols.")
    return symbols


def _parse_nifty500_csv(raw_csv: str) -> list[str]:
    return _parse_index_csv(raw_csv, min_count=400, index_name="NIFTY 500")


def _parse_nifty100_csv(raw_csv: str) -> list[str]:
    return _parse_index_csv(raw_csv, min_count=90, index_name="NIFTY 100")


def _parse_nifty50_csv(raw_csv: str) -> list[str]:
    return _parse_index_csv(raw_csv, min_count=45, index_name="NIFTY 50")


def _load_index_symbols(
    *,
    url: str,
    cache_path: Path | str | None,
    parser,
    fallback_symbols: list[str],
    name: str,
) -> UniverseResult:
    cache = Path(cache_path) if cache_path else None
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.niftyindices.com/",
    }
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        symbols = parser(response.text)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(response.text, encoding="utf-8")
        return UniverseResult(symbols, url, None)
    except Exception as exc:
        if cache and cache.exists():
            symbols = parser(cache.read_text(encoding="utf-8"))
            return UniverseResult(
                symbols,
                f"Cached {name} list: {cache}",
                f"Using cached {name} list because the official CSV could not be fetched: {exc}",
            )
        return UniverseResult(
            fallback_symbols,
            "Built-in fallback symbols",
            f"Could not fetch {name} list; using fallback symbols only: {exc}",
        )


def load_nifty500_symbols(cache_path: Path | str | None = None) -> UniverseResult:
    """Fetch the current official NIFTY 500 universe, with a local cache fallback."""
    return _load_index_symbols(
        url=NIFTY500_CSV_URL,
        cache_path=cache_path,
        parser=_parse_nifty500_csv,
        fallback_symbols=DEFAULT_SYMBOLS,
        name="NIFTY 500",
    )


def load_nifty100_symbols(cache_path: Path | str | None = None) -> UniverseResult:
    """Fetch the current official NIFTY 100 universe, with a local cache fallback."""
    return _load_index_symbols(
        url=NIFTY100_CSV_URL,
        cache_path=cache_path,
        parser=_parse_nifty100_csv,
        fallback_symbols=DEFAULT_SYMBOLS,
        name="NIFTY 100",
    )


def load_nifty50_symbols(cache_path: Path | str | None = None) -> UniverseResult:
    """Fetch the current official NIFTY 50 universe, with a local cache fallback."""
    return _load_index_symbols(
        url=NIFTY50_CSV_URL,
        cache_path=cache_path,
        parser=_parse_nifty50_csv,
        fallback_symbols=DEFAULT_SYMBOLS,
        name="NIFTY 50",
    )


def load_commodity_symbols() -> UniverseResult:
    """Return a built-in Yahoo Finance commodity futures research universe."""
    return UniverseResult(
        symbols=DEFAULT_COMMODITY_SYMBOLS,
        source="Built-in Yahoo Finance commodity futures universe",
        warning=None,
    )


def to_nse_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    if clean.endswith(".NS") or clean.endswith(".BO"):
        clean = clean.rsplit(".", 1)[0]
    return clean


class NsePublicQuoteClient:
    name = "NSE public quote endpoint"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/",
            }
        )

    def quote(self, symbol: str) -> PublicQuoteResult:
        nse_symbol = to_nse_symbol(symbol)
        try:
            response = self.session.get(NSE_QUOTE_URL, params={"symbol": nse_symbol}, timeout=10)
            response.raise_for_status()
            data = response.json()
            price_info = data.get("priceInfo", {})
            metadata = data.get("metadata", {})
            last_price = price_info.get("lastPrice")
            last_update = metadata.get("lastUpdateTime")
            age_seconds = None
            if last_update:
                quote_time = _parse_nse_time(last_update)
                age_seconds = (datetime.now(IST) - quote_time).total_seconds()
            return PublicQuoteResult(
                symbol=to_yahoo_symbol(nse_symbol),
                last_price=float(last_price) if last_price is not None else None,
                last_update_time=last_update,
                age_seconds=age_seconds,
                source=self.name,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            return PublicQuoteResult(
                symbol=to_yahoo_symbol(nse_symbol),
                last_price=None,
                last_update_time=None,
                age_seconds=None,
                source=self.name,
                warning=str(exc),
            )

    def index_quotes(
        self, symbols: list[str], index_name: str = "NIFTY 50"
    ) -> dict[str, PublicQuoteResult]:
        wanted = {to_yahoo_symbol(symbol) for symbol in symbols}
        try:
            response = self.session.get(
                NSE_INDEX_URL,
                params={"index": index_name},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            timestamp = data.get("timestamp")
            quote_time = _parse_nse_time(timestamp)
            age_seconds = (
                (datetime.now(IST) - quote_time).total_seconds() if quote_time is not None else None
            )
            quotes: dict[str, PublicQuoteResult] = {}
            for item in data.get("data", []):
                symbol = item.get("symbol")
                if not symbol or symbol == index_name:
                    continue
                yahoo_symbol = to_yahoo_symbol(symbol)
                if yahoo_symbol not in wanted:
                    continue
                last_price = item.get("lastPrice")
                quotes[yahoo_symbol] = PublicQuoteResult(
                    symbol=yahoo_symbol,
                    last_price=float(last_price) if last_price is not None else None,
                    last_update_time=timestamp,
                    age_seconds=age_seconds,
                    source=f"{self.name}: {index_name} snapshot",
                )
            return quotes
        except Exception as exc:  # pragma: no cover - network dependent
            return {
                to_yahoo_symbol(symbol): PublicQuoteResult(
                    symbol=to_yahoo_symbol(symbol),
                    last_price=None,
                    last_update_time=None,
                    age_seconds=None,
                    source=f"{self.name}: {index_name} snapshot",
                    warning=str(exc),
                )
                for symbol in symbols
            }


class GoogleFinanceQuoteClient:
    name = "Google Finance quote page"
    timeout_seconds = 4

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    def quote(self, symbol: str) -> PublicQuoteResult:
        nse_symbol = to_nse_symbol(symbol)
        yahoo_symbol = to_yahoo_symbol(nse_symbol)
        try:
            response = self.session.get(
                GOOGLE_FINANCE_QUOTE_URL.format(symbol=nse_symbol),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            html = response.text
            price_match = re.search(r"₹\s*([0-9,]+(?:\.\d+)?)", html)
            if not price_match:
                raise ValueError("Google Finance page did not contain an INR price.")

            time_match = re.search(
                r'<div class="jZZ2de">(.*?)&nbsp;\s*&middot;\s*&nbsp;\s*INR',
                html,
            )
            quote_time = _parse_google_time(time_match.group(1) if time_match else None)
            age_seconds = (
                (datetime.now(IST) - quote_time).total_seconds()
                if quote_time is not None
                else None
            )
            return PublicQuoteResult(
                symbol=yahoo_symbol,
                last_price=float(price_match.group(1).replace(",", "")),
                last_update_time=quote_time.isoformat(timespec="seconds") if quote_time else None,
                age_seconds=age_seconds,
                source=f"{self.name}: NSE:{nse_symbol}",
            )
        except Exception as exc:  # pragma: no cover - network dependent
            return PublicQuoteResult(
                symbol=yahoo_symbol,
                last_price=None,
                last_update_time=None,
                age_seconds=None,
                source=f"{self.name}: NSE:{nse_symbol}",
                warning=str(exc),
            )

    def index_quotes(self, symbols: list[str]) -> dict[str, PublicQuoteResult]:
        wanted = [to_yahoo_symbol(symbol) for symbol in symbols]
        quotes: dict[str, PublicQuoteResult] = {}
        workers = max(1, min(16, len(wanted)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.quote, symbol): symbol for symbol in wanted}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    result = PublicQuoteResult(
                        symbol=to_yahoo_symbol(symbol),
                        last_price=None,
                        last_update_time=None,
                        age_seconds=None,
                        source=self.name,
                        warning=str(exc),
                    )
                quotes[to_yahoo_symbol(symbol)] = result
        return quotes


class YFinanceProvider:
    """Yahoo Finance data provider with transparent offline candle cache.

    Every successful fetch is persisted to ``CandleCache``.  When the network
    is unavailable the cache is served automatically, so the app keeps working
    and accumulates historical data across sessions for future ML/backtesting.
    """

    name = "Yahoo Finance"

    def __init__(self, cache: "CandleCache | None" = None) -> None:
        if cache is None:
            try:
                from nse_intraday_ai.candle_cache import CandleCache as _CC
                _default_path = Path(__file__).resolve().parents[2] / "data" / "candles.sqlite3"
                cache = _CC(_default_path)
            except Exception:
                cache = None
        self._cache = cache

    def _save(self, symbol: str, interval: str, frame: pd.DataFrame) -> None:
        if self._cache is not None and not frame.empty:
            try:
                self._cache.save(symbol, interval, frame)
            except Exception:
                pass

    def _load_cache(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        if self._cache is None:
            return pd.DataFrame()
        try:
            return self._cache.load_period(symbol, interval, period)
        except Exception:
            return pd.DataFrame()

    def history(self, symbol: str, period: str = "1d", interval: str = "1m") -> DataResult:
        yahoo_symbol = to_yahoo_symbol(symbol)
        try:
            frame = yf.download(
                tickers=yahoo_symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            frame = normalize_ohlcv(frame)
            if not frame.empty:
                self._save(yahoo_symbol, interval, frame)
                return DataResult(yahoo_symbol, frame, self.name, None)
        except Exception:
            pass

        # Network failed — serve from cache
        cached = self._load_cache(yahoo_symbol, interval, period)
        if not cached.empty:
            return DataResult(yahoo_symbol, cached, "Offline cache", "Network unavailable — serving cached candles.")
        return DataResult(yahoo_symbol, pd.DataFrame(), self.name, "No market data returned. Try demo mode or another symbol.")

    def batch_history(
        self,
        symbols: list[str],
        period: str = "1d",
        interval: str = "1m",
        retry_missing: bool = True,
    ) -> dict[str, DataResult]:
        yahoo_symbols = _unique_symbols(symbols)
        if not yahoo_symbols:
            return {}
        try:
            frame = yf.download(
                tickers=" ".join(yahoo_symbols),
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            frame = pd.DataFrame()

        results: dict[str, DataResult] = {}
        for symbol in yahoo_symbols:
            subframe = pd.DataFrame()
            if not frame.empty and isinstance(frame.columns, pd.MultiIndex):
                level0 = set(frame.columns.get_level_values(0))
                level1 = set(frame.columns.get_level_values(1))
                if symbol in level0:
                    subframe = frame[symbol]
                elif symbol in level1:
                    subframe = frame.xs(symbol, level=1, axis=1)
            elif len(yahoo_symbols) == 1:
                subframe = frame

            subframe = normalize_ohlcv(subframe)
            if not subframe.empty:
                self._save(symbol, interval, subframe)
                results[symbol] = DataResult(symbol, subframe, self.name, None)
            else:
                results[symbol] = DataResult(symbol, pd.DataFrame(), self.name, "No market data returned for this symbol.")

        missing = [symbol for symbol, result in results.items() if result.frame.empty]
        if retry_missing and missing:
            workers = min(6, len(missing))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.history, symbol, period, interval): symbol
                    for symbol in missing
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        retry_result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        retry_result = DataResult(symbol, pd.DataFrame(), self.name, str(exc))
                    results[symbol] = retry_result
        return results


class DemoProvider:
    name = "Demo synthetic stream"

    def history(self, symbol: str, period: str = "1d", interval: str = "1m") -> DataResult:
        minutes = 375 if interval == "1m" else 120
        now = datetime.now(IST).replace(second=0, microsecond=0)
        idx = pd.date_range(end=now, periods=minutes, freq="min", tz=IST)
        seed = abs(hash(symbol)) % (2**32)
        rng = np.random.default_rng(seed)
        drift = rng.normal(0.00002, 0.00004)
        volatility = rng.uniform(0.0008, 0.0022)
        base = rng.uniform(250, 2500)
        returns = rng.normal(drift, volatility, minutes)
        trend_burst = np.sin(np.linspace(0, 4 * np.pi, minutes)) * rng.uniform(0.0001, 0.0005)
        price = base * np.exp(np.cumsum(returns + trend_burst))
        spread = price * rng.uniform(0.0006, 0.0025, minutes)
        open_ = np.r_[price[0], price[:-1]]
        high = np.maximum(open_, price) + spread
        low = np.minimum(open_, price) - spread
        volume = rng.integers(50_000, 900_000, minutes).astype(float)
        frame = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": price, "volume": volume},
            index=idx,
        )
        return DataResult(to_yahoo_symbol(symbol), frame, self.name, None)

    def batch_history(
        self, symbols: list[str], period: str = "1d", interval: str = "1m"
    ) -> dict[str, DataResult]:
        return {
            to_yahoo_symbol(symbol): self.history(symbol, period=period, interval=interval)
            for symbol in symbols
        }
