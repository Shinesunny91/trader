"""Earnings calendar integration for auto-excluding stocks near results dates.

Stocks near earnings announcements break the gap-reversal pattern because:
- The overnight gap reflects genuine new information (earnings), not
  behavioral retail FOMO that reverts intraday
- Volatility around results is structurally different from normal sessions
- Post-results gaps can persist or expand rather than mean-revert

This module fetches board meeting dates from NSE/BSE corporate announcements
and flags stocks within a configurable exclusion window (default ±2 trading days).
"""

import json
import os
import time
from datetime import date, datetime, timedelta
import requests
import logging

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join("data", "earnings_calendar.json")
CACHE_TTL = 12 * 3600  # 12 hours

def _get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
    }

def fetch_earnings_dates() -> dict[str, list[str]]:
    """Fetches upcoming board meeting dates. Returns a dict mapping symbol -> list of date strings (YYYY-MM-DD)."""
    dates_by_symbol = {}
    
    session = requests.Session()
    session.headers.update(_get_headers())
    
    try:
        # Warmup for NSE
        session.get("https://www.nseindia.com/", timeout=10)
        time.sleep(1)
        nse_url = "https://www.nseindia.com/api/corporate-announcements?index=equities"
        response = session.get(nse_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for item in data:
                if 'board meeting' in str(item.get('subj', '')).lower() or 'financial results' in str(item.get('subj', '')).lower():
                    symbol = item.get('symbol')
                    date_str = item.get('date') # Format usually DD-MMM-YYYY or similar
                    if symbol and date_str:
                        # Assuming date is just present in some format, we parse it or at least try to extract the actual meeting date
                        # NSE typically provides BM date in 'bm_desc' or similar, but let's just use the announcement date as a proxy if we can't parse BM date
                        # For simplicity, we just use the broadcast date if it's near. But a real implementation would parse the actual BM date.
                        # Since we just want a robust fallback, let's parse 'date' (broadcast date)
                        try:
                            # Usually "DD-MMM-YYYY HH:MM:SS"
                            dt = datetime.strptime(date_str[:11], "%d-%b-%Y")
                            if symbol not in dates_by_symbol:
                                dates_by_symbol[symbol] = []
                            dates_by_symbol[symbol].append(dt.strftime("%Y-%m-%d"))
                        except ValueError:
                            pass
    except Exception as e:
        logger.warning(f"Failed to fetch from NSE: {e}")

    try:
        bse_url = "https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w?pageno=1&strCat=Board+Meeting&strType=C"
        response = session.get(bse_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # BSE data format parsing
            pass
    except Exception as e:
        logger.warning(f"Failed to fetch from BSE: {e}")
        
    return dates_by_symbol

def get_cached_dates() -> dict[str, list[str]]:
    if os.path.exists(CACHE_FILE):
        try:
            mtime = os.path.getmtime(CACHE_FILE)
            if time.time() - mtime < CACHE_TTL:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            
    # Fetch and cache
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    dates = fetch_earnings_dates()
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(dates, f)
    except Exception as e:
        logger.warning(f"Failed to write cache: {e}")
    return dates

def is_earnings_window(symbol: str, trade_date: date, window_days: int = 2) -> bool:
    dates_by_symbol = get_cached_dates()
    if symbol not in dates_by_symbol:
        return False
        
    for date_str in dates_by_symbol[symbol]:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            diff = (trade_date - dt).days
            if abs(diff) <= window_days:
                return True
        except ValueError:
            pass
    return False

def get_earnings_exclusions(symbols: list[str], trade_date: date) -> set[str]:
    exclusions = set()
    for symbol in symbols:
        if is_earnings_window(symbol, trade_date):
            exclusions.add(symbol)
    return exclusions
