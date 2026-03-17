"""
data_manager.py - Market Data Fetcher & Cache
Supports stocks, ETFs, options chains, futures proxies, forex, and crypto via yfinance.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── Asset catalogs ──────────────────────────────────────────────────────────

POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "WMT", "JNJ", "PG", "MA", "HD", "DIS", "NFLX", "PYPL",
    "INTC", "AMD", "CRM", "ADBE", "ORCL", "CSCO", "PEP", "KO",
]

CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "MATIC-USD",
]

FOREX_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "NZDUSD=X", "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "AUDJPY=X", "EURAUD=X", "EURCHF=X", "GBPCHF=X", "CADJPY=X",
    "AUDNZD=X", "NZDJPY=X", "GBPAUD=X", "EURCAD=X", "AUDCAD=X",
]

FOREX_LABELS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X", "AUD/USD": "AUDUSD=X", "NZD/USD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "AUD/JPY": "AUDJPY=X", "EUR/AUD": "EURAUD=X",
    "EUR/CHF": "EURCHF=X", "GBP/CHF": "GBPCHF=X", "CAD/JPY": "CADJPY=X",
    "AUD/NZD": "AUDNZD=X", "NZD/JPY": "NZDJPY=X", "GBP/AUD": "GBPAUD=X",
    "EUR/CAD": "EURCAD=X", "AUD/CAD": "AUDCAD=X",
}

# ETFs that track futures indices
FUTURES_PROXIES = {
    "S&P 500 Futures":  "ES=F",
    "Nasdaq Futures":   "NQ=F",
    "Dow Futures":      "YM=F",
    "Russell 2000":     "RTY=F",
    "Crude Oil":        "CL=F",
    "Gold":             "GC=F",
    "Silver":           "SI=F",
    "Natural Gas":      "NG=F",
    "Corn":             "ZC=F",
    "Wheat":            "ZW=F",
    "10-Year T-Note":   "ZN=F",
    "Euro FX":          "6E=F",
    "VIX":              "^VIX",
}

INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"]
PERIODS   = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]


def _cache_key(symbol: str, period: str, interval: str) -> str:
    raw = f"{symbol}_{period}_{interval}"
    return hashlib.md5(raw.encode()).hexdigest()


def fetch_ohlcv(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True,
    cache_hours: int = 1,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance with local file caching.

    Returns a DataFrame with columns:
        Date (index), Open, High, Low, Close, Volume
    """
    key = _cache_key(symbol, period, interval)
    cache_path = CACHE_DIR / f"{key}.parquet"

    # Return cached data if fresh enough
    if use_cache and cache_path.exists():
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if age < cache_hours * 3600:
            df = pd.read_parquet(cache_path)
            if not df.empty:
                return df

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Normalize column names
        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)

        # Cache to disk
        df.to_parquet(cache_path)
        return df

    except Exception as e:
        # Fall back to cache even if stale
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        raise RuntimeError(f"Failed to fetch {symbol}: {e}")


def fetch_multiple(symbols: list[str], period="1y", interval="1d",
                   progress_callback=None) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple symbols, calling *progress_callback(i, n)* on each."""
    results = {}
    for i, sym in enumerate(symbols):
        try:
            results[sym] = fetch_ohlcv(sym, period=period, interval=interval)
        except Exception:
            pass
        if progress_callback:
            progress_callback(i + 1, len(symbols))
    return results


def get_options_chain(symbol: str) -> Optional[dict]:
    """
    Return a dict with keys 'calls' and 'puts', each a DataFrame,
    for the nearest expiration date.
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return None
        chain = ticker.option_chain(expirations[0])
        return {"calls": chain.calls, "puts": chain.puts, "expiration": expirations[0]}
    except Exception:
        return None


def get_ticker_info(symbol: str) -> dict:
    """Return metadata dict for a symbol (sector, market cap, etc.)."""
    try:
        return dict(yf.Ticker(symbol).info)
    except Exception:
        return {}


def search_symbols(query: str) -> list[dict]:
    """Search Yahoo Finance for matching tickers."""
    try:
        from yfinance import Tickers  # noqa
        # yfinance doesn't expose search directly; fall back to a simple filter
        all_syms = POPULAR_STOCKS + CRYPTO_SYMBOLS + list(FUTURES_PROXIES.values()) + FOREX_SYMBOLS
        q = query.upper()
        return [{"symbol": s} for s in all_syms if q in s]
    except Exception:
        return []


def clear_cache():
    """Delete all cached data files."""
    for f in CACHE_DIR.glob("*.parquet"):
        f.unlink()
