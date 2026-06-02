"""Fetch and cache daily OHLCV price history via yfinance."""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from . import cache

logger = logging.getLogger(__name__)

_REQUIRED = {"Open", "High", "Low", "Close", "Volume"}


def fetch_prices(ticker: str, *, refresh: bool = False) -> pd.DataFrame:
    """Return 10 years of daily OHLCV for *ticker*.

    The returned DataFrame:
    - Is indexed by a DatetimeIndex named ``date`` (UTC midnight, timezone-naive).
    - Has lower-case columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
    - Is adjusted for splits/dividends (``auto_adjust=True``).
    - Is empty (correct schema, zero rows) for unknown or delisted tickers.

    Results are cached to disk; pass ``refresh=True`` to force a re-fetch.
    """
    key = f"prices_{ticker.upper()}"

    if not refresh:
        cached = cache.load_df(key)
        if cached is not None:
            return cached

    logger.info("fetching prices for %s via yfinance", ticker)
    try:
        raw = yf.download(
            ticker,
            period="10y",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.warning("yfinance price fetch failed for %s: %s", ticker, exc)
        return _empty()

    if raw is None or raw.empty:
        logger.warning("no price data returned for %s", ticker)
        return _empty()

    # yfinance ≥ 0.2 with a single ticker can still return MultiIndex columns.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    missing = _REQUIRED - set(raw.columns)
    if missing:
        logger.warning("price data for %s is missing columns: %s", ticker, missing)
        return _empty()

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = pd.Index(["open", "high", "low", "close", "volume"])
    df.index = pd.DatetimeIndex(df.index).normalize().tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()

    cache.save_df(df, key)
    return df


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="date"),
    ).astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
