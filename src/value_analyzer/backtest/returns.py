"""Forward return computation for the backtest.

IMPORTANT: forward returns are MEASUREMENT outputs, not scoring inputs.
The as_of() guard must NOT be applied here — we deliberately read prices
*after* the as-of date to measure what actually happened.

The lookahead-bias firewall applies only to data used in scoring
(fundamentals and prices up to as_of_date).  Measuring outcomes is not
lookahead bias; it is the point of the backtest.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from value_analyzer.data.prices import fetch_prices

logger = logging.getLogger(__name__)


def forward_return(
    ticker: str,
    as_of_date: date,
    days: int,
    transaction_cost_bps: float = 20.0,
) -> tuple[float | None, float | None]:
    """Return (gross_return, net_return) for *ticker* over *days* calendar days
    starting the first trading day after *as_of_date*.

    Returns (None, None) when the price history does not cover the full window
    (e.g. ticker delisted during the measurement period, or window extends
    into the future).

    Parameters
    ----------
    ticker:
        Stock ticker symbol.
    as_of_date:
        The analysis date.  We buy on the first trading day after this date.
    days:
        Measurement window length in calendar days.
    transaction_cost_bps:
        Round-trip cost in basis points, deducted from gross return.
    """
    prices = fetch_prices(ticker)
    if prices.empty:
        return None, None

    # Entry: first closing price AFTER as_of_date (next trading day)
    after_start = prices[prices.index > pd.Timestamp(as_of_date)]
    if after_start.empty:
        return None, None
    entry_price = float(after_start["close"].iloc[0])
    entry_date = after_start.index[0]

    # Exit: last closing price on or before (as_of_date + days)
    exit_cutoff = pd.Timestamp(as_of_date) + pd.Timedelta(days=days)
    before_exit = prices[prices.index <= exit_cutoff]
    if before_exit.empty:
        return None, None
    # The exit must be at least one trading day after entry
    valid_exit = before_exit[before_exit.index >= entry_date + pd.Timedelta(days=1)]
    if valid_exit.empty:
        return None, None
    exit_price = float(valid_exit["close"].iloc[-1])

    # Check we are close to the target window — if exit is more than 20 days
    # short of the target (trading halt, delisting) treat as unavailable
    actual_days = (valid_exit.index[-1] - pd.Timestamp(as_of_date)).days
    if actual_days < days - 20:
        logger.debug(
            "%s: forward return window %dd ended early at %s (%dd) — skipping",
            ticker, days, valid_exit.index[-1].date(), actual_days,
        )
        return None, None

    gross = exit_price / entry_price - 1.0
    net = gross - transaction_cost_bps / 10_000
    return gross, net


def benchmark_forward_return(
    benchmark_ticker: str,
    as_of_date: date,
    days: int,
) -> float | None:
    """Return the gross total-return of the benchmark over the same window.

    No transaction cost is deducted — the benchmark is a theoretical
    buy-and-hold without rebalancing, so round-trip costs don't apply.
    (In practice SPY has a management fee ~0.03%/yr, which we ignore.)
    """
    gross, _ = forward_return(benchmark_ticker, as_of_date, days, transaction_cost_bps=0.0)
    return gross
