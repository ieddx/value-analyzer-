"""Data layer — fetch, cache, and point-in-time-filter financial data.

Public API
----------
fetch_prices(ticker, *, refresh) -> pd.DataFrame
fetch_fundamentals(ticker, *, refresh) -> pd.DataFrame
lookup_cik(ticker) -> str | None
as_of(df, cutoff) -> pd.DataFrame
assert_no_lookahead(df, cutoff) -> None   (raises AssertionError on violation)
"""

from .fundamentals import fetch_edgar_facts, fetch_fundamentals, lookup_cik
from .point_in_time import as_of, assert_no_lookahead
from .prices import fetch_prices

__all__ = [
    "fetch_prices",
    "fetch_fundamentals",
    "fetch_edgar_facts",
    "lookup_cik",
    "as_of",
    "assert_no_lookahead",
]
