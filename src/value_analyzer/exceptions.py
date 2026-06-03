"""Project-level exceptions for value-analyzer."""

from __future__ import annotations


class TickerNotFoundError(ValueError):
    """Raised when no price or fundamental data can be fetched for a ticker.

    Usually indicates a misspelled or delisted ticker symbol.
    """

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        super().__init__(
            f"No data found for ticker {ticker!r}. "
            "Verify the symbol is correct and that it is listed on a major exchange. "
            "Delisted or over-the-counter tickers may not be supported."
        )


class DataUnavailableError(RuntimeError):
    """Raised when required data cannot be fetched due to a network or API error."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
