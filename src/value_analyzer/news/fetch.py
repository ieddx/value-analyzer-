"""News fetching layer for value-analyzer.

Fetches recent company news headlines from a configurable provider.
Never raises — all errors are captured in NewsResult.error.
Never hardcodes API keys.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from value_analyzer.news.models import NewsItem, NewsResult

logger = logging.getLogger(__name__)


# ── Provider protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class NewsProvider(Protocol):
    def fetch(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
        api_key: str,
    ) -> list[dict]:
        """Return a list of raw news dicts from the data source."""
        ...


# ── Finnhub implementation ─────────────────────────────────────────────────────

class FinnhubProvider:
    """Fetches company news from the Finnhub REST API."""

    _BASE_URL = "https://finnhub.io/api/v1/company-news"
    _HEADERS = {"User-Agent": "value-analyzer/0.1 educational-use"}

    def fetch(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
        api_key: str,
    ) -> list[dict]:
        import urllib.request
        import urllib.parse
        import json

        params = urllib.parse.urlencode({
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": api_key,
        })
        url = f"{self._BASE_URL}?{params}"

        req = urllib.request.Request(url, headers=self._HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_news(
    ticker: str,
    *,
    days: int = 30,
    as_of_date: date | None = None,
    provider: "NewsProvider | None" = None,
) -> NewsResult:
    """Fetch recent news for *ticker*.

    Returns a NewsResult; never raises.  If the API key is missing or any
    error occurs, NewsResult.error is set and NewsResult.items is empty.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.
    days:
        How many calendar days of news to request (counting back from as_of_date).
    as_of_date:
        The analysis date ceiling. Defaults to today.
    provider:
        Inject a custom NewsProvider for testing. Defaults to FinnhubProvider.
    """
    as_of = as_of_date or date.today()
    from_date = as_of - timedelta(days=days)
    fetched_at = date.today()

    # Resolve the provider first so injected (test) providers bypass the key check.
    # The API key is only required when using the default FinnhubProvider.
    if provider is None:
        api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
        if not api_key:
            return NewsResult(
                ticker=ticker,
                fetched_at=fetched_at,
                provider="finnhub",
                error=(
                    "FINNHUB_API_KEY not set — news unavailable. "
                    "Set the environment variable to enable this feature."
                ),
            )
        provider = FinnhubProvider()
    else:
        # Injected provider — pass an empty key; providers that need one
        # should accept it as a positional argument and handle it themselves.
        api_key = os.environ.get("FINNHUB_API_KEY", "")

    try:
        raw_items = provider.fetch(ticker, from_date, as_of, api_key)
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s: %s", ticker, type(exc).__name__, exc)
        return NewsResult(
            ticker=ticker,
            fetched_at=fetched_at,
            provider="finnhub",
            error=f"News fetch failed: {exc}",
        )

    # Parse, deduplicate, sort
    items: list[NewsItem] = []
    seen: set[tuple[str, str]] = set()

    for raw in raw_items:
        headline = (raw.get("headline") or "").strip()
        if not headline:
            continue

        source = (raw.get("source") or "").strip()
        url = raw.get("url") or ""
        summary = raw.get("summary") or ""

        # Deduplicate by (headline.lower(), source.lower())
        key = (headline.lower(), source.lower())
        if key in seen:
            continue
        seen.add(key)

        # Parse datetime field (unix timestamp)
        ts = raw.get("datetime")
        try:
            published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
        except (TypeError, ValueError, OSError):
            published_at = from_date

        items.append(NewsItem(
            headline=headline,
            source=source,
            published_at=published_at,
            url=url,
            summary=summary,
        ))

    # Sort descending by published_at
    items.sort(key=lambda x: x.published_at, reverse=True)

    return NewsResult(
        ticker=ticker,
        fetched_at=fetched_at,
        provider="finnhub",
        items=items,
    )
