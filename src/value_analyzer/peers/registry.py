"""Peer registry — curated value investor holdings, classified and cached.

OFFLINE BUILD (one-time, run when you want fresh peer data):
    python -c "
    from value_analyzer.peers.registry import build_peer_registry
    from datetime import date
    build_peer_registry(date(2024, 12, 31))
    "

SCORING PIPELINE (fast, reads from cache — gracefully returns None if no cache):
    from value_analyzer.peers.registry import get_peer_stats
    stats = get_peer_stats("cyclical", as_of_date=date(2024, 12, 31))
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from value_analyzer.data.cache import CACHE_DIR, _ensure
from .models import CategoryPeerStats, PeerComparison, PeerSnapshot

logger = logging.getLogger(__name__)

# ── Curated seed list ──────────────────────────────────────────────────────
# Tickers drawn from Berkshire Hathaway (CIK 1067983) and Markel Corporation
# (CIK 1096343) 13F-HR equity holdings, verified against public filings.
# Categories are assigned at build time by our classifier — NOT hard-coded here.
INVESTOR_HOLDINGS: list[str] = [
    # Berkshire Hathaway core equity portfolio
    "KO",    # Coca-Cola
    "AXP",   # American Express
    "MCO",   # Moody's
    "AAPL",  # Apple
    "DVA",   # DaVita
    "OXY",   # Occidental Petroleum
    "CVX",   # Chevron
    "BAC",   # Bank of America
    "BK",    # Bank of New York Mellon
    # Markel Corporation equity portfolio
    "GOOG",  # Alphabet
    "HD",    # Home Depot
    "NKE",   # Nike
    "DIS",   # Disney
    # Canonical value investor holdings across both
    "JNJ",   # Johnson & Johnson
    "PG",    # Procter & Gamble
    "V",     # Visa
    "MA",    # Mastercard
    "WMT",   # Walmart
    "NUE",   # Nucor Steel
    "CAT",   # Caterpillar
    "XOM",   # ExxonMobil
]

_CACHE_PREFIX = "peer_registry_"


# ── Public API ─────────────────────────────────────────────────────────────

def build_peer_registry(
    as_of_date: date,
    tickers: list[str] | None = None,
) -> list[PeerSnapshot]:
    """Classify all seed tickers and persist the results to cache.

    Expensive: calls classify() and fetches data for every ticker in the list.
    Run offline; do not call from the scoring hot path.

    Parameters
    ----------
    as_of_date:
        Point-in-time cutoff for classification and price history.
    tickers:
        Override the default INVESTOR_HOLDINGS list.
    """
    from value_analyzer.classify.classifier import classify
    from value_analyzer.score.composite import _weight_profile
    from value_analyzer.data import as_of as pit_filter
    from value_analyzer.data import fetch_fundamentals, fetch_prices

    seed = tickers or INVESTOR_HOLDINGS
    snapshots: list[PeerSnapshot] = []

    for ticker in seed:
        try:
            cat = classify(ticker, as_of_date=as_of_date)
            profile = _weight_profile(cat)

            raw_fund = fetch_fundamentals(ticker)
            fund = pit_filter(raw_fund, as_of_date)
            raw_prices = fetch_prices(ticker)
            prices = pit_filter(raw_prices, as_of_date)

            pe_vals = _pe_history(fund, prices)
            pfcf_vals = _pfcf_history(fund, prices)

            snap = PeerSnapshot(
                ticker=ticker.upper(),
                weight_profile=profile,
                as_of_date=as_of_date,
                pe_median_10y=float(np.median(pe_vals)) if pe_vals else None,
                pfcf_median_10y=float(np.median(pfcf_vals)) if pfcf_vals else None,
                gross_margin_avg=cat.metrics.gross_margin_avg,
                roic_avg=cat.metrics.roic_avg,
                fcf_margin_avg=cat.metrics.fcf_margin_avg,
            )
            snapshots.append(snap)
            logger.info("peer classified: %s → %s", ticker, profile)
        except Exception as exc:
            logger.warning("skipping peer %s during registry build: %s", ticker, exc)

    _save(snapshots, as_of_date)
    logger.info("peer registry built: %d snapshots for %s", len(snapshots), as_of_date)
    return snapshots


def load_peer_snapshots(as_of_date: date) -> list[PeerSnapshot]:
    """Load cached peer snapshots, searching up to 180 days back from *as_of_date*.

    Returns an empty list if no cache file is found — callers must degrade
    gracefully (scoring continues without peer comparison).
    """
    for delta in range(0, 181):
        candidate = as_of_date - timedelta(days=delta)
        path = _cache_path(candidate)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return [PeerSnapshot.model_validate(s) for s in data]
            except Exception as exc:
                logger.warning("failed to load peer registry from %s: %s", path, exc)
                return []
    return []


def get_peer_stats(
    weight_profile: str,
    as_of_date: date,
    snapshots: list[PeerSnapshot] | None = None,
) -> CategoryPeerStats | None:
    """Return aggregate P/E and quality stats for *weight_profile* peers.

    Parameters
    ----------
    weight_profile:
        One of "compounder", "stable", "cyclical", "declining".
    as_of_date:
        Used to locate the nearest cached registry.
    snapshots:
        Inject pre-loaded snapshots (useful for testing without disk I/O).

    Returns None when no peer data is available for the requested category.
    """
    if snapshots is None:
        snapshots = load_peer_snapshots(as_of_date)
    return _aggregate(snapshots, weight_profile, as_of_date)


def build_peer_comparison(
    fund,
    prices,
    weight_profile: str,
    peer_stats: CategoryPeerStats | None,
    snapshots: list[PeerSnapshot] | None = None,
) -> PeerComparison | None:
    """Build a PeerComparison for the report layer.

    Computes the subject's current P/E and P/FCF from *fund* and *prices*,
    then combines them with *peer_stats* for side-by-side context.

    Returns None when *peer_stats* is None (no registry available).
    """
    if peer_stats is None:
        return None

    subject_pe = subject_pfcf = None
    if not prices.empty:
        price = float(prices["close"].iloc[-1])
        eps = _latest(fund, "eps_diluted")
        if eps and eps > 0:
            subject_pe = price / eps

        op_cf = _latest(fund, "operating_cf")
        capex = _latest(fund, "capex")
        shares = _latest(fund, "shares_outstanding") or _latest(fund, "shares_diluted")
        if op_cf is not None and capex is not None and shares and shares > 0:
            fcf_ps = (op_cf - abs(capex)) / shares
            if fcf_ps > 0:
                subject_pfcf = price / fcf_ps

    return PeerComparison(
        weight_profile=weight_profile,
        peer_count=len(peer_stats.peer_tickers),
        peer_tickers=list(peer_stats.peer_tickers),
        subject_pe=subject_pe,
        subject_pfcf=subject_pfcf,
        peer_pe_median=peer_stats.pe_median,
        peer_pe_p25=peer_stats.pe_p25,
        peer_pe_p75=peer_stats.pe_p75,
        peer_pfcf_median=peer_stats.pfcf_median,
        peer_gross_margin_median=peer_stats.gross_margin_median,
        peer_roic_median=peer_stats.roic_median,
        context_note=(
            f"Reference: {len(peer_stats.peer_tickers)} {weight_profile} stocks "
            "from Berkshire/Markel value investor portfolios. "
            "Historical context only — not investment advice."
        ),
    )


# ── Private helpers ────────────────────────────────────────────────────────

def _cache_path(as_of: date) -> Path:
    _ensure()
    return CACHE_DIR / f"{_CACHE_PREFIX}{as_of.isoformat()}.json"


def _save(snapshots: list[PeerSnapshot], as_of_date: date) -> None:
    path = _cache_path(as_of_date)
    path.write_text(json.dumps([s.model_dump(mode="json") for s in snapshots], default=str))
    logger.debug("peer registry saved: %s (%d entries)", path, len(snapshots))


def _aggregate(
    snapshots: list[PeerSnapshot],
    weight_profile: str,
    as_of_date: date,
) -> CategoryPeerStats | None:
    peers = [s for s in snapshots if s.weight_profile == weight_profile]
    if not peers:
        return None

    pes = [s.pe_median_10y for s in peers if s.pe_median_10y is not None]
    pfcfs = [s.pfcf_median_10y for s in peers if s.pfcf_median_10y is not None]
    gms = [s.gross_margin_avg for s in peers if s.gross_margin_avg is not None]
    roics = [s.roic_avg for s in peers if s.roic_avg is not None]

    return CategoryPeerStats(
        weight_profile=weight_profile,
        as_of_date=as_of_date,
        peer_tickers=[s.ticker for s in peers],
        pe_p25=float(np.percentile(pes, 25)) if pes else None,
        pe_median=float(np.median(pes)) if pes else None,
        pe_p75=float(np.percentile(pes, 75)) if pes else None,
        pfcf_median=float(np.median(pfcfs)) if pfcfs else None,
        gross_margin_median=float(np.median(gms)) if gms else None,
        roic_median=float(np.median(roics)) if roics else None,
    )


def _latest(fund, concept: str) -> float | None:
    """Most recent annual value for *concept*, None if unavailable."""
    import pandas as pd
    sub = fund[
        fund["concept"].eq(concept)
        & fund["form"].isin({"10-K", "10-K/A"})
    ]
    if sub.empty:
        return None
    sub = (
        sub.sort_values("period_end")
        .drop_duplicates("period_end", keep="last")
    )
    sub = sub.copy()
    sub["_y"] = sub["period_end"].dt.year
    sub = sub.drop_duplicates("_y", keep="last")
    if sub.empty:
        return None
    return float(sub["value"].iloc[-1])


def _pe_history(fund, prices) -> list[float]:
    """Annual P/E values over the available price and EPS history."""
    import pandas as pd
    eps_map = _annual_map(fund, "eps_diluted")
    result = []
    for year, eps in eps_map.items():
        if eps <= 0:
            continue
        sub = prices[prices.index <= pd.Timestamp(f"{year}-12-31")]
        if sub.empty:
            continue
        result.append(float(sub["close"].iloc[-1]) / eps)
    return result


def _pfcf_history(fund, prices) -> list[float]:
    """Annual P/FCF values over the available history."""
    import pandas as pd
    op_cf_map = _annual_map(fund, "operating_cf")
    capex_map = _annual_map(fund, "capex")
    shares_map = _annual_map(fund, "shares_outstanding")
    if not shares_map:
        shares_map = _annual_map(fund, "shares_diluted")
    result = []
    for year in set(op_cf_map) & set(capex_map) & set(shares_map):
        sh = shares_map[year]
        if sh <= 0:
            continue
        fcf_ps = (op_cf_map[year] - abs(capex_map[year])) / sh
        if fcf_ps <= 0:
            continue
        sub = prices[prices.index <= pd.Timestamp(f"{year}-12-31")]
        if sub.empty:
            continue
        result.append(float(sub["close"].iloc[-1]) / fcf_ps)
    return result


def _annual_map(fund, concept: str) -> dict[int, float]:
    """Return {year: value} for annual filings of *concept*."""
    sub = fund[
        fund["concept"].eq(concept)
        & fund["form"].isin({"10-K", "10-K/A"})
    ]
    if sub.empty:
        return {}
    sub = (
        sub.sort_values("period_end")
        .drop_duplicates("period_end", keep="last")
        .copy()
    )
    sub["_y"] = sub["period_end"].dt.year
    sub = sub.drop_duplicates("_y", keep="last")
    return dict(zip(sub["_y"].to_numpy(), sub["value"].to_numpy(dtype=float)))
