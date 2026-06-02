"""Classify layer orchestrator.

Usage
-----
    from value_analyzer.classify import classify
    from datetime import date

    cat = classify("KO", as_of=date(2024, 12, 31))
    print(cat.moat_type)          # MoatType.brand
    print(cat.traces["moat_type"].rationale)
"""

from __future__ import annotations

import logging
from datetime import date

from value_analyzer.data import as_of, fetch_fundamentals
from .industry import fetch_sic
from .metrics import compute_metrics
from .models import Category, CapitalIntensity, GrowthProfile, MoatType, RevenueType
from .rules import apply_all_rules

logger = logging.getLogger(__name__)


def classify(ticker: str, *, as_of_date: date | None = None) -> Category:
    """Run the full classify pipeline for *ticker* as of *as_of_date*.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (e.g. "KO", "NUE").
    as_of_date:
        Point-in-time cutoff.  Defaults to today.  No data filed after this
        date will influence the classification (lookahead-bias firewall).

    Returns
    -------
    Category
        Fully populated Category with classification, confidence scores,
        human-readable rationale traces, and the underlying metrics.
    """
    if as_of_date is None:
        as_of_date = date.today()

    logger.info("classifying %s as of %s", ticker, as_of_date)

    # ── 1. Fetch and point-in-time filter fundamentals ────────────────────
    raw_fund = fetch_fundamentals(ticker)
    fund = as_of(raw_fund, as_of_date)

    # ── 2. Compute financial ratios ───────────────────────────────────────
    metrics = compute_metrics(fund, ticker, as_of_date)

    # ── 3. Fetch SIC hint ─────────────────────────────────────────────────
    sic_hint = fetch_sic(ticker)

    # ── 4. Apply classification rules ────────────────────────────────────
    traces = apply_all_rules(metrics, sic_hint)

    # ── 5. Assemble Category ──────────────────────────────────────────────
    return Category(
        ticker=ticker.upper(),
        as_of_date=as_of_date,
        capital_intensity=CapitalIntensity(traces["capital_intensity"].result),
        revenue_type=RevenueType(traces["revenue_type"].result),
        moat_type=MoatType(traces["moat_type"].result),
        growth_profile=GrowthProfile(traces["growth_profile"].result),
        traces=traces,
        metrics=metrics,
        sic_hint=sic_hint,
    )
