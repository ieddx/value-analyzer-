"""Composite score orchestrator.

Usage
-----
    from value_analyzer.score import score
    from datetime import date

    result = score("KO", as_of_date=date(2024, 12, 31))
    print(result.composite)
    print(result.moat.score, result.moat.reasons)
"""

from __future__ import annotations

import logging
from datetime import date

from value_analyzer.classify import classify
from value_analyzer.classify.models import GrowthProfile, RevenueType
from value_analyzer.data import as_of, fetch_fundamentals, fetch_prices

from .config import CATEGORY_WEIGHTS
from .health import score_health
from .management import score_management
from .models import CompositeScore
from .moat import score_moat
from .valuation import score_valuation

logger = logging.getLogger(__name__)


def score(ticker: str, *, as_of_date: date | None = None) -> CompositeScore:
    """Run the full score pipeline for *ticker* as of *as_of_date*.

    Pipeline
    --------
    1. Classify the business (reuses data layer, applies as_of guard).
    2. Fetch and filter price history.
    3. Run four sub-scorers in isolation (each can be tested independently).
    4. Select category-appropriate weights from config.CATEGORY_WEIGHTS.
    5. Compute weighted composite score.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.
    as_of_date:
        Analysis date.  Defaults to today.  No data after this date is used.

    Returns
    -------
    CompositeScore
        Fully populated with sub-scores, reasons, flags, and the Category
        object that drove the weight selection.
    """
    if as_of_date is None:
        as_of_date = date.today()

    logger.info("scoring %s as of %s", ticker, as_of_date)

    # ── 1. Classify (fetch + filter + compute metrics inside) ─────────────
    category = classify(ticker, as_of_date=as_of_date)
    metrics = category.metrics

    # ── 2. Fetch and filter data ──────────────────────────────────────────
    raw_fund = fetch_fundamentals(ticker)
    fund = as_of(raw_fund, as_of_date)

    raw_prices = fetch_prices(ticker)
    prices = as_of(raw_prices, as_of_date)

    # ── 3. Sub-scores ─────────────────────────────────────────────────────
    moat = score_moat(fund, metrics)
    health = score_health(fund, metrics)
    valuation = score_valuation(fund, prices, metrics, category)
    management = score_management(fund, metrics)

    # ── 4. Weight profile ─────────────────────────────────────────────────
    profile = _weight_profile(category)
    weights = CATEGORY_WEIGHTS.get(profile, CATEGORY_WEIGHTS["default"])

    # ── 5. Composite ──────────────────────────────────────────────────────
    composite = (
        moat.score       * weights["moat"]
        + health.score   * weights["health"]
        + valuation.score * weights["valuation"]
        + management.score * weights["management"]
    )

    logger.info(
        "%s composite=%.1f (moat=%.1f h=%.1f val=%.1f mgmt=%.1f profile=%s)",
        ticker, composite,
        moat.score, health.score, valuation.score, management.score, profile,
    )

    return CompositeScore(
        ticker=ticker.upper(),
        as_of_date=as_of_date,
        composite=round(composite, 1),
        moat=moat,
        health=health,
        valuation=valuation,
        management=management,
        weight_profile=profile,
        weights_used=weights,
        category=category,
    )


def _weight_profile(category) -> str:
    """Select the weight profile name from the classified Category."""
    if category.revenue_type == RevenueType.cyclical_commodity:
        return "cyclical"
    if category.growth_profile == GrowthProfile.declining:
        return "declining"
    if category.growth_profile == GrowthProfile.compounder:
        return "compounder"
    return "stable"
