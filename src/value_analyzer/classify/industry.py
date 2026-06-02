"""Fetch SIC code from SEC EDGAR and map it to classification hints.

The SIC code is used as a *prior*, not as the answer.  Financial metrics
always override a SIC-based hint when they point clearly in a different
direction.  See rules.py for how hints are weighted.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from value_analyzer.data import cache, lookup_cik
from .models import CapitalIntensity, MoatType, RevenueType, SicHint

logger = logging.getLogger(__name__)

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_HEADERS = {
    "User-Agent": "value-analyzer/0.1 personal-educational-use contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# ── SIC → economic-character hint table ───────────────────────────────────
# Format: (sic_low, sic_high_inclusive, {field: EnumValue})
# Ranges listed from specific to general; first match wins.
# Rationale for each entry is in the inline comment.
_SIC_HINTS: list[tuple[int, int, dict]] = [
    # ── Energy / resources ────────────────────────────────────────────────
    (1000, 1499, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.cyclical_commodity,
        moat_type=MoatType.cost_advantage,
    )),  # Mining: commodity prices dominate; capex-intensive extraction
    (1311, 1311, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.cyclical_commodity,
        moat_type=MoatType.cost_advantage,
    )),  # Crude petroleum & natural gas

    # ── Food & beverages (brand-moat CPG) ─────────────────────────────────
    (2000, 2099, dict(
        revenue_type=RevenueType.recurring,
        moat_type=MoatType.brand,
    )),  # Food manufacturing: stable demand, brand pricing power
    (2080, 2089, dict(
        capital_intensity=CapitalIntensity.asset_light,
        revenue_type=RevenueType.recurring,
        moat_type=MoatType.brand,
    )),  # Beverages (KO, PEP): concentrate model = asset-light + brand

    # ── Tobacco ───────────────────────────────────────────────────────────
    (2100, 2111, dict(
        revenue_type=RevenueType.recurring,
        moat_type=MoatType.brand,
    )),

    # ── Chemicals / petroleum refining ────────────────────────────────────
    (2800, 2899, dict(
        revenue_type=RevenueType.transactional,
    )),  # Specialty chemicals: not commodity but not brand either
    (2900, 2999, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.cyclical_commodity,
        moat_type=MoatType.cost_advantage,
    )),  # Petroleum refining: refinery capex, crack-spread cycles

    # ── Metals / primary materials ────────────────────────────────────────
    (3300, 3399, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.cyclical_commodity,
        moat_type=MoatType.cost_advantage,
    )),  # Steel, aluminum: commodity pricing, asset-heavy mills
    (3000, 3299, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.cyclical_commodity,
    )),  # Rubber, stone, glass, concrete

    # ── Electronics / tech hardware ───────────────────────────────────────
    (3670, 3679, dict(
        moat_type=MoatType.switching_cost,
        revenue_type=RevenueType.transactional,
    )),  # Electronic components: often switching costs in supply chains
    (3571, 3579, dict(
        moat_type=MoatType.switching_cost,
    )),  # Computer hardware

    # ── Autos ─────────────────────────────────────────────────────────────
    (3711, 3716, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.cyclical_commodity,
    )),  # Motor vehicles: capex-heavy assembly, credit-cycle sensitivity

    # ── Transportation ────────────────────────────────────────────────────
    (4011, 4013, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.transactional,
        moat_type=MoatType.cost_advantage,
    )),  # Railroads: natural oligopoly, regulated returns, huge capex
    (4400, 4499, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.cyclical_commodity,
    )),  # Water / marine transport
    (4512, 4522, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.cyclical_commodity,
        moat_type=MoatType.none,
    )),  # Airlines: extreme capex, fuel/macro cyclicality, no moat

    # ── Telecom ───────────────────────────────────────────────────────────
    (4810, 4899, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.recurring,
        moat_type=MoatType.network,
    )),  # Telephone / wireless: recurring subs, huge spectrum capex

    # ── Utilities ─────────────────────────────────────────────────────────
    (4900, 4999, dict(
        capital_intensity=CapitalIntensity.asset_heavy,
        revenue_type=RevenueType.recurring,
        moat_type=MoatType.cost_advantage,
    )),  # Regulated utilities: capex-heavy infrastructure, stable demand

    # ── Retail ────────────────────────────────────────────────────────────
    (5200, 5999, dict(
        revenue_type=RevenueType.transactional,
    )),  # Retail: transactional by nature; moat varies (omit from hint)

    # ── Computer services / software ──────────────────────────────────────
    (7370, 7379, dict(
        capital_intensity=CapitalIntensity.asset_light,
        revenue_type=RevenueType.recurring,
        moat_type=MoatType.switching_cost,
    )),  # Software & data processing: recurring SaaS, low capex, high switching costs

    # ── Entertainment / media ─────────────────────────────────────────────
    (7800, 7999, dict(
        revenue_type=RevenueType.transactional,
        moat_type=MoatType.brand,
    )),

    # ── Healthcare services ───────────────────────────────────────────────
    (8000, 8099, dict(
        capital_intensity=CapitalIntensity.asset_light,
        revenue_type=RevenueType.recurring,
    )),
]


def fetch_sic(ticker: str, *, refresh: bool = False) -> SicHint:
    """Return the SIC-based economic hint for *ticker*.

    Fetches SIC from EDGAR submissions JSON (cached).  Returns a SicHint
    with all hint fields set to None if the SIC cannot be determined.
    """
    cik = lookup_cik(ticker)
    if not cik:
        return SicHint()

    sic_code, sic_desc = _fetch_sic_from_edgar(cik, refresh=refresh)
    if sic_code is None:
        return SicHint()

    hints = _sic_to_hints(sic_code)
    return SicHint(sic_code=sic_code, sic_description=sic_desc, **hints)


def _fetch_sic_from_edgar(
    cik: str, *, refresh: bool = False
) -> tuple[Optional[int], Optional[str]]:
    key = f"submissions_{cik}"
    data: dict | None = None if refresh else cache.load_json(key)  # type: ignore

    if data is None:
        url = _SUBMISSIONS_URL.format(cik=cik)
        logger.info("fetching EDGAR submissions for CIK %s", cik)
        try:
            time.sleep(0.15)
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            cache.save_json(data, key)
        except Exception as exc:
            logger.warning("EDGAR submissions fetch failed for %s: %s", cik, exc)
            return None, None

    sic = data.get("sic")
    desc = data.get("sicDescription")
    return (int(sic), str(desc)) if sic else (None, None)


def _sic_to_hints(sic: int) -> dict:
    """Return a dict of hint fields for the first matching SIC range."""
    for low, high, fields in _SIC_HINTS:
        if low <= sic <= high:
            return fields
    return {}
