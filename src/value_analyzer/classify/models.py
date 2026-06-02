"""Pydantic models for the classify layer."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CapitalIntensity(str, Enum):
    asset_light = "asset_light"   # capex/revenue < 5%; brand, software, platforms
    moderate = "moderate"          # capex/revenue 5-10%; most industrials, consumer staples
    asset_heavy = "asset_heavy"   # capex/revenue > 10%; utilities, airlines, telco, rails


class RevenueType(str, Enum):
    recurring = "recurring"               # stable, visible; SaaS, consumer staples, insurance
    transactional = "transactional"       # project/unit-based; retail, services, industrials
    cyclical_commodity = "cyclical_commodity"  # price-driven swings; steel, oil, airlines


class MoatType(str, Enum):
    brand = "brand"                   # pricing power via customer preference
    network = "network"               # value grows with users; payments, social, exchanges
    switching_cost = "switching_cost" # high cost/risk of changing vendor; enterprise software
    cost_advantage = "cost_advantage" # structural lower cost; scale, location, proprietary process
    none = "none"                     # no evident durable competitive advantage


class GrowthProfile(str, Enum):
    compounder = "compounder"  # revenue CAGR > 8%, margins stable or expanding
    stable = "stable"          # CAGR 0-8%, mature business, cash generative
    declining = "declining"    # negative or deeply stagnating revenue trend


class Metrics(BaseModel):
    """Financial ratios computed from annual fundamentals, used as classifier inputs."""

    ticker: str
    as_of_date: date
    years_of_data: int

    # Profitability
    gross_margin_avg: Optional[float] = None   # mean of annual (gross_profit / revenue)
    gross_margin_std: Optional[float] = None   # std — lower = more stable pricing power
    ebit_margin_avg: Optional[float] = None    # mean of annual (operating_income / revenue)
    fcf_margin_avg: Optional[float] = None     # mean of annual ((op_cf - capex) / revenue)

    # Capital intensity
    capex_pct_revenue: Optional[float] = None  # mean of annual (|capex| / revenue)
    asset_turnover: Optional[float] = None     # mean of annual (revenue / total_assets)

    # Revenue dynamics
    revenue_cagr: Optional[float] = None       # compound annual growth rate over available period
    revenue_growth_cv: Optional[float] = None  # coeff. of variation of annual growth rates

    # Returns (used to confirm moat quality)
    roe_avg: Optional[float] = None            # mean of annual (net_income / equity)
    roe_std: Optional[float] = None
    roic_avg: Optional[float] = None           # mean of annual NOPAT / invested_capital
    roic_std: Optional[float] = None

    data_sources: list[str] = Field(default_factory=list)


class SicHint(BaseModel):
    """Economic character implied by the SIC code, used as a prior in classification."""

    sic_code: Optional[int] = None
    sic_description: Optional[str] = None
    # Each field is a suggested value or None (no opinion from SIC alone).
    capital_intensity: Optional[CapitalIntensity] = None
    revenue_type: Optional[RevenueType] = None
    moat_type: Optional[MoatType] = None


class RuleTrace(BaseModel):
    """Records which rule fired and why — lets you audit or override any decision."""

    rule_name: str
    result: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str   # plain English: which metric values triggered which threshold


class Category(BaseModel):
    """Business-economics classification for a single ticker at a point in time.

    Read ``traces`` to understand exactly why each dimension was classified as it was.
    Override any field and re-run the report layer with the corrected Category.
    """

    ticker: str
    as_of_date: date

    capital_intensity: CapitalIntensity
    revenue_type: RevenueType
    moat_type: MoatType
    growth_profile: GrowthProfile

    # Audit trail — one RuleTrace per dimension
    traces: dict[str, RuleTrace] = Field(default_factory=dict)

    # Inputs that produced this classification
    metrics: Metrics
    sic_hint: SicHint
