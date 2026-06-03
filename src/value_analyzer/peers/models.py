"""Pydantic models for the peers layer."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class PeerSnapshot(BaseModel):
    """Classified metrics snapshot for one peer ticker at a point in time."""

    ticker: str
    weight_profile: str  # "compounder" | "stable" | "cyclical" | "declining"
    as_of_date: date
    pe_median_10y: Optional[float] = None
    pfcf_median_10y: Optional[float] = None
    gross_margin_avg: Optional[float] = None
    roic_avg: Optional[float] = None
    fcf_margin_avg: Optional[float] = None


class CategoryPeerStats(BaseModel):
    """Aggregate P/E and quality statistics across same-category peers."""

    weight_profile: str
    as_of_date: date
    peer_tickers: list[str] = Field(default_factory=list)

    pe_p25: Optional[float] = None
    pe_median: Optional[float] = None
    pe_p75: Optional[float] = None
    pfcf_median: Optional[float] = None
    gross_margin_median: Optional[float] = None
    roic_median: Optional[float] = None


class PeerComparison(BaseModel):
    """Subject stock vs same-category peers drawn from value investor portfolios.

    Intended for the report layer as sanity-check context only.
    "Stocks great value investors held in this category looked like X."
    This is NOT training signal and NOT a scoring input — purely analytical framing.
    """

    weight_profile: str
    peer_count: int
    peer_tickers: list[str] = Field(default_factory=list)

    # Subject metrics (the stock under analysis)
    subject_pe: Optional[float] = None
    subject_pfcf: Optional[float] = None

    # Same-category peer aggregate stats
    peer_pe_median: Optional[float] = None
    peer_pe_p25: Optional[float] = None
    peer_pe_p75: Optional[float] = None
    peer_pfcf_median: Optional[float] = None
    peer_gross_margin_median: Optional[float] = None
    peer_roic_median: Optional[float] = None

    context_note: str = ""
