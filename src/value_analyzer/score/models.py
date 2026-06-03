"""Pydantic models for the score layer."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from value_analyzer.classify.models import Category
from value_analyzer.peers.models import PeerComparison


class SubScore(BaseModel):
    """Score for one analytical dimension, with a full audit trail.

    Every point awarded or withheld is recorded in ``reasons``.  Read the
    reasons list to understand exactly why the score is what it is.
    Data-quality caveats (missing data, short history, etc.) go in ``flags``.
    """

    name: str
    score: float = Field(ge=0.0, le=100.0)  # 0–100
    reasons: list[str]   # "[+N/Max] human explanation" for every component
    flags: list[str]     # ⚠ data-quality or methodology warnings

    # Data-completeness tracking — set by Scorer.build().
    # real_inputs: components scored from real fetched data.
    # total_inputs: all components attempted (real + missing-data floors).
    real_inputs: int = 0
    total_inputs: int = 0


class CompositeScore(BaseModel):
    """Full scored analysis for a single ticker at a point in time.

    ``composite`` is the category-weighted average of the four sub-scores.
    The weighting rationale lives in ``weights_used`` and ``weight_profile``.

    This object is the primary input to the report layer.
    """

    ticker: str
    as_of_date: date

    composite: float = Field(ge=0.0, le=100.0)
    moat: SubScore
    health: SubScore
    valuation: SubScore
    management: SubScore

    weight_profile: str          # which CATEGORY_WEIGHTS key was selected
    weights_used: dict[str, float]
    category: Category           # the classify result that drove weights
    peer_comparison: Optional[PeerComparison] = None  # same-category peer context for report layer

    # Data-completeness aggregated across all four sub-scores.
    # Populated by composite.py; 0/0 if not yet computed.
    completeness_real: int = 0
    completeness_total: int = 0

    # Dispersion flag from the valuation layer.
    # Set when IV estimates span > VAL_IV_DISPERSION_RATIO.  None otherwise.
    iv_dispersion_flag: Optional[str] = None
