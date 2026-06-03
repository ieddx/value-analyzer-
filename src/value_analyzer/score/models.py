"""Pydantic models for the score layer."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from value_analyzer.classify.models import Category


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
