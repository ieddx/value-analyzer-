"""Classify layer — assign business-economics categories to a ticker.

Public API
----------
classify(ticker, *, as_of_date) -> Category
"""

from .classifier import classify
from .models import (
    CapitalIntensity,
    Category,
    GrowthProfile,
    Metrics,
    MoatType,
    RevenueType,
    RuleTrace,
    SicHint,
)

__all__ = [
    "classify",
    "Category",
    "CapitalIntensity",
    "RevenueType",
    "MoatType",
    "GrowthProfile",
    "Metrics",
    "RuleTrace",
    "SicHint",
]
