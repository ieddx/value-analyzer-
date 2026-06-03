"""Score layer — produce interpretable 0–100 sub-scores for a ticker.

Public API
----------
score(ticker, *, as_of_date) -> CompositeScore
"""

from .composite import score
from .models import CompositeScore, SubScore

__all__ = ["score", "CompositeScore", "SubScore"]
