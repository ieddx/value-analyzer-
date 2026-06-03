"""Shared helpers used across score sub-modules.

Not part of the public API — import from score/__init__.py instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from value_analyzer.score.models import SubScore

_ANNUAL_FORMS = {"10-K", "10-K/A"}


# ── Data extraction ────────────────────────────────────────────────────────

def annual_series(fund: pd.DataFrame, concept: str) -> pd.Series:
    """Annual values for *concept*, indexed by fiscal year (int).

    Deduplicates by period_end then by year so fiscal-year-change companies
    (which might file two partial 10-Ks in one calendar year) don't cause
    duplicate-index arithmetic errors.
    """
    sub = fund[fund["concept"].eq(concept) & fund["form"].isin(_ANNUAL_FORMS)]
    if sub.empty:
        return pd.Series(dtype=float, name=concept)
    sub = (
        sub.sort_values("period_end")
        .drop_duplicates("period_end", keep="last")
        .copy()
    )
    sub["_y"] = sub["period_end"].dt.year
    sub = sub.drop_duplicates("_y", keep="last")
    return pd.Series(sub["value"].to_numpy(dtype=float), index=sub["_y"].to_numpy(), name=concept)


def latest(fund: pd.DataFrame, concept: str) -> float | None:
    """Most recent annual value for *concept*, or None if unavailable."""
    s = annual_series(fund, concept)
    if s.empty:
        return None
    return float(s.iloc[-1])


def price_at(prices: pd.DataFrame, cutoff: pd.Timestamp) -> float | None:
    """Last closing price on or before *cutoff*."""
    sub = prices[prices.index <= cutoff]
    if sub.empty:
        return None
    return float(sub["close"].iloc[-1])


# ── Point accumulator ──────────────────────────────────────────────────────

class Scorer:
    """Accumulates scored components into a SubScore.

    Usage::

        s = Scorer("moat")
        s.add(pts=20, max_pts=25, reason="gross margin 61% > 60% excellent threshold")
        s.flag("Only 5 years of gross-profit data — stability score may be overstated")
        return s.build()

    Every call to ``add`` appends a ``[+N/Max] reason`` entry so the caller can
    see exactly which components drove the total.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._total = 0.0
        self.reasons: list[str] = []
        self.flags: list[str] = []

    def add(self, pts: float, max_pts: float, reason: str) -> None:
        """Award *pts* (capped at *max_pts*) and record *reason*."""
        pts = float(max(0.0, min(pts, max_pts)))
        self._total += pts
        self.reasons.append(f"[+{pts:.1f}/{max_pts:.0f}] {reason}")

    def flag(self, msg: str) -> None:
        self.flags.append(f"⚠ {msg}")

    def build(self) -> SubScore:
        return SubScore(
            name=self.name,
            score=round(min(100.0, max(0.0, self._total)), 1),
            reasons=self.reasons,
            flags=self.flags,
        )


# ── Numeric helpers ────────────────────────────────────────────────────────

def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def series_cv(s: pd.Series) -> float | None:
    """Coefficient of variation (std / |mean|). Returns None if insufficient data."""
    clean = s.dropna()
    if len(clean) < 3:
        return None
    mu = float(clean.mean())
    if abs(mu) < 1e-6:
        return None
    return float(clean.std() / abs(mu))


def pct_positive(s: pd.Series) -> float | None:
    """Fraction of non-NaN values that are > 0."""
    clean = s.dropna()
    if clean.empty:
        return None
    return float((clean > 0).mean())


def cagr(s: pd.Series) -> float | None:
    """CAGR of series *s* over its index span (assumes index = year ints)."""
    clean = s.dropna().sort_index()
    if len(clean) < 2:
        return None
    n = clean.index[-1] - clean.index[0]
    if n <= 0 or clean.iloc[0] <= 0 or clean.iloc[-1] <= 0:
        return None
    return float((clean.iloc[-1] / clean.iloc[0]) ** (1 / n) - 1)
