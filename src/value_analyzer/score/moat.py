"""Moat score (0–100).

Components and max points
─────────────────────────
  gross_margin_level      25 pts  — how much pricing power does the P&L show?
  gross_margin_stability  25 pts  — is that pricing power durable over time?
  roic_level              30 pts  — are returns above the cost of capital?
  roic_consistency        10 pts  — is ROIC consistently above the hurdle?
  revenue_trend           10 pts  — is the moat translating into growth?
  ─────────────────────────────────────────────────────────────────────
  Total max               100 pts
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from value_analyzer.classify.models import Metrics

from .config import (
    MOAT_GM_CV_OK,
    MOAT_GM_CV_STABLE,
    MOAT_GM_EXCELLENT,
    MOAT_GM_FAIR,
    MOAT_GM_GOOD,
    MOAT_ROIC_CONSISTENCY,
    MOAT_ROIC_EXCELLENT,
    MOAT_ROIC_GOOD,
    MOAT_ROIC_HURDLE,
)
from ._helpers import Scorer, annual_series, cagr, pct_positive, series_cv
from .models import SubScore


def score_moat(fund: pd.DataFrame, metrics: Metrics) -> SubScore:
    """Compute the moat sub-score from financial data.

    Parameters
    ----------
    fund:
        Point-in-time-filtered annual fundamentals (output of ``data.as_of``).
    metrics:
        Pre-computed Metrics from the classify layer — reused to avoid
        redundant computation.
    """
    s = Scorer("moat")

    # ── 1. Gross margin level (max 25) ────────────────────────────────────
    gm = metrics.gross_margin_avg
    if gm is None:
        s.flag("Gross profit data unavailable — gross-margin components skipped.")
    elif gm > MOAT_GM_EXCELLENT:
        s.add(25, 25, f"Gross margin avg {gm:.1%} > {MOAT_GM_EXCELLENT:.0%} (excellent threshold) — "
                       "pricing power is structural, not cyclical.")
    elif gm > MOAT_GM_GOOD:
        pts = 15 + (gm - MOAT_GM_GOOD) / (MOAT_GM_EXCELLENT - MOAT_GM_GOOD) * 10
        s.add(pts, 25, f"Gross margin avg {gm:.1%} — good product differentiation, "
                        f"above {MOAT_GM_GOOD:.0%} threshold.")
    elif gm > MOAT_GM_FAIR:
        pts = 6 + (gm - MOAT_GM_FAIR) / (MOAT_GM_GOOD - MOAT_GM_FAIR) * 9
        s.add(pts, 25, f"Gross margin avg {gm:.1%} — thin but positive margins; "
                        "limited pricing power.")
    else:
        s.add(2, 25, f"Gross margin avg {gm:.1%} < {MOAT_GM_FAIR:.0%} — "
                      "commodity-like pricing; no structural advantage evident.")

    # ── 2. Gross margin stability (max 25) ────────────────────────────────
    gm_cv = None
    if metrics.gross_margin_avg and metrics.gross_margin_std:
        gm_cv = metrics.gross_margin_std / abs(metrics.gross_margin_avg)

    if gm_cv is None:
        s.flag("Insufficient data to measure gross-margin stability (< 3 years).")
    elif gm_cv < MOAT_GM_CV_STABLE:
        s.add(25, 25, f"Gross-margin CV = {gm_cv:.3f} < {MOAT_GM_CV_STABLE} — "
                       "extremely stable pricing power; competitors cannot erode it.")
    elif gm_cv < MOAT_GM_CV_OK:
        pts = 12 + (MOAT_GM_CV_OK - gm_cv) / (MOAT_GM_CV_OK - MOAT_GM_CV_STABLE) * 13
        s.add(pts, 25, f"Gross-margin CV = {gm_cv:.3f} — acceptable stability; "
                        "some year-to-year pricing variation.")
    else:
        pts = max(0, 12 * (1 - (gm_cv - MOAT_GM_CV_OK) / 0.20))
        s.add(pts, 25, f"Gross-margin CV = {gm_cv:.3f} > {MOAT_GM_CV_OK} — "
                        "meaningful pricing variability; moat quality uncertain.")

    # ── 3. ROIC level (max 30) ────────────────────────────────────────────
    roic = metrics.roic_avg
    if roic is None:
        s.flag("ROIC could not be computed (missing operating income or equity data).")
    elif roic > MOAT_ROIC_EXCELLENT:
        s.add(30, 30, f"ROIC avg {roic:.1%} > {MOAT_ROIC_EXCELLENT:.0%} — "
                       "clearly earns above cost of capital; strong economic moat.")
    elif roic > MOAT_ROIC_GOOD:
        pts = 20 + (roic - MOAT_ROIC_GOOD) / (MOAT_ROIC_EXCELLENT - MOAT_ROIC_GOOD) * 10
        s.add(pts, 30, f"ROIC avg {roic:.1%} — good returns above cost of capital "
                        f"({MOAT_ROIC_GOOD:.0%}–{MOAT_ROIC_EXCELLENT:.0%} range).")
    elif roic > MOAT_ROIC_HURDLE:
        pts = 10 + (roic - MOAT_ROIC_HURDLE) / (MOAT_ROIC_GOOD - MOAT_ROIC_HURDLE) * 10
        s.add(pts, 30, f"ROIC avg {roic:.1%} — above the {MOAT_ROIC_HURDLE:.0%} hurdle "
                        "but not exceptional.")
    else:
        s.add(2, 30, f"ROIC avg {roic:.1%} < {MOAT_ROIC_HURDLE:.0%} hurdle — "
                      "returns do not clearly exceed cost of capital.")

    # ── 4. ROIC consistency (max 10) ─────────────────────────────────────
    roic_std = metrics.roic_std
    if roic is not None and roic_std is not None:
        roic_cv = roic_std / abs(roic) if abs(roic) > 0.005 else None
        if roic_cv is not None:
            if roic_cv < MOAT_ROIC_CONSISTENCY:
                s.add(10, 10, f"ROIC std = {roic_std:.2%}, CV = {roic_cv:.2f} — "
                               "ROIC is consistent year-to-year; moat is durable, not cyclical.")
            elif roic_cv < 0.20:
                s.add(5, 10, f"ROIC std = {roic_std:.2%}, CV = {roic_cv:.2f} — "
                               "moderate ROIC variability.")
            else:
                s.add(1, 10, f"ROIC std = {roic_std:.2%}, CV = {roic_cv:.2f} — "
                               "high ROIC variability; returns may be cyclical rather than structural.")
        else:
            s.flag("ROIC CV undefined (ROIC near zero); consistency score skipped.")
    else:
        s.flag("ROIC std not available; consistency score skipped.")

    # ── 5. Revenue trend (max 10) ─────────────────────────────────────────
    rev_cagr = metrics.revenue_cagr
    years = metrics.years_of_data
    if rev_cagr is None or years < 3:
        s.flag(f"Revenue CAGR unreliable (only {years} year(s) of data).")
        s.add(3, 10, "Insufficient history for revenue trend scoring; awarding floor.")
    elif rev_cagr > 0.08:
        s.add(10, 10, f"Revenue CAGR {rev_cagr:+.1%} over {years} years — "
                       "moat translates into meaningful growth.")
    elif rev_cagr > 0.03:
        pts = 6 + (rev_cagr - 0.03) / 0.05 * 4
        s.add(pts, 10, f"Revenue CAGR {rev_cagr:+.1%} over {years} years — "
                        "steady, moat-supported growth.")
    elif rev_cagr > 0:
        s.add(4, 10, f"Revenue CAGR {rev_cagr:+.1%} — slow but positive growth; "
                      "moat is defensive rather than expansionary.")
    else:
        s.add(0, 10, f"Revenue CAGR {rev_cagr:+.1%} — revenue is shrinking; "
                      "moat may be eroding.")

    if years < 7:
        s.flag(f"Only {years} years of fundamentals data — scores may not reflect "
               "full business cycle.")

    return s.build()
