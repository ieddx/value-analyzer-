"""Financial health score (0–100).

Components and max points
─────────────────────────
  leverage (debt/equity)    25 pts  — how much debt is on the balance sheet?
  interest_coverage         25 pts  — can the business comfortably service debt?
  fcf_consistency           25 pts  — does free cash flow reliably materialise?
  fcf_level                 25 pts  — how large is the FCF margin?
  ─────────────────────────────────────────────────────────────────────
  Total max                100 pts
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from value_analyzer.classify.models import Metrics

from .config import (
    HEALTH_COVERAGE_LOW,
    HEALTH_COVERAGE_OK,
    HEALTH_COVERAGE_SAFE,
    HEALTH_DE_HIGH,
    HEALTH_DE_OK,
    HEALTH_DE_SAFE,
    HEALTH_FCF_GOOD,
    HEALTH_FCF_OK,
)
from ._helpers import Scorer, annual_series, latest, pct_positive, safe_div
from .models import SubScore


def score_health(fund: pd.DataFrame, metrics: Metrics) -> SubScore:
    """Compute the financial health sub-score.

    Parameters
    ----------
    fund:
        Point-in-time-filtered fundamentals.
    metrics:
        Pre-computed Metrics from the classify layer.
    """
    s = Scorer("health")

    # ── 1. Leverage — Debt / Equity (max 25) ─────────────────────────────
    ltd = latest(fund, "long_term_debt")
    std_ = latest(fund, "short_term_debt")
    eq = latest(fund, "equity")

    total_debt = (ltd or 0.0) + (std_ or 0.0)
    de = safe_div(total_debt, eq) if eq and eq > 0 else None

    if de is None:
        s.flag("Debt/Equity ratio unavailable (missing balance-sheet data).")
        s.add(10, 25, "D/E ratio unavailable; awarding neutral floor.")
    elif de < HEALTH_DE_SAFE:
        s.add(25, 25, f"D/E = {de:.2f} < {HEALTH_DE_SAFE} — conservative balance sheet; "
                       "ample cushion to weather downturns.")
    elif de < HEALTH_DE_OK:
        pts = 14 + (HEALTH_DE_OK - de) / (HEALTH_DE_OK - HEALTH_DE_SAFE) * 11
        s.add(pts, 25, f"D/E = {de:.2f} — manageable leverage within the "
                        f"{HEALTH_DE_SAFE}–{HEALTH_DE_OK} acceptable range.")
    elif de < HEALTH_DE_HIGH:
        pts = max(3, 14 * (HEALTH_DE_HIGH - de) / (HEALTH_DE_HIGH - HEALTH_DE_OK))
        s.add(pts, 25, f"D/E = {de:.2f} — elevated leverage; "
                        "refinancing risk rises if earnings soften.")
    else:
        s.add(0, 25, f"D/E = {de:.2f} > {HEALTH_DE_HIGH} — high leverage; "
                      "financial stress possible in an economic downturn.")

    if de is not None and de > HEALTH_DE_HIGH:
        s.flag(f"D/E = {de:.2f} exceeds {HEALTH_DE_HIGH} — treat debt level as a key risk.")

    # ── 2. Interest coverage — EBIT / Interest expense (max 25) ──────────
    ebit = latest(fund, "operating_income")
    interest = latest(fund, "interest_expense")
    coverage = safe_div(ebit, interest) if interest and interest > 0 else None

    if coverage is None and interest is None:
        s.add(20, 25, "No interest expense found — company appears to carry no "
                       "significant interest-bearing debt. Full credit for coverage.")
    elif coverage is None:
        s.flag("Interest coverage ratio unavailable.")
        s.add(10, 25, "Interest coverage unavailable; awarding neutral floor.")
    elif coverage > HEALTH_COVERAGE_SAFE:
        s.add(25, 25, f"Interest coverage = {coverage:.1f}× > {HEALTH_COVERAGE_SAFE}× — "
                       "debt service is trivially comfortable.")
    elif coverage > HEALTH_COVERAGE_OK:
        pts = 16 + (coverage - HEALTH_COVERAGE_OK) / (HEALTH_COVERAGE_SAFE - HEALTH_COVERAGE_OK) * 9
        s.add(pts, 25, f"Interest coverage = {coverage:.1f}× — adequate; "
                        f"in the {HEALTH_COVERAGE_OK}–{HEALTH_COVERAGE_SAFE}× safe range.")
    elif coverage > HEALTH_COVERAGE_LOW:
        pts = 7 + (coverage - HEALTH_COVERAGE_LOW) / (HEALTH_COVERAGE_OK - HEALTH_COVERAGE_LOW) * 9
        s.add(pts, 25, f"Interest coverage = {coverage:.1f}× — thin; "
                        "a revenue decline could strain debt service.")
    elif coverage > 1.0:
        s.add(2, 25, f"Interest coverage = {coverage:.1f}× — barely covers interest; "
                      "high bankruptcy risk in stress scenario.")
    else:
        s.add(0, 25, f"Interest coverage = {coverage:.1f}× < 1 — "
                      "EBIT does not cover interest; technically insolvent on an operating basis.")
        s.flag(f"Interest coverage < 1× ({coverage:.1f}) — distress signal.")

    # ── 3. FCF consistency — % years with positive FCF (max 25) ──────────
    op_cf = annual_series(fund, "operating_cf")
    capex = annual_series(fund, "capex")
    common = op_cf.index.intersection(capex.index)
    fcf_series: pd.Series | None = None

    if common.empty:
        s.flag("Cannot compute FCF consistency — missing operating cash flow or capex data.")
        s.add(8, 25, "FCF data unavailable; awarding conservative floor.")
    else:
        fcf_series = op_cf.loc[common] - capex.abs().loc[common]
        pct_pos = float((fcf_series > 0).mean())
        years_fcf = len(fcf_series)

        if pct_pos >= 0.90:
            s.add(25, 25, f"FCF positive in {pct_pos:.0%} of {years_fcf} measured years — "
                           "extremely reliable cash generation.")
        elif pct_pos >= 0.75:
            pts = 16 + (pct_pos - 0.75) / 0.15 * 9
            s.add(pts, 25, f"FCF positive in {pct_pos:.0%} of {years_fcf} years — "
                            "mostly consistent cash generation with occasional soft years.")
        elif pct_pos >= 0.60:
            pts = 8 + (pct_pos - 0.60) / 0.15 * 8
            s.add(pts, 25, f"FCF positive in {pct_pos:.0%} of {years_fcf} years — "
                            "inconsistent; some years of cash burn.")
        else:
            s.add(0, 25, f"FCF positive in only {pct_pos:.0%} of {years_fcf} years — "
                          "frequent cash burn; business model may not be self-financing.")
            s.flag(f"FCF positive in only {pct_pos:.0%} of measured years.")

        if years_fcf < 5:
            s.flag(f"Only {years_fcf} years of FCF data — consistency score may not "
                   "capture a full business cycle.")

    # ── 4. FCF margin level (max 25) ─────────────────────────────────────
    fcf_margin = metrics.fcf_margin_avg
    if fcf_margin is None:
        s.flag("FCF margin average unavailable.")
        s.add(8, 25, "FCF margin unavailable; awarding conservative floor.")
    elif fcf_margin > HEALTH_FCF_GOOD:
        s.add(25, 25, f"FCF margin avg {fcf_margin:.1%} > {HEALTH_FCF_GOOD:.0%} — "
                       "strong cash conversion; business generates substantial excess cash.")
    elif fcf_margin > HEALTH_FCF_OK:
        pts = 13 + (fcf_margin - HEALTH_FCF_OK) / (HEALTH_FCF_GOOD - HEALTH_FCF_OK) * 12
        s.add(pts, 25, f"FCF margin avg {fcf_margin:.1%} — adequate cash generation "
                        f"in the {HEALTH_FCF_OK:.0%}–{HEALTH_FCF_GOOD:.0%} range.")
    elif fcf_margin > 0:
        pts = 5 + (fcf_margin / HEALTH_FCF_OK) * 8
        s.add(pts, 25, f"FCF margin avg {fcf_margin:.1%} — thin but positive; "
                        "limited buffer against capital expenditure increases.")
    else:
        s.add(0, 25, f"FCF margin avg {fcf_margin:.1%} — negative FCF on average; "
                      "business has been a net cash consumer.")
        s.flag(f"Negative average FCF margin ({fcf_margin:.1%}).")

    return s.build()
