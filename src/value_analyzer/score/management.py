"""Management / capital allocation score (0–100).

Components and max points
─────────────────────────
  share_count_trend    30 pts  — buybacks vs dilution over 5+ years
  return_on_equity     25 pts  — level and stability of ROE
  return_on_retained   25 pts  — how much value was created from reinvested earnings?
  dividend_consistency 20 pts  — dividend track record (if applicable)
  ─────────────────────────────────────────────────────────────────────
  Total max            100 pts

Note on insider ownership: yfinance provides this in ticker.info["heldPercentInsiders"]
but the EDGAR fundamentals pipeline does not capture it.  Add a data.insiders module
if you want to include this signal in future iterations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from value_analyzer.classify.models import Metrics

from .config import (
    MGMT_BUYBACK_OK,
    MGMT_BUYBACK_STRONG,
    MGMT_DILUTION_BAD,
    MGMT_DILUTION_WARN,
    MGMT_ROE_EXCELLENT,
    MGMT_ROE_GOOD,
    MGMT_ROE_OK,
)
from ._helpers import Scorer, annual_series, cagr, latest, safe_div
from .models import SubScore


def score_management(fund: pd.DataFrame, metrics: Metrics) -> SubScore:
    """Compute the management / capital allocation sub-score.

    Parameters
    ----------
    fund:
        Point-in-time-filtered fundamentals.
    metrics:
        Pre-computed Metrics from the classify layer.
    """
    s = Scorer("management")

    shares = annual_series(fund, "shares_outstanding")
    if shares.empty:
        shares = annual_series(fund, "shares_diluted")

    net_inc = annual_series(fund, "net_income")
    eps = annual_series(fund, "eps_diluted")
    divs = annual_series(fund, "dividends_paid")

    # ── 1. Share count trend (max 30) ────────────────────────────────────
    shares_cagr = cagr(shares)
    n_shares = len(shares)

    if shares_cagr is None or n_shares < 3:
        s.flag(f"Share count history too short ({n_shares} year(s)) to score dilution trend.")
        s.add(10, 30, "Insufficient share-count history; awarding neutral floor.")
    else:
        if shares_cagr < MGMT_BUYBACK_STRONG:
            s.add(30, 30, f"Share count CAGR = {shares_cagr:+.1%}/yr over {n_shares} years — "
                           "consistent buybacks; management is returning capital to shareholders.")
        elif shares_cagr < MGMT_BUYBACK_OK:
            pts = 20 + abs(shares_cagr - MGMT_BUYBACK_OK) / abs(MGMT_BUYBACK_STRONG - MGMT_BUYBACK_OK) * 10
            s.add(pts, 30, f"Share count CAGR = {shares_cagr:+.1%}/yr — "
                            "roughly flat; capital is disciplined.")
        elif shares_cagr < MGMT_DILUTION_WARN:
            pts = 10 + (MGMT_DILUTION_WARN - shares_cagr) / MGMT_DILUTION_WARN * 10
            s.add(pts, 30, f"Share count CAGR = {shares_cagr:+.1%}/yr — "
                            "slight dilution from stock compensation or small equity issuances.")
        elif shares_cagr < MGMT_DILUTION_BAD:
            s.add(4, 30, f"Share count CAGR = {shares_cagr:+.1%}/yr — "
                          "meaningful dilution; shareholders' ownership is being eroded.")
            s.flag(f"Share count growing at {shares_cagr:+.1%}/yr — watch SBC and equity issuance.")
        else:
            s.add(0, 30, f"Share count CAGR = {shares_cagr:+.1%}/yr > {MGMT_DILUTION_BAD:.0%} — "
                          "material dilution; each share represents a shrinking piece of the business.")
            s.flag(f"Heavy dilution: share count growing {shares_cagr:+.1%}/yr.")

    # ── 2. Return on equity — level and stability (max 25) ────────────────
    roe_avg = metrics.roe_avg
    roe_std = metrics.roe_std

    if roe_avg is None:
        s.flag("ROE data unavailable (missing net income or equity).")
        s.add(8, 25, "ROE unavailable; awarding conservative floor.")
    else:
        roe_cv = (roe_std / abs(roe_avg)) if (roe_std and abs(roe_avg) > 0.005) else None

        if roe_avg > MGMT_ROE_EXCELLENT:
            base = 22
        elif roe_avg > MGMT_ROE_GOOD:
            base = 14 + (roe_avg - MGMT_ROE_GOOD) / (MGMT_ROE_EXCELLENT - MGMT_ROE_GOOD) * 8
        elif roe_avg > MGMT_ROE_OK:
            base = 7 + (roe_avg - MGMT_ROE_OK) / (MGMT_ROE_GOOD - MGMT_ROE_OK) * 7
        else:
            base = max(0, roe_avg / MGMT_ROE_OK * 7)

        # Stability bonus: low ROE CV = consistent capital allocation
        stability_bonus = 0.0
        stability_note = ""
        if roe_cv is not None:
            if roe_cv < 0.20:
                stability_bonus = 3.0
                stability_note = f"; ROE CV = {roe_cv:.2f} (consistent)"
            elif roe_cv > 0.50:
                stability_note = f"; ROE CV = {roe_cv:.2f} (highly variable)"

        pts = min(25.0, base + stability_bonus)
        s.add(pts, 25, f"ROE avg = {roe_avg:.1%}{stability_note} — "
                        + (f"excellent capital allocation." if roe_avg > MGMT_ROE_EXCELLENT
                           else f"good returns on shareholders' equity." if roe_avg > MGMT_ROE_GOOD
                           else f"adequate returns." if roe_avg > MGMT_ROE_OK
                           else "subpar returns on equity."))

    # ── 3. Return on retained earnings (max 25) ───────────────────────────
    # RORE ≈ (EPS_now - EPS_5y_ago) / cumulative_retained_earnings_per_share
    # Proxy: increase in EPS / increase in book-value-per-share (using equity/shares)
    # If EPS has grown faster than book value growth, management created value.
    if len(eps) >= 5:
        eps_start = float(eps.iloc[-5]) if len(eps) >= 5 else float(eps.iloc[0])
        eps_end = float(eps.iloc[-1])
        eps_delta = eps_end - eps_start

        shares_latest = float(shares.iloc[-1]) if not shares.empty else None

        net_start = float(net_inc.iloc[-5]) if len(net_inc) >= 5 else None
        net_end = float(net_inc.iloc[-1]) if not net_inc.empty else None

        # Retained earnings proxy: sum of net income minus dividends over period
        div_total = float(divs.abs().sum()) if not divs.empty else 0.0
        net_total = float(net_inc.iloc[-5:].sum()) if len(net_inc) >= 5 else None

        if net_total and shares_latest and shares_latest > 0:
            retained_ps = (net_total - div_total) / shares_latest
            if retained_ps > 0 and eps_delta != 0:
                rore = eps_delta / retained_ps
                if rore > 0.20:
                    s.add(25, 25, f"Return on retained earnings ≈ {rore:.1%} — "
                                   "management has created substantial value from reinvested capital.")
                elif rore > 0.10:
                    pts = 12 + (rore - 0.10) / 0.10 * 13
                    s.add(pts, 25, f"Return on retained earnings ≈ {rore:.1%} — "
                                    "capital reinvestment is creating value.")
                elif rore > 0:
                    pts = 4 + rore / 0.10 * 8
                    s.add(pts, 25, f"Return on retained earnings ≈ {rore:.1%} — "
                                    "limited value creation from retained earnings.")
                else:
                    s.add(0, 25, f"Return on retained earnings ≈ {rore:.1%} — "
                                  "EPS has not grown despite retained earnings; "
                                  "capital reinvestment is not creating value.")
                    s.flag(f"Negative RORE ({rore:.1%}): retained capital may be destroying value.")
            else:
                s.add(8, 25, "Retained earnings positive but EPS delta near zero; "
                               "reinvestment returns inconclusive.")
        else:
            s.flag("Cannot compute return on retained earnings (missing net income or shares data).")
            s.add(8, 25, "RORE unavailable; awarding conservative floor.")
    else:
        s.flag(f"EPS history only {len(eps)} year(s) — 5 years required for RORE calculation.")
        s.add(8, 25, "Insufficient EPS history for RORE; awarding conservative floor.")

    # ── 4. Dividend consistency (max 20) ─────────────────────────────────
    if divs.empty:
        s.add(8, 20, "No dividend data found — company either pays no dividends or "
                      "data is unavailable. Neither confirms nor denies capital discipline.")
        s.flag("No dividend data available in EDGAR filings.")
    else:
        div_pos = divs[divs.abs() > 0]
        n_div_years = len(div_pos)
        n_years = len(divs)
        consistency = n_div_years / n_years if n_years > 0 else 0.0

        div_cagr = cagr(div_pos.abs()) if len(div_pos) >= 3 else None

        if consistency >= 0.90 and div_cagr is not None and div_cagr > 0.03:
            s.add(20, 20, f"Dividends paid in {n_div_years}/{n_years} years with "
                           f"CAGR {div_cagr:+.1%} — consistent and growing dividend; "
                           "strong signal of management confidence in cash flows.")
        elif consistency >= 0.80:
            pts = 12 + (consistency - 0.80) / 0.10 * 8
            s.add(pts, 20, f"Dividends paid in {n_div_years}/{n_years} years — "
                            "mostly consistent dividend track record.")
        elif consistency >= 0.50:
            s.add(6, 20, f"Dividends paid in {n_div_years}/{n_years} years — "
                          "inconsistent dividend history; may indicate capital allocation variability.")
            s.flag("Dividend history inconsistent — check if cuts were strategic or forced.")
        else:
            s.add(3, 20, f"Dividends paid in only {n_div_years}/{n_years} years — "
                          "company rarely distributes dividends; focus on buybacks or reinvestment.")

    return s.build()
