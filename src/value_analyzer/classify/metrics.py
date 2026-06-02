"""Compute financial ratios from the fundamentals DataFrame.

All inputs must already be point-in-time filtered (via ``data.as_of``) before
being passed here.  This module never fetches data — it only crunches numbers.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .models import Metrics

logger = logging.getLogger(__name__)

_ANNUAL_FORMS = {"10-K", "10-K/A"}


def compute_metrics(fund: pd.DataFrame, ticker: str, as_of_date: date) -> Metrics:
    """Derive classification-relevant ratios from an annual fundamentals DataFrame.

    Parameters
    ----------
    fund:
        Output of ``data.as_of(fetch_fundamentals(ticker), cutoff)``.
        Must contain columns: concept, period_end, value, form, filed.
    ticker:
        Used for logging only.
    as_of_date:
        Recorded in the returned Metrics for audit purposes.

    Returns
    -------
    Metrics
        All fields may be None if the underlying data is unavailable.
    """
    annual = fund[fund["form"].isin(_ANNUAL_FORMS)].copy()

    rev = _series(annual, "revenue")
    gp = _series(annual, "gross_profit")
    opinc = _series(annual, "operating_income")
    net_inc = _series(annual, "net_income")
    capex = _series(annual, "capex")
    op_cf = _series(annual, "operating_cf")
    equity = _series(annual, "equity")
    assets = _series(annual, "total_assets")
    ltd = _series(annual, "long_term_debt")
    cash_s = _series(annual, "cash")
    tax_s = _series(annual, "income_tax")

    # ── Profitability ──────────────────────────────────────────────────────
    gm = _div(gp, rev)
    gm_avg = _mean(gm)
    gm_std = _std(gm)

    ebit_margin = _div(opinc, rev)
    ebit_avg = _mean(ebit_margin)

    fcf_margin_avg = _compute_fcf_margin(op_cf, capex, rev)

    # ── Capital intensity ──────────────────────────────────────────────────
    # capex from EDGAR's PaymentsToAcquirePropertyPlantAndEquipment is a
    # positive number (absolute cash outflow), so abs() is a safety measure.
    capex_pct = _div(capex.abs(), rev)
    capex_pct_avg = _mean(capex_pct)

    at = _div(rev, assets)
    at_avg = _mean(at)

    # ── Revenue dynamics ───────────────────────────────────────────────────
    revenue_cagr = _cagr(rev)
    rev_growth_cv = _growth_cv(rev)

    # ── Returns ────────────────────────────────────────────────────────────
    roe = _div(_align(net_inc, equity), _align(equity, net_inc))
    roe_avg = _mean(roe)
    roe_std = _std(roe)

    roic_avg, roic_std = _compute_roic(opinc, equity, ltd, cash_s, net_inc, tax_s)

    sources = list(fund["source"].unique()) if "source" in fund.columns else []
    years = len(rev)

    logger.debug(
        "%s metrics: gm=%.2f capex_pct=%.2f cagr=%.2f cv=%.2f years=%d",
        ticker,
        gm_avg or 0,
        capex_pct_avg or 0,
        revenue_cagr or 0,
        rev_growth_cv or 0,
        years,
    )

    return Metrics(
        ticker=ticker,
        as_of_date=as_of_date,
        years_of_data=years,
        gross_margin_avg=gm_avg,
        gross_margin_std=gm_std,
        ebit_margin_avg=ebit_avg,
        fcf_margin_avg=fcf_margin_avg,
        capex_pct_revenue=capex_pct_avg,
        asset_turnover=at_avg,
        revenue_cagr=revenue_cagr,
        revenue_growth_cv=rev_growth_cv,
        roe_avg=roe_avg,
        roe_std=roe_std,
        roic_avg=roic_avg,
        roic_std=roic_std,
        data_sources=sources,
    )


# ── Private helpers ────────────────────────────────────────────────────────

def _series(df: pd.DataFrame, concept: str) -> pd.Series:
    """Return an annual Series indexed by fiscal year (period_end.year).

    Deduplicates first by period_end (taking the latest filing), then by
    year (taking the later period_end within the same calendar year).  This
    handles companies that changed fiscal-year-end dates, which would otherwise
    produce duplicate year-index entries and break arithmetic alignment.
    """
    sub = df[df["concept"] == concept]
    if sub.empty:
        return pd.Series(dtype=float, name=concept)
    sub = sub.copy()
    sub = sub.sort_values("period_end").drop_duplicates("period_end", keep="last")
    sub["_year"] = sub["period_end"].dt.year
    # If two distinct period_ends fall in the same calendar year (fiscal-year change),
    # keep the one with the later period_end (usually the full-year filing).
    sub = sub.drop_duplicates("_year", keep="last")
    return pd.Series(
        sub["value"].to_numpy(dtype=float),
        index=sub["_year"].to_numpy(),
        name=concept,
    )


def _align(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return the slice of *a* aligned to *b*'s index."""
    common = a.index.intersection(b.index)
    return a.loc[common]


def _div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division on aligned index; returns NaN where den == 0."""
    common = num.index.intersection(den.index)
    if common.empty:
        return pd.Series(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num.loc[common].values / den.loc[common].values
    return pd.Series(
        np.where(np.isinf(result) | (den.loc[common].values == 0), np.nan, result),
        index=common,
    )


def _mean(s: pd.Series) -> Optional[float]:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if len(clean) >= 1 else None


def _std(s: pd.Series) -> Optional[float]:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.std()) if len(clean) >= 2 else None


def _cagr(rev: pd.Series) -> Optional[float]:
    """Revenue CAGR over the available period."""
    clean = rev.dropna().sort_index()
    if len(clean) < 2:
        return None
    n = clean.index[-1] - clean.index[0]
    if n <= 0 or clean.iloc[0] <= 0 or clean.iloc[-1] <= 0:
        return None
    return float((clean.iloc[-1] / clean.iloc[0]) ** (1 / n) - 1)


def _growth_cv(rev: pd.Series) -> Optional[float]:
    """Coefficient of variation of YoY revenue growth rates.

    High CV (> 0.25) signals cyclicality; low CV (< 0.12) signals stability.
    CV is undefined when the mean growth rate is near zero, so we return None
    in that case rather than an unstable large number.
    """
    clean = rev.dropna().sort_index()
    if len(clean) < 3:
        return None
    growth = clean.pct_change().dropna()
    if len(growth) < 2:
        return None
    mu = float(growth.mean())
    sigma = float(growth.std())
    if abs(mu) < 0.005:
        return None
    return float(abs(sigma / mu))


def _compute_fcf_margin(
    op_cf: pd.Series,
    capex: pd.Series,
    rev: pd.Series,
) -> Optional[float]:
    common = op_cf.index.intersection(capex.index).intersection(rev.index)
    if common.empty:
        return None
    fcf = op_cf.loc[common] - capex.abs().loc[common]
    fcf_margin = _div(fcf, rev.loc[common])
    return _mean(fcf_margin)


def _compute_roic(
    opinc: pd.Series,
    equity: pd.Series,
    ltd: pd.Series,
    cash_s: pd.Series,
    net_inc: pd.Series,
    tax_s: pd.Series,
) -> tuple[Optional[float], Optional[float]]:
    """ROIC = NOPAT / Invested Capital.

    NOPAT = operating_income × (1 − effective_tax_rate).
    Invested capital = equity + long_term_debt − cash.
    Falls back to ROE if invested capital cannot be computed.
    """
    common = opinc.index.intersection(equity.index)
    if common.empty:
        return None, None

    # Effective tax rate from income tax / pre-tax income
    tax_common = tax_s.index.intersection(net_inc.index)
    if len(tax_common) >= 2:
        ebt = (net_inc.loc[tax_common] + tax_s.loc[tax_common]).replace(0, np.nan)
        eff_tax = (tax_s.loc[tax_common] / ebt).clip(0.0, 0.50)
        tax_rate = float(eff_tax.mean())
    else:
        tax_rate = 0.21  # US statutory rate as fallback

    nopat = opinc.loc[common] * (1 - tax_rate)

    ic_eq = equity.loc[common]
    ic_ltd = ltd.reindex(common).fillna(0)
    ic_cash = cash_s.reindex(common).fillna(0)
    ic = (ic_eq + ic_ltd - ic_cash).replace(0, np.nan)

    roic_series = nopat / ic
    return _mean(roic_series), _std(roic_series)
