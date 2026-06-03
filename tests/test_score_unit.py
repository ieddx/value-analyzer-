"""Unit tests for the score layer.  No network access — all inputs are synthetic."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from value_analyzer.classify.models import (
    CapitalIntensity, Category, GrowthProfile, Metrics,
    MoatType, RevenueType, RuleTrace, SicHint,
)
from value_analyzer.score.moat import score_moat
from value_analyzer.score.health import score_health
from value_analyzer.score.management import score_management
from value_analyzer.score.valuation import score_valuation
from value_analyzer.score.config import WACC

TODAY = date(2024, 12, 31)
_FORMS = {"10-K"}


# ── Synthetic data factories ───────────────────────────────────────────────

def _metrics(**kwargs) -> Metrics:
    defaults = dict(ticker="TEST", as_of_date=TODAY, years_of_data=10)
    defaults.update(kwargs)
    return Metrics(**defaults)


def _fund_rows(concept: str, values: list[float], years: list[int],
               form: str = "10-K") -> pd.DataFrame:
    rows = []
    for year, val in zip(years, values):
        rows.append({
            "concept": concept,
            "period_end": pd.Timestamp(f"{year}-12-31"),
            "filed": pd.Timestamp(f"{year+1}-02-01"),
            "value": val,
            "form": form,
            "source": "test",
            "unit": "USD",
        })
    return pd.DataFrame(rows)


def _fund(*frames: pd.DataFrame) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["concept", "period_end", "filed", "value", "form"])
    return pd.concat(frames, ignore_index=True)


def _make_prices(years: list[int], prices: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(f"{y}-12-31") for y in years], name="date")
    return pd.DataFrame({"close": prices}, index=idx)


def _make_category(
    moat: MoatType = MoatType.brand,
    intensity: CapitalIntensity = CapitalIntensity.asset_light,
    rev_type: RevenueType = RevenueType.recurring,
    growth: GrowthProfile = GrowthProfile.stable,
) -> Category:
    m = _metrics()
    trace = RuleTrace(rule_name="test", result="brand", confidence=0.8, rationale="test")
    return Category(
        ticker="TEST", as_of_date=TODAY,
        capital_intensity=intensity, revenue_type=rev_type,
        moat_type=moat, growth_profile=growth,
        traces={
            "capital_intensity": trace, "revenue_type": trace,
            "moat_type": trace, "growth_profile": trace,
        },
        metrics=m, sic_hint=SicHint(),
    )


YEARS = list(range(2014, 2025))


# ══════════════════════════════════════════════════════════════════════════════
# MOAT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMoatScore:
    def test_excellent_business_scores_high(self):
        m = _metrics(
            gross_margin_avg=0.62,
            gross_margin_std=0.015,
            roic_avg=0.20,
            roic_std=0.03,
            revenue_cagr=0.07,
            years_of_data=10,
        )
        result = score_moat(_fund(), m)
        assert result.score >= 70, f"Expected ≥70, got {result.score}. Reasons: {result.reasons}"

    def test_commodity_business_scores_low(self):
        m = _metrics(
            gross_margin_avg=0.08,
            gross_margin_std=0.06,
            roic_avg=0.05,
            roic_std=0.08,
            revenue_cagr=-0.02,
            years_of_data=10,
        )
        result = score_moat(_fund(), m)
        assert result.score <= 30, f"Expected ≤30, got {result.score}"

    def test_all_reasons_populated(self):
        m = _metrics(gross_margin_avg=0.55, roic_avg=0.14, revenue_cagr=0.05)
        result = score_moat(_fund(), m)
        assert len(result.reasons) >= 3, "Expected reasons for each scored component"

    def test_every_reason_has_score_prefix(self):
        m = _metrics(gross_margin_avg=0.55, roic_avg=0.12)
        result = score_moat(_fund(), m)
        for r in result.reasons:
            assert r.startswith("[+"), f"Reason missing score prefix: {r!r}"

    def test_score_within_bounds(self):
        m = _metrics(gross_margin_avg=0.99, roic_avg=0.99, revenue_cagr=0.99)
        result = score_moat(_fund(), m)
        assert 0 <= result.score <= 100

    def test_missing_data_flags_added(self):
        m = _metrics()  # all None
        result = score_moat(_fund(), m)
        assert len(result.flags) > 0, "Expected flags when data is missing"

    def test_roic_below_hurdle_scores_low(self):
        m = _metrics(gross_margin_avg=0.40, roic_avg=0.05, revenue_cagr=0.03)
        result = score_moat(_fund(), m)
        # ROIC below hurdle should cap the moat score
        assert result.score < 65


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthScore:
    def _strong_fund(self) -> pd.DataFrame:
        yrs = YEARS
        return _fund(
            _fund_rows("long_term_debt", [500e6] * len(yrs), yrs),
            _fund_rows("equity", [5000e6] * len(yrs), yrs),
            _fund_rows("operating_income", [1200e6] * len(yrs), yrs),
            _fund_rows("interest_expense", [50e6] * len(yrs), yrs),
            _fund_rows("operating_cf", [1100e6] * len(yrs), yrs),
            _fund_rows("capex", [150e6] * len(yrs), yrs),
        )

    def _weak_fund(self) -> pd.DataFrame:
        yrs = YEARS
        return _fund(
            _fund_rows("long_term_debt", [8000e6] * len(yrs), yrs),
            _fund_rows("equity", [1000e6] * len(yrs), yrs),
            _fund_rows("operating_income", [200e6] * len(yrs), yrs),
            _fund_rows("interest_expense", [300e6] * len(yrs), yrs),
            _fund_rows("operating_cf", [-50e6] * len(yrs), yrs),
            _fund_rows("capex", [200e6] * len(yrs), yrs),
        )

    def test_strong_balance_sheet_scores_high(self):
        m = _metrics(fcf_margin_avg=0.18, years_of_data=10)
        result = score_health(self._strong_fund(), m)
        assert result.score >= 65, f"Expected ≥65, got {result.score}. Reasons: {result.reasons}"

    def test_distressed_balance_sheet_scores_low(self):
        m = _metrics(fcf_margin_avg=-0.05, years_of_data=10)
        result = score_health(self._weak_fund(), m)
        assert result.score <= 25, f"Expected ≤25, got {result.score}"

    def test_no_interest_expense_gets_coverage_credit(self):
        yrs = YEARS
        fund_data = _fund(
            _fund_rows("long_term_debt", [0.0] * len(yrs), yrs),
            _fund_rows("equity", [5000e6] * len(yrs), yrs),
            _fund_rows("operating_cf", [800e6] * len(yrs), yrs),
            _fund_rows("capex", [100e6] * len(yrs), yrs),
        )
        m = _metrics(fcf_margin_avg=0.12)
        result = score_health(fund_data, m)
        # No interest expense → should award coverage points
        coverage_reason = next((r for r in result.reasons if "interest" in r.lower()), None)
        assert coverage_reason is not None

    def test_all_reasons_present(self):
        m = _metrics(fcf_margin_avg=0.10)
        result = score_health(self._strong_fund(), m)
        assert len(result.reasons) >= 4

    def test_score_in_bounds(self):
        for fcf in [-0.20, 0.0, 0.10, 0.30]:
            m = _metrics(fcf_margin_avg=fcf)
            result = score_health(self._strong_fund(), m)
            assert 0 <= result.score <= 100


# ══════════════════════════════════════════════════════════════════════════════
# VALUATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestValuationScore:
    def _make_inputs(self, price: float, eps: float, fcf_ps: float, shares: float):
        yrs = list(range(2015, 2025))
        fund_data = _fund(
            _fund_rows("eps_diluted", [eps * (1 + 0.05 * (y - 2015)) for y in yrs], yrs),
            _fund_rows("shares_outstanding", [shares] * len(yrs), yrs),
            _fund_rows("operating_cf", [fcf_ps * shares * (1 + 0.03 * (y - 2015)) for y in yrs], yrs),
            _fund_rows("capex", [fcf_ps * shares * 0.1] * len(yrs), yrs),
            _fund_rows("equity", [shares * 20] * len(yrs), yrs),
        )
        prices = _make_prices(yrs, [price * (1 + 0.05 * (y - 2015)) for y in yrs])
        # Set last price to our target
        prices.loc[prices.index[-1], "close"] = price
        return fund_data, prices

    def test_cheap_stock_scores_high(self):
        # Very low P/E (price = 5× EPS) vs 10y median P/E likely ~15×
        fund_data, prices = self._make_inputs(price=50.0, eps=10.0, fcf_ps=8.0, shares=1e9)
        m = _metrics(revenue_cagr=0.04, years_of_data=10)
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)
        assert result.score >= 55, f"Cheap stock should score ≥55, got {result.score}"

    def test_expensive_stock_scores_low(self):
        # Build history: stock traded at P/E ~10x for years (price $50, EPS $5),
        # but current price has shot to $500 making current P/E = 100×.
        # This is unmistakably expensive vs its own history.
        yrs = list(range(2015, 2025))
        shares = 1e9
        eps_val = 5.0
        fund_data = _fund(
            _fund_rows("eps_diluted", [eps_val] * len(yrs), yrs),
            _fund_rows("shares_outstanding", [shares] * len(yrs), yrs),
            _fund_rows("operating_cf", [4.0 * shares] * len(yrs), yrs),
            _fund_rows("capex", [0.5 * shares] * len(yrs), yrs),
            _fund_rows("equity", [shares * 15] * len(yrs), yrs),
        )
        # Historical prices all ~$50 (P/E ~10×); current year = $500 (P/E ~100×)
        hist_prices = [50.0] * (len(yrs) - 1) + [500.0]
        prices = _make_prices(yrs, hist_prices)
        m = _metrics(revenue_cagr=0.03, years_of_data=10)
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)
        assert result.score <= 30, f"Expensive stock (P/E 100× vs 10× history) should score ≤30, got {result.score}"

    def test_no_price_data_returns_valid_score(self):
        fund_data, _ = self._make_inputs(50.0, 5.0, 4.0, 1e9)
        m = _metrics()
        cat = _make_category()
        result = score_valuation(fund_data, pd.DataFrame(columns=["close"]), m, cat)
        assert 0 <= result.score <= 100
        assert len(result.flags) > 0

    def test_assumptions_appear_in_flags(self):
        fund_data, prices = self._make_inputs(50.0, 5.0, 4.0, 1e9)
        m = _metrics(revenue_cagr=0.04)
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)
        all_flags = " ".join(result.flags)
        assert "WACC" in all_flags or "9%" in all_flags
        assert "terminal" in all_flags.lower() or "2.5%" in all_flags

    def test_all_reasons_have_prefix(self):
        fund_data, prices = self._make_inputs(50.0, 5.0, 4.0, 1e9)
        m = _metrics()
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)
        for r in result.reasons:
            assert r.startswith("[+"), f"Reason missing prefix: {r!r}"


# ══════════════════════════════════════════════════════════════════════════════
# MANAGEMENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestManagementScore:
    def _buyback_fund(self) -> pd.DataFrame:
        yrs = YEARS
        # Shares declining 3%/yr → strong buybacks
        shares = [1e9 * (0.97 ** i) for i in range(len(yrs))]
        net = [500e6] * len(yrs)
        eps = [0.5 * (1.06 ** i) for i in range(len(yrs))]
        divs = [100e6] * len(yrs)
        return _fund(
            _fund_rows("shares_outstanding", shares, yrs),
            _fund_rows("net_income", net, yrs),
            _fund_rows("eps_diluted", eps, yrs),
            _fund_rows("dividends_paid", divs, yrs),
        )

    def _dilution_fund(self) -> pd.DataFrame:
        yrs = YEARS
        shares = [1e9 * (1.06 ** i) for i in range(len(yrs))]  # +6%/yr
        net = [200e6] * len(yrs)
        eps = [0.2] * len(yrs)
        return _fund(
            _fund_rows("shares_outstanding", shares, yrs),
            _fund_rows("net_income", net, yrs),
            _fund_rows("eps_diluted", eps, yrs),
        )

    def test_buyback_company_scores_high(self):
        m = _metrics(roe_avg=0.25, roe_std=0.04, years_of_data=10)
        result = score_management(self._buyback_fund(), m)
        assert result.score >= 60, f"Buyback company should score ≥60, got {result.score}"

    def test_dilution_company_scores_lower(self):
        m = _metrics(roe_avg=0.08, roe_std=0.05, years_of_data=10)
        r_dilution = score_management(self._dilution_fund(), m)
        r_buyback = score_management(self._buyback_fund(), m)
        assert r_buyback.score > r_dilution.score, (
            f"Buyback ({r_buyback.score}) should beat dilution ({r_dilution.score})"
        )

    def test_reasons_all_present(self):
        m = _metrics(roe_avg=0.15, roe_std=0.03)
        result = score_management(self._buyback_fund(), m)
        assert len(result.reasons) >= 3

    def test_score_in_bounds(self):
        m = _metrics(roe_avg=0.30, roe_std=0.02)
        result = score_management(self._buyback_fund(), m)
        assert 0 <= result.score <= 100
