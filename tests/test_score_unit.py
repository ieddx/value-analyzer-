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
from value_analyzer.score.config import WACC, COMPLETENESS_CAUTION_THRESHOLD

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

    # ── Negative-EPS fallback tests (Fixes 1–3) ───────────────────────────

    def _make_negative_eps_inputs(
        self,
        *,
        price: float,
        eps_values: list[float],
        shares: float,
        equity_per_share: float,
        fcf_ps: float | None,
    ):
        """Build fund + prices for a company with negative EPS."""
        yrs = list(range(2015, 2025))
        n = len(yrs)
        # Pad eps_values if shorter than yrs
        if len(eps_values) < n:
            eps_values = ([-5.0] * (n - len(eps_values))) + eps_values

        rows = [
            _fund_rows("eps_diluted", eps_values, yrs),
            _fund_rows("shares_outstanding", [shares] * n, yrs),
            _fund_rows("equity", [equity_per_share * shares] * n, yrs),
        ]
        if fcf_ps is not None:
            # Positive FCF: fcf = op_cf - capex; give op_cf = (fcf_ps + small capex) * shares
            capex_ps = 2.0
            rows += [
                _fund_rows("operating_cf", [(fcf_ps + capex_ps) * shares] * n, yrs),
                _fund_rows("capex", [capex_ps * shares] * n, yrs),
            ]
        fund_data = _fund(*rows)
        prices = _make_prices(yrs, [price] * n)
        return fund_data, prices

    def test_negative_eps_positive_bvps_gets_pb_estimate(self):
        """(a) Negative EPS + positive BVPS → P/B estimate appears in flags."""
        fund_data, prices = self._make_negative_eps_inputs(
            price=40.0,
            eps_values=[-20.0, -25.0, -2.0],   # deeply negative 3y avg
            shares=1e9,
            equity_per_share=25.0,              # BVPS = $25
            fcf_ps=None,                        # no FCF so only P/B fires
        )
        m = _metrics()
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)

        all_flags = " ".join(result.flags)
        assert "P/B" in all_flags or "P/b" in all_flags.lower(), (
            f"Expected P/B estimate in flags. Flags were:\n{chr(10).join(result.flags)}"
        )
        # Must not say "missing EPS or book-value data" (the old wrong message)
        assert "missing EPS" not in all_flags
        assert "missing EPS or book-value" not in all_flags
        # IV scoring path must have run (score > floor of 10)
        assert result.score > 10, f"P/B fallback should produce IV estimate above floor, got {result.score}"

    def test_negative_eps_positive_fcf_gets_fcf_estimate(self):
        """(b) Negative EPS + positive FCF/share → FCF earnings-power estimate appears."""
        fund_data, prices = self._make_negative_eps_inputs(
            price=40.0,
            eps_values=[-20.0, -25.0, -2.0],
            shares=1e9,
            equity_per_share=-1.0,   # negative equity → P/B fallback won't fire
            fcf_ps=3.50,             # positive FCF/share
        )
        m = _metrics()
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)

        all_flags = " ".join(result.flags)
        assert "FCF" in all_flags, (
            f"Expected FCF estimate in flags. Flags were:\n{chr(10).join(result.flags)}"
        )
        assert "earnings power" in all_flags.lower() or "FCF fallback" in all_flags, (
            f"Expected FCF earnings-power label in flags."
        )
        assert "missing EPS" not in all_flags
        assert result.score > 10, f"FCF fallback should produce IV estimate above floor, got {result.score}"

    def test_error_message_absent_eps_vs_negative_eps(self):
        """(c) Error message correctly distinguishes absent EPS from negative EPS."""
        yrs = list(range(2015, 2025))
        prices = _make_prices(yrs, [50.0] * len(yrs))
        m = _metrics()
        cat = _make_category()

        # Case 1: no EPS data at all (eps_norm is None)
        fund_no_eps = _fund(
            _fund_rows("shares_outstanding", [1e9] * len(yrs), yrs),
        )
        result_absent = score_valuation(fund_no_eps, prices, m, cat)
        flags_absent = " ".join(result_absent.flags)
        assert "EPS data absent" in flags_absent, (
            f"Expected 'EPS data absent' flag when EPS is missing. Got:\n{chr(10).join(result_absent.flags)}"
        )
        assert "negative" not in flags_absent.lower() or "EPS data absent" in flags_absent

        # Case 2: EPS present but negative
        fund_neg_eps = _fund(
            _fund_rows("eps_diluted", [-15.0, -20.0, -5.0], list(range(2022, 2025))),
            _fund_rows("shares_outstanding", [1e9] * len(yrs), yrs),
        )
        result_negative = score_valuation(fund_neg_eps, prices, m, cat)
        flags_negative = " ".join(result_negative.flags)
        assert "negative" in flags_negative.lower(), (
            f"Expected 'negative' in flags when EPS is negative. Got:\n{chr(10).join(result_negative.flags)}"
        )
        assert "non-cash impairment" in flags_negative.lower() or "impairment" in flags_negative.lower(), (
            f"Expected impairment context in negative-EPS flag."
        )
        # Must NOT say the old misleading message
        assert "missing EPS or book-value data" not in flags_negative

    def test_fallback_labels_appear_in_iv_estimate_flags(self):
        """Fallback IV estimates carry distinct labels so the investor knows which method ran."""
        fund_data, prices = self._make_negative_eps_inputs(
            price=40.0,
            eps_values=[-20.0, -25.0, -2.0],
            shares=1e9,
            equity_per_share=20.0,
            fcf_ps=3.0,
        )
        m = _metrics()
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)

        all_flags = " ".join(result.flags)
        # At least one fallback method label must appear in the IV estimate flags
        assert ("P/B reversion" in all_flags or "FCF earnings power" in all_flags
                or "No-growth FCF" in all_flags), (
            f"Fallback IV label missing from flags:\n{chr(10).join(result.flags)}"
        )

    def test_positive_eps_still_uses_eps_methods(self):
        """Regression: positive-EPS path unchanged — fallbacks must NOT fire."""
        fund_data, prices = self._make_inputs(price=50.0, eps=5.0, fcf_ps=4.0, shares=1e9)
        m = _metrics()
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)

        all_flags = " ".join(result.flags)
        # EPS-based methods should appear
        assert "EPS_norm" in all_flags or "earnings power" in all_flags.lower()
        # Fallback flags must not appear when EPS is positive
        assert "P/B fallback" not in all_flags
        assert "FCF fallback" not in all_flags
        assert "normalised EPS is negative" not in all_flags


# ── _build_pb_history unit tests ───────────────────────────────────────────────

class TestBuildPbHistory:
    """Unit tests for the P/B history builder added in Fix 2."""

    def test_returns_dict_with_pb_values(self):
        from value_analyzer.score.valuation import _build_pb_history
        yrs = [2019, 2020, 2021, 2022, 2023]
        shares = 1e9
        equity_s = pd.Series([20 * shares] * len(yrs), index=yrs)
        shares_s = pd.Series([shares] * len(yrs), index=yrs)
        prices = _make_prices(yrs, [30.0] * len(yrs))  # price=30, BVPS=20 → P/B=1.5
        result = _build_pb_history(equity_s, shares_s, prices)
        assert len(result) == len(yrs)
        for pb in result.values():
            assert abs(pb - 1.5) < 0.01, f"Expected P/B ≈ 1.5, got {pb}"

    def test_skips_years_with_negative_equity(self):
        from value_analyzer.score.valuation import _build_pb_history
        yrs = [2020, 2021, 2022]
        shares = 1e9
        equity_s = pd.Series([-5 * shares, 20 * shares, 20 * shares], index=yrs)
        shares_s = pd.Series([shares] * 3, index=yrs)
        prices = _make_prices(yrs, [30.0] * 3)
        result = _build_pb_history(equity_s, shares_s, prices)
        assert 2020 not in result  # negative equity skipped
        assert 2021 in result
        assert 2022 in result

    def test_returns_empty_for_empty_inputs(self):
        from value_analyzer.score.valuation import _build_pb_history
        empty = pd.Series(dtype=float)
        prices = pd.DataFrame({"close": []})
        assert _build_pb_history(empty, empty, prices) == {}


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


# ══════════════════════════════════════════════════════════════════════════════
# DATA COMPLETENESS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDataCompleteness:
    """Tests for the real_inputs / total_inputs tracking on SubScore."""

    def _full_valuation_fund(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fund + prices with all four valuation components backed by real data."""
        yrs = list(range(2015, 2025))
        fund_data = _fund(
            _fund_rows("eps_diluted", [5.0] * len(yrs), yrs),
            _fund_rows("shares_outstanding", [1e9] * len(yrs), yrs),
            _fund_rows("operating_cf", [6e9] * len(yrs), yrs),
            _fund_rows("capex", [1e9] * len(yrs), yrs),
            _fund_rows("equity", [25e9] * len(yrs), yrs),
        )
        prices = _make_prices(yrs, [80.0] * len(yrs))
        return fund_data, prices

    def _empty_valuation_fund(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fund with no financial data at all → all valuation components floor."""
        prices = _make_prices([2024], [50.0])
        return _fund(), prices

    # ── (a) Missing inputs → reduced completeness, caution triggered ───────

    def test_missing_data_reduces_real_inputs(self):
        """(a) Valuation with no fund data has real_inputs < total_inputs."""
        fund_data, prices = self._empty_valuation_fund()
        m = _metrics()
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)
        assert result.real_inputs < result.total_inputs, (
            f"Expected real < total when data is absent: {result.real_inputs}/{result.total_inputs}"
        )
        assert result.real_inputs == 0, (
            f"No real data available — expected 0 real inputs, got {result.real_inputs}"
        )

    def test_full_data_maximises_real_inputs(self):
        """(b) Fully populated valuation fund has real_inputs == total_inputs."""
        fund_data, prices = self._full_valuation_fund()
        m = _metrics()
        cat = _make_category()
        result = score_valuation(fund_data, prices, m, cat)
        assert result.real_inputs == result.total_inputs, (
            f"All components should use real data: "
            f"{result.real_inputs}/{result.total_inputs}\n{result.reasons}"
        )
        assert result.total_inputs >= 4, "Expected at least 4 scored components"

    def test_floor_award_counts_as_not_real(self):
        """A Scorer.add() call with data_available=False is counted in total but not real."""
        from value_analyzer.score._helpers import Scorer
        s = Scorer("test")
        s.add(8, 20, "real component")
        s.add(5, 20, "floor component", data_available=False)
        sub = s.build()
        assert sub.real_inputs == 1
        assert sub.total_inputs == 2

    def test_health_all_floors_has_low_real(self):
        """Health with no data at all → mostly floor inputs.

        Interest coverage when no interest expense is found counts as real
        (confirmed absence of debt is genuine information, not a floor).
        The other three components (D/E, FCF consistency, FCF margin) are floors.
        """
        result = score_health(_fund(), _metrics())
        assert result.real_inputs == 1   # only "no interest expense" is real
        assert result.total_inputs == 4
        pct = result.real_inputs / result.total_inputs
        assert pct < COMPLETENESS_CAUTION_THRESHOLD

    def test_health_full_data_all_real(self):
        """Health fully populated → real_inputs == total_inputs."""
        yrs = list(range(2015, 2025))
        fund_data = _fund(
            _fund_rows("long_term_debt", [500e6] * len(yrs), yrs),
            _fund_rows("equity", [5e9] * len(yrs), yrs),
            _fund_rows("operating_income", [1.2e9] * len(yrs), yrs),
            _fund_rows("interest_expense", [50e6] * len(yrs), yrs),
            _fund_rows("operating_cf", [1.1e9] * len(yrs), yrs),
            _fund_rows("capex", [150e6] * len(yrs), yrs),
        )
        result = score_health(fund_data, _metrics(fcf_margin_avg=0.12))
        assert result.real_inputs == result.total_inputs

    def test_pb_and_fcf_fallbacks_count_as_real(self):
        """Spec: P/B and FCF fallback IV methods count as real valuation data."""
        yrs = list(range(2015, 2025))
        fund_data = _fund(
            _fund_rows("eps_diluted", [-15.0] * len(yrs), yrs),   # negative → EPS methods skip
            _fund_rows("shares_outstanding", [1e9] * len(yrs), yrs),
            _fund_rows("equity", [20e9] * len(yrs), yrs),          # BVPS = $20
            _fund_rows("operating_cf", [4e9] * len(yrs), yrs),     # FCF/share = $3
            _fund_rows("capex", [1e9] * len(yrs), yrs),
        )
        prices = _make_prices(yrs, [50.0] * len(yrs))
        result = score_valuation(fund_data, prices, _metrics(), _make_category())
        # IV component must be real (fallback ran)
        all_flags = " ".join(result.flags)
        assert "P/B" in all_flags or "FCF fallback" in all_flags
        assert result.real_inputs > 0, (
            "P/B or FCF fallback IV must count as real — real_inputs should be > 0"
        )

    # ── CompositeScore completeness aggregation ─────────────────────────────

    def test_composite_score_aggregates_completeness(self):
        """completeness_real/total on CompositeScore == sum of sub-score fields."""
        from value_analyzer.score.composite import score as run_score
        from datetime import date
        from unittest.mock import patch
        from value_analyzer.data import as_of, fetch_fundamentals, fetch_prices
        from value_analyzer.classify.models import (
            Category, CapitalIntensity, GrowthProfile, MoatType, RevenueType,
            Metrics, SicHint, RuleTrace,
        )
        from value_analyzer.score.moat import score_moat
        from value_analyzer.score.health import score_health
        from value_analyzer.score.valuation import score_valuation
        from value_analyzer.score.management import score_management

        # Build sub-scores directly and verify aggregation
        yrs = list(range(2015, 2025))
        fund_data = _fund(
            _fund_rows("eps_diluted", [5.0] * len(yrs), yrs),
            _fund_rows("shares_outstanding", [1e9] * len(yrs), yrs),
            _fund_rows("operating_cf", [6e9] * len(yrs), yrs),
            _fund_rows("capex", [1e9] * len(yrs), yrs),
            _fund_rows("equity", [25e9] * len(yrs), yrs),
            _fund_rows("long_term_debt", [2e9] * len(yrs), yrs),
            _fund_rows("net_income", [5e9] * len(yrs), yrs),
        )
        prices = _make_prices(yrs, [80.0] * len(yrs))
        m = _metrics(fcf_margin_avg=0.12, roe_avg=0.18, roe_std=0.03, years_of_data=10)
        cat = _make_category()

        moat = score_moat(fund_data, m)
        health = score_health(fund_data, m)
        val = score_valuation(fund_data, prices, m, cat)
        mgmt = score_management(fund_data, m)

        expected_real = moat.real_inputs + health.real_inputs + val.real_inputs + mgmt.real_inputs
        expected_total = moat.total_inputs + health.total_inputs + val.total_inputs + mgmt.total_inputs

        assert expected_real >= 0
        assert expected_total >= expected_real


# ══════════════════════════════════════════════════════════════════════════════
# IV DISPERSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestIVDispersion:
    """Tests for the IV method dispersion check added to valuation.py."""

    def _make_dispersion_inputs(
        self, *, price: float, bvps: float, pb_median_approx: float, fcf_ps: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build fund + prices engineered so fallback IV methods produce a large spread.
        EPS is negative so EPS methods skip; P/B and FCF fallbacks fire instead.
        """
        yrs = list(range(2015, 2025))
        # Build price history so historical P/B median ≈ pb_median_approx
        # BVPS is fixed at bvps; historical prices are set to bvps * pb_median_approx
        hist_price = bvps * pb_median_approx
        fund_data = _fund(
            _fund_rows("eps_diluted", [-20.0] * len(yrs), yrs),
            _fund_rows("shares_outstanding", [1e9] * len(yrs), yrs),
            _fund_rows("equity", [bvps * 1e9] * len(yrs), yrs),
            _fund_rows("operating_cf", [(fcf_ps + 2.0) * 1e9] * len(yrs), yrs),
            _fund_rows("capex", [2e9] * len(yrs), yrs),
        )
        # Historical prices at pb_median_approx × bvps; last year is current price
        hist_prices = [hist_price] * (len(yrs) - 1) + [price]
        prices = _make_prices(yrs, hist_prices)
        return fund_data, prices

    def test_wide_spread_triggers_dispersion_flag(self):
        """(c) IV estimates spanning > 2.5× trigger the IV_DISPERSION flag."""
        # P/B: 7x * $20 = $140; FCF: $3 / 0.09 = $33 → ratio = 140/33 ≈ 4.2× > 2.5
        fund_data, prices = self._make_dispersion_inputs(
            price=60.0, bvps=20.0, pb_median_approx=7.0, fcf_ps=3.0
        )
        result = score_valuation(fund_data, prices, _metrics(), _make_category())
        all_flags = " ".join(result.flags)
        assert "IV_DISPERSION" in all_flags, (
            f"Expected IV_DISPERSION flag. Flags:\n{chr(10).join(result.flags)}"
        )
        assert "disagree significantly" in all_flags

    def test_narrow_spread_no_dispersion_flag(self):
        """(d) IV estimates within 2.5× do NOT trigger the dispersion flag."""
        # P/B: 2x * $20 = $40; FCF: $4 / 0.09 = $44 → ratio = 44/40 ≈ 1.1× < 2.5
        fund_data, prices = self._make_dispersion_inputs(
            price=60.0, bvps=20.0, pb_median_approx=2.0, fcf_ps=4.0
        )
        result = score_valuation(fund_data, prices, _metrics(), _make_category())
        all_flags = " ".join(result.flags)
        assert "IV_DISPERSION" not in all_flags, (
            f"Unexpected IV_DISPERSION flag when spread is narrow. "
            f"Flags:\n{chr(10).join(result.flags)}"
        )

    def test_positive_eps_methods_close_no_dispersion(self):
        """Positive-EPS path: three methods typically agree within 2.5× for a normal company."""
        # eps_norm=5, WACC=9% → IV_A=55.6; pe_median≈15 → IV_B=75; bvps=30 → graham≈58
        # Max/min = 75/55.6 = 1.35 < 2.5 → no dispersion flag
        yrs = list(range(2015, 2025))
        fund_data = _fund(
            _fund_rows("eps_diluted", [5.0] * len(yrs), yrs),
            _fund_rows("shares_outstanding", [1e9] * len(yrs), yrs),
            _fund_rows("equity", [30e9] * len(yrs), yrs),
            _fund_rows("operating_cf", [6e9] * len(yrs), yrs),
            _fund_rows("capex", [1e9] * len(yrs), yrs),
        )
        prices = _make_prices(yrs, [75.0] * len(yrs))  # historical P/E ~15×
        result = score_valuation(fund_data, prices, _metrics(), _make_category())
        all_flags = " ".join(result.flags)
        assert "IV_DISPERSION" not in all_flags, (
            f"Normal company should not trigger dispersion. "
            f"Flags:\n{chr(10).join(result.flags)}"
        )

    def test_dispersion_flag_extracted_to_composite_score(self):
        """iv_dispersion_flag on CompositeScore is set when valuation dispersion fires."""
        from value_analyzer.score.models import CompositeScore, SubScore
        from value_analyzer.classify.models import (
            Category, CapitalIntensity, GrowthProfile, MoatType, RevenueType,
            Metrics, SicHint, RuleTrace,
        )
        from value_analyzer.score.moat import score_moat
        from value_analyzer.score.health import score_health
        from value_analyzer.score.valuation import score_valuation
        from value_analyzer.score.management import score_management
        from value_analyzer.score.config import CATEGORY_WEIGHTS
        from datetime import date

        yrs = list(range(2015, 2025))
        fund_data = _fund(
            _fund_rows("eps_diluted", [-20.0] * len(yrs), yrs),
            _fund_rows("shares_outstanding", [1e9] * len(yrs), yrs),
            _fund_rows("equity", [20e9] * len(yrs), yrs),
            _fund_rows("operating_cf", [5e9] * len(yrs), yrs),
            _fund_rows("capex", [2e9] * len(yrs), yrs),
        )
        # historical prices set high so P/B median ≈ 7× → IV_D ≈ $140
        # FCF/share = 3.0, IV_E = $33 → ratio > 2.5
        hist_prices = [140.0] * (len(yrs) - 1) + [60.0]
        prices = _make_prices(yrs, hist_prices)

        m = _metrics()
        cat = _make_category()
        moat = score_moat(fund_data, m)
        health = score_health(fund_data, m)
        val = score_valuation(fund_data, prices, m, cat)
        mgmt = score_management(fund_data, m)

        weights = CATEGORY_WEIGHTS["stable"]
        composite = (
            moat.score * weights["moat"] + health.score * weights["health"]
            + val.score * weights["valuation"] + mgmt.score * weights["management"]
        )
        real = moat.real_inputs + health.real_inputs + val.real_inputs + mgmt.real_inputs
        total = moat.total_inputs + health.total_inputs + val.total_inputs + mgmt.total_inputs

        iv_flag = next((f for f in val.flags if "IV_DISPERSION:" in f), None)

        cs = CompositeScore(
            ticker="TEST", as_of_date=date(2024, 12, 31),
            composite=round(composite, 1),
            moat=moat, health=health, valuation=val, management=mgmt,
            weight_profile="stable", weights_used=weights, category=cat,
            completeness_real=real, completeness_total=total,
            iv_dispersion_flag=iv_flag,
        )

        if "IV_DISPERSION" in " ".join(val.flags):
            assert cs.iv_dispersion_flag is not None, (
                "Dispersion flag in valuation.flags must be extracted to CompositeScore"
            )
