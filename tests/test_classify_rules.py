"""Unit tests for classify rules.  No network access — all inputs are synthetic."""

from datetime import date

import pytest

from value_analyzer.classify.models import (
    CapitalIntensity,
    GrowthProfile,
    Metrics,
    MoatType,
    RevenueType,
    SicHint,
)
from value_analyzer.classify.rules import (
    BRAND_GM_FLOOR,
    CAPEX_ASSET_HEAVY,
    CAPEX_ASSET_LIGHT,
    COMPOUNDER_CAGR,
    CYCLICAL_CV_THRESHOLD,
    HIGH_GM_FLOOR,
    LOW_CV_THRESHOLD,
    RECURRING_GM_FLOOR,
    ROIC_MOAT_CONFIRM,
    STABLE_CAGR_MIN,
    classify_capital_intensity,
    classify_growth_profile,
    classify_moat,
    classify_revenue_type,
    apply_all_rules,
)

TODAY = date(2024, 12, 31)
NO_SIC = SicHint()


def _metrics(**kwargs) -> Metrics:
    defaults = dict(ticker="TEST", as_of_date=TODAY, years_of_data=7)
    defaults.update(kwargs)
    return Metrics(**defaults)


# ── Capital intensity ──────────────────────────────────────────────────────

class TestCapitalIntensity:
    def test_clearly_asset_light(self):
        m = _metrics(capex_pct_revenue=0.02)
        result, conf, rationale = classify_capital_intensity(m, NO_SIC)
        assert result == CapitalIntensity.asset_light
        assert conf >= 0.60
        assert "2.0%" in rationale or "<" in rationale

    def test_clearly_asset_heavy(self):
        m = _metrics(capex_pct_revenue=0.18)
        result, conf, _ = classify_capital_intensity(m, NO_SIC)
        assert result == CapitalIntensity.asset_heavy
        assert conf >= 0.60

    def test_borderline_moderate_no_sic(self):
        m = _metrics(capex_pct_revenue=0.07)
        result, _, _ = classify_capital_intensity(m, NO_SIC)
        assert result == CapitalIntensity.moderate

    def test_borderline_sic_pushes_to_heavy(self):
        m = _metrics(capex_pct_revenue=0.07)
        sic = SicHint(sic_code=4512, sic_description="Air Transportation",
                      capital_intensity=CapitalIntensity.asset_heavy)
        result, conf, rationale = classify_capital_intensity(m, sic)
        assert result == CapitalIntensity.asset_heavy
        assert conf >= 0.55
        assert "4512" in rationale

    def test_borderline_sic_pushes_to_light(self):
        m = _metrics(capex_pct_revenue=0.07)
        sic = SicHint(sic_code=7375, sic_description="Computer Processing",
                      capital_intensity=CapitalIntensity.asset_light)
        result, _, _ = classify_capital_intensity(m, sic)
        assert result == CapitalIntensity.asset_light

    def test_no_capex_uses_sic(self):
        sic = SicHint(sic_code=4911, sic_description="Electric Services",
                      capital_intensity=CapitalIntensity.asset_heavy)
        result, conf, _ = classify_capital_intensity(_metrics(), sic)
        assert result == CapitalIntensity.asset_heavy
        assert conf < 0.60  # lower confidence without financial data

    def test_threshold_boundary_exactly_at_light(self):
        m = _metrics(capex_pct_revenue=CAPEX_ASSET_LIGHT)
        # At exactly the threshold it falls into moderate (not strictly <)
        result, _, _ = classify_capital_intensity(m, NO_SIC)
        assert result == CapitalIntensity.moderate

    def test_threshold_just_below_light(self):
        m = _metrics(capex_pct_revenue=CAPEX_ASSET_LIGHT - 0.001)
        result, _, _ = classify_capital_intensity(m, NO_SIC)
        assert result == CapitalIntensity.asset_light


# ── Revenue type ───────────────────────────────────────────────────────────

class TestRevenueType:
    def test_high_cv_is_cyclical(self):
        m = _metrics(revenue_growth_cv=0.40, gross_margin_avg=0.15)
        result, conf, rationale = classify_revenue_type(m, NO_SIC)
        assert result == RevenueType.cyclical_commodity
        assert conf >= 0.55
        assert "CV" in rationale or "cv" in rationale.lower()

    def test_low_cv_high_gm_is_recurring(self):
        m = _metrics(revenue_growth_cv=0.08, gross_margin_avg=0.60)
        result, _, _ = classify_revenue_type(m, NO_SIC)
        assert result == RevenueType.recurring

    def test_high_gm_plus_sic_recurring(self):
        m = _metrics(revenue_growth_cv=None, gross_margin_avg=0.55)
        sic = SicHint(sic_code=2086, sic_description="Bottled Beverages",
                      revenue_type=RevenueType.recurring)
        result, _, _ = classify_revenue_type(m, sic)
        assert result == RevenueType.recurring

    def test_low_gm_defaults_transactional(self):
        m = _metrics(revenue_growth_cv=0.10, gross_margin_avg=0.20)
        result, _, _ = classify_revenue_type(m, NO_SIC)
        assert result == RevenueType.transactional

    def test_cyclical_sic_without_contradicting_data(self):
        m = _metrics(revenue_growth_cv=0.18, gross_margin_avg=0.10)
        sic = SicHint(sic_code=3312, sic_description="Steel Works",
                      revenue_type=RevenueType.cyclical_commodity)
        result, _, _ = classify_revenue_type(m, sic)
        assert result == RevenueType.cyclical_commodity

    def test_cyclical_cv_at_exact_threshold(self):
        m = _metrics(revenue_growth_cv=CYCLICAL_CV_THRESHOLD, gross_margin_avg=0.15)
        # Rule uses >=, so exactly at threshold fires cyclical.
        result, _, _ = classify_revenue_type(m, NO_SIC)
        assert result == RevenueType.cyclical_commodity

    def test_just_below_cyclical_threshold_is_not_cyclical(self):
        m = _metrics(revenue_growth_cv=CYCLICAL_CV_THRESHOLD - 0.01, gross_margin_avg=0.15)
        result, _, _ = classify_revenue_type(m, NO_SIC)
        assert result != RevenueType.cyclical_commodity


# ── Moat type ──────────────────────────────────────────────────────────────

class TestMoatType:
    def test_high_gm_brand_sic_is_brand(self):
        m = _metrics(gross_margin_avg=0.60, roic_avg=0.20)
        sic = SicHint(sic_code=2086, sic_description="Bottled Beverages",
                      moat_type=MoatType.brand)
        result, conf, rationale = classify_moat(m, sic)
        assert result == MoatType.brand
        assert conf >= 0.75
        assert "60" in rationale or "brand" in rationale.lower()

    def test_very_high_gm_switching_cost(self):
        m = _metrics(gross_margin_avg=0.72, roic_avg=0.25)
        sic = SicHint(sic_code=7372, sic_description="Prepackaged Software",
                      moat_type=MoatType.switching_cost)
        result, conf, _ = classify_moat(m, sic)
        assert result == MoatType.switching_cost
        assert conf >= 0.75

    def test_cost_advantage_with_roic(self):
        m = _metrics(gross_margin_avg=0.15, roic_avg=0.18)
        sic = SicHint(sic_code=3312, sic_description="Steel Works",
                      moat_type=MoatType.cost_advantage)
        result, conf, _ = classify_moat(m, sic)
        assert result == MoatType.cost_advantage
        assert conf >= 0.60

    def test_cost_advantage_without_roic_lower_conf(self):
        m = _metrics(gross_margin_avg=0.15, roic_avg=0.08)
        sic = SicHint(sic_code=3312, moat_type=MoatType.cost_advantage)
        result, conf, _ = classify_moat(m, sic)
        assert result == MoatType.cost_advantage
        assert conf < 0.60  # ROIC doesn't confirm

    def test_low_gm_no_sic_is_none(self):
        m = _metrics(gross_margin_avg=0.25, roic_avg=0.08)
        result, _, _ = classify_moat(m, NO_SIC)
        assert result == MoatType.none

    def test_network_moat_from_sic(self):
        m = _metrics(gross_margin_avg=0.50)
        sic = SicHint(sic_code=4813, sic_description="Telephone Communications",
                      moat_type=MoatType.network)
        result, _, _ = classify_moat(m, sic)
        assert result == MoatType.network

    def test_brand_requires_gm_above_floor(self):
        m = _metrics(gross_margin_avg=BRAND_GM_FLOOR - 0.01, roic_avg=0.05)
        sic = SicHint(moat_type=MoatType.brand)
        result, _, _ = classify_moat(m, sic)
        # Below brand GM floor — should fall through to none or cost_advantage
        assert result != MoatType.brand


# ── Growth profile ─────────────────────────────────────────────────────────

class TestGrowthProfile:
    def test_high_cagr_is_compounder(self):
        m = _metrics(revenue_cagr=0.12, years_of_data=7)
        result, conf, rationale = classify_growth_profile(m, NO_SIC)
        assert result == GrowthProfile.compounder
        assert conf >= 0.60
        assert "+12" in rationale or "12" in rationale

    def test_moderate_cagr_is_stable(self):
        m = _metrics(revenue_cagr=0.04, years_of_data=7)
        result, _, _ = classify_growth_profile(m, NO_SIC)
        assert result == GrowthProfile.stable

    def test_negative_cagr_is_declining(self):
        m = _metrics(revenue_cagr=-0.05, years_of_data=7)
        result, conf, _ = classify_growth_profile(m, NO_SIC)
        assert result == GrowthProfile.declining
        assert conf >= 0.55

    def test_borderline_stable_upper(self):
        m = _metrics(revenue_cagr=COMPOUNDER_CAGR - 0.001, years_of_data=7)
        result, _, _ = classify_growth_profile(m, NO_SIC)
        assert result == GrowthProfile.stable

    def test_exactly_at_compounder_threshold(self):
        m = _metrics(revenue_cagr=COMPOUNDER_CAGR + 0.001, years_of_data=7)
        result, _, _ = classify_growth_profile(m, NO_SIC)
        assert result == GrowthProfile.compounder

    def test_limited_data_caps_confidence(self):
        m = _metrics(revenue_cagr=0.15, years_of_data=2)
        _, conf, rationale = classify_growth_profile(m, NO_SIC)
        assert conf <= 0.40
        assert "caution" in rationale or "limited" in rationale.lower()

    def test_no_cagr_defaults_stable_low_conf(self):
        m = _metrics(revenue_cagr=None, years_of_data=1)
        result, conf, _ = classify_growth_profile(m, NO_SIC)
        assert result == GrowthProfile.stable
        assert conf <= 0.20


# ── apply_all_rules ────────────────────────────────────────────────────────

class TestApplyAllRules:
    def test_returns_all_four_dimensions(self):
        m = _metrics(
            gross_margin_avg=0.60,
            capex_pct_revenue=0.03,
            revenue_growth_cv=0.08,
            revenue_cagr=0.04,
            roic_avg=0.18,
        )
        sic = SicHint(sic_code=2086, sic_description="Beverages",
                      capital_intensity=CapitalIntensity.asset_light,
                      revenue_type=RevenueType.recurring,
                      moat_type=MoatType.brand)
        traces = apply_all_rules(m, sic)
        assert set(traces.keys()) == {"capital_intensity", "revenue_type", "moat_type", "growth_profile"}
        for trace in traces.values():
            assert trace.rationale
            assert 0.0 <= trace.confidence <= 1.0
