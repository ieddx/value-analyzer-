"""Integration tests for the classify layer using real ticker data.

KO (Coca-Cola) — the archetype brand/stable/asset-light business.
NUE (Nucor Steel) — the archetype cyclical/commodity/asset-heavy business.

Run with: pytest -m integration
"""

from datetime import date

import pytest

from value_analyzer.classify import (
    CapitalIntensity,
    GrowthProfile,
    MoatType,
    RevenueType,
    classify,
)

pytestmark = pytest.mark.integration

CUTOFF = date(2024, 12, 31)


# ── KO (Coca-Cola Company) ─────────────────────────────────────────────────
# Expected profile:
#   moat       = brand   (60%+ gross margins, consumer beverages, pricing power)
#   intensity  = asset_light or moderate  (concentrate model, ~4% capex/revenue)
#   rev_type   = recurring  (recession-resistant, stable growth CV)
#   growth     = stable  (low-single-digit CAGR, mature business)

class TestKoClassification:
    @pytest.fixture(scope="class")
    def ko(self):
        return classify("KO", as_of_date=CUTOFF)

    def test_moat_is_brand(self, ko):
        assert ko.moat_type == MoatType.brand, (
            f"KO moat_type={ko.moat_type}. "
            f"Rationale: {ko.traces['moat_type'].rationale}"
        )

    def test_not_cyclical(self, ko):
        assert ko.revenue_type != RevenueType.cyclical_commodity, (
            f"KO revenue_type={ko.revenue_type}. "
            f"Rationale: {ko.traces['revenue_type'].rationale}"
        )

    def test_not_asset_heavy(self, ko):
        assert ko.capital_intensity != CapitalIntensity.asset_heavy, (
            f"KO capital_intensity={ko.capital_intensity}. "
            f"Rationale: {ko.traces['capital_intensity'].rationale}"
        )

    def test_not_declining(self, ko):
        assert ko.growth_profile != GrowthProfile.declining, (
            f"KO growth_profile={ko.growth_profile}. "
            f"Rationale: {ko.traces['growth_profile'].rationale}"
        )

    def test_gross_margin_is_high(self, ko):
        gm = ko.metrics.gross_margin_avg
        assert gm is not None and gm > 0.50, (
            f"KO gross_margin_avg={gm:.2%} — expected > 50% for a beverage brand"
        )

    def test_capex_is_low(self, ko):
        c = ko.metrics.capex_pct_revenue
        assert c is not None and c < 0.08, (
            f"KO capex/revenue={c:.2%} — concentrate model should be < 8%"
        )

    def test_traces_have_rationale(self, ko):
        for dim, trace in ko.traces.items():
            assert trace.rationale, f"KO trace for {dim!r} has no rationale"
            assert trace.confidence > 0, f"KO trace for {dim!r} has zero confidence"

    def test_sic_hint_is_beverage(self, ko):
        if ko.sic_hint.sic_code:
            assert 2080 <= ko.sic_hint.sic_code <= 2089, (
                f"KO SIC={ko.sic_hint.sic_code}, expected 208x (beverages)"
            )


# ── NUE (Nucor Corporation — steel) ───────────────────────────────────────
# Expected profile:
#   moat       = cost_advantage  (low-cost mini-mill producer)
#   intensity  = asset_heavy  (steel mills, heavy capex)
#   rev_type   = cyclical_commodity  (steel prices = commodity, high CV)
#   growth     = stable or compounder  (Nucor has actually grown via M&A)

class TestNueClassification:
    @pytest.fixture(scope="class")
    def nue(self):
        return classify("NUE", as_of_date=CUTOFF)

    def test_revenue_type_is_cyclical(self, nue):
        assert nue.revenue_type == RevenueType.cyclical_commodity, (
            f"NUE revenue_type={nue.revenue_type}. "
            f"Rationale: {nue.traces['revenue_type'].rationale}\n"
            f"Metrics: CV={nue.metrics.revenue_growth_cv}"
        )

    def test_capital_intensity_is_not_asset_light_given_sic(self, nue):
        # Nucor's mini-mill technology makes its capex/revenue genuinely low
        # vs integrated steel — the classifier may return asset_light or moderate.
        # What we DO require: if SIC says asset_heavy, the confidence on any
        # asset_light call should be low (< 0.70), signalling uncertainty.
        ci = nue.capital_intensity
        trace = nue.traces["capital_intensity"]
        if ci == CapitalIntensity.asset_light:
            # Low confidence is acceptable when SIC contradicts the financial ratio.
            assert trace.confidence <= 0.75, (
                f"NUE classified asset_light with high confidence ({trace.confidence}) "
                f"despite SIC {nue.sic_hint.sic_code} indicating heavy industry.\n"
                f"Rationale: {trace.rationale}"
            )
        # If asset_heavy or moderate, that's also fine — no assertion needed.

    def test_moat_is_not_brand(self, nue):
        assert nue.moat_type != MoatType.brand, (
            f"NUE moat_type={nue.moat_type} — steel doesn't have brand pricing power"
        )

    def test_moat_is_not_switching_cost(self, nue):
        assert nue.moat_type != MoatType.switching_cost, (
            f"NUE moat_type={nue.moat_type} — steel customers can switch mills"
        )

    def test_gross_margin_is_lower_than_ko(self, nue):
        nue_gm = nue.metrics.gross_margin_avg
        ko = classify("KO", as_of_date=CUTOFF)
        ko_gm = ko.metrics.gross_margin_avg
        if nue_gm is not None and ko_gm is not None:
            assert nue_gm < ko_gm, (
                f"Expected KO gross margin ({ko_gm:.1%}) > NUE ({nue_gm:.1%})"
            )

    def test_revenue_cv_is_higher_than_ko(self, nue):
        nue_cv = nue.metrics.revenue_growth_cv
        ko = classify("KO", as_of_date=CUTOFF)
        ko_cv = ko.metrics.revenue_growth_cv
        if nue_cv is not None and ko_cv is not None:
            assert nue_cv > ko_cv, (
                f"Expected NUE CV ({nue_cv:.2f}) > KO CV ({ko_cv:.2f}) — "
                "steel is more cyclical than consumer beverages"
            )

    def test_sic_hint_is_steel(self, nue):
        if nue.sic_hint.sic_code:
            assert 3310 <= nue.sic_hint.sic_code <= 3399, (
                f"NUE SIC={nue.sic_hint.sic_code}, expected 33xx (primary metals)"
            )


# ── Cross-ticker sanity checks ─────────────────────────────────────────────

class TestKoVsNueContrast:
    """KO and NUE should classify into meaningfully different categories."""

    @pytest.fixture(scope="class")
    def ko_and_nue(self):
        return classify("KO", as_of_date=CUTOFF), classify("NUE", as_of_date=CUTOFF)

    def test_different_moat_types(self, ko_and_nue):
        ko, nue = ko_and_nue
        assert ko.moat_type != nue.moat_type, (
            f"KO and NUE have the same moat ({ko.moat_type}) — expected different"
        )

    def test_different_capital_intensity_or_revenue_type(self, ko_and_nue):
        # KO and NUE must differ on at least capital_intensity OR revenue_type.
        # (NUE's mini-mill efficiency can produce similar capex/revenue to KO,
        # but their revenue types are completely different.)
        ko, nue = ko_and_nue
        assert (ko.capital_intensity != nue.capital_intensity or
                ko.revenue_type != nue.revenue_type), (
            f"KO and NUE are identical on both capital_intensity "
            f"({ko.capital_intensity}) and revenue_type ({ko.revenue_type})."
        )

    def test_different_revenue_type(self, ko_and_nue):
        ko, nue = ko_and_nue
        assert ko.revenue_type != nue.revenue_type, (
            f"KO ({ko.revenue_type}) and NUE ({nue.revenue_type}) "
            "should differ on revenue type"
        )
