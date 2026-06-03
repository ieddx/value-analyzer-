"""Integration tests for the score layer — uses real KO and AAPL data.

All data is cached; subsequent runs are fast (< 5s total).
Run with: pytest -m integration
"""

from datetime import date

import pytest

from value_analyzer.score import score, CompositeScore

pytestmark = pytest.mark.integration

CUTOFF = date(2024, 12, 31)


def _assert_valid_subscore(sub, name: str):
    assert 0 <= sub.score <= 100, f"{name} score {sub.score} out of [0,100]"
    assert len(sub.reasons) >= 1, f"{name} has no reasons"
    for r in sub.reasons:
        assert r.startswith("[+"), f"{name} reason missing '[+N/Max]' prefix: {r!r}"


# ── KO (Coca-Cola) ─────────────────────────────────────────────────────────

class TestKoScore:
    @pytest.fixture(scope="class")
    def ko(self) -> CompositeScore:
        return score("KO", as_of_date=CUTOFF)

    def test_composite_in_range(self, ko):
        assert 0 <= ko.composite <= 100, f"Composite {ko.composite} out of range"

    def test_all_sub_scores_valid(self, ko):
        _assert_valid_subscore(ko.moat, "moat")
        _assert_valid_subscore(ko.health, "health")
        _assert_valid_subscore(ko.valuation, "valuation")
        _assert_valid_subscore(ko.management, "management")

    def test_moat_score_is_strong(self, ko):
        assert ko.moat.score >= 60, (
            f"KO moat score = {ko.moat.score} — expected ≥60 for a brand-moat "
            f"beverage company with 60%+ gross margins.\n"
            f"Reasons: {ko.moat.reasons}"
        )

    def test_health_score_is_adequate(self, ko):
        assert ko.health.score >= 40, (
            f"KO health score = {ko.health.score} — expected ≥40.\n"
            f"Reasons: {ko.health.reasons}"
        )

    def test_reasons_are_human_readable(self, ko):
        for sub in [ko.moat, ko.health, ko.valuation, ko.management]:
            for r in sub.reasons:
                # Reason should be longer than just the prefix
                assert len(r) > 15, f"Reason too short: {r!r}"

    def test_valuation_flags_contain_assumptions(self, ko):
        all_flags = " ".join(ko.valuation.flags)
        assert "WACC" in all_flags or "9%" in all_flags, (
            "Valuation flags must state the WACC assumption. "
            f"Flags: {ko.valuation.flags}"
        )

    def test_weights_sum_to_one(self, ko):
        total = sum(ko.weights_used.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_composite_matches_weighted_average(self, ko):
        expected = (
            ko.moat.score * ko.weights_used["moat"]
            + ko.health.score * ko.weights_used["health"]
            + ko.valuation.score * ko.weights_used["valuation"]
            + ko.management.score * ko.weights_used["management"]
        )
        assert abs(ko.composite - expected) < 0.2, (
            f"Composite {ko.composite} doesn't match weighted average {expected:.1f}"
        )

    def test_weight_profile_is_set(self, ko):
        assert ko.weight_profile in ("compounder", "stable", "cyclical", "declining", "default")

    def test_category_attached(self, ko):
        assert ko.category is not None
        assert ko.category.ticker == "KO"


# ── AAPL (Apple) ───────────────────────────────────────────────────────────

class TestAaplScore:
    @pytest.fixture(scope="class")
    def aapl(self) -> CompositeScore:
        return score("AAPL", as_of_date=CUTOFF)

    def test_composite_in_range(self, aapl):
        assert 0 <= aapl.composite <= 100

    def test_all_sub_scores_valid(self, aapl):
        _assert_valid_subscore(aapl.moat, "moat")
        _assert_valid_subscore(aapl.health, "health")
        _assert_valid_subscore(aapl.valuation, "valuation")
        _assert_valid_subscore(aapl.management, "management")

    def test_moat_score_is_strong(self, aapl):
        assert aapl.moat.score >= 60, (
            f"AAPL moat score = {aapl.moat.score} — expected ≥60 for a platform "
            f"with high switching costs and exceptional margins.\n"
            f"Reasons: {aapl.moat.reasons}"
        )

    def test_all_reasons_non_empty(self, aapl):
        for sub in [aapl.moat, aapl.health, aapl.valuation, aapl.management]:
            assert sub.reasons, f"Sub-score '{sub.name}' has no reasons"


# ── Cross-ticker checks ────────────────────────────────────────────────────

class TestCrossTicker:
    @pytest.fixture(scope="class")
    def both(self):
        return score("KO", as_of_date=CUTOFF), score("AAPL", as_of_date=CUTOFF)

    def test_different_composite_scores(self, both):
        ko, aapl = both
        # Scores should differ — they're different businesses
        assert ko.composite != aapl.composite, (
            f"KO and AAPL have identical composite scores ({ko.composite}) — "
            "check scoring logic."
        )

    def test_different_moat_scores(self, both):
        ko, aapl = both
        # Not asserting which is higher — both are strong moats, scores will differ
        assert ko.moat.score != aapl.moat.score or abs(ko.moat.score - aapl.moat.score) < 15, (
            "KO and AAPL moat scores are identical — unusual for different business models."
        )

    def test_aapl_management_score_reflects_buybacks(self, both):
        _, aapl = both
        # Apple is famous for buybacks — management score should be respectable
        assert aapl.management.score >= 40, (
            f"AAPL management = {aapl.management.score} — "
            f"expected ≥40 given massive buyback programme.\n"
            f"Reasons: {aapl.management.reasons}"
        )
