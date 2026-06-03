"""Unit tests for the peers layer and its wiring into score/valuation.

All tests are purely synthetic — no network access, no disk I/O.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from value_analyzer.classify.models import (
    CapitalIntensity, Category, GrowthProfile, Metrics,
    MoatType, RevenueType, RuleTrace, SicHint,
)
from value_analyzer.peers.models import CategoryPeerStats, PeerSnapshot
from value_analyzer.peers.registry import _aggregate, build_peer_comparison, get_peer_stats
from value_analyzer.score.valuation import score_valuation

TODAY = date(2024, 12, 31)


# ── Synthetic data factories ───────────────────────────────────────────────

def _metrics(**kwargs) -> Metrics:
    defaults = dict(ticker="TEST", as_of_date=TODAY, years_of_data=10)
    defaults.update(kwargs)
    return Metrics(**defaults)


def _make_trace(result: str = "brand") -> RuleTrace:
    return RuleTrace(rule_name="test", result=result, confidence=0.8, rationale="test")


def _make_category(
    moat: MoatType = MoatType.brand,
    intensity: CapitalIntensity = CapitalIntensity.asset_light,
    rev_type: RevenueType = RevenueType.recurring,
    growth: GrowthProfile = GrowthProfile.stable,
) -> Category:
    return Category(
        ticker="TEST", as_of_date=TODAY,
        capital_intensity=intensity, revenue_type=rev_type,
        moat_type=moat, growth_profile=growth,
        traces={
            "capital_intensity": _make_trace(intensity.value),
            "revenue_type": _make_trace(rev_type.value),
            "moat_type": _make_trace(moat.value),
            "growth_profile": _make_trace(growth.value),
        },
        metrics=_metrics(), sic_hint=SicHint(),
    )


def _fund_rows(concept: str, values: list[float], years: list[int]) -> pd.DataFrame:
    rows = []
    for year, val in zip(years, values):
        rows.append({
            "concept": concept,
            "period_end": pd.Timestamp(f"{year}-12-31"),
            "filed": pd.Timestamp(f"{year + 1}-02-01"),
            "value": val,
            "form": "10-K",
            "source": "test",
            "unit": "USD",
        })
    return pd.DataFrame(rows)


def _make_fund_and_prices():
    yrs = list(range(2015, 2025))
    shares = 1e9
    fund = pd.concat([
        _fund_rows("eps_diluted", [5.0] * len(yrs), yrs),
        _fund_rows("shares_outstanding", [shares] * len(yrs), yrs),
        _fund_rows("operating_cf", [4.0 * shares] * len(yrs), yrs),
        _fund_rows("capex", [0.4 * shares] * len(yrs), yrs),
        _fund_rows("equity", [shares * 15] * len(yrs), yrs),
    ], ignore_index=True)
    idx = pd.DatetimeIndex([pd.Timestamp(f"{y}-12-31") for y in yrs])
    prices = pd.DataFrame({"close": [75.0] * len(yrs)}, index=idx)
    return fund, prices


def _make_snapshots() -> list[PeerSnapshot]:
    """Four snapshots: two compounders and two cyclicals."""
    return [
        PeerSnapshot(ticker="KO",  weight_profile="compounder", as_of_date=TODAY,
                     pe_median_10y=22.0, pfcf_median_10y=20.0,
                     gross_margin_avg=0.60, roic_avg=0.18),
        PeerSnapshot(ticker="AXP", weight_profile="compounder", as_of_date=TODAY,
                     pe_median_10y=18.0, pfcf_median_10y=16.0,
                     gross_margin_avg=0.55, roic_avg=0.25),
        PeerSnapshot(ticker="CVX", weight_profile="cyclical", as_of_date=TODAY,
                     pe_median_10y=12.0, pfcf_median_10y=10.0,
                     gross_margin_avg=0.32, roic_avg=0.09),
        PeerSnapshot(ticker="OXY", weight_profile="cyclical", as_of_date=TODAY,
                     pe_median_10y=9.0, pfcf_median_10y=8.0,
                     gross_margin_avg=0.28, roic_avg=0.07),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY / AGGREGATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregate:
    def test_filters_by_weight_profile(self):
        snapshots = _make_snapshots()
        cyclical = _aggregate(snapshots, "cyclical", TODAY)
        compounder = _aggregate(snapshots, "compounder", TODAY)

        assert cyclical is not None
        assert compounder is not None
        assert set(cyclical.peer_tickers) == {"CVX", "OXY"}
        assert set(compounder.peer_tickers) == {"KO", "AXP"}

    def test_cyclical_pe_median_is_low(self):
        cyclical = _aggregate(_make_snapshots(), "cyclical", TODAY)
        # Median of [12, 9] = 10.5
        assert cyclical.pe_median == pytest.approx(10.5, abs=0.1)

    def test_compounder_pe_median_is_higher_than_cyclical(self):
        snapshots = _make_snapshots()
        cyclical = _aggregate(snapshots, "cyclical", TODAY)
        compounder = _aggregate(snapshots, "compounder", TODAY)
        assert compounder.pe_median > cyclical.pe_median, (
            f"Compounder P/E median ({compounder.pe_median}) should exceed "
            f"cyclical ({cyclical.pe_median})"
        )

    def test_returns_none_for_unknown_profile(self):
        assert _aggregate(_make_snapshots(), "nonexistent", TODAY) is None

    def test_returns_none_for_empty_snapshots(self):
        assert _aggregate([], "cyclical", TODAY) is None

    def test_quartiles_are_ordered(self):
        stats = _aggregate(_make_snapshots(), "cyclical", TODAY)
        if stats.pe_p25 is not None and stats.pe_p75 is not None:
            assert stats.pe_p25 <= stats.pe_median <= stats.pe_p75

    def test_gross_margin_median_computed(self):
        stats = _aggregate(_make_snapshots(), "compounder", TODAY)
        # Median of [0.60, 0.55] = 0.575
        assert stats.gross_margin_median == pytest.approx(0.575, abs=0.01)


class TestGetPeerStats:
    def test_injectable_snapshots_bypass_disk(self):
        snapshots = _make_snapshots()
        stats = get_peer_stats("cyclical", TODAY, snapshots=snapshots)
        assert stats is not None
        assert stats.weight_profile == "cyclical"
        assert set(stats.peer_tickers) == {"CVX", "OXY"}

    def test_missing_category_returns_none(self):
        snapshots = _make_snapshots()
        stats = get_peer_stats("declining", TODAY, snapshots=snapshots)
        assert stats is None

    def test_empty_registry_returns_none(self):
        stats = get_peer_stats("compounder", TODAY, snapshots=[])
        assert stats is None


# ══════════════════════════════════════════════════════════════════════════════
# PEER COMPARISON TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildPeerComparison:
    def test_returns_none_when_no_peer_stats(self):
        fund, prices = _make_fund_and_prices()
        result = build_peer_comparison(fund, prices, "stable", peer_stats=None)
        assert result is None

    def test_subject_pe_computed(self):
        fund, prices = _make_fund_and_prices()
        # Price=75, EPS=5 → P/E=15
        peer_stats = _aggregate(_make_snapshots(), "compounder", TODAY)
        result = build_peer_comparison(fund, prices, "compounder", peer_stats)
        assert result is not None
        assert result.subject_pe == pytest.approx(15.0, abs=0.1)

    def test_peer_tickers_propagated(self):
        fund, prices = _make_fund_and_prices()
        peer_stats = _aggregate(_make_snapshots(), "cyclical", TODAY)
        result = build_peer_comparison(fund, prices, "cyclical", peer_stats)
        assert result is not None
        assert set(result.peer_tickers) == {"CVX", "OXY"}

    def test_weight_profile_matches_category(self):
        fund, prices = _make_fund_and_prices()
        stats = _aggregate(_make_snapshots(), "compounder", TODAY)
        result = build_peer_comparison(fund, prices, "compounder", stats)
        assert result.weight_profile == "compounder"

    def test_context_note_mentions_value_investors(self):
        fund, prices = _make_fund_and_prices()
        stats = _aggregate(_make_snapshots(), "cyclical", TODAY)
        result = build_peer_comparison(fund, prices, "cyclical", stats)
        note = result.context_note.lower()
        assert "berkshire" in note or "value investor" in note

    def test_no_price_data_still_returns_comparison(self):
        fund, _ = _make_fund_and_prices()
        empty_prices = pd.DataFrame(columns=["close"])
        stats = _aggregate(_make_snapshots(), "compounder", TODAY)
        result = build_peer_comparison(fund, empty_prices, "compounder", stats)
        assert result is not None
        assert result.subject_pe is None


# ══════════════════════════════════════════════════════════════════════════════
# CYCLICAL vs COMPOUNDER PEER WIRING  (the core correctness test)
# ══════════════════════════════════════════════════════════════════════════════

class TestCyclicalGetsCyclicalPeers:
    """Verify that cyclical stocks are compared against cyclical peers, not compounders."""

    CYCLICAL_PEER_PE = 10.5   # median of [12, 9] in _make_snapshots()
    COMPOUNDER_PEER_PE = 20.0  # median of [22, 18] in _make_snapshots()

    def _score_with_profile(self, rev_type: RevenueType, profile: str):
        snapshots = _make_snapshots()
        peer_stats = get_peer_stats(profile, TODAY, snapshots=snapshots)
        fund, prices = _make_fund_and_prices()
        cat = _make_category(rev_type=rev_type)
        m = _metrics(revenue_cagr=0.03, years_of_data=10)
        return score_valuation(fund, prices, m, cat, peer_stats=peer_stats)

    def test_cyclical_stock_gets_cyclical_peer_pe_in_flags(self):
        result = self._score_with_profile(RevenueType.cyclical_commodity, "cyclical")
        flags_text = " ".join(result.flags)
        # Should mention "cyclical" category
        assert "cyclical" in flags_text, (
            f"Cyclical stock should see 'cyclical' peer context in flags.\nFlags: {result.flags}"
        )
        # Should show the cyclical P/E range (10–12), not compounder range (18–22)
        assert any(
            str(int(self.CYCLICAL_PEER_PE)) in f or f"{self.CYCLICAL_PEER_PE:.1f}" in f
            for f in result.flags
        ), (
            f"Expected cyclical P/E median ~{self.CYCLICAL_PEER_PE} in flags.\nFlags: {result.flags}"
        )

    def test_cyclical_stock_does_not_see_compounder_peers(self):
        result = self._score_with_profile(RevenueType.cyclical_commodity, "cyclical")
        flags_text = " ".join(result.flags)
        assert "compounder" not in flags_text, (
            f"Cyclical stock should NOT reference compounder peers.\nFlags: {result.flags}"
        )

    def test_compounder_stock_gets_compounder_peer_pe_in_flags(self):
        result = self._score_with_profile(RevenueType.recurring, "compounder")
        flags_text = " ".join(result.flags)
        assert "compounder" in flags_text, (
            f"Compounder stock should see 'compounder' peer context in flags.\nFlags: {result.flags}"
        )

    def test_cyclical_and_compounder_see_different_peer_pe_medians(self):
        cyclical_result = self._score_with_profile(RevenueType.cyclical_commodity, "cyclical")
        compounder_result = self._score_with_profile(RevenueType.recurring, "compounder")

        cyclical_flags = " ".join(cyclical_result.flags)
        compounder_flags = " ".join(compounder_result.flags)

        # The P/E medians shown in flags should differ between categories
        assert cyclical_flags != compounder_flags, (
            "Cyclical and compounder stocks should see different peer P/E context in their flags."
        )

    def test_no_peer_registry_falls_back_to_static_reference(self):
        fund, prices = _make_fund_and_prices()
        cat = _make_category(rev_type=RevenueType.cyclical_commodity)
        m = _metrics(revenue_cagr=0.03)
        result = score_valuation(fund, prices, m, cat, peer_stats=None)
        flags_text = " ".join(result.flags)
        # Should fall back gracefully — static reference mentioned
        assert "static reference" in flags_text or "PEER_PE" in flags_text or "long-run" in flags_text, (
            f"Expected fallback language when no peer registry.\nFlags: {result.flags}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PEER COMPARISON ISOLATION — SCORES ARE NOT AFFECTED
# ══════════════════════════════════════════════════════════════════════════════

class TestPeerComparisonDoesNotAffectScore:
    """Peer context is display-only — it must not change the numeric score."""

    def test_score_identical_with_and_without_peer_stats(self):
        fund, prices = _make_fund_and_prices()
        cat = _make_category()
        m = _metrics(revenue_cagr=0.05, years_of_data=10)

        result_no_peers = score_valuation(fund, prices, m, cat, peer_stats=None)
        peer_stats = _aggregate(_make_snapshots(), "compounder", TODAY)
        result_with_peers = score_valuation(fund, prices, m, cat, peer_stats=peer_stats)

        assert result_no_peers.score == result_with_peers.score, (
            f"Peer stats must not change the valuation score "
            f"({result_no_peers.score} vs {result_with_peers.score})."
        )
