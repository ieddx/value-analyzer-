"""Tests for the Evaluation & Framework Context panel in renderer.py.

Critical invariants:
  (a) No directive advice phrases appear in the panel or full render output.
  (b) DISCLAIMER_TEXT is still present after the new section is added.
  (c) The new section is actually present and contains both Part 1 and Part 2.
  (d) Confidence tier logic maps correctly to data completeness / IV dispersion.
  (e) Backtest calibration note respects the stored summary (or falls back gracefully).
"""

from __future__ import annotations

import io
from datetime import date
from typing import Optional

import pytest
from rich.console import Console

from value_analyzer.classify.models import (
    CapitalIntensity, Category, GrowthProfile, Metrics,
    MoatType, RevenueType, RuleTrace, SicHint,
)
from value_analyzer.score.models import CompositeScore, SubScore
from value_analyzer.report.renderer import (
    DISCLAIMER_TEXT,
    _evaluation_panel,
    render,
)

TODAY = date(2024, 12, 31)


# ── Synthetic fixture helpers ──────────────────────────────────────────────────

def _trace(result: str = "brand") -> RuleTrace:
    return RuleTrace(rule_name="test", result=result, confidence=0.8, rationale="test")


def _category(ticker: str = "TEST", moat: MoatType = MoatType.brand) -> Category:
    return Category(
        ticker=ticker, as_of_date=TODAY,
        capital_intensity=CapitalIntensity.asset_light,
        revenue_type=RevenueType.recurring,
        moat_type=moat,
        growth_profile=GrowthProfile.stable,
        traces={
            "capital_intensity": _trace("asset_light"),
            "revenue_type": _trace("recurring"),
            "moat_type": _trace(moat.value),
            "growth_profile": _trace("stable"),
        },
        metrics=Metrics(ticker=ticker, as_of_date=TODAY, years_of_data=10),
        sic_hint=SicHint(),
    )


def _subscore(
    name: str,
    score: float = 60.0,
    flags: Optional[list[str]] = None,
    real: int = 4,
    total: int = 4,
) -> SubScore:
    return SubScore(
        name=name,
        score=score,
        reasons=[f"[+{score:.1f}/100] synthetic"],
        flags=flags or [],
        real_inputs=real,
        total_inputs=total,
    )


def _composite(
    ticker: str = "TEST",
    composite: float = 65.0,
    moat: float = 70.0,
    health: float = 70.0,
    valuation: float = 55.0,
    management: float = 60.0,
    weight_profile: str = "stable",
    completeness_real: int = 17,
    completeness_total: int = 17,
    iv_dispersion_flag: Optional[str] = None,
    val_flags: Optional[list[str]] = None,
) -> CompositeScore:
    return CompositeScore(
        ticker=ticker,
        as_of_date=TODAY,
        composite=composite,
        moat=_subscore("moat", moat, real=4, total=4),
        health=_subscore("health", health, real=4, total=4),
        valuation=_subscore("valuation", valuation, flags=val_flags or [], real=5, total=5),
        management=_subscore("management", management, real=4, total=4),
        weight_profile=weight_profile,
        weights_used={"moat": 0.25, "health": 0.25, "valuation": 0.25, "management": 0.25},
        category=_category(ticker),
        completeness_real=completeness_real,
        completeness_total=completeness_total,
        iv_dispersion_flag=iv_dispersion_flag,
    )


def _render_to_str(cs: CompositeScore) -> str:
    """Render a full report to a plain-text string (markup stripped)."""
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=False, width=300)
    render(cs, console=con, ai_attempted=False)
    return buf.getvalue()


def _panel_to_str(cs: CompositeScore) -> str:
    """Render only the evaluation panel to a plain-text string."""
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=False, width=300)
    con.print(_evaluation_panel(cs))
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# (a) NO DIRECTIVE ADVICE PHRASES
# These tests are the safety check. If any of them fail, the report is giving
# investment advice, which violates the cardinal rule.
# ══════════════════════════════════════════════════════════════════════════════

_DIRECTIVE_PHRASES = [
    "you should buy",
    "you should sell",
    "you should hold",
    "buy now",
    "sell now",
    "buy the stock",
    "sell the stock",
    "we recommend",
    "i recommend",
    "strong buy",
    "strong sell",
    "time to buy",
    "time to sell",
    "go long",
    "go short",
    "purchase shares",
    "divest",
]


class TestNoDirectiveLanguage:
    """
    THESE TESTS MUST NEVER BE SKIPPED.

    The report layer must never emit directive language. Any phrase that
    instructs the reader to take a specific action with this specific stock
    is a violation of the cardinal rule.
    """

    def test_evaluation_panel_contains_no_directive_phrases(self):
        """The evaluation panel must contain no directive advice phrases."""
        cs = _composite()
        text = _panel_to_str(cs).lower()
        violations = [p for p in _DIRECTIVE_PHRASES if p in text]
        assert not violations, (
            f"Evaluation panel contains directive phrase(s): {violations}\n"
            "The panel must never instruct the reader to buy, sell, or take "
            "a specific action with this stock."
        )

    def test_full_render_contains_no_directive_phrases(self):
        """The full rendered report must contain no directive advice phrases."""
        cs = _composite()
        text = _render_to_str(cs).lower()
        violations = [p for p in _DIRECTIVE_PHRASES if p in text]
        assert not violations, (
            f"Full render contains directive phrase(s): {violations}\n"
            "The report must never instruct the reader to take action."
        )

    def test_evaluation_panel_high_score_still_no_directive(self):
        """Even a very high composite score must not produce directive language."""
        cs = _composite(composite=95.0, moat=95.0, health=90.0,
                        valuation=90.0, management=90.0)
        text = _panel_to_str(cs).lower()
        violations = [p for p in _DIRECTIVE_PHRASES if p in text]
        assert not violations, (
            f"High-score report contains directive phrase(s): {violations}"
        )

    def test_evaluation_panel_low_score_still_no_directive(self):
        """A very low composite score must not produce 'sell' directive language."""
        cs = _composite(composite=15.0, moat=10.0, health=20.0,
                        valuation=15.0, management=10.0)
        text = _panel_to_str(cs).lower()
        violations = [p for p in _DIRECTIVE_PHRASES if p in text]
        assert not violations, (
            f"Low-score report contains directive phrase(s): {violations}"
        )

    def test_evaluation_panel_low_confidence_still_no_directive(self):
        """Low-confidence result (dispersion + low completeness) must not say 'sell' or 'avoid'."""
        disp_flag = (
            "IV_DISPERSION: Valuation methods disagree significantly "
            "($28.00–$135.00, 4.8× spread) — treat average IV as low-confidence."
        )
        cs = _composite(
            completeness_real=10, completeness_total=17,
            iv_dispersion_flag=disp_flag,
        )
        text = _panel_to_str(cs).lower()
        violations = [p for p in _DIRECTIVE_PHRASES if p in text]
        assert not violations, (
            f"Low-confidence report contains directive phrase(s): {violations}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# (b) DISCLAIMER IS STILL PRESENT
# ══════════════════════════════════════════════════════════════════════════════

class TestDisclaimerPresence:
    def test_disclaimer_present_in_full_render(self):
        """DISCLAIMER_TEXT must appear in every full render output."""
        cs = _composite()
        text = _render_to_str(cs)
        assert DISCLAIMER_TEXT in text, (
            "DISCLAIMER_TEXT is missing from the rendered report.\n"
            "Adding the evaluation panel must not remove the disclaimer."
        )

    def test_disclaimer_present_for_high_score(self):
        cs = _composite(composite=92.0, moat=92.0, health=88.0,
                        valuation=85.0, management=88.0)
        text = _render_to_str(cs)
        assert DISCLAIMER_TEXT in text

    def test_disclaimer_present_for_low_confidence(self):
        disp_flag = (
            "IV_DISPERSION: Valuation methods disagree significantly "
            "($20.00–$90.00, 4.5× spread) — treat average IV as low-confidence."
        )
        cs = _composite(
            completeness_real=8, completeness_total=17,
            iv_dispersion_flag=disp_flag,
        )
        text = _render_to_str(cs)
        assert DISCLAIMER_TEXT in text


# ══════════════════════════════════════════════════════════════════════════════
# (c) SECTION PRESENCE AND STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

class TestSectionPresence:
    def test_evaluation_section_in_full_render(self):
        """'Evaluation' heading must appear in the full rendered report."""
        cs = _composite()
        text = _render_to_str(cs)
        assert "Evaluation" in text or "evaluation" in text.lower(), (
            "The Evaluation & Framework Context section is missing from the render."
        )

    def test_part1_label_present(self):
        cs = _composite()
        text = _panel_to_str(cs)
        assert "Part 1" in text, "Part 1 label is missing from the evaluation panel."

    def test_part2_label_present(self):
        cs = _composite()
        text = _panel_to_str(cs)
        assert "Part 2" in text, "Part 2 label is missing from the evaluation panel."

    def test_confidence_tier_mentioned(self):
        cs = _composite()
        text = _panel_to_str(cs)
        assert any(t in text for t in ("HIGH", "MEDIUM", "LOW")), (
            "Confidence tier (HIGH/MEDIUM/LOW) must appear in evaluation panel."
        )

    def test_backtest_calibration_note_present(self):
        cs = _composite()
        text = _panel_to_str(cs).lower()
        assert "backtest" in text or "calibration" in text, (
            "Evaluation panel must include a backtest calibration note."
        )

    def test_framework_context_present(self):
        """Part 2 must mention at least two named frameworks."""
        cs = _composite()
        text = _panel_to_str(cs).lower()
        framework_keywords = ["margin of safety", "kelly", "diversif", "conviction"]
        found = [kw for kw in framework_keywords if kw in text]
        assert len(found) >= 2, (
            f"Expected ≥2 framework keywords in evaluation panel, found: {found}"
        )

    def test_investor_autonomy_language_present(self):
        """Panel must explicitly state the decision is the investor's own."""
        cs = _composite()
        text = _panel_to_str(cs).lower()
        assert "investor" in text and (
            "own" in text or "themselves" in text or "differ" in text
        ), "Panel must state that the decision belongs to the investor."


# ══════════════════════════════════════════════════════════════════════════════
# (d) CONFIDENCE TIER LOGIC
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidenceTierLogic:
    def test_high_completeness_no_dispersion_gives_high_tier(self):
        """17/17 inputs, no IV dispersion → confidence tier HIGH."""
        cs = _composite(completeness_real=17, completeness_total=17,
                        iv_dispersion_flag=None)
        text = _panel_to_str(cs)
        assert "HIGH" in text, (
            "Full data + no dispersion should produce HIGH confidence tier."
        )

    def test_iv_dispersion_flag_downgrades_to_medium(self):
        """IV dispersion with high completeness → MEDIUM, not HIGH."""
        disp_flag = (
            "IV_DISPERSION: Valuation methods disagree significantly "
            "($30.00–$90.00, 3.0× spread) — treat average IV as low-confidence."
        )
        cs = _composite(
            completeness_real=17, completeness_total=17,
            iv_dispersion_flag=disp_flag,
        )
        text = _panel_to_str(cs)
        assert "MEDIUM" in text or "LOW" in text, (
            "IV dispersion should downgrade confidence below HIGH."
        )
        assert "HIGH" not in text.split("MEDIUM")[0].split("LOW")[0] or "HIGH" not in text, (
            "IV dispersion prevents HIGH confidence tier."
        )

    def test_low_completeness_gives_low_tier(self):
        """Completeness below 70% → confidence tier LOW."""
        cs = _composite(
            completeness_real=8, completeness_total=17,   # 47% < 70%
            iv_dispersion_flag=None,
        )
        text = _panel_to_str(cs)
        assert "LOW" in text, (
            "Completeness below COMPLETENESS_CAUTION_THRESHOLD must give LOW confidence."
        )

    def test_medium_completeness_gives_medium_tier(self):
        """Completeness 75% (between 70% and 85%) → MEDIUM."""
        cs = _composite(
            completeness_real=13, completeness_total=17,  # 76%
            iv_dispersion_flag=None,
        )
        text = _panel_to_str(cs)
        assert "MEDIUM" in text, (
            "Completeness between 70% and 85% (with no other issues) should give MEDIUM."
        )

    def test_dispersion_flag_iv_range_shown_in_panel(self):
        """When IV dispersion fires, the IV range should appear in the panel."""
        disp_flag = (
            "IV_DISPERSION: Valuation methods disagree significantly "
            "($28.00–$135.00, 4.8× spread) — treat average IV as low-confidence."
        )
        cs = _composite(iv_dispersion_flag=disp_flag)
        text = _panel_to_str(cs)
        assert "$28" in text or "28.00" in text, (
            "IV dispersion range should appear in the confidence assessment."
        )

    def test_both_issues_still_gives_low_or_medium(self):
        """Both dispersion and partial completeness combined → not HIGH."""
        disp_flag = (
            "IV_DISPERSION: Valuation methods disagree significantly "
            "($30.00–$100.00, 3.3× spread)."
        )
        cs = _composite(
            completeness_real=13, completeness_total=17,
            iv_dispersion_flag=disp_flag,
        )
        text = _panel_to_str(cs)
        # Should be MEDIUM or LOW, not HIGH
        assert "HIGH" not in text or (
            # HIGH could appear in educational text — only check the tier line
            "Confidence in this assessment: HIGH" not in text
        ), "Both dispersion + partial completeness should not produce HIGH confidence."


# ══════════════════════════════════════════════════════════════════════════════
# (e) BACKTEST CALIBRATION NOTE
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestCalibrationNote:
    def test_calibration_note_mentions_long_horizon(self):
        """Calibration note must mention long-horizon or 5-year context."""
        cs = _composite()
        text = _panel_to_str(cs).lower()
        assert "5-year" in text or "long-horizon" in text or "long horizon" in text, (
            "Calibration note must reference the 5-year horizon finding."
        )

    def test_calibration_note_mentions_short_horizon_limitation(self):
        """Calibration note must mention lack of 1-year signal."""
        cs = _composite()
        text = _panel_to_str(cs).lower()
        assert "1-year" in text or "short" in text or "not a short" in text, (
            "Calibration note must reference the 1-year non-signal finding."
        )

    def test_calibration_note_present_without_backtest_file(self):
        """Panel should render a calibration note even when no backtest summary exists."""
        import value_analyzer.report.renderer as _rmod
        from pathlib import Path
        orig = _rmod._BACKTEST_SUMMARY_PATH
        _rmod._BACKTEST_SUMMARY_PATH = Path("/tmp/nonexistent_backtest_12345.json")
        try:
            cs = _composite()
            text = _panel_to_str(cs).lower()
            assert "backtest" in text or "calibration" in text, (
                "Calibration note must appear even without a stored backtest summary."
            )
        finally:
            _rmod._BACKTEST_SUMMARY_PATH = orig

    def test_calibration_note_does_not_overstate_edge(self):
        """Calibration text must not claim the edge is proven or guaranteed."""
        cs = _composite()
        text = _panel_to_str(cs).lower()
        strong_claims = ["proven edge", "guaranteed", "will outperform", "always outperform"]
        violations = [c for c in strong_claims if c in text]
        assert not violations, (
            f"Calibration note overstates edge: {violations}"
        )
