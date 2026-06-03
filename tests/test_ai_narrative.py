"""Tests for the AI narrative layer (Section 8).

All tests must pass without ANTHROPIC_API_KEY set.  The validation gate is:
  - report renders fully with AI disabled
  - generate_commentary returns None gracefully when no API key
  - when ai_attempted=True and commentary is None, report shows unavailable note
  - when commentary is supplied, it appears in its own labelled section
  - no directive language introduced by the AI panel
"""

from __future__ import annotations

import io
import os
from datetime import date

import pytest
from rich.console import Console

from value_analyzer.ai import generate_commentary
from value_analyzer.report import DISCLAIMER_TEXT, render, render_markdown
from value_analyzer.report.render import _ai_commentary_panel
from value_analyzer.score.models import CompositeScore, SubScore
from value_analyzer.classify.models import (
    Category, CapitalIntensity, GrowthProfile, MoatType, RevenueType,
    Metrics, SicHint, RuleTrace,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_subscore(name: str) -> SubScore:
    return SubScore(
        name=name,
        score=55.0,
        reasons=[f"[+15.0/25] {name} component one"],
        flags=[],
    )


def _make_composite() -> CompositeScore:
    metrics = Metrics(
        ticker="TEST",
        as_of_date=date(2024, 1, 1),
        years_of_data=8,
        gross_margin_avg=0.52,
        revenue_cagr=0.06,
    )
    sic_hint = SicHint(sic_code=2080, sic_description="Beverages")
    traces = {
        "moat_type": RuleTrace(
            rule_name="gross_margin_rule",
            result="brand",
            confidence=0.8,
            rationale="Gross margin 52% > 40% threshold.",
        ),
    }
    category = Category(
        ticker="TEST",
        as_of_date=date(2024, 1, 1),
        capital_intensity=CapitalIntensity.asset_light,
        revenue_type=RevenueType.recurring,
        moat_type=MoatType.brand,
        growth_profile=GrowthProfile.stable,
        traces=traces,
        metrics=metrics,
        sic_hint=sic_hint,
    )
    return CompositeScore(
        ticker="TEST",
        as_of_date=date(2024, 1, 1),
        composite=62.5,
        moat=_make_subscore("moat"),
        health=_make_subscore("health"),
        valuation=SubScore(
            name="valuation",
            score=58.0,
            reasons=["[+20.0/35] Margin of safety = +18.5%"],
            flags=["⚠ Average IV estimate: $51.30 | Current price: $42.00 | Margin of safety: +22.1%."],
        ),
        management=_make_subscore("management"),
        weight_profile="stable",
        weights_used={"moat": 0.25, "health": 0.25, "valuation": 0.30, "management": 0.20},
        category=category,
        peer_comparison=None,
    )


def _render_to_str(cs: CompositeScore, **kwargs) -> str:
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, width=300)
    render(cs, console=con, **kwargs)
    return buf.getvalue()


# ── generate_commentary: returns None without API key ─────────────────────────

def test_generate_commentary_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = generate_commentary(_make_composite())
    assert result is None


# ── render() with AI disabled (default) ───────────────────────────────────────

def test_render_no_ai_section_by_default():
    """No AI section at all when ai_attempted=False (the default)."""
    output = _render_to_str(_make_composite())
    assert "AI Commentary" not in output


def test_render_no_ai_section_explicit_false():
    output = _render_to_str(_make_composite(), ai_attempted=False)
    assert "AI Commentary" not in output


def test_disclaimer_still_present_with_ai_disabled():
    output = _render_to_str(_make_composite())
    assert DISCLAIMER_TEXT in output


# ── render() with ai_attempted=True, commentary=None ─────────────────────────

def test_render_shows_unavailable_note_when_attempted_but_none():
    output = _render_to_str(_make_composite(), ai_attempted=True, ai_commentary=None)
    assert "AI Commentary" in output
    assert "unavailable" in output.lower()


def test_disclaimer_still_present_when_ai_unavailable():
    output = _render_to_str(_make_composite(), ai_attempted=True, ai_commentary=None)
    assert DISCLAIMER_TEXT in output


# ── render() with ai_attempted=True and real commentary ───────────────────────

_SAMPLE_COMMENTARY = (
    "The composite score of 62.5/100 reflects a stable business with a recognised brand moat. "
    "The valuation sub-score of 58.0/100 incorporates an average IV estimate of $51.30 against "
    "a current price of $42.00, implying a margin of safety of 22.1%. "
    "If an investor applied a Graham-style framework, this gap would represent a meaningful cushion "
    "for a business with recurring revenue. The health and management sub-scores both sit at 55.0/100, "
    "suggesting adequate but not exceptional capital discipline."
)


def test_render_shows_ai_commentary_panel():
    output = _render_to_str(_make_composite(), ai_attempted=True, ai_commentary=_SAMPLE_COMMENTARY)
    assert "AI Commentary" in output
    assert "Interpretation only" in output


def test_render_ai_commentary_content_present():
    output = _render_to_str(_make_composite(), ai_attempted=True, ai_commentary=_SAMPLE_COMMENTARY)
    assert "composite score" in output.lower()


def test_disclaimer_still_present_with_ai_commentary():
    output = _render_to_str(_make_composite(), ai_attempted=True, ai_commentary=_SAMPLE_COMMENTARY)
    assert DISCLAIMER_TEXT in output


# ── No directive language from the AI panel ───────────────────────────────────

_DIRECTIVE_PHRASES = [
    "you should buy",
    "you should sell",
    "we recommend",
    "strong buy",
    "strong sell",
    "is a buy",
    "is a sell",
    "time to buy",
    "time to sell",
    "add to your portfolio",
    "sell your shares",
]


def test_no_directive_language_in_ai_panel():
    output = _render_to_str(
        _make_composite(), ai_attempted=True, ai_commentary=_SAMPLE_COMMENTARY
    ).lower()
    for phrase in _DIRECTIVE_PHRASES:
        assert phrase not in output, (
            f"Directive phrase {phrase!r} found in AI commentary output."
        )


# ── AI commentary must not introduce numbers not in the report ────────────────

def test_ai_commentary_numbers_all_present_in_base_report():
    """Every number in the sample AI commentary must also appear in the base report.

    This validates the system-prompt rule: the model must only reference figures
    already present in the structured analysis.  We verify this property on
    controlled commentary in tests; the system prompt enforces it at runtime.
    """
    import re
    base_output = _render_to_str(_make_composite())
    numbers_in_base = set(re.findall(r"\d+\.?\d*", base_output))
    numbers_in_commentary = set(re.findall(r"\d+\.?\d*", _SAMPLE_COMMENTARY))
    new_numbers = numbers_in_commentary - numbers_in_base
    assert not new_numbers, (
        f"AI commentary references numbers not in base report: {new_numbers!r}. "
        "The model must only interpret figures already shown in the analysis."
    )


# ── render_markdown passes AI params through ──────────────────────────────────

def test_render_markdown_no_ai_by_default():
    text = render_markdown(_make_composite())
    assert "AI Commentary" not in text
    assert DISCLAIMER_TEXT in text


def test_render_markdown_shows_unavailable_when_attempted():
    text = render_markdown(_make_composite(), ai_attempted=True, ai_commentary=None)
    assert "AI Commentary" in text
    assert "unavailable" in text.lower()
    assert DISCLAIMER_TEXT in text


def test_render_markdown_shows_commentary_when_supplied():
    text = render_markdown(_make_composite(), ai_attempted=True, ai_commentary=_SAMPLE_COMMENTARY)
    assert "AI Commentary" in text
    assert DISCLAIMER_TEXT in text


# ── _ai_commentary_panel unit tests ───────────────────────────────────────────

def test_panel_builder_returns_none_when_not_attempted():
    assert _ai_commentary_panel(None, False) is None
    assert _ai_commentary_panel("some text", False) is None


def test_panel_builder_unavailable_panel():
    panel = _ai_commentary_panel(None, True)
    assert panel is not None
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, width=300)
    con.print(panel)
    assert "unavailable" in buf.getvalue().lower()


def test_panel_builder_commentary_panel():
    panel = _ai_commentary_panel("Interpretation text here.", True)
    assert panel is not None
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, width=300)
    con.print(panel)
    assert "Interpretation text here." in buf.getvalue()
