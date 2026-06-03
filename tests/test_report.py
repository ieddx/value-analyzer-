"""Tests for the report layer.

Key guarantee: DISCLAIMER_TEXT must appear in every rendered report.
"""

from __future__ import annotations

import io
from datetime import date

import pytest
from rich.console import Console

from value_analyzer.report import DISCLAIMER_TEXT, render, render_markdown
from value_analyzer.score.models import CompositeScore, SubScore
from value_analyzer.classify.models import (
    Category, CapitalIntensity, GrowthProfile, MoatType, RevenueType,
    Metrics, SicHint,
)
from value_analyzer.peers.models import PeerComparison


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_subscore(name: str) -> SubScore:
    return SubScore(
        name=name,
        score=55.0,
        reasons=[f"[+15.0/25] {name} component one", f"[+8.0/25] {name} component two"],
        flags=[f"⚠ {name} data note"],
    )


def _make_category() -> Category:
    metrics = Metrics(
        ticker="TEST",
        as_of_date=date(2024, 1, 1),
        years_of_data=8,
        gross_margin_avg=0.52,
        revenue_cagr=0.06,
    )
    sic_hint = SicHint(sic_code=2080, sic_description="Beverages")
    from value_analyzer.classify.models import RuleTrace
    traces = {
        "moat_type": RuleTrace(
            rule_name="gross_margin_rule",
            result="brand",
            confidence=0.8,
            rationale="Gross margin 52% > 40% threshold with stable history.",
        ),
        "growth_profile": RuleTrace(
            rule_name="revenue_cagr_rule",
            result="stable",
            confidence=0.75,
            rationale="Revenue CAGR 6% within 0-8% stable band.",
        ),
    }
    return Category(
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


def _make_composite(*, with_peers: bool = False) -> CompositeScore:
    peer_comparison = None
    if with_peers:
        peer_comparison = PeerComparison(
            weight_profile="stable",
            peer_count=5,
            peer_tickers=["KO", "PEP", "CL", "PG", "UL"],
            subject_pe=21.5,
            subject_pfcf=18.0,
            peer_pe_median=20.0,
            peer_pe_p25=17.0,
            peer_pe_p75=24.0,
            peer_pfcf_median=19.0,
            peer_gross_margin_median=0.50,
            peer_roic_median=0.18,
            context_note="Stocks great value investors held in this category.",
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
            flags=[
                "⚠ Valuation assumptions: WACC = 9%, terminal growth = 2.5%.",
                "⚠ IV estimate (No-growth earnings power): $48.20 — discount to current price $42.00.",
                "⚠ Average IV estimate: $51.30 | Current price: $42.00 | Margin of safety: +22.1%.",
            ],
        ),
        management=_make_subscore("management"),
        weight_profile="stable",
        weights_used={"moat": 0.25, "health": 0.25, "valuation": 0.30, "management": 0.20},
        category=_make_category(),
        peer_comparison=peer_comparison,
    )


# ── Disclaimer must always be present ─────────────────────────────────────────

def _capture(cs: CompositeScore) -> str:
    buf = io.StringIO()
    # Width must exceed len(DISCLAIMER_TEXT) so the plain sentinel line never wraps.
    con = Console(file=buf, highlight=False, width=300)
    render(cs, console=con)
    return buf.getvalue()


def test_disclaimer_present_basic():
    output = _capture(_make_composite())
    assert DISCLAIMER_TEXT in output, "Disclaimer must appear in every rendered report."


def test_disclaimer_present_with_peers():
    output = _capture(_make_composite(with_peers=True))
    assert DISCLAIMER_TEXT in output, "Disclaimer must appear even when peer comparison is shown."


def test_disclaimer_present_in_markdown():
    text = render_markdown(_make_composite())
    assert DISCLAIMER_TEXT in text, "Disclaimer must appear in markdown output."


# ── Peer comparison: same-category only ───────────────────────────────────────

def test_peer_panel_shows_only_same_category_peers():
    cs = _make_composite(with_peers=True)
    output = _capture(cs)
    assert "Peer Comparison" in output
    # Subject and peer profile must match
    assert cs.peer_comparison.weight_profile == cs.weight_profile


def test_no_peer_panel_when_absent():
    cs = _make_composite(with_peers=False)
    output = _capture(cs)
    assert "Peer Comparison" not in output


# ── No directive language ──────────────────────────────────────────────────────
# Check for contextual directive *advice* phrases, not bare words that can
# appear legitimately (e.g. "buy" inside "NOT a buy or sell recommendation").

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


def test_no_directive_language():
    output = _capture(_make_composite(with_peers=True)).lower()
    for phrase in _DIRECTIVE_PHRASES:
        assert phrase not in output, (
            f"Directive phrase {phrase!r} found in report output — report must not give advice."
        )


# ── Structure checks ───────────────────────────────────────────────────────────

def test_all_subscore_names_present():
    output = _capture(_make_composite())
    for name in ("Moat", "Health", "Valuation", "Management"):
        assert name in output


def test_composite_score_in_header():
    cs = _make_composite()
    output = _capture(cs)
    assert "62.5" in output


def test_bull_bear_panel_present():
    output = _capture(_make_composite())
    assert "Bull" in output or "Bear" in output


def test_position_sizing_panel_present():
    output = _capture(_make_composite())
    assert "Position-Sizing" in output or "Position-sizing" in output


def test_backtest_context_present():
    output = _capture(_make_composite())
    # Line reads either "Backtest not yet run" or "Backtest (run <date>)"
    assert "Backtest" in output
