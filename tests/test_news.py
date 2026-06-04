"""Tests for the news layer.

All tests must pass with NO API keys set in the environment.

Three critical invariants:
  (a) Graceful degradation — no API key / network failure → NewsResult returned,
      never an exception; report renders normally with "unavailable" note.
  (b) No directive advice — news panel contains no buy/sell directives.
  (c) Score isolation — news data never enters composite.score(); the score
      module never imports from the news module.
"""

from __future__ import annotations

import ast
import io
import os
import pathlib
from datetime import date, timedelta
from typing import Optional

import pytest
from rich.console import Console

from value_analyzer.news.fetch import fetch_news
from value_analyzer.news.models import NewsItem, NewsResult
from value_analyzer.report.renderer import (
    DISCLAIMER_TEXT,
    _news_panel,
    _news_unavailable_line,
    render,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

TODAY = date(2024, 12, 31)

_DIRECTIVE_PHRASES = [
    "you should buy",
    "you should sell",
    "buy now",
    "sell now",
    "we recommend",
    "i recommend",
    "strong buy",
    "strong sell",
    "time to buy",
    "time to sell",
]


def _news_result(n_items: int = 3, error: Optional[str] = None) -> NewsResult:
    items = [
        NewsItem(
            headline=f"Company announces Q{i} earnings beat",
            source="Reuters",
            published_at=TODAY - timedelta(days=i),
            url=f"https://example.com/{i}",
            summary="",
        )
        for i in range(n_items)
    ]
    return NewsResult(
        ticker="TEST",
        fetched_at=TODAY,
        provider="test",
        items=[] if error else items,
        error=error,
    )


# Minimal CompositeScore builder — copied from test_evaluation_panel.py pattern
from value_analyzer.classify.models import (
    CapitalIntensity, Category, GrowthProfile, Metrics,
    MoatType, RevenueType, RuleTrace, SicHint,
)
from value_analyzer.score.models import CompositeScore, SubScore


def _trace(result: str = "brand") -> RuleTrace:
    return RuleTrace(rule_name="test", result=result, confidence=0.8, rationale="test")


def _category(ticker: str = "TEST") -> Category:
    return Category(
        ticker=ticker, as_of_date=TODAY,
        capital_intensity=CapitalIntensity.asset_light,
        revenue_type=RevenueType.recurring,
        moat_type=MoatType.brand,
        growth_profile=GrowthProfile.stable,
        traces={
            "capital_intensity": _trace("asset_light"),
            "revenue_type": _trace("recurring"),
            "moat_type": _trace("brand"),
            "growth_profile": _trace("stable"),
        },
        metrics=Metrics(ticker=ticker, as_of_date=TODAY, years_of_data=10),
        sic_hint=SicHint(),
    )


def _subscore(name: str, score: float = 60.0) -> SubScore:
    return SubScore(
        name=name, score=score,
        reasons=[f"[+{score:.1f}/100] synthetic"],
        flags=[],
        real_inputs=4, total_inputs=4,
    )


def _composite(ticker: str = "TEST", score: float = 65.0) -> CompositeScore:
    return CompositeScore(
        ticker=ticker, as_of_date=TODAY,
        composite=score,
        moat=_subscore("moat", score),
        health=_subscore("health", score),
        valuation=_subscore("valuation", score),
        management=_subscore("management", score),
        weight_profile="stable",
        weights_used={"moat": 0.25, "health": 0.25, "valuation": 0.25, "management": 0.25},
        category=_category(ticker),
        completeness_real=16, completeness_total=16,
    )


def _render_to_str(cs: CompositeScore, news: Optional[NewsResult] = None) -> str:
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=False, width=300)
    render(cs, console=con, ai_attempted=False, news=news)
    return buf.getvalue()


def _panel_to_str(news: NewsResult) -> str:
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=False, width=300)
    con.print(_news_panel(news))
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# (a) GRACEFUL DEGRADATION — no API key, network errors
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchNewsNoDependencies:
    """All tests here must pass with FINNHUB_API_KEY unset."""

    def test_no_api_key_returns_news_result_not_exception(self):
        """fetch_news with no API key returns a NewsResult, never raises."""
        old = os.environ.pop("FINNHUB_API_KEY", None)
        try:
            result = fetch_news("AAPL")
        finally:
            if old is not None:
                os.environ["FINNHUB_API_KEY"] = old

        assert result is not None
        assert isinstance(result, NewsResult)

    def test_no_api_key_error_field_set(self):
        """NewsResult.error is set when no API key is available."""
        old = os.environ.pop("FINNHUB_API_KEY", None)
        try:
            result = fetch_news("AAPL")
        finally:
            if old is not None:
                os.environ["FINNHUB_API_KEY"] = old

        assert result.error is not None
        assert "FINNHUB_API_KEY" in result.error

    def test_no_api_key_items_empty(self):
        """NewsResult.items is empty when no API key is available."""
        old = os.environ.pop("FINNHUB_API_KEY", None)
        try:
            result = fetch_news("AAPL")
        finally:
            if old is not None:
                os.environ["FINNHUB_API_KEY"] = old

        assert result.items == []

    def test_no_api_key_available_is_false(self):
        """NewsResult.available is False when no API key is available."""
        old = os.environ.pop("FINNHUB_API_KEY", None)
        try:
            result = fetch_news("AAPL")
        finally:
            if old is not None:
                os.environ["FINNHUB_API_KEY"] = old

        assert result.available is False

    def test_provider_network_exception_caught(self):
        """A provider that raises ConnectionError is caught; returns error NewsResult."""
        class FailingProvider:
            def fetch(self, ticker, from_date, to_date, api_key):
                raise ConnectionError("simulated network failure")

        result = fetch_news("AAPL", provider=FailingProvider())
        assert result is not None
        assert result.error is not None
        assert result.items == []

    def test_provider_value_error_caught(self):
        """Any exception from provider is caught, never propagates."""
        class BrokenProvider:
            def fetch(self, ticker, from_date, to_date, api_key):
                raise ValueError("malformed response")

        result = fetch_news("AAPL", provider=BrokenProvider())
        assert result.error is not None

    def test_provider_returns_empty_list(self):
        """Provider returning [] gives NewsResult with no items and no error."""
        class EmptyProvider:
            def fetch(self, ticker, from_date, to_date, api_key):
                return []

        result = fetch_news("AAPL", provider=EmptyProvider())
        assert result.error is None
        assert result.items == []
        assert result.available is False

    def test_deduplication_removes_duplicate_headlines(self):
        """Identical (headline, source) pairs are deduplicated."""
        class DupProvider:
            def fetch(self, ticker, from_date, to_date, api_key):
                item = {
                    "headline": "Big news",
                    "source": "Reuters",
                    "datetime": 1700000000,
                    "url": "https://example.com",
                    "summary": "",
                }
                return [item, item, item]

        result = fetch_news("AAPL", provider=DupProvider())
        assert len(result.items) == 1

    def test_deduplication_keeps_different_headlines(self):
        """Different headlines from the same source are both kept."""
        class TwoProvider:
            def fetch(self, ticker, from_date, to_date, api_key):
                return [
                    {"headline": "H1", "source": "Reuters",
                     "datetime": 1700000000, "url": "", "summary": ""},
                    {"headline": "H2", "source": "Reuters",
                     "datetime": 1700000001, "url": "", "summary": ""},
                ]

        result = fetch_news("AAPL", provider=TwoProvider())
        assert len(result.items) == 2

    def test_items_sorted_descending_by_date(self):
        """Items are returned sorted most-recent first."""
        class UnsortedProvider:
            def fetch(self, ticker, from_date, to_date, api_key):
                return [
                    {"headline": "Old news", "source": "AP",
                     "datetime": 1690000000, "url": "", "summary": ""},  # older
                    {"headline": "New news", "source": "AP",
                     "datetime": 1700000000, "url": "", "summary": ""},  # newer
                ]

        result = fetch_news("AAPL", provider=UnsortedProvider())
        assert result.items[0].headline == "New news"
        assert result.items[1].headline == "Old news"

    def test_empty_headline_skipped(self):
        """Entries with no headline are silently skipped."""
        class NoisyProvider:
            def fetch(self, ticker, from_date, to_date, api_key):
                return [
                    {"headline": "", "source": "AP",
                     "datetime": 1700000000, "url": "", "summary": ""},
                    {"headline": "Real headline", "source": "AP",
                     "datetime": 1700000000, "url": "", "summary": ""},
                ]

        result = fetch_news("AAPL", provider=NoisyProvider())
        assert len(result.items) == 1
        assert result.items[0].headline == "Real headline"

    def test_bad_timestamp_falls_back_to_from_date(self):
        """Unparseable datetime field defaults to from_date, no exception."""
        class BadTsProvider:
            def fetch(self, ticker, from_date, to_date, api_key):
                return [
                    {"headline": "News", "source": "AP",
                     "datetime": None, "url": "", "summary": ""},
                ]

        result = fetch_news("AAPL", days=30, provider=BadTsProvider())
        assert len(result.items) == 1  # didn't crash
        assert result.items[0].published_at is not None


# ══════════════════════════════════════════════════════════════════════════════
# (b) RENDERING — no directives, correct framing, graceful absent news
# ══════════════════════════════════════════════════════════════════════════════

class TestNewsRendering:
    def test_render_with_news_none_does_not_crash(self):
        """render() with news=None works normally — no crash."""
        cs = _composite()
        text = _render_to_str(cs, news=None)
        assert len(text) > 100

    def test_render_with_error_news_result_does_not_crash(self):
        """render() with error NewsResult shows unavailable note, no crash."""
        cs = _composite()
        news = _news_result(error="FINNHUB_API_KEY not set")
        text = _render_to_str(cs, news=news)
        assert "unavailable" in text.lower()

    def test_render_with_available_news_shows_panel(self):
        """render() with available news shows the panel."""
        cs = _composite()
        news = _news_result(n_items=3)
        text = _render_to_str(cs, news=news)
        assert "Recent News" in text or "recent news" in text.lower()

    def test_news_panel_has_no_directive_phrases(self):
        """The news panel contains no directive advice phrases."""
        news = _news_result(n_items=3)
        text = _panel_to_str(news).lower()
        violations = [p for p in _DIRECTIVE_PHRASES if p in text]
        assert not violations, (
            f"News panel contains directive phrase(s): {violations}"
        )

    def test_news_panel_notes_no_score_effect(self):
        """News panel states it does not affect the score."""
        news = _news_result(n_items=3)
        text = _panel_to_str(news).lower()
        assert "does not affect" in text or "not affect" in text, (
            "News panel must state it does not affect the score."
        )

    def test_news_panel_notes_post_date_gap(self):
        """News panel notes that events may post-date SEC filings."""
        news = _news_result(n_items=3)
        text = _panel_to_str(news).lower()
        assert any(kw in text for kw in ("post-date", "sec filing", "filed", "10-q", "10-k")), (
            "News panel must mention the SEC-filing staleness gap."
        )

    def test_news_capped_at_8_items(self):
        """At most 8 headlines appear in the news panel even with 12 items."""
        news = NewsResult(
            ticker="TEST",
            fetched_at=TODAY,
            provider="test",
            items=[
                NewsItem(
                    headline=f"Headline {i}",
                    source="AP",
                    published_at=TODAY - timedelta(days=i),
                )
                for i in range(12)
            ],
        )
        text = _panel_to_str(news)
        # Count occurrences of "AP" (source appears once per item)
        ap_count = text.count("AP")
        assert ap_count <= 8, (
            f"Expected ≤8 news items displayed, found {ap_count} 'AP' occurrences."
        )

    def test_news_unavailable_line_contains_error(self):
        """_news_unavailable_line includes the error message."""
        news = _news_result(error="FINNHUB_API_KEY not set")
        line = _news_unavailable_line(news)
        assert "unavailable" in line.lower()
        assert "FINNHUB_API_KEY" in line

    def test_disclaimer_still_present_with_news(self):
        """DISCLAIMER_TEXT is still present in a full render that includes news."""
        cs = _composite()
        news = _news_result(n_items=3)
        text = _render_to_str(cs, news=news)
        assert DISCLAIMER_TEXT in text

    def test_disclaimer_still_present_with_error_news(self):
        """DISCLAIMER_TEXT is still present when news is unavailable."""
        cs = _composite()
        news = _news_result(error="key missing")
        text = _render_to_str(cs, news=news)
        assert DISCLAIMER_TEXT in text

    def test_news_panel_shows_provider_and_date(self):
        """Panel footer shows provider name and fetch date."""
        news = _news_result(n_items=2)
        text = _panel_to_str(news)
        assert "test" in text.lower()           # provider name
        assert str(TODAY) in text               # fetched_at date

    def test_full_render_no_directive_with_news(self):
        """Full render output with news contains no directive phrases."""
        cs = _composite()
        news = _news_result(n_items=3)
        text = _render_to_str(cs, news=news).lower()
        violations = [p for p in _DIRECTIVE_PHRASES if p in text]
        assert not violations, (
            f"Full render with news contains directive phrase(s): {violations}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# (c) SCORE ISOLATION — news never enters scoring math
# ══════════════════════════════════════════════════════════════════════════════

class TestNewsScoreIsolation:
    """
    THESE TESTS MUST NEVER BE SKIPPED.

    The news layer is purely additive context for the human reader.
    It must never influence the composite score or any valuation math.
    """

    def test_score_module_does_not_import_news(self):
        """The score layer must have zero imports from value_analyzer.news.

        Walk every .py file under src/value_analyzer/score/ with the AST
        parser and assert no import references the news module.
        """
        score_dir = pathlib.Path(
            __file__
        ).parent.parent / "src" / "value_analyzer" / "score"

        violations: list[str] = []
        for py_file in sorted(score_dir.glob("*.py")):
            source = py_file.read_text()
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if "news" in mod.lower():
                        violations.append(f"{py_file.name}: ImportFrom {mod!r}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if "news" in alias.name.lower():
                            violations.append(f"{py_file.name}: Import {alias.name!r}")

        assert not violations, (
            "Score layer imports from news module — this is forbidden.\n"
            "News must never influence scoring math.\n"
            f"Violations: {violations}"
        )

    def test_composite_score_not_mutated_by_news(self):
        """Passing news to render() must not change cs.composite."""
        cs = _composite(score=72.5)
        original_composite = cs.composite

        news = _news_result(n_items=5)
        _render_to_str(cs, news=news)

        assert cs.composite == original_composite, (
            f"cs.composite changed from {original_composite} to {cs.composite} "
            "after render() with news — news must never affect the score."
        )

    def test_same_composite_value_regardless_of_news(self):
        """The composite value in the rendered output is identical with or without news."""
        cs = _composite(score=58.3)
        news = _news_result(n_items=4)

        text_no_news = _render_to_str(cs, news=None)
        text_with_news = _render_to_str(cs, news=news)

        # Both renders must show the same composite score
        score_str = f"{cs.composite:.1f}"
        assert score_str in text_no_news, "Score not found in no-news render."
        assert score_str in text_with_news, "Score not found in news render."
        # Composite value on cs is unchanged
        assert cs.composite == 58.3

    def test_news_not_in_classify_module(self):
        """The classify layer must not import news."""
        classify_dir = pathlib.Path(
            __file__
        ).parent.parent / "src" / "value_analyzer" / "classify"

        for py_file in sorted(classify_dir.glob("*.py")):
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "news" not in (node.module or "").lower(), (
                        f"classify/{py_file.name} imports from news: {node.module!r}"
                    )

    def test_news_not_in_data_module(self):
        """The data layer must not import news."""
        data_dir = pathlib.Path(
            __file__
        ).parent.parent / "src" / "value_analyzer" / "data"

        for py_file in sorted(data_dir.glob("*.py")):
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "news" not in (node.module or "").lower(), (
                        f"data/{py_file.name} imports from news: {node.module!r}"
                    )
