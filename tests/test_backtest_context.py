"""Tests for the live backtest-context line in renderer.py.

Four required cases:
  (a) "not yet run" message when the summary file is absent
  (b) correctly displays a stored summary
  (c) low t-stat renders with "not statistically robust" caveat
  (d) malformed JSON falls back gracefully

All tests patch _BACKTEST_SUMMARY_PATH so they never touch the real filesystem.
The module is now named renderer.py, so the package-level `render` function no
longer shadows the submodule — a plain import and direct patch() both work.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

import value_analyzer.report.renderer as _render_mod
import value_analyzer.backtest.engine as _engine_mod

from value_analyzer.report.renderer import (
    _backtest_context_line,
    _load_backtest_summary,
    DISCLAIMER_TEXT,
    render,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _with_summary_file(data: dict | str | None):
    """Context manager: redirect _BACKTEST_SUMMARY_PATH to a temp file.

    Pass None to simulate a missing file.
    Pass a dict to write valid JSON.
    Pass a str to write raw content (use for malformed JSON tests).

    Patches the module-level variable directly on the imported module object
    to avoid the name-collision between the `render` function and the
    `render` submodule that makes patch() with a dotted string fail.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "backtest_summary.json"
        if data is not None:
            content = json.dumps(data) if isinstance(data, dict) else data
            summary_path.write_text(content)

        orig = _render_mod._BACKTEST_SUMMARY_PATH
        _render_mod._BACKTEST_SUMMARY_PATH = summary_path
        try:
            yield summary_path
        finally:
            _render_mod._BACKTEST_SUMMARY_PATH = orig


def _render_context_line(**kwargs) -> str:
    """Render just the backtest context line and strip rich markup."""
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=False, width=300)
    con.print(_backtest_context_line(**kwargs))
    return buf.getvalue()


# ── (a) File absent → "not yet run" ───────────────────────────────────────────

def test_absent_file_shows_not_yet_run():
    with _with_summary_file(None):
        line = _render_context_line()
    assert "not yet run" in line.lower()
    assert "unvalidated" in line.lower()


def test_absent_file_still_suggests_backtest_command():
    with _with_summary_file(None):
        line = _render_context_line()
    assert "--backtest" in line


# ── (b) Valid summary → displays stored values ─────────────────────────────────

_GOOD_SUMMARY = {
    "run_date": "2026-05-30",
    "date_range": "2013-01-01–2021-12-31",
    "n_scored": 342,
    "n_attempted": 360,
    "universe_size": 45,
    "benchmark_ticker": "SPY",
    "transaction_cost_bps": 20.0,
    "q1_vs_benchmark_1y": 0.032,    # +3.2% vs SPY
    "q1_q5_spread_1y": 0.048,
    "t_stat_1y": 2.31,
    "p_value_1y": 0.041,
    "tuning_train_val_gap": None,
}


def test_valid_summary_shows_run_date():
    with _with_summary_file(_GOOD_SUMMARY):
        line = _render_context_line()
    assert "2026-05-30" in line


def test_valid_summary_shows_edge_vs_benchmark():
    with _with_summary_file(_GOOD_SUMMARY):
        line = _render_context_line()
    # +3.2% edge vs SPY
    assert "+3.2%" in line
    assert "SPY" in line


def test_valid_summary_shows_t_stat():
    with _with_summary_file(_GOOD_SUMMARY):
        line = _render_context_line()
    assert "t=2.31" in line


def test_valid_summary_shows_sample_size():
    with _with_summary_file(_GOOD_SUMMARY):
        line = _render_context_line()
    assert "n=342" in line


def test_valid_summary_shows_date_range():
    with _with_summary_file(_GOOD_SUMMARY):
        line = _render_context_line()
    assert "2013-01-01" in line
    assert "2021-12-31" in line


def test_valid_summary_always_has_caveat():
    with _with_summary_file(_GOOD_SUMMARY):
        line = _render_context_line()
    assert "evidence" in line.lower() or "guarantee" in line.lower() or "decay" in line.lower()


# ── (c) Low t-stat → "not statistically robust" ───────────────────────────────

_LOW_TSTAT_SUMMARY = {
    **_GOOD_SUMMARY,
    "t_stat_1y": 1.12,
    "p_value_1y": 0.29,
    "q1_vs_benchmark_1y": 0.015,
}


def test_low_tstat_shows_not_robust():
    with _with_summary_file(_LOW_TSTAT_SUMMARY):
        line = _render_context_line()
    lower = line.lower()
    assert "not statistically robust" in lower or "not robust" in lower


def test_low_tstat_still_shows_caveat():
    with _with_summary_file(_LOW_TSTAT_SUMMARY):
        line = _render_context_line()
    assert "evidence" in line.lower() or "guarantee" in line.lower() or "decay" in line.lower()


def test_high_tstat_does_not_claim_strong_edge_without_caveat():
    """A t-stat above 2 must not print 'strong edge' without the caveat phrase."""
    with _with_summary_file(_GOOD_SUMMARY):
        line = _render_context_line()
    lower = line.lower()
    # If "strong edge" appears, the caveat must also appear on the same line
    if "strong edge" in lower:
        assert any(w in lower for w in ("evidence", "guarantee", "caveat", "small-sample"))


def test_negative_edge_small_n_shows_not_robust():
    """Small n_scored always triggers not-robust even if t looks OK."""
    small_summary = {
        **_GOOD_SUMMARY,
        "n_scored": 15,
        "t_stat_1y": 2.5,   # t > 2 but n < 30
        "p_value_1y": 0.03,
    }
    with _with_summary_file(small_summary):
        line = _render_context_line()
    lower = line.lower()
    assert "not statistically robust" in lower or "not robust" in lower


# ── (d) Malformed JSON → graceful fallback ─────────────────────────────────────

def test_malformed_json_falls_back_to_not_yet_run():
    with _with_summary_file("{this is not valid json!!!"):
        line = _render_context_line()
    assert "not yet run" in line.lower() or "unvalidated" in line.lower()


def test_wrong_type_json_falls_back():
    """JSON that parses but is a list, not a dict."""
    with _with_summary_file('[1, 2, 3]'):
        # _load_backtest_summary returns None for non-dict
        line = _render_context_line()
    assert "not yet run" in line.lower() or "unvalidated" in line.lower()


def test_partial_json_renders_gracefully():
    """Summary with only some keys — must not crash."""
    partial = {"run_date": "2026-01-01", "n_scored": 50}
    with _with_summary_file(partial):
        line = _render_context_line()
    # Should render something, not crash
    assert "2026-01-01" in line or "not yet run" in line.lower()


# ── _load_backtest_summary unit tests ─────────────────────────────────────────

def test_load_returns_none_when_file_missing():
    with _with_summary_file(None):
        result = _load_backtest_summary()
    assert result is None


def test_load_returns_dict_for_valid_file():
    with _with_summary_file(_GOOD_SUMMARY):
        result = _load_backtest_summary()
    assert isinstance(result, dict)
    assert result["run_date"] == "2026-05-30"


def test_load_returns_none_for_malformed_json():
    with _with_summary_file("not json"):
        result = _load_backtest_summary()
    assert result is None


# ── write_backtest_summary integration ────────────────────────────────────────

def test_write_backtest_summary_produces_readable_file():
    """write_backtest_summary() writes a file that _load_backtest_summary() can read."""
    from datetime import date
    from value_analyzer.backtest.engine import write_backtest_summary

    mock_result = MagicMock()
    mock_result.run_date = date(2026, 5, 30)
    mock_result.as_of_dates = [date(2013, 12, 31), date(2021, 12, 31)]
    mock_result.n_scored = 200
    mock_result.n_attempted = 210
    mock_result.universe = ["KO", "PEP"]
    mock_result.benchmark_ticker = "SPY"
    mock_result.transaction_cost_bps = 20.0
    mock_result.q1_vs_benchmark_1y = 0.025
    mock_result.q1_q5_spread_1y = 0.040
    mock_result.t_stat_1y = 1.8
    mock_result.p_value_1y = 0.10

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "backtest_summary.json"

        # Patch both module-level path constants directly on their modules
        orig_engine = _engine_mod._SUMMARY_PATH
        orig_render = _render_mod._BACKTEST_SUMMARY_PATH
        _engine_mod._SUMMARY_PATH = tmp_path
        _render_mod._BACKTEST_SUMMARY_PATH = tmp_path
        try:
            write_backtest_summary(mock_result)
            assert tmp_path.exists(), "Summary file should be written"
            loaded = _load_backtest_summary()
        finally:
            _engine_mod._SUMMARY_PATH = orig_engine
            _render_mod._BACKTEST_SUMMARY_PATH = orig_render

    assert loaded is not None
    assert loaded["run_date"] == "2026-05-30"
    assert loaded["n_scored"] == 200
    assert abs(loaded["q1_vs_benchmark_1y"] - 0.025) < 1e-9
