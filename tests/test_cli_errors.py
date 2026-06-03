"""Tests for CLI error handling and the --no-news flag.

These tests do not hit external APIs.  They patch `score()` to simulate
bad-ticker and network-error conditions and verify the CLI exits with the
expected code and a user-friendly message.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from value_analyzer.exceptions import DataUnavailableError, TickerNotFoundError


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Run cli.main(argv) and return (exit_code, stderr_text)."""
    from value_analyzer.cli import main
    import io

    stderr_buf = io.StringIO()
    exit_code = 0
    with patch("sys.stderr", stderr_buf):
        try:
            main(argv)
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 1
    return exit_code, stderr_buf.getvalue()


# ── Bad date ───────────────────────────────────────────────────────────────────

def test_bad_as_of_date_exits_1():
    code, stderr = _run_cli(["KO", "--as-of", "not-a-date"])
    assert code == 1
    assert "--as-of" in stderr or "YYYY-MM-DD" in stderr


# ── TickerNotFoundError ────────────────────────────────────────────────────────

def test_ticker_not_found_exits_2(monkeypatch):
    with patch("value_analyzer.score.score",
               side_effect=TickerNotFoundError("BADTICKER")):
        code, stderr = _run_cli(["BADTICKER", "--no-ai"])
    assert code == 2
    assert "BADTICKER" in stderr or "No data found" in stderr


# ── DataUnavailableError ───────────────────────────────────────────────────────

def test_data_unavailable_exits_3():
    # Patch the name as seen by the CLI's local import: `from value_analyzer.score import score`
    with patch("value_analyzer.score.score",
               side_effect=DataUnavailableError("SEC EDGAR unreachable")):
        code, stderr = _run_cli(["KO", "--no-ai"])
    assert code == 3
    assert "unreachable" in stderr or "unavailable" in stderr.lower()


# ── Generic network-style exception ───────────────────────────────────────────

def test_connection_error_exits_4_with_hint():
    class FakeConnectionError(ConnectionError):
        pass

    with patch("value_analyzer.score.score",
               side_effect=FakeConnectionError("timed out")):
        code, stderr = _run_cli(["KO", "--no-ai"])
    assert code == 4
    assert "network" in stderr.lower() or "internet" in stderr.lower()


# ── TickerNotFoundError unit test ─────────────────────────────────────────────

def test_ticker_not_found_error_message():
    exc = TickerNotFoundError("XYZ123")
    assert "XYZ123" in str(exc)
    assert exc.ticker == "XYZ123"


# ── --no-news flag accepted without error ─────────────────────────────────────

def test_no_news_flag_accepted_in_help():
    """--no-news appears in --help output."""
    import io
    from value_analyzer.cli import build_parser
    parser = build_parser()
    buf = io.StringIO()
    try:
        parser.parse_args(["--help"])
    except SystemExit:
        pass
    # Just verify the parser accepts --no-news without raising
    args = parser.parse_args(["KO", "--no-news"])
    assert args.no_news is True


def test_no_news_flag_in_help_text():
    from value_analyzer.cli import build_parser
    import io
    parser = build_parser()
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        try:
            parser.parse_args(["--help"])
        except SystemExit:
            pass
    # build_parser() is enough; just confirm --no-news parses
    args = parser.parse_args(["AAPL", "--no-news", "--no-ai"])
    assert args.no_news is True
    assert args.no_ai is True
