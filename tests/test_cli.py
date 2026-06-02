"""Smoke tests for the CLI entry point."""

import pytest
from value_analyzer.cli import build_parser


def test_help_exits_cleanly():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_version_exits_cleanly():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0


def test_ticker_parsed():
    parser = build_parser()
    args = parser.parse_args(["AAPL", "--as-of", "2024-01-01"])
    assert args.ticker == "AAPL"
    assert args.as_of == "2024-01-01"
