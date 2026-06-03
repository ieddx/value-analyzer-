"""Entry point for the value-analyzer CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="value-analyzer",
        description=(
            "Deep value-investing analysis for a single stock ticker.\n\n"
            "OUTPUT IS ANALYSIS ONLY — not financial advice or a buy/sell recommendation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        metavar="TICKER",
        help="Stock ticker symbol to analyse (e.g. AAPL, KO, BRK-B).",
    )
    parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        default=str(date.today()),
        help="Analysis date (default: today). No data after this date will be used.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit plain-text markdown instead of rich terminal output.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run the backtest engine on the default universe and date range.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run walk-forward weight tuning (train 2013–2017, validate 2018–2021).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force cache invalidation before fetching data.",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip the optional AI commentary layer (no API call made).",
    )
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Skip news fetching (reserved for a future news-sentiment layer).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="value-analyzer 0.1.0",
    )
    return parser


def _run_analysis(ticker: str, as_of_str: str, markdown: bool, no_ai: bool) -> None:
    from datetime import date as _date
    from value_analyzer.exceptions import DataUnavailableError, TickerNotFoundError
    from value_analyzer.score import score
    from value_analyzer.report import render, render_markdown
    from value_analyzer.ai import generate_commentary

    try:
        as_of = _date.fromisoformat(as_of_str)
    except ValueError:
        print(f"Error: --as-of must be YYYY-MM-DD, got {as_of_str!r}", file=sys.stderr)
        sys.exit(1)

    try:
        result = score(ticker, as_of_date=as_of)
    except TickerNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except DataUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(3)
    except Exception as exc:  # network errors, unexpected failures
        _maybe_network_error(exc)
        print(f"Unexpected error fetching data for {ticker!r}: {exc}", file=sys.stderr)
        sys.exit(4)

    ai_commentary: str | None = None
    ai_attempted = not no_ai
    if ai_attempted:
        ai_commentary = generate_commentary(result)

    if markdown:
        print(render_markdown(result, ai_commentary=ai_commentary, ai_attempted=ai_attempted))
    else:
        render(result, ai_commentary=ai_commentary, ai_attempted=ai_attempted)


def _maybe_network_error(exc: Exception) -> None:
    """Print a friendlier hint when exc looks like a network failure."""
    name = type(exc).__name__
    if any(kw in name.lower() for kw in ("connection", "timeout", "ssl", "request", "http")):
        print(
            "Hint: this looks like a network error. Check your internet connection "
            "and try again. Data is fetched from Yahoo Finance and SEC EDGAR.",
            file=sys.stderr,
        )


def _run_backtest() -> None:
    from value_analyzer.backtest.engine import run
    from value_analyzer.backtest.report import format_report
    from value_analyzer.backtest.config import DEFAULT_AS_OF_DATES
    from value_analyzer.backtest.universe import UNIVERSE

    result = run(UNIVERSE, DEFAULT_AS_OF_DATES)
    print(format_report(result))


def _run_tune() -> None:
    from value_analyzer.backtest.engine import run
    from value_analyzer.backtest.config import DEFAULT_AS_OF_DATES
    from value_analyzer.backtest.universe import UNIVERSE
    from value_analyzer.backtest.tuning import tune_weights
    from value_analyzer.score.config import CATEGORY_WEIGHTS

    all_dates = DEFAULT_AS_OF_DATES
    train_dates = [d for d in all_dates if d.year <= 2017]
    val_dates   = [d for d in all_dates if d.year >= 2018]

    train_snaps = run(UNIVERSE, train_dates).snapshots
    val_snaps   = run(UNIVERSE, val_dates).snapshots

    result = tune_weights(train_snaps, val_snaps, CATEGORY_WEIGHTS)
    print(result)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.refresh:
        import shutil
        from value_analyzer.data.cache import CACHE_DIR
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.backtest:
        _run_backtest()
        return

    if args.tune:
        _run_tune()
        return

    if args.ticker is None:
        parser.print_help()
        sys.exit(0)

    _run_analysis(args.ticker, args.as_of, args.markdown, args.no_ai)


if __name__ == "__main__":
    main()
