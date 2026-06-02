"""Entry point for the value-analyzer CLI."""

from __future__ import annotations

import argparse
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
        help="Stock ticker symbol to analyse (e.g. AAPL, BRK-B).",
    )
    parser.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        default=str(date.today()),
        help="Analysis date (default: today). No data after this date will be used.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="value-analyzer 0.1.0",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.ticker is None:
        parser.print_help()
        sys.exit(0)

    # Placeholder — analysis layers not yet implemented.
    print(f"[value-analyzer] ticker={args.ticker!r}  as_of={args.as_of}")
    print("Analysis layers not yet implemented. See CLAUDE.md.")


if __name__ == "__main__":
    main()
