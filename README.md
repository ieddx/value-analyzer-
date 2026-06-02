# value-analyzer

A command-line tool for deep value-investing analysis of a single stock ticker.

**Educational use only. Not financial advice.**

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
value-analyzer --help
```

## Usage

```bash
value-analyzer AAPL
value-analyzer AAPL --as-of 2023-12-31
```

## Architecture

See [CLAUDE.md](CLAUDE.md) for the four-layer design and all development rules.
