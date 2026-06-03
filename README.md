# value-analyzer

A command-line tool for deep value-investing analysis of a single stock ticker.
Produces a structured, multi-framework analysis grounded entirely in public data.

> **This tool is for educational use only.**
> It is NOT financial advice and NOT a buy or sell recommendation.
> All output is analysis — the investor makes every decision.

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd value-analyzer

# 2. Create a virtual environment and install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Analyse a ticker
value-analyzer KO
```

Or with Make:

```bash
make install
make run TICKER=KO
```

---

## Usage

```
value-analyzer TICKER [options]
```

| Flag | Description |
|------|-------------|
| `TICKER` | Stock ticker symbol (e.g. `AAPL`, `KO`, `BRK-B`) |
| `--as-of YYYY-MM-DD` | Analysis date — no data after this date is used (default: today) |
| `--markdown` | Emit plain-text output instead of rich terminal rendering |
| `--no-ai` | Skip the optional AI commentary layer; no API call is made |
| `--no-news` | Skip news fetching (reserved for a future news-sentiment layer) |
| `--backtest` | Run the offline backtest engine on the built-in universe |
| `--tune` | Run walk-forward weight tuning (train 2013–2017, validate 2018–2021) |
| `--refresh` | Force cache invalidation before fetching data |
| `--verbose` | Enable debug logging |
| `--version` | Print version and exit |

### Examples

```bash
# Terminal report for Coca-Cola, today
value-analyzer KO

# Point-in-time analysis — no data after 2022-12-31
value-analyzer KO --as-of 2022-12-31

# Pipe plain-text report to a file
value-analyzer KO --markdown > ko_report.txt

# Disable AI commentary (no ANTHROPIC_API_KEY needed)
value-analyzer KO --no-ai

# Run the offline backtest
value-analyzer --backtest

# Walk-forward weight tuning
value-analyzer --tune
```

### AI commentary (optional)

Set `ANTHROPIC_API_KEY` in your environment to enable qualitative AI interpretation
of the analysis. The model is instructed to interpret only — it never introduces
figures not already present in the structured analysis, and it never gives
buy/sell guidance. If the key is absent or the call fails the report renders
normally without it.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
value-analyzer KO
```

---

## What the report contains

Each report runs the following analytical layers in order:

1. **Classification** — moat type, revenue model, growth profile, capital intensity,
   derived from the business's historical financials and sector.
2. **Moat score** — assesses durability of competitive advantage: gross-margin
   stability, ROIC vs. cost of capital, pricing power signals.
3. **Financial health score** — balance-sheet strength, debt service, cash-flow
   quality, Altman Z-score proxy.
4. **Valuation score** — three intrinsic-value frameworks (no-growth earnings power,
   normalised DCF, asset-based), combined into an average IV estimate with a
   margin-of-safety calculation.
5. **Management quality score** — capital allocation history, buyback/dilution
   patterns, return on invested capital trend.
6. **Peer comparison** — same-category peers drawn from value-investor 13F filings;
   presented as reference context, not a scoring input.
7. **Bull / bear summary** — automatic extraction of the strongest positive and
   negative signals from the four sub-scores.
8. **Position-sizing context** — illustrative outputs of Graham, Kelly, and
   equal-weight frameworks; never a directive.
9. **AI commentary** (optional) — qualitative interpretation using Claude;
   interpretation only, no new numbers introduced.

---

## Data sources

| Source | What it provides | Latency / caveats |
|--------|-----------------|-------------------|
| **Yahoo Finance** (via `yfinance`) | Price history, basic fundamentals | Free; rate-limited; subject to change |
| **SEC EDGAR** (XBRL API) | Filed financial statements | Free; US-listed companies only; filing date used (not period-end) |

**Important**: data is fetched using the *filing date*, not the period-end date, to
prevent lookahead bias. A 2022-Q4 filing submitted on 2023-02-15 will not appear in
an analysis with `--as-of 2023-02-01`.

Fetched data is cached locally under `~/.cache/value_analyzer/` (prices) and
`/tmp/value_analyzer_cache/` (fundamentals). Use `--refresh` to invalidate the cache.

---

## Architecture

```
data  →  classify  →  score  →  report
              ↑
           backtest  (offline validation layer)
```

Each layer may only import from layers to its left. See [CLAUDE.md](CLAUDE.md) for
the full design rules, including the no-lookahead constraint and the no-advice rule.

---

## Backtest

The built-in backtest runs a walk-forward analysis on a small hand-picked universe
of 20 tickers over 2013–2021, using annual analysis dates. It measures whether
composite scores correlated with 1-year forward returns over this period.

```bash
value-analyzer --backtest    # run and print the backtest report
value-analyzer --tune        # walk-forward weight tuning
```

The tuning uses 2013–2017 as a training window and 2018–2021 as a validation window.
Weights are selected by Spearman rank-correlation on the training set and evaluated
on held-out data.

---

## LIMITATIONS

Read this section before drawing any conclusions from the output.

### 1. Very small sample
The backtest universe contains ~20 tickers over ~9 years. This is far too small
to draw statistically robust conclusions about the analytical framework. A result
that looks promising may be driven entirely by a handful of stocks or one unusual
market regime.

### 2. Short free-data history
Yahoo Finance and SEC EDGAR provide reliable machine-readable data back to roughly
2010–2013 for most tickers. This covers only two full market cycles. Longer,
cleaner data (e.g. CRSP/Compustat via academic access) would substantially
increase the reliability of backtest results.

### 3. Survivorship bias
The backtest universe was selected by a human who already knows these companies
exist and are well-known. This introduces survivorship bias: failed or obscure
companies are absent, which almost certainly inflates any apparent edge.

### 4. Costs not modeled
The backtest does not account for trading commissions, bid-ask spreads, market
impact, short-term capital gains tax, or the cost of the analyst's time. Any
gross return signal would be smaller net of these real-world frictions.

### 5. Framework assumptions
Every intrinsic-value estimate rests on explicit assumptions (WACC, terminal
growth rate, normalisation periods). Small changes in these inputs produce large
changes in the output. The tool displays its assumptions but cannot validate them
for any specific company or market environment.

### 6. Not financial advice
This tool is an educational aid for an investor doing their own research. It does
not know your tax situation, time horizon, risk tolerance, existing portfolio, or
personal circumstances. A high score is not a recommendation to buy; a low score
is not a recommendation to sell.

### 7. A passing backtest is evidence, not a guarantee
Even if the composite score showed meaningful correlation with forward returns in
the sample, markets change, edges decay, and past performance does not predict
future results. Treat any backtest finding as a hypothesis to investigate further,
not a confirmed edge.

---

## Development

```bash
make install      # set up venv
make test         # run all 237 tests
make test-fast    # skip slow integration tests
make lint         # ruff + mypy
make run          # analyse KO
make run TICKER=AAPL
```

### Running tests directly

```bash
source .venv/bin/activate
pytest                        # all tests
pytest tests/test_report.py   # one module
pytest -m "not integration"   # skip network-hitting tests
```

### Project structure

```
src/value_analyzer/
├── data/          # fetch + cache + as_of() firewall
├── classify/      # rule-based business classification
├── score/         # four sub-scorers + composite
├── peers/         # same-category peer comparison
├── report/        # rich terminal renderer
├── backtest/      # offline walk-forward validation
├── ai/            # optional AI commentary (claude-opus-4-8)
└── exceptions.py  # TickerNotFoundError, DataUnavailableError

tests/             # 237 tests; pytest markers: integration
```

---

## License

MIT. See LICENSE for details.
