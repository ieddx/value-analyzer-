# value-analyzer

A command-line tool for educational value-investing analysis of a single stock ticker.
It fetches public financial data, runs four valuation frameworks, and produces a
structured terminal report with optional AI interpretation.

> **This tool is for educational use only.**
> It is NOT financial advice and NOT a buy or sell recommendation.
> All output is analysis — every decision and every risk is the investor's own.

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd value-analyzer

# 2. Create a virtual environment and install
make install

# 3. Analyse a ticker
make run TICKER=KO
```

Or without Make:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m value_analyzer.cli KO
```

---

## CLI flags

```
value-analyzer TICKER [options]
```

| Flag | Description |
|------|-------------|
| `TICKER` | Stock ticker symbol (e.g. `AAPL`, `KO`, `BRK-B`) |
| `--as-of YYYY-MM-DD` | Analysis date — no data after this date is used (default: today) |
| `--markdown` | Emit plain-text output instead of rich terminal rendering |
| `--no-ai` | Skip AI commentary; no API call is made |
| `--no-news` | Skip Finnhub news fetching |
| `--backtest` | Run the offline walk-forward backtest on the built-in universe |
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

# Skip AI and news (no API keys needed)
value-analyzer KO --no-ai --no-news

# Pipe plain-text report to a file
value-analyzer KO --markdown > ko_report.txt

# Run the offline backtest
value-analyzer --backtest

# Walk-forward weight tuning
value-analyzer --tune
```

---

## API keys (optional)

Both keys are optional. Set them as environment variables — never put them in a file
that could be committed.

| Key | Purpose |
|-----|---------|
| `ANTHROPIC_API_KEY` | Enables AI commentary (Claude). Without it the report renders normally. |
| `FINNHUB_API_KEY` | Enables the recent-news panel. Without it the news section is skipped gracefully. |

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export FINNHUB_API_KEY=...
value-analyzer KO
```

Use `--no-ai` and `--no-news` to suppress the corresponding sections regardless of
whether the keys are set.

---

## What the report contains

Each report runs the following analytical layers in order:

1. **Classification** — moat type, revenue model, growth profile, capital intensity,
   derived from historical financials and sector.
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
   reference context only, not a scoring input.
7. **Recent news** *(optional)* — up to 8 headlines from Finnhub; framed as context
   that may post-date the most recent SEC filing. Has no effect on any score.
8. **Bull / bear summary** — automatic extraction of the strongest positive and
   negative signals from the four sub-scores.
9. **Position-sizing context** — illustrative outputs of Graham, Kelly, and
   equal-weight frameworks; framed as "if a value investor applied framework X,
   the implied sizing would be Y" — never a directive.
10. **AI commentary** *(optional)* — qualitative interpretation using Claude; no new
    numbers, no buy/sell guidance.

---

## Data sources

| Source | What it provides | Notes |
|--------|-----------------|-------|
| **SEC EDGAR** (XBRL API) | Filed financial statements | Free; US-listed only; filing date used (not period-end date) |
| **Yahoo Finance** (via `yfinance`) | Price history, basic fundamentals | Free; rate-limited; subject to change without notice |
| **Finnhub** *(optional)* | Recent news headlines | Requires `FINNHUB_API_KEY`; no API key → news section skipped gracefully |

**Lookahead firewall**: data is fetched using the *filing date*, not the period-end
date. A 2022-Q4 filing submitted on 2023-02-15 will not appear in an analysis run
with `--as-of 2023-02-01`. This constraint is enforced throughout the codebase and
tested explicitly.

Fetched data is cached locally under `~/.cache/value_analyzer/` (prices) and
`/tmp/value_analyzer_cache/` (fundamentals). Use `--refresh` to invalidate.

---

## Architecture

```
data  →  classify  →  peers  →  score  →  report
                                              ↑
                                   news  ────┘  (additive only; no score effect)

backtest  (offline validation — never runs in the live analysis path)
```

Layer import rules:
- Each layer may only import from layers to its left.
- `news/` is a sibling of `data/`; only `cli`, `report`, and `ai` import from it.
- `score/`, `classify/`, and `data/` are structurally prevented from importing `news/`
  (enforced by an AST-walk test in `tests/test_news.py`).
- `report/` never imports from `backtest/`.

See [CLAUDE.md](CLAUDE.md) for the full design rules.

---

## Backtest results (honest summary)

The built-in backtest runs a walk-forward analysis on a universe of 43 tickers over
annual snapshots from 2013-12-31 to 2021-12-31 (373 scored ticker-date pairs).

```bash
value-analyzer --backtest    # run and print results
value-analyzer --tune        # walk-forward weight tuning
```

### What the numbers say

| Horizon | Q1 mean return | Q5 mean return | Q1−Q5 spread | Hit rate | t-statistic |
|---------|---------------|---------------|-------------|----------|-------------|
| 1-year  | +13.0%        | +13.1%        | −0.1%        | 43%      | −0.02       |
| 3-year  | +45.5%        | +39.2%        | +5.8%        | 44%      | +0.56       |
| 5-year  | +112.1%       | +70.6%        | +40.6%       | 88%      | +2.11       |

Hit rate = fraction of snapshot dates where Q1 mean > Q5 mean. 50% = chance.
Returns are net of a 20 bps round-trip transaction cost model.

### Honest interpretation

- **1-year: no signal.** Q1 underperformed Q5 on average; hit rate below chance.
  The framework does not demonstrate useful 1-year stock-picking ability.
- **5-year: borderline.** t = 2.11 with ~8 degrees of freedom (9 snapshot dates).
  This is suggestive but falls short of conventional significance thresholds on
  such a small sample. It is not a proven edge.
- **Walk-forward tuning is not reliable** on this dataset. With 9 dates and a 43-ticker
  universe, any weight tuning is likely to overfit the training window.
- **Do not trade on this.** These results are directional context — an existence proof
  that the framework produces differentiated estimates, not evidence of a replicable
  edge.

---

## Known limitations

Read this section before drawing conclusions from the output.

### 1. Very small sample
43 tickers over 9 annual snapshots. This is far too small for statistically robust
conclusions. A result that looks promising may be driven by a handful of stocks or
one unusual market regime (e.g. the post-2020 recovery heavily influenced 5-year
returns in this window).

### 2. Short free-data history
Yahoo Finance and SEC EDGAR provide reliable machine-readable data back to roughly
2010–2013 for most tickers — covering only two full market cycles. Longer, cleaner
data (e.g. CRSP/Compustat via academic access) would substantially change the picture.

### 3. Survivorship bias
The backtest universe was selected by a human who already knows these companies exist
and are well-known. Failed, delisted, or obscure companies are absent. This almost
certainly inflates apparent returns.

### 4. Costs not modeled
The backtest deducts 20 bps round-trip but does not account for bid-ask spreads,
market impact, short-term capital gains tax, or the analyst's time. Real-world costs
would reduce any gross signal further.

### 5. Framework is value/Graham-style — it struggles with tech compounders
The valuation frameworks (earnings power, normalised DCF, asset-based) are designed
for businesses with stable, legible economics. They routinely undervalue high-growth
companies that reinvest aggressively, depressing scores for names like AMZN or GOOG
precisely when they were generating the strongest returns.

### 6. Known share-count bug on multi-class-share companies
Diluted share counts for companies with multiple share classes (e.g. BRK-B, GOOGL)
may be inaccurate due to how XBRL filers report share classes separately. This affects
per-share IV estimates for those tickers.

### 7. News layer is untested in live production
The Finnhub news panel was implemented and unit-tested but has not been validated
against a live key over extended use. Edge cases (ticker aliases, non-US companies,
rate limits) may produce missing or incomplete results.

---

## Development

```bash
make install      # create venv and install all dependencies
make test         # run the full test suite (336 tests)
make test-fast    # skip slow integration tests
make lint         # ruff + mypy
make run          # analyse KO (default)
make run TICKER=AAPL
make run-md       # plain-text markdown output
```

### Running tests directly

```bash
source .venv/bin/activate
pytest                          # all tests
pytest tests/test_news.py       # news layer only
pytest -m "not integration"     # skip network-hitting tests
```

### Project structure

```
src/value_analyzer/
├── data/          # fetch + cache + as_of() lookahead firewall
├── classify/      # rule-based business classification
├── peers/         # same-category peer comparison via 13F filings
├── score/         # four sub-scorers + composite
├── news/          # optional Finnhub news layer (additive, no score effect)
├── report/        # rich terminal renderer
├── backtest/      # offline walk-forward validation
├── ai/            # optional AI commentary (Claude)
└── exceptions.py  # TickerNotFoundError, DataUnavailableError

tests/             # 336 tests; pytest markers: integration
```

---

## License

MIT. See LICENSE for details.
