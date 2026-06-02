# value-analyzer — Claude context

**Read this file first.** Every new session working on this project should start here
before touching any code.

---

## Project goal

A command-line tool that accepts a single stock ticker and produces a deep
value-investing analysis grounded in publicly available financial data. The output
is educational and analytical — it is **never** a buy or sell recommendation.

Target users: individual investors doing their own research. The tool surfaces
valuation frameworks and position-sizing context; the human makes all decisions.

---

## Four-layer architecture

```
data  →  classify  →  score  →  report
              ↑
           backtest  (validation layer, runs offline)
```

| Layer | Package | Responsibility |
|-------|---------|----------------|
| **data** | `value_analyzer/data/` | Fetch and cache raw financial data (price history, fundamentals, filings). Returns only data available as of the analysis date. |
| **classify** | `value_analyzer/classify/` | Categorise the business: sector, moat type, cyclicality, capital intensity, accounting quality flags. |
| **score** | `value_analyzer/score/` | Apply valuation frameworks (DCF, earnings power, asset-based, relative). Produce a composite intrinsic-value estimate with a confidence band. |
| **report** | `value_analyzer/report/` | Render the terminal report via `rich`. Translate score outputs into human-readable analysis and position-sizing *context* (never directives). |
| **backtest** | `value_analyzer/backtest/` | Offline validation: given historical analysis dates, measure how framework estimates compared to subsequent outcomes. Used to calibrate — never run in the live analysis path. |

Each layer may only import from layers to its left. `report` never imports from
`backtest`. `backtest` may import from `data`, `classify`, and `score`.

---

## Cardinal rules

### 1. No lookahead bias — ever

**No function may use data that was not publicly available as of the analysis date.**

- All data-fetch functions accept an `as_of: date` parameter and must respect it.
- Financial statements use the *filing date*, not the period-end date.
- Price data is cut off at market close on `as_of`.
- Violations of this rule corrupt every backtest result and make the tool useless.
  Treat it as the highest-priority correctness constraint.

### 2. Output is analysis, not advice

The `report` layer must never emit directive language: no "buy", "sell", "hold",
"strong buy", "avoid", or equivalent. It may describe valuation gaps, margin of
safety estimates, and how a given position size relates to standard frameworks
(e.g. Kelly, equal-weight, concentration limits) — but the framing is always
"if an investor were applying framework X, the implied sizing would be Y", not
"you should do Y".

Any new report component must pass a read-aloud test: does it sound like advice?
If yes, rewrite it.

---

## Development conventions

- Python ≥ 3.11, use `pyproject.toml` for all packaging.
- Source lives in `src/value_analyzer/`; tests in `tests/`.
- Virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Run tests: `pytest`.
- Entry point: `python -m value_analyzer.cli` or `value-analyzer` after install.
- Use `pydantic` models for all data structures crossing layer boundaries.
- Use `rich` for all terminal output; no bare `print()` in `report/`.

---

## What is NOT implemented yet

Everything. The scaffold is in place; no analysis logic exists. Start with the
`data` layer before touching anything else.
