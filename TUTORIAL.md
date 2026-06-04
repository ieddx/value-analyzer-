# value-analyzer — Usage Tutorial

A walkthrough of how to use the tool, how to read the report, and how to think about
the output. This is an educational tool — it produces analysis, not investment advice.
Every decision and its risk stays with you.

---

## What this tool does

You give it one stock ticker. It fetches 10 years of financial data from SEC EDGAR and
yfinance, classifies the business by its economic characteristics, scores it across four
value-investing pillars (moat, financial health, valuation, management), runs three
intrinsic-value estimates, and produces a structured report explaining every number.

It does **not** tell you to buy or sell anything. It tells you what the data says, how
confident it is in that reading, and what value-investing frameworks would say about the
result. The judgment is yours.

---

## Setup (one time only)

```bash
git clone https://github.com/ieddx/value-analyzer-.git
cd value-analyzer-
make install
```

That creates a virtual environment and installs all dependencies. You only do this once.

To activate the environment in future sessions:

```bash
source .venv/bin/activate
```

---

## Running the tool

The basic command:

```bash
make run TICKER=KO
```

Or directly:

```bash
python -m value_analyzer.cli KO
```

Replace `KO` with any US stock ticker. The tool fetches fresh data (or uses its cache
if it ran recently), scores the stock, and prints the full report to your terminal.

### All flags

| Flag | What it does |
|------|-------------|
| `--as-of YYYY-MM-DD` | Score the stock as it looked on a past date (uses only data filed before that date — no lookahead) |
| `--markdown` | Export the report as a plain-text markdown file |
| `--no-ai` | Skip the AI commentary section (no API key needed) |
| `--no-news` | Skip the news panel |
| `--backtest` | Run the backtest engine across the full universe |
| `--tune` | Run the walk-forward weight tuner (read the limitations section first) |
| `--refresh` | Clear the cache and re-fetch all data |
| `--verbose` | Show detailed logging |

---

## A real example: running KO (Coca-Cola)

```bash
make run TICKER=KO
```

Here is what each section of the report means.

---

### Header panel

```
KO    as of 2026-06-03    profile: compounder
Composite score: 61.1/100  ████████░░░░░░░░░░░░

Moat: brand  |  Revenue: recurring  |  Growth: compounder  |  Capital: asset_light
Weights — moat 35%  health 15%  valuation 30%  management 20%
Data completeness: 17/17 inputs (100%)
```

**What to read here:**

- The **composite score** (61.1) is a weighted sum of the four sub-scores below. Higher
  is better, but read the sub-scores — two stocks can share the same composite for very
  different reasons.
- The **profile** (compounder, cyclical, stable, declining) determines the weight
  profile. A compounder weights moat and management higher; a cyclical weights valuation
  higher. The weights are shown explicitly so you always know what the score is
  prioritising.
- **Data completeness** tells you how much of the score is built on real data vs
  missing-data placeholder floors. 100% means every input was found. Below ~80%, treat
  the score as low-confidence and read the notes carefully.
- If a **valuation dispersion caution** appears here, it means the intrinsic-value
  methods disagree significantly — the average IV is unreliable and you should read each
  method separately.

---

### Classification table

```
Dimension         Result        Confidence   Rationale
capital_intensity asset_light   91%          capex/revenue = 2.1% < 5% threshold
revenue_type      recurring     87%          Gross margin CV = 0.08 < 0.12 threshold
moat_type         brand         78%          Gross margin avg 59.3% > 50% brand floor...
growth_profile    compounder    72%          Revenue CAGR = +4.2% > 3% threshold
```

**What to read here:**

The tool classifies the business by its *economic character*, not its sector label.
These classifications determine which peer benchmarks and weight profiles apply.

- **Confidence** tells you how cleanly the data fit the rule. 90%+ is decisive; 60-70%
  means the signal was mixed and the rationale is worth reading.
- Read the **rationale** column when confidence is below 75% — it shows the exact metric
  and threshold that drove the call, so you can judge whether you agree.
- A **"cyclical_commodity"** revenue classification on a company you know isn't a
  commodity usually means its revenue swings are unusually wide (restructuring, hits-
  driven business, etc.). Read the rationale.

---

### Sub-scores

Four panels, each scored out of 100 with every component's points shown explicitly.

#### Moat (weight depends on profile)

```
85.0/100  ████████████████░░░░  (weight 35%)

[+25.0/25] Gross margin avg 59.3% > 50% brand-moat threshold
[+18.0/20] ROIC avg 28.5% > 8% hurdle — clearly earns above cost of capital
[+12.0/15] ROIC std 4.2%, CV = 0.15 — consistent returns; structural moat likely
[+10.0/10] Revenue CAGR +4.2% — moat translates into meaningful growth
[+20.0/30] Gross-margin CV = 0.08 < 0.12 — stable pricing power
```

**What to read here:**

The bracketed format `[+points_earned/points_available]` is the full scoring trace —
every number traces to a reason. A high moat score means the data shows consistent
above-hurdle returns, stable margins, and durable growth. These are the signals Graham
and Buffett associated with a durable competitive advantage.

A low moat score (below ~40) usually means ROIC is below the cost-of-capital hurdle,
or margins are thin or volatile. The tool won't call something a moat just because it's
a well-known brand — it looks for the *financial evidence* of a moat.

#### Health

```
85.3/100  ████████████████░░░░  (weight 15%)

[+22.6/25] D/E = 0.45 — manageable leverage within 0.5-1.5 acceptable range
[+20.0/25] Interest coverage = 27.9x > 10x — debt service trivially comfortable
[+17.7/25] FCF positive in 93% of 15 years — highly consistent cash generation
[+25.0/25] FCF margin avg 21.3% > 15% — strong cash conversion
```

**What to read here:**

Health measures whether the business can survive adversity. High scores mean low debt,
comfortable coverage, and consistent free cash flow. This is the section where you want
to see no red flags — a low health score on any other high-scoring stock is a warning
that quality is being financed with leverage.

FCF consistency is particularly important: a business that generates free cash flow in
90%+ of years is qualitatively different from one that has great years and terrible
years.

#### Valuation

```
25.9/100  ████░░░░░░░░░░░░░░░░  (weight 30%)

[+8.0/20]  P/E vs history: current 23.4x is 118% of 10y median 19.8x — slight premium
[+0.0/20]  P/FCF: current 27.1x is 142% of median 19.1x — expensive vs own history
[+17.9/35] IV estimates (see table below)
[+0.0/25]  Reverse-DCF: implied growth 6.8%/yr — ambitious expectations in the price

⚠ Valuation assumptions: WACC = 9%, terminal growth = 2.5%
⚠ All intrinsic-value estimates are analytical frameworks, not price targets
```

**What to read here:**

This is the section most likely to score low on genuinely good businesses — the market
usually prices quality at a premium. A low valuation score doesn't mean the business is
bad; it means the price is high relative to historical multiples and earnings-power
estimates. A high valuation score means the market is pricing it cheaply relative to
those benchmarks.

The **reverse-DCF implied growth** is a useful sanity check: it asks "what growth rate
must be true forever for today's price to make sense?" If that number seems optimistic
given the business, the price is likely rich.

**Intrinsic-value estimates table:**

```
Item                                          Value
IV (No-growth earnings power EPS/WACC)        $38.20 — premium to current $62.50
IV (Normalised P/E reversion median×EPS)      $52.10 — premium to current $62.50
IV (Graham Number √(22.5 × EPS × BVPS))      $41.30 — premium to current $62.50
Average IV estimate                           $43.87 | Current: $62.50 | MoS: -29.8%
```

Three methods, each making different assumptions, averaged. When all three cluster
together (as here), the average is meaningful. When they diverge widely (a dispersion
caution fires), the average is not reliable — read each method separately and understand
why they disagree.

**Margin of safety** is (IV − price) / price. Negative means you're paying above the
estimated intrinsic value. Value investors typically want a positive margin of safety
(buying below IV) as a cushion against being wrong. How much cushion depends on your
conviction and the uncertainty in the estimate.

#### Management

```
54.0/100  ██████████░░░░░░░░░░  (weight 20%)

[+20.0/30] Share count CAGR = -1.2%/yr — buybacks; each share represents growing piece
[+12.0/25] ROE avg 38.5% — strong but highly variable (CV = 0.82)
[+14.0/25] Return on retained earnings positive — reinvestment is creating value
[+8.0/20]  Dividend consistency — dividends paid in 15/15 years; consistent capital return
```

**What to read here:**

Management scores what capital allocators have *done* with the business's earnings —
not what they've said. The key signals: is share count shrinking (buybacks, good) or
growing (dilution, bad)? Is retained capital generating returns? Is there a pattern of
consistent cash return to shareholders?

A low management score is often the most actionable finding — it can reflect a
management team that dilutes shareholders, destroys capital through bad reinvestment, or
maintains a business without growing it. A high score reflects disciplined capital
allocation over many years.

---

### Peer comparison

```
Same-category peers (brand-moat / asset_light / compounder) — 8 peers
Metric          This stock    Peer median    vs peers
Gross margin    59.3%         54.1%          +5.2pp above
ROIC            28.5%         21.3%          +7.2pp above
P/FCF           27.1x         24.8x          +9% premium
FCF margin      21.3%         18.7%          +2.6pp above
```

**What to read here:**

Your stock compared only against companies in the same economic category — not the
whole market. This is how you know whether "good margins" are good *for this type of
business*. A 22% gross margin looks different for a software company vs a grocery chain.

This panel also surfaces the 13F reference context: how stocks that Berkshire Hathaway
and aligned value investors held in this category looked on these same metrics. Framed
as context, not as a signal.

---

### Bull / Bear summary

```
Bull case
+  ROIC avg 28.5% > 15% — clearly earns above cost of capital; strong economic moat
+  Revenue CAGR +4.2% over 15 years — moat translates into meaningful growth
+  D/E = 0.45 — conservative balance sheet; ample cushion to weather downturns
+  FCF positive in 93% of 15 years — highly consistent cash generation

Bear case
-  Current P/FCF (27.1x) is 142% of median — expensive vs own cash-flow history
-  Gross-margin CV = 0.08 > 0.05 — some pricing variability; moat quality uncertain
-  Average IV $43.87 vs price $62.50 — trading at significant premium to earnings power
```

**What to read here:**

The strongest positive and negative signals from the sub-scores, assembled into a plain-
language summary. Nothing new is introduced here — every line traces back to a number in
the sub-scores above. This is the tool's synthesis of what the data said.

---

### Evaluation & framework context

```
ANALYSIS CONFIDENCE: Moderate-High
Score built on: 17/17 real inputs (100% completeness). IV methods agree (no dispersion).
Backtest context: this framework showed borderline signal at 5-year horizons (t=2.11,
n=42, not statistically robust at α=0.05). No signal at 1-year. Treat as a long-horizon
analytical lens, not a proven predictive model.

FRAMEWORK CONTEXT (educational — not a recommendation):
KO scores 61.1/100 — above-median for brand compounders on quality metrics, held back
by a rich valuation relative to own history and earnings-power estimates.

An investor applying Graham's margin-of-safety framework would note the -29.8% margin of
safety and typically require a positive cushion before acting. How much cushion depends
on conviction and uncertainty tolerance.

An investor applying conviction-based sizing might note that a 61.1 score with strong
moat/health but weak valuation suggests the quality is real but the price is full —
consistent with a "watch and wait for a better entry" posture in that framework.

These are illustrative framework applications. Risk tolerance, portfolio context, and
tax situation differ per investor. Apply independent judgment.
```

**What to read here:**

This section synthesises the report into a confidence assessment (how much to trust the
score) and shows how the result would be *interpreted* under value-investing frameworks
— explicitly framed as illustration, never as instruction. The confidence level draws on
completeness, dispersion, and the backtest context. A low-confidence label here means
the score's inputs were weak — read the notes in the sub-scores before relying on it.

---

### Recent news

```
Recent News (last 30 days — Finnhub) — context only, does not affect score
[Jun 01] KO raises quarterly dividend to $0.515/share  —  Reuters
[May 28] Coca-Cola reaffirms full-year guidance at investor day  —  Bloomberg
[May 15] KO Q1 earnings: organic revenue +6%, EPS beat consensus  —  CNBC
```

**What to read here:**

Recent headlines from a structured financial news API — surfaced for your awareness,
not fed into the score. The score is based on filed fundamentals (SEC EDGAR); news can
reflect material events (earnings, capital raises, guidance changes, M&A) that haven't
reached a filing yet. **You** weigh the news against the analysis. The score is
unchanged regardless of what appears here.

Requires a `FINNHUB_API_KEY` environment variable. Without it, this panel is skipped.

---

### AI Commentary

```
AI Commentary (claude-opus-4-8 — interpret only, never directive)

The report shows a high-quality business — strong and consistent ROIC well above cost of
capital, dominant brand moat with stable margins, and near-zero leverage. The primary
tension is valuation: all three IV methods place fair value meaningfully below the current
price, implying the market is pricing in continued premium growth. The news panel shows
a dividend raise and guidance reaffirmation, consistent with management confidence but
not materially changing the valuation picture. An investor in this framework would be
watching for a price pullback toward the IV range before acting.

This is interpretation of the analysis above — not financial advice.
```

**What to read here:**

The AI layer interprets the numbers the tool already computed — it never introduces new
figures or overrides the deterministic scores. Its job is to synthesise the report in
plain language and flag any material event from the news panel that might affect the
thesis. Treat it as a reading aid, not an authority — it reasons over the same data you
can read above.

Requires an `ANTHROPIC_API_KEY` environment variable. The full report works without it.

---

### Position-sizing context and disclaimer

```
Position-Sizing Context (educational frameworks only)

Composite score:      61.1/100
Implied margin of safety: -29.8%
Weight profile:       compounder

Graham / Buffett tradition: a margin of safety > 25% is generally considered a
meaningful cushion for a stable business... [frameworks described, not applied]

THIS REPORT IS ANALYSIS ONLY — it is NOT financial advice and NOT a buy or sell
recommendation. All intrinsic-value estimates are the output of stated frameworks
applied to public data. The investor must apply their own judgment.
```

The disclaimer is present on every report, every time. It means what it says.

---

## What to do with the output

The tool narrows where you look — it doesn't replace looking. A high composite score
and a positive margin of safety together identify a stock worth *researching further*,
not one to act on automatically. A low score doesn't mean a bad company; it often means
a good company at a full price, or a company outside the tool's lane (large-cap tech
compounders, pre-revenue growth stocks, and financials are all outside its designed
competence).

The useful workflow:

1. Run the tool on a stock you're already curious about.
2. Read the sub-scores — find the strongest bull and bear signals.
3. Check the confidence level and completeness — know how much to trust the score.
4. Read the news panel — flag anything material the filings haven't caught.
5. Form your own thesis: do you agree with the bull case? Can you accept the bear case?
6. Apply your own judgment, risk tolerance, and portfolio context.

The tool informs step 1-4. Steps 5 and 6 are yours.

---

## Known limitations

Be honest about these when presenting or using the tool:

1. **Small backtest sample** — 42 tickers, borderline t-stat (2.11). The 5-year signal
   is suggestive, not proven. Do not treat a high score as a prediction.
2. **No 1-year signal** — the framework showed no edge at 1-year horizons. Short-term
   results are noise by the tool's own evidence.
3. **Survivorship bias** — the backtest universe excludes delisted companies. Real
   returns are likely lower than the backtest suggests.
4. **Framework mismatch for growth/tech** — the Graham/value framework systematically
   undervalues large-cap tech compounders (GOOG, META, AMZN). Moat classifications and
   IV estimates are unreliable for these. The tool is best pointed at stable,
   cash-generative businesses with real book value and steady multiples.
5. **Share-count bug on multi-class shares** — companies with Class A/B/C structures
   (Alphabet, Meta) may show incorrect share-count CAGRs, corrupting the management
   score. Known bug, not yet fixed.
6. **News layer untested live** — the news module is built and passes tests, but has not
   been run with a live Finnhub API key. First live run may surface edge cases.
7. **Not financial advice** — this is an educational tool. Decisions and risk are yours.

---

## Quick reference

```bash
# Basic run
make run TICKER=AAPL

# Past date (point-in-time — uses only data available then)
python -m value_analyzer.cli AAPL --as-of 2023-01-01

# No AI, no news (no API keys needed)
python -m value_analyzer.cli AAPL --no-ai --no-news

# Save as markdown file
python -m value_analyzer.cli AAPL --markdown

# Run full test suite
make test

# Run backtest
python -m value_analyzer.cli --backtest
```
