"""Backtest universe — fixed broad list with explicit survivorship-bias documentation.

POINT-IN-TIME UNIVERSE LIMITATION
══════════════════════════════════════════════════════════════════════════════
A true point-in-time universe would contain exactly the tickers that existed
and were investable on each as-of date.  That requires a commercial data
source (e.g. Compustat point-in-time, CRSP).  Free sources (yfinance, EDGAR)
do not provide historical constituent lists.

This module uses a fixed universe chosen WITHOUT knowledge of returns.
The following biases remain after our best-effort mitigations:

  1. DELISTED / BANKRUPT STOCKS — Companies that were removed from exchanges
     during the backtest window are absent unless they still trade today.
     This understates the true failure rate and biases returns upward.
     Examples of known omissions: Sears (SHLD), Chesapeake Energy (pre-2020).

  2. RECONSTITUTION BIAS — S&P 500 adds winners and removes losers.  We do
     not replicate those changes, so our fixed list may include stocks that
     were added late (mild forward-looking bias) and may omit stocks removed
     early (survival bias in the opposite direction).

  3. ANALYST / MEDIA BIAS — Tickers in this list are well-known companies.
     Obscure small-caps that failed are absent entirely.

MITIGATION STEPS TAKEN
  - Deliberately included known historical UNDERPERFORMERS: GE (massive
    value destruction 2016-2020), IBM (decade of underperformance), INTC
    (lost semiconductor leadership), M/Macy's (secular retail decline),
    WBA/Walgreens (declining CVS/retail pharmacy), T/AT&T (debt-laden,
    dividend cut 2022), VZ (low growth).
  - Universe was documented in mid-2024 and was NOT screened for
    subsequent performance.
  - The list spans all major sectors and economic categories.

HONEST INTERPRETATION
  Results on this universe should be treated as rough directional context
  only.  A statistically rigorous conclusion requires a point-in-time
  universe with several hundred tickers over 20+ years.
"""

from __future__ import annotations

# ── Universe constant ──────────────────────────────────────────────────────
# ~45 large-cap US tickers, primarily S&P 500 members throughout 2013–2021.
# Intentionally includes known underperformers (marked ↓) to partially offset
# survivorship bias.
UNIVERSE: list[str] = [
    # ── Technology ─────────────────────────────────────────────────────────
    "AAPL",   # Apple — dominant compounder
    "MSFT",   # Microsoft — cloud transition success
    "INTC",   # Intel ↓ — lost process leadership
    "IBM",    # IBM ↓ — decade of revenue decline
    "CSCO",   # Cisco — stable but slow growth
    "ORCL",   # Oracle — cloud transition, mixed results
    # ── Consumer Discretionary ─────────────────────────────────────────────
    "AMZN",   # Amazon — rapid compounder
    "HD",     # Home Depot — strong compounder
    "TGT",    # Target — cyclical retailer
    "MCD",    # McDonald's — stable franchise compounder
    "M",      # Macy's ↓ — secular retail decline
    "NKE",    # Nike — brand compounder
    "DIS",    # Disney — mixed results
    # ── Consumer Staples ───────────────────────────────────────────────────
    "KO",     # Coca-Cola — stable brand compounder
    "PEP",    # PepsiCo — stable brand compounder
    "PG",     # Procter & Gamble — defensive compounder
    "JNJ",    # Johnson & Johnson — diversified healthcare/staples
    "CL",     # Colgate-Palmolive — stable
    "WMT",    # Walmart — cost-advantage compounder
    "WBA",    # Walgreens ↓ — pharmacy retail decline
    # ── Financials ─────────────────────────────────────────────────────────
    "JPM",    # JPMorgan Chase — dominant bank
    "BAC",    # Bank of America — cyclical recovery post-2013
    "WFC",    # Wells Fargo — mixed (scandal 2016, recovery)
    "BRK-B",  # Berkshire Hathaway — holding company benchmark
    "AXP",    # American Express — network moat
    "BK",     # Bank of New York Mellon — trust bank
    # ── Energy / Cyclical ──────────────────────────────────────────────────
    "CVX",    # Chevron — major integrated oil
    "XOM",    # ExxonMobil — major integrated oil
    "NUE",    # Nucor — cyclical steel (cost-advantage)
    "CAT",    # Caterpillar — cyclical industrial
    "FCX",    # Freeport-McMoRan — cyclical copper miner
    # ── Healthcare ─────────────────────────────────────────────────────────
    "ABT",    # Abbott Laboratories
    "MDT",    # Medtronic — medical devices
    "UNH",    # UnitedHealth — managed care compounder
    "TMO",    # Thermo Fisher — scientific instruments compounder
    # ── Industrials ────────────────────────────────────────────────────────
    "MMM",    # 3M ↓ — litigation, margin pressure post-2019
    "HON",    # Honeywell — stable industrial compounder
    "GE",     # General Electric ↓ — massive value destruction
    "EMR",    # Emerson Electric — stable industrial
    # ── Telecom ────────────────────────────────────────────────────────────
    "VZ",     # Verizon ↓ — slow growth, high debt
    "T",      # AT&T ↓ — debt-laden, 2022 dividend cut, spinoff
    # ── Payment Networks ───────────────────────────────────────────────────
    "V",      # Visa — network-moat compounder
    "MA",     # Mastercard — network-moat compounder
]

SURVIVORSHIP_BIAS_WARNING = """\
⚠ SURVIVORSHIP BIAS — this backtest uses a fixed universe, not a true
  point-in-time constituent list.  Results are directional context only.
  See value_analyzer/backtest/universe.py for the full methodology note."""
