"""Backtest configuration — all default assumptions live here."""

from __future__ import annotations

from datetime import date

# ── Default as-of dates ────────────────────────────────────────────────────
# Year-end snapshots 2013–2021.  As of today (2026), every window (1y, 3y,
# 5y) is fully observable for all nine dates.  Earlier dates would require
# EDGAR fundamentals that may pre-date the XBRL era.
DEFAULT_AS_OF_DATES: list[date] = [
    date(2013, 12, 31),
    date(2014, 12, 31),
    date(2015, 12, 31),
    date(2016, 12, 31),
    date(2017, 12, 31),
    date(2018, 12, 31),
    date(2019, 12, 31),
    date(2020, 12, 31),
    date(2021, 12, 31),
]

# ── Transaction costs ──────────────────────────────────────────────────────
# Round-trip cost in basis points (1 bp = 0.01%).
# 20 bps = 0.20% per round trip — conservative for liquid large-caps.
# Note: bid-ask spreads, market impact, slippage, and taxes are NOT modelled.
# Actual costs for a retail investor will be higher than this estimate.
TRANSACTION_COST_BPS: float = 20.0

# ── Benchmark ─────────────────────────────────────────────────────────────
DEFAULT_BENCHMARK: str = "SPY"

# ── Forward-return horizons ────────────────────────────────────────────────
# Calendar-day counts: 252 trading days ≈ 365 calendar days.
HORIZON_DAYS: dict[str, int] = {
    "1y": 365,
    "3y": 3 * 365,
    "5y": 5 * 365,
}

# ── Quintile settings ──────────────────────────────────────────────────────
N_QUINTILES: int = 5  # Q1 = top scores, Q5 = bottom scores

# ── Statistical significance ───────────────────────────────────────────────
ALPHA: float = 0.10  # threshold for labelling a result "possibly significant"
# NOTE: with ~9 annual dates, virtually no result will clear this bar.
# The honest conclusion is almost always "not enough data."

# ── Minimum observations ───────────────────────────────────────────────────
MIN_QUINTILE_OBS: int = 5  # skip quintile stats if fewer observations
