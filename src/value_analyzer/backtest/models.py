"""Pydantic models for the backtest layer."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class SnapshotResult(BaseModel):
    """Score and forward return for one (ticker, as_of_date) pair."""

    ticker: str
    as_of_date: date
    composite_score: Optional[float] = None
    weight_profile: Optional[str] = None
    quintile: Optional[int] = None          # 1 (top) – 5 (bottom), assigned after ranking
    error: Optional[str] = None             # set when scoring failed; other fields are None

    # Sub-scores stored so the tuner can reweight without re-running the full pipeline
    moat_score: Optional[float] = None
    health_score: Optional[float] = None
    valuation_score: Optional[float] = None
    management_score: Optional[float] = None

    # Gross forward returns — price appreciation only, no dividends
    fwd_return_1y: Optional[float] = None
    fwd_return_3y: Optional[float] = None
    fwd_return_5y: Optional[float] = None

    # Net forward returns after subtracting round-trip transaction cost
    net_return_1y: Optional[float] = None
    net_return_3y: Optional[float] = None
    net_return_5y: Optional[float] = None

    # Benchmark (SPY) return for the identical measurement window
    benchmark_return_1y: Optional[float] = None
    benchmark_return_3y: Optional[float] = None
    benchmark_return_5y: Optional[float] = None


class QuintileStats(BaseModel):
    """Aggregate forward-return statistics for one score quintile."""

    quintile: int
    label: str        # "Q1 (top 20%)" … "Q5 (bottom 20%)"
    n_obs: int

    mean_net_return_1y: Optional[float] = None
    mean_net_return_3y: Optional[float] = None
    mean_net_return_5y: Optional[float] = None

    median_net_return_1y: Optional[float] = None
    median_net_return_3y: Optional[float] = None
    median_net_return_5y: Optional[float] = None


class BacktestResult(BaseModel):
    """Full backtest output — scores, forward returns, and analysis."""

    run_date: date
    universe: list[str]
    as_of_dates: list[date]
    benchmark_ticker: str
    transaction_cost_bps: float

    # Execution summary
    n_attempted: int
    n_scored: int               # successfully scored
    n_errors: int               # scoring failures (data gaps, delisted, etc.)
    n_with_1y_return: int
    n_with_3y_return: int
    n_with_5y_return: int

    # Per-(ticker, date) records — sub-scores stored so tuner can reweight without re-scoring
    snapshots: list[SnapshotResult] = Field(default_factory=list)

    # Quintile aggregate stats
    quintile_stats: list[QuintileStats] = Field(default_factory=list)

    # Top-line Q1 vs Q5 spread (mean difference in net returns)
    q1_q5_spread_1y: Optional[float] = None
    q1_q5_spread_3y: Optional[float] = None
    q1_q5_spread_5y: Optional[float] = None

    # Hit rate: fraction of snapshot dates where Q1 mean return > Q5 mean return
    # Ranges 0–1; 0.5 means no better than chance at the date level.
    hit_rate_1y: Optional[float] = None
    hit_rate_3y: Optional[float] = None
    hit_rate_5y: Optional[float] = None

    # Q1 vs SPY benchmark
    q1_vs_benchmark_1y: Optional[float] = None
    q1_vs_benchmark_3y: Optional[float] = None
    q1_vs_benchmark_5y: Optional[float] = None

    # Benchmark average return across all measurement periods
    benchmark_avg_1y: Optional[float] = None
    benchmark_avg_3y: Optional[float] = None
    benchmark_avg_5y: Optional[float] = None

    # Statistical test on per-date Q1-Q5 spread (one-sample t-test, H0: spread=0)
    t_stat_1y: Optional[float] = None
    p_value_1y: Optional[float] = None
    t_stat_3y: Optional[float] = None
    p_value_3y: Optional[float] = None
    t_stat_5y: Optional[float] = None
    p_value_5y: Optional[float] = None

    # Plain-English notes (always set)
    survivorship_bias_note: str = ""
    cost_model_note: str = ""
    sample_size_note: str = ""
    conclusion: str = ""
