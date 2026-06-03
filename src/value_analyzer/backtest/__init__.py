"""Backtest layer — offline validation of the scoring framework.

This module is NOT part of the live analysis pipeline.  It imports from
data, classify, peers, and score — never from report.

Public API
----------
run(universe, as_of_dates, ...)  — execute the full backtest
format_report(result)            — plain-text report string
to_csv(result)                   — CSV string of per-snapshot data
BacktestResult                   — the result model
UNIVERSE                         — default ~45-ticker universe
DEFAULT_AS_OF_DATES              — year-end 2013–2021
"""

from .config import DEFAULT_AS_OF_DATES, TRANSACTION_COST_BPS
from .engine import run
from .models import BacktestResult, QuintileStats, SnapshotResult
from .report import format_report, to_csv
from .tuning import (
    TuningConfig,
    TuningResult,
    WalkForwardPeriod,
    run_noise_check,
    split_snapshots,
    tune_weights,
)
from .universe import UNIVERSE

__all__ = [
    "run",
    "format_report",
    "to_csv",
    "tune_weights",
    "run_noise_check",
    "split_snapshots",
    "TuningConfig",
    "TuningResult",
    "WalkForwardPeriod",
    "BacktestResult",
    "QuintileStats",
    "SnapshotResult",
    "UNIVERSE",
    "DEFAULT_AS_OF_DATES",
    "TRANSACTION_COST_BPS",
]
