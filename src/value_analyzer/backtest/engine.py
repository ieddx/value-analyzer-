"""Backtest engine — runs the full pipeline over a universe and date grid.

POINT-IN-TIME FIREWALL
══════════════════════════════════════════════════════════════════════════════
For every (ticker, as_of_date) pair, the engine:

  1. Calls score(ticker, as_of_date=as_of_date) which applies as_of()
     internally before any calculation.

  2. Re-fetches the fundamentals and prices that score() used, filters them
     through as_of(), and then calls assert_no_lookahead() on each filtered
     DataFrame.  This belt-and-suspenders check confirms the filter worked.
     If a bug in the data layer ever produced future-dated rows after as_of(),
     the backtest would raise immediately rather than silently computing
     lookahead-biased scores.

  3. Forward returns are computed on UN-filtered price history.  This is
     intentional: measuring outcomes requires post-as_of prices.  It is NOT
     lookahead bias — it is the measurement step.

══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from value_analyzer.data import as_of, assert_no_lookahead, fetch_fundamentals, fetch_prices
from value_analyzer.score import CompositeScore

from .config import (
    ALPHA,
    DEFAULT_AS_OF_DATES,
    DEFAULT_BENCHMARK,
    HORIZON_DAYS,
    MIN_QUINTILE_OBS,
    N_QUINTILES,
    TRANSACTION_COST_BPS,
)
from .models import BacktestResult, QuintileStats, SnapshotResult
from .returns import benchmark_forward_return, forward_return
from .universe import SURVIVORSHIP_BIAS_WARNING, UNIVERSE

logger = logging.getLogger(__name__)


def run(
    universe: list[str] | None = None,
    as_of_dates: list[date] | None = None,
    *,
    benchmark_ticker: str = DEFAULT_BENCHMARK,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    score_fn: Callable[[str, date], CompositeScore] | None = None,
    show_progress: bool = True,
) -> BacktestResult:
    """Run the backtest and return a fully populated BacktestResult.

    Parameters
    ----------
    universe:
        List of ticker symbols.  Defaults to the ~45-ticker UNIVERSE constant.
        The universe is treated as fixed for all dates; see universe.py for the
        survivorship-bias caveats.
    as_of_dates:
        Historical snapshot dates.  Defaults to year-ends 2013–2021.
        For each date, scores are computed using ONLY data available on that date.
    benchmark_ticker:
        Ticker to use as the passive-buy-and-hold benchmark (default: SPY).
    transaction_cost_bps:
        Round-trip cost in basis points deducted from every forward return.
    score_fn:
        Override the scoring function (useful for testing without network access).
        Must accept (ticker: str, as_of_date: date) → CompositeScore.
    show_progress:
        Log progress at INFO level.
    """
    universe = list(universe or UNIVERSE)
    as_of_dates = sorted(as_of_dates or DEFAULT_AS_OF_DATES)

    if score_fn is None:
        from value_analyzer.score import score as _score

        def score_fn(ticker: str, as_of_date: date) -> CompositeScore:  # type: ignore[misc]
            return _score(ticker, as_of_date=as_of_date)

    snapshots: list[SnapshotResult] = []
    total = len(universe) * len(as_of_dates)

    for i, as_of_date in enumerate(as_of_dates):
        for ticker in universe:
            snap = _score_one(ticker, as_of_date, score_fn, benchmark_ticker, transaction_cost_bps)
            snapshots.append(snap)

        if show_progress:
            done = (i + 1) * len(universe)
            logger.info(
                "backtest progress: %d/%d pairs (as_of %s complete)",
                done, total, as_of_date,
            )

    # Assign cross-sectional quintiles within each as_of_date
    snapshots = _assign_quintiles(snapshots, as_of_dates)

    # Quintile aggregates
    quintile_stats = _compute_quintile_stats(snapshots)

    # Benchmark averages across all as_of_dates
    bm_avgs = _benchmark_averages(snapshots)

    # Top-line spreads, hit rates, and significance tests
    spreads, t_stats, p_vals, hit_rates = _compute_spreads_and_stats(snapshots, as_of_dates)

    # Q1 vs benchmark
    q1_vs_bm = _q1_vs_benchmark(quintile_stats, bm_avgs)

    n_scored = sum(1 for s in snapshots if s.error is None)
    n_errors = sum(1 for s in snapshots if s.error is not None)

    result = BacktestResult(
        run_date=date.today(),
        universe=universe,
        as_of_dates=as_of_dates,
        benchmark_ticker=benchmark_ticker,
        transaction_cost_bps=transaction_cost_bps,
        n_attempted=len(snapshots),
        n_scored=n_scored,
        n_errors=n_errors,
        n_with_1y_return=sum(1 for s in snapshots if s.net_return_1y is not None),
        n_with_3y_return=sum(1 for s in snapshots if s.net_return_3y is not None),
        n_with_5y_return=sum(1 for s in snapshots if s.net_return_5y is not None),
        snapshots=snapshots,
        quintile_stats=quintile_stats,
        q1_q5_spread_1y=spreads.get("1y"),
        q1_q5_spread_3y=spreads.get("3y"),
        q1_q5_spread_5y=spreads.get("5y"),
        hit_rate_1y=hit_rates.get("1y"),
        hit_rate_3y=hit_rates.get("3y"),
        hit_rate_5y=hit_rates.get("5y"),
        q1_vs_benchmark_1y=q1_vs_bm.get("1y"),
        q1_vs_benchmark_3y=q1_vs_bm.get("3y"),
        q1_vs_benchmark_5y=q1_vs_bm.get("5y"),
        benchmark_avg_1y=bm_avgs.get("1y"),
        benchmark_avg_3y=bm_avgs.get("3y"),
        benchmark_avg_5y=bm_avgs.get("5y"),
        t_stat_1y=t_stats.get("1y"),
        p_value_1y=p_vals.get("1y"),
        t_stat_3y=t_stats.get("3y"),
        p_value_3y=p_vals.get("3y"),
        t_stat_5y=t_stats.get("5y"),
        p_value_5y=p_vals.get("5y"),
        survivorship_bias_note=SURVIVORSHIP_BIAS_WARNING,
        cost_model_note=(
            f"Transaction costs: {transaction_cost_bps:.0f} bps round-trip deducted from "
            "each forward return.  Bid-ask spreads, market impact, slippage, and "
            "taxes are NOT modelled.  Actual costs for a retail investor will be higher."
        ),
        sample_size_note=_sample_size_note(as_of_dates, n_scored),
        conclusion=_generate_conclusion(spreads, p_vals, bm_avgs, q1_vs_bm, n_scored, hit_rates),
    )

    logger.info(
        "backtest complete: %d scored, %d errors, Q1-Q5 1y spread=%.1f%%",
        n_scored, n_errors,
        (spreads.get("1y") or 0) * 100,
    )

    write_backtest_summary(result)
    return result


# ── Per-ticker scoring ─────────────────────────────────────────────────────

def _score_one(
    ticker: str,
    as_of_date: date,
    score_fn: Callable,
    benchmark_ticker: str,
    transaction_cost_bps: float,
) -> SnapshotResult:
    """Score one ticker and compute its forward returns."""
    try:
        composite = score_fn(ticker, as_of_date)

        # Belt-and-suspenders PIT check: re-fetch and assert no lookahead
        # in the data that score() would have used.
        _assert_pit_clean(fetch_fundamentals(ticker), as_of_date, tag=f"{ticker}/fund")
        _assert_pit_clean(fetch_prices(ticker),       as_of_date, tag=f"{ticker}/prices")

        gross_1y, net_1y = forward_return(ticker, as_of_date, HORIZON_DAYS["1y"], transaction_cost_bps)
        gross_3y, net_3y = forward_return(ticker, as_of_date, HORIZON_DAYS["3y"], transaction_cost_bps)
        gross_5y, net_5y = forward_return(ticker, as_of_date, HORIZON_DAYS["5y"], transaction_cost_bps)

        bm_1y = benchmark_forward_return(benchmark_ticker, as_of_date, HORIZON_DAYS["1y"])
        bm_3y = benchmark_forward_return(benchmark_ticker, as_of_date, HORIZON_DAYS["3y"])
        bm_5y = benchmark_forward_return(benchmark_ticker, as_of_date, HORIZON_DAYS["5y"])

        return SnapshotResult(
            ticker=ticker.upper(),
            as_of_date=as_of_date,
            composite_score=composite.composite,
            weight_profile=composite.weight_profile,
            moat_score=composite.moat.score,
            health_score=composite.health.score,
            valuation_score=composite.valuation.score,
            management_score=composite.management.score,
            fwd_return_1y=gross_1y,
            fwd_return_3y=gross_3y,
            fwd_return_5y=gross_5y,
            net_return_1y=net_1y,
            net_return_3y=net_3y,
            net_return_5y=net_5y,
            benchmark_return_1y=bm_1y,
            benchmark_return_3y=bm_3y,
            benchmark_return_5y=bm_5y,
        )

    except AssertionError:
        raise  # PIT violations must propagate — never swallow them

    except Exception as exc:
        logger.warning("backtest: %s at %s failed: %s", ticker, as_of_date, exc)
        return SnapshotResult(
            ticker=ticker.upper(),
            as_of_date=as_of_date,
            error=str(exc),
        )


def _assert_pit_clean(df: pd.DataFrame, cutoff: date, tag: str = "") -> None:
    """Apply as_of() then assert_no_lookahead() — raises AssertionError on violation.

    This is the belt-and-suspenders check inside the backtest engine.  Any
    bug in the data layer that lets future-dated rows past as_of() will be
    caught here immediately, before it can silently corrupt a score.
    """
    filtered = as_of(df, cutoff)
    try:
        assert_no_lookahead(filtered, cutoff)
    except AssertionError as exc:
        raise AssertionError(
            f"Lookahead bias detected in backtest data ({tag}): {exc}"
        ) from exc


# ── Quintile assignment ────────────────────────────────────────────────────

def _assign_quintiles(
    snapshots: list[SnapshotResult],
    as_of_dates: list[date],
) -> list[SnapshotResult]:
    """Assign cross-sectional quintile ranks within each as_of_date.

    Only snapshots with a composite_score are ranked.  Ties share the lower
    quintile number (i.e. ties at the boundary go to the better quintile).
    """
    by_date: dict[date, list[SnapshotResult]] = {d: [] for d in as_of_dates}
    for snap in snapshots:
        by_date.setdefault(snap.as_of_date, []).append(snap)

    result: list[SnapshotResult] = []
    for snaps in by_date.values():
        scored = [(s, s.composite_score) for s in snaps if s.composite_score is not None]
        scored.sort(key=lambda x: x[1], reverse=True)
        n = len(scored)
        for rank, (snap, _) in enumerate(scored):
            q = min(N_QUINTILES, int(rank * N_QUINTILES / n) + 1) if n > 0 else None
            result.append(snap.model_copy(update={"quintile": q}))
        # Unscored snapshots: quintile stays None
        result.extend(s for s in snaps if s.composite_score is None)

    return result


# ── Quintile stats ─────────────────────────────────────────────────────────

def _compute_quintile_stats(snapshots: list[SnapshotResult]) -> list[QuintileStats]:
    """Aggregate net forward returns by quintile across all dates."""
    groups: dict[int, list[SnapshotResult]] = {q: [] for q in range(1, N_QUINTILES + 1)}
    for s in snapshots:
        if s.quintile is not None:
            groups[s.quintile].append(s)

    stats: list[QuintileStats] = []
    labels = {1: "Q1 (top 20%)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (bottom 20%)"}

    for q in range(1, N_QUINTILES + 1):
        snaps = groups[q]
        n = len(snaps)
        stats.append(QuintileStats(
            quintile=q,
            label=labels[q],
            n_obs=n,
            mean_net_return_1y=_mean([s.net_return_1y for s in snaps]),
            mean_net_return_3y=_mean([s.net_return_3y for s in snaps]),
            mean_net_return_5y=_mean([s.net_return_5y for s in snaps]),
            median_net_return_1y=_median([s.net_return_1y for s in snaps]),
            median_net_return_3y=_median([s.net_return_3y for s in snaps]),
            median_net_return_5y=_median([s.net_return_5y for s in snaps]),
        ))

    return stats


# ── Spreads, significance, and benchmarks ─────────────────────────────────

def _compute_spreads_and_stats(
    snapshots: list[SnapshotResult],
    as_of_dates: list[date],
) -> tuple[
    dict[str, float | None],   # spreads
    dict[str, float | None],   # t_stats
    dict[str, float | None],   # p_vals
    dict[str, float | None],   # hit_rates
]:
    """Compute Q1-Q5 spread, hit rate, and significance per horizon.

    Returns (spreads, t_stats, p_vals, hit_rates) dicts keyed by horizon.

    Hit rate: fraction of snapshot dates where Q1 mean return > Q5 mean return.
    A hit rate of 0.5 is indistinguishable from chance at the date level.
    """
    spreads: dict[str, float | None] = {}
    t_stats: dict[str, float | None] = {}
    p_vals: dict[str, float | None] = {}
    hit_rates: dict[str, float | None] = {}

    horizon_fields = {
        "1y": ("net_return_1y",),
        "3y": ("net_return_3y",),
        "5y": ("net_return_5y",),
    }

    for hz, (ret_field,) in horizon_fields.items():
        per_date_spreads: list[float] = []
        hits: list[bool] = []

        for d in as_of_dates:
            day_snaps = [s for s in snapshots if s.as_of_date == d and s.quintile is not None]
            q1_returns = [getattr(s, ret_field) for s in day_snaps if s.quintile == 1
                          and getattr(s, ret_field) is not None]
            q5_returns = [getattr(s, ret_field) for s in day_snaps if s.quintile == 5
                          and getattr(s, ret_field) is not None]

            if len(q1_returns) >= 2 and len(q5_returns) >= 2:
                q1_mean = float(np.mean(q1_returns))
                q5_mean = float(np.mean(q5_returns))
                per_date_spreads.append(q1_mean - q5_mean)
                hits.append(q1_mean > q5_mean)

        if len(per_date_spreads) >= 2:
            spreads[hz] = float(np.mean(per_date_spreads))
            hit_rates[hz] = float(np.mean(hits))
            t, p = _ttest_1samp(per_date_spreads)
            t_stats[hz] = t
            p_vals[hz] = p
        else:
            spreads[hz] = None
            hit_rates[hz] = None
            t_stats[hz] = None
            p_vals[hz] = None

    return spreads, t_stats, p_vals, hit_rates


def _benchmark_averages(snapshots: list[SnapshotResult]) -> dict[str, float | None]:
    bm_1y = _mean([s.benchmark_return_1y for s in snapshots])
    bm_3y = _mean([s.benchmark_return_3y for s in snapshots])
    bm_5y = _mean([s.benchmark_return_5y for s in snapshots])
    return {"1y": bm_1y, "3y": bm_3y, "5y": bm_5y}


def _q1_vs_benchmark(
    quintile_stats: list[QuintileStats],
    bm_avgs: dict[str, float | None],
) -> dict[str, float | None]:
    q1 = next((q for q in quintile_stats if q.quintile == 1), None)
    if q1 is None:
        return {"1y": None, "3y": None, "5y": None}

    def _diff(q_ret, bm_ret):
        return (q_ret - bm_ret) if q_ret is not None and bm_ret is not None else None

    return {
        "1y": _diff(q1.mean_net_return_1y, bm_avgs.get("1y")),
        "3y": _diff(q1.mean_net_return_3y, bm_avgs.get("3y")),
        "5y": _diff(q1.mean_net_return_5y, bm_avgs.get("5y")),
    }


# ── Statistical helpers ────────────────────────────────────────────────────

def _ttest_1samp(values: list[float]) -> tuple[float | None, float | None]:
    """Two-tailed one-sample t-test: H₀ mean(values) == 0.

    Uses scipy.stats.ttest_1samp when available; returns (t_stat, None) when
    scipy is not installed (install with: pip install scipy).
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        return None, None
    std = float(arr.std(ddof=1))
    if std == 0:
        return None, None
    t_stat = float(arr.mean()) / (std / np.sqrt(n))

    try:
        from scipy.stats import t as t_dist
        p_val = float(2 * t_dist.sf(abs(t_stat), df=n - 1))
    except ImportError:
        p_val = None

    return t_stat, p_val


def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def _median(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.median(clean)) if clean else None


# ── Conclusion generation ──────────────────────────────────────────────────

_SUMMARY_PATH = Path.home() / ".value_analyzer" / "backtest_summary.json"


def write_backtest_summary(result: "BacktestResult") -> None:
    """Persist a compact summary of *result* so the report layer can display it.

    Written to ~/.value_analyzer/backtest_summary.json.  The report layer reads
    this file directly using stdlib json — no import of the backtest layer.
    """
    summary: dict = {
        "run_date": result.run_date.isoformat(),
        "date_range": (
            f"{result.as_of_dates[0].isoformat()}–{result.as_of_dates[-1].isoformat()}"
            if result.as_of_dates else "unknown"
        ),
        "n_scored": result.n_scored,
        "n_attempted": result.n_attempted,
        "universe_size": len(result.universe),
        "benchmark_ticker": result.benchmark_ticker,
        "transaction_cost_bps": result.transaction_cost_bps,
        # Primary headline metric: Q1 net return minus benchmark, 1-year horizon
        "q1_vs_benchmark_1y": result.q1_vs_benchmark_1y,
        # Q1-Q5 spread for readers who prefer that framing
        "q1_q5_spread_1y": result.q1_q5_spread_1y,
        # Statistical test on per-date spread
        "t_stat_1y": result.t_stat_1y,
        "p_value_1y": result.p_value_1y,
        # Included so the report can note walk-forward discipline if tuning was run
        "tuning_train_val_gap": None,  # populated by write_tuning_supplement() if called
    }
    try:
        _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("backtest summary written → %s", _SUMMARY_PATH)
    except OSError as exc:
        logger.warning("could not write backtest summary: %s", exc)


def write_tuning_supplement(train_corr: float, val_corr: float) -> None:
    """Add train/validation Spearman gap to the existing summary file.

    Called by the CLI after a --tune run so the report can mention whether
    the walk-forward gap looks reasonable.  No-ops silently if the summary
    file does not yet exist.
    """
    if not _SUMMARY_PATH.exists():
        logger.debug("no backtest summary to supplement — run --backtest first")
        return
    try:
        data = json.loads(_SUMMARY_PATH.read_text())
        data["tuning_train_val_gap"] = round(train_corr - val_corr, 4)
        data["tuning_train_corr"] = round(train_corr, 4)
        data["tuning_val_corr"] = round(val_corr, 4)
        _SUMMARY_PATH.write_text(json.dumps(data, indent=2, default=str))
        logger.info("tuning supplement written to backtest summary")
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not update backtest summary with tuning data: %s", exc)


def _sample_size_note(as_of_dates: list[date], n_scored: int) -> str:
    return (
        f"Sample: {len(as_of_dates)} annual snapshots × universe.  "
        f"{n_scored} (ticker, date) pairs scored successfully.  "
        "With 9 snapshot dates, per-date Q1-Q5 spread tests have ~8 degrees "
        "of freedom — insufficient for statistical significance at any "
        "conventional threshold.  Results are directional context only."
    )


def _generate_conclusion(
    spreads: dict[str, float | None],
    p_vals: dict[str, float | None],
    bm_avgs: dict[str, float | None],
    q1_vs_bm: dict[str, float | None],
    n_scored: int,
    hit_rates: dict[str, float | None] | None = None,
) -> str:
    lines: list[str] = []

    hit_rates = hit_rates or {}
    primary_hz = "1y"
    spread = spreads.get(primary_hz)
    p_val = p_vals.get(primary_hz)
    q1_bm = q1_vs_bm.get(primary_hz)
    bm_avg = bm_avgs.get(primary_hz)
    hit_rate = hit_rates.get(primary_hz)

    # Did Q1 beat Q5?
    if spread is None:
        lines.append(
            "INCONCLUSIVE — insufficient data to compute Q1-Q5 spread.  "
            "More ticker-date pairs with complete forward returns are needed."
        )
        return " ".join(lines)

    spread_pct = spread * 100
    if spread > 0:
        hr_str = f" (hit rate: {hit_rate:.0%} of dates)" if hit_rate is not None else ""
        lines.append(
            f"Q1 outperformed Q5 by {spread_pct:+.1f}% on average over 1-year windows{hr_str}."
        )
    else:
        hr_str = f" (Q1 > Q5 on only {hit_rate:.0%} of dates)" if hit_rate is not None else ""
        lines.append(
            f"Q1 UNDERPERFORMED Q5 by {abs(spread_pct):.1f}% on average over 1-year windows{hr_str}.  "
            "The framework showed NO positive selection ability on this horizon and universe."
        )

    # Significance
    if p_val is not None:
        if p_val < ALPHA:
            lines.append(
                f"The per-date spread is possibly significant (p={p_val:.2f} < {ALPHA}).  "
                "Caveat: very small sample; this is suggestive at best."
            )
        else:
            lines.append(
                f"The per-date spread is NOT statistically significant (p={p_val:.2f}, "
                f"threshold {ALPHA}).  Small-sample chance variation cannot be ruled out."
            )
    else:
        lines.append(
            "Statistical significance: install scipy for p-values.  "
            f"With ~{len(spreads)} snapshot dates, most results will not be significant."
        )

    # Did Q1 beat the benchmark?
    if q1_bm is not None and bm_avg is not None:
        if q1_bm > 0:
            lines.append(
                f"Q1 net of costs beat the {DEFAULT_BENCHMARK} benchmark by "
                f"{q1_bm * 100:+.1f}% on average (1-year horizon)."
            )
        else:
            lines.append(
                f"Q1 DID NOT BEAT the {DEFAULT_BENCHMARK} benchmark after costs "
                f"({q1_bm * 100:+.1f}% vs benchmark, 1-year horizon).  "
                "A passive index fund would have been at least as good on this period and universe."
            )

    # Standing caveat
    lines.append(
        "Remember: survivorship bias, small sample size, and the specific "
        "2013–2021 period make definitive conclusions impossible.  "
        "This tool is for calibration and framework audit, not strategy validation."
    )

    return "  ".join(lines)
