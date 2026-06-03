"""Walk-forward self-tuning for CATEGORY_WEIGHTS.

DESIGN PHILOSOPHY (read before modifying)
══════════════════════════════════════════════════════════════════════════════
This module refines the interpretable category weights in score/config.py
against historical data.  The goal is calibration — finding whether any
weight combination consistently outperforms the baseline.  It is NOT goal is
to maximise in-sample profit; that produces overfit garbage.

Structural safeguards that enforce this discipline:

  STRICT TRAIN/VAL SPLIT
    _optimize_weights() only receives train_snapshots.  It has no way to
    access val_snapshots.  An assertion inside the function checks that every
    snapshot's as_of_date is in train_dates; if val data sneaks in, it raises
    immediately.  val_snapshots are evaluated ONCE after optimisation via
    _evaluate_weights() called explicitly from tune_weights().

  ITERATION CAP
    max_weight_iterations limits the grid search.  Larger caps = more fitting
    capacity = more overfitting risk.  Default: 50.  The cap is reported in
    TuningResult so the reader knows how much search was done.

  MINIMUM IMPROVEMENT GATE
    If the best tuned spread exceeds the baseline (default config weights) by
    less than min_improvement (default 0.5%), the tuner returns the DEFAULT
    weights unchanged and calls the result "no meaningful improvement."  The
    system must be capable of this verdict; never let it keep searching until
    it manufactures a positive number.

  TRAIN–VALIDATION GAP FLAG
    If train_spread_tuned − val_spread_tuned > overfit_gap_warn (3%), the
    result is flagged overfit and the conclusion says so explicitly.

  NOISE CHECK
    run_noise_check() shuffles the return assignments (breaking any real
    score-return correlation) and runs optimisation on each shuffle.  If the
    tuner finds apparent improvement on noise, the guard is broken.
    Validation gate: this check MUST produce "no edge" on shuffled returns.

  WALK-FORWARD STABILITY
    Weights are tuned on rolling train windows and the per-period values are
    compared.  Weights that swing > 0.15 across periods indicate no stable
    pattern in the data.  This is reported, not hidden.

WHAT IS NOT TUNED
  Scoring thresholds (MOAT_GM_EXCELLENT etc.) are not tuned here.  They are
  interpretable human judgments.  Tuning them requires re-running the full
  scoring pipeline for each candidate, which multiplies compute cost and
  overfitting risk.  If you want to explore threshold sensitivity, use a
  manual grid in score/config.py and compare BacktestResults.
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from pydantic import BaseModel, Field

from value_analyzer.score.config import CATEGORY_WEIGHTS

from .engine import _assign_quintiles
from .models import SnapshotResult

logger = logging.getLogger(__name__)

# ── Types ──────────────────────────────────────────────────────────────────

WeightProfile = dict[str, float]   # {"moat": 0.25, "health": 0.25, ...}
WeightMap = dict[str, WeightProfile]  # {profile_name: WeightProfile}


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class TuningConfig:
    """All tunable knobs for the walk-forward tuner."""

    train_dates: list[date]
    val_dates: list[date]

    horizon: str = "1y"                  # "1y", "3y", or "5y"
    max_weight_iterations: int = 50      # cap on grid-search evaluations per profile
    weight_floor: float = 0.10          # minimum weight per scoring dimension
    weight_step: float = 0.05           # grid resolution
    min_improvement: float = 0.005      # 0.5% absolute spread; below = "no improvement"
    overfit_gap_warn: float = 0.03      # 3% train-val gap triggers overfit flag
    noise_check_n_shuffles: int = 10    # shuffles for the noise sanity check
    rng_seed: int = 42                  # for reproducibility


# ── Result models ──────────────────────────────────────────────────────────

class WalkForwardPeriod(BaseModel):
    """One roll of the expanding train/validation window."""

    train_dates: list[date]
    val_dates: list[date]
    tuned_weights: WeightMap
    train_spread: Optional[float] = None
    val_spread: Optional[float] = None
    gap: Optional[float] = None         # train_spread - val_spread; positive = overfit


class TuningResult(BaseModel):
    """Full output of a tune_weights() call."""

    train_dates: list[date]
    val_dates: list[date]
    horizon: str
    iterations_used: int
    val_peek_count: int    # how many times val was evaluated; 1 is the minimum honest use

    # Weights before and after tuning
    baseline_weights: WeightMap    # from score/config.py
    tuned_weights: WeightMap       # best found on training window (or baseline if no gain)

    # Training-window performance
    train_spread_baseline: Optional[float] = None
    train_spread_tuned: Optional[float] = None
    train_improvement: Optional[float] = None   # tuned - baseline on training

    # Validation-window performance (evaluated ONCE)
    val_spread_baseline: Optional[float] = None
    val_spread_tuned: Optional[float] = None
    val_improvement: Optional[float] = None     # tuned - baseline on validation

    # Overfit signal
    train_val_gap: Optional[float] = None       # train_spread_tuned - val_spread_tuned
    overfit_flagged: bool = False

    # Walk-forward analysis
    walk_forward_periods: list[WalkForwardPeriod] = Field(default_factory=list)
    weight_stability: dict[str, float] = Field(default_factory=dict)
    # std of each weight component across walk-forward periods, keyed as "profile/dim"

    # Noise check
    noise_check_passed: Optional[bool] = None
    noise_check_note: str = ""

    # Plain-English findings
    stability_note: str = ""
    conclusion: str = ""


# ── Public API ─────────────────────────────────────────────────────────────

def tune_weights(
    train_snapshots: list[SnapshotResult],
    val_snapshots: list[SnapshotResult],
    config: TuningConfig,
    prior_val_peek_count: int = 0,
) -> TuningResult:
    """Tune CATEGORY_WEIGHTS on training data, evaluate once on validation.

    Parameters
    ----------
    train_snapshots:
        Snapshots from the training window only — the optimiser never sees
        val_snapshots during the search phase.
    val_snapshots:
        Held-out validation snapshots.  Evaluated ONCE after optimisation.
        Every additional call to tune_weights() that re-uses the same val
        window increments val_peek_count and erodes the window's integrity.
    config:
        TuningConfig with train/val dates, iteration cap, etc.
    prior_val_peek_count:
        Pass the val_peek_count from a previous TuningResult to track
        cumulative peeks across multiple tuning runs.

    Returns
    -------
    TuningResult
    """
    baseline = _baseline_weights()

    # ── 1. Tune on training window ────────────────────────────────────────
    best_weights, n_iters = _optimize_weights(
        train_snapshots, config.train_dates, config
    )

    train_spread_baseline = _evaluate_weights(
        train_snapshots, config.train_dates, baseline, config.horizon
    )
    train_spread_tuned = _evaluate_weights(
        train_snapshots, config.train_dates, best_weights, config.horizon
    )

    train_improvement = (
        (train_spread_tuned - train_spread_baseline)
        if train_spread_tuned is not None and train_spread_baseline is not None
        else None
    )

    # Apply minimum improvement gate: if gain is below threshold, revert to baseline
    weights_to_report = best_weights
    if train_improvement is None or train_improvement < config.min_improvement:
        weights_to_report = baseline
        logger.info(
            "tuning: improvement %.3f%% below threshold %.3f%% — keeping default weights",
            (train_improvement or 0) * 100, config.min_improvement * 100,
        )

    # ── 2. Evaluate on validation (one peek) ─────────────────────────────
    val_peek_count = prior_val_peek_count + 1
    val_spread_baseline = _evaluate_weights(
        val_snapshots, config.val_dates, baseline, config.horizon
    )
    val_spread_tuned = _evaluate_weights(
        val_snapshots, config.val_dates, weights_to_report, config.horizon
    )
    val_improvement = (
        (val_spread_tuned - val_spread_baseline)
        if val_spread_tuned is not None and val_spread_baseline is not None
        else None
    )

    train_val_gap = (
        (train_spread_tuned - val_spread_tuned)
        if train_spread_tuned is not None and val_spread_tuned is not None
        else None
    )
    overfit_flagged = (
        train_val_gap is not None and train_val_gap > config.overfit_gap_warn
    )

    # ── 3. Walk-forward stability ─────────────────────────────────────────
    all_snapshots = train_snapshots + val_snapshots
    all_dates = sorted(set(s.as_of_date for s in all_snapshots))
    wf_periods = _walk_forward(all_snapshots, all_dates, config)
    stability = _weight_stability(wf_periods)

    # ── 4. Noise check ────────────────────────────────────────────────────
    noise_passed, noise_note = _run_noise_check(train_snapshots, config)

    conclusion = _generate_conclusion(
        train_improvement, val_improvement, train_val_gap,
        overfit_flagged, stability, noise_passed, config,
    )

    return TuningResult(
        train_dates=config.train_dates,
        val_dates=config.val_dates,
        horizon=config.horizon,
        iterations_used=n_iters,
        val_peek_count=val_peek_count,
        baseline_weights=baseline,
        tuned_weights=weights_to_report,
        train_spread_baseline=train_spread_baseline,
        train_spread_tuned=train_spread_tuned,
        train_improvement=train_improvement,
        val_spread_baseline=val_spread_baseline,
        val_spread_tuned=val_spread_tuned,
        val_improvement=val_improvement,
        train_val_gap=train_val_gap,
        overfit_flagged=overfit_flagged,
        walk_forward_periods=wf_periods,
        weight_stability=stability,
        noise_check_passed=noise_passed,
        noise_check_note=noise_note,
        stability_note=_stability_note(stability, wf_periods),
        conclusion=conclusion,
    )


def run_noise_check(
    train_snapshots: list[SnapshotResult],
    config: TuningConfig,
) -> tuple[bool, str]:
    """Sanity check: tuner must NOT find improvement on shuffled returns.

    Shuffles forward-return assignments while keeping scores intact, then
    runs optimisation.  If any shuffle produces improvement > min_improvement,
    the guard is broken — the tuner is exploiting noise.

    Returns (passed: bool, explanation: str).
    """
    return _run_noise_check(train_snapshots, config)


def split_snapshots(
    snapshots: list[SnapshotResult],
    train_dates: list[date],
    val_dates: list[date],
) -> tuple[list[SnapshotResult], list[SnapshotResult]]:
    """Split a snapshot list into (train, val) by as_of_date."""
    train_set = set(train_dates)
    val_set = set(val_dates)
    train = [s for s in snapshots if s.as_of_date in train_set]
    val = [s for s in snapshots if s.as_of_date in val_set]
    return train, val


# ── Core optimisation ──────────────────────────────────────────────────────

def _optimize_weights(
    train_snapshots: list[SnapshotResult],
    train_dates: list[date],
    config: TuningConfig,
) -> tuple[WeightMap, int]:
    """Grid-search weight combinations on the training window only.

    CONTAMINATION GUARD: asserts every snapshot's as_of_date is in
    train_dates before doing any work.  This will raise immediately if
    validation data is accidentally passed here.

    Returns (best_weights, n_evaluations).
    """
    # ── Contamination guard ───────────────────────────────────────────────
    train_date_set = set(train_dates)
    contamination = [
        s.as_of_date for s in train_snapshots
        if s.as_of_date not in train_date_set
    ]
    if contamination:
        raise AssertionError(
            f"Training contamination: {len(contamination)} snapshots have as_of_date "
            f"outside train_dates. First offender: {contamination[0]}. "
            "Validation data must NEVER be passed to _optimize_weights()."
        )

    baseline_weights = _baseline_weights()
    best_weights = baseline_weights
    best_spread = _evaluate_weights(
        train_snapshots, train_dates, baseline_weights, config.horizon
    ) or -float("inf")

    # Generate weight candidates (each profile tuned independently, then assembled)
    candidates = _weight_combinations(
        config.weight_floor, config.weight_step, config.max_weight_iterations, config.rng_seed
    )

    profiles = list(CATEGORY_WEIGHTS.keys())
    n_evals = 0

    for cand in candidates:
        # Apply candidate to ALL profiles simultaneously
        candidate_map: WeightMap = {p: cand for p in profiles}
        spread = _evaluate_weights(
            train_snapshots, train_dates, candidate_map, config.horizon
        )
        n_evals += 1
        if spread is not None and spread > best_spread:
            best_spread = spread
            best_weights = candidate_map

    logger.info(
        "weight optimisation: %d evaluations, best spread=%.3f%%",
        n_evals, best_spread * 100,
    )
    return best_weights, n_evals


def _evaluate_weights(
    snapshots: list[SnapshotResult],
    dates: list[date],
    weights: WeightMap,
    horizon: str,
) -> float | None:
    """Recompute composite scores with *weights*, rank into quintiles, return Q1-Q5 spread.

    This is the core evaluation: O(N) reweighting of pre-computed sub-scores,
    followed by cross-sectional ranking and spread computation.
    Returns None when there are fewer than 2 usable dates.
    """
    ret_field = f"net_return_{horizon}"
    date_set = set(dates)
    relevant = [s for s in snapshots if s.as_of_date in date_set]

    if not relevant:
        return None

    # Recompute composite for each snapshot using the candidate weights
    recomputed = [
        s.model_copy(update={"composite_score": _recompute_composite(s, weights)})
        for s in relevant
    ]

    ranked = _assign_quintiles(recomputed, dates)

    per_date_spreads: list[float] = []
    for d in dates:
        day = [s for s in ranked if s.as_of_date == d and s.quintile is not None]
        q1 = [getattr(s, ret_field) for s in day
              if s.quintile == 1 and getattr(s, ret_field) is not None]
        q5 = [getattr(s, ret_field) for s in day
              if s.quintile == 5 and getattr(s, ret_field) is not None]
        if len(q1) >= 2 and len(q5) >= 2:
            per_date_spreads.append(float(np.mean(q1)) - float(np.mean(q5)))

    if len(per_date_spreads) < 2:
        return None
    return float(np.mean(per_date_spreads))


def _recompute_composite(snap: SnapshotResult, weights: WeightMap) -> float:
    """Apply weights to stored sub-scores; fall back to original composite if any sub-score is None."""
    w = weights.get(snap.weight_profile or "default",
                    weights.get("default", _flat_weights()))
    subs = [snap.moat_score, snap.health_score, snap.valuation_score, snap.management_score]
    if any(v is None for v in subs):
        return snap.composite_score or 50.0
    return (
        snap.moat_score       * w["moat"]         # type: ignore[operator]
        + snap.health_score   * w["health"]        # type: ignore[operator]
        + snap.valuation_score * w["valuation"]    # type: ignore[operator]
        + snap.management_score * w["management"]  # type: ignore[operator]
    )


# ── Walk-forward ───────────────────────────────────────────────────────────

def _walk_forward(
    snapshots: list[SnapshotResult],
    all_dates: list[date],
    config: TuningConfig,
) -> list[WalkForwardPeriod]:
    """Expanding-window walk-forward: train grows one date at a time, val=next 2 dates.

    Requires at least 5 total dates to produce any periods.
    """
    periods: list[WalkForwardPeriod] = []
    min_train = 3    # minimum training dates per period
    val_size = 2     # validation window size in dates

    n = len(all_dates)
    if n < min_train + val_size:
        return periods

    for split in range(min_train, n - val_size + 1):
        train_dates = all_dates[:split]
        val_dates = all_dates[split: split + val_size]
        if len(val_dates) < val_size:
            break

        # Filter snapshots by window
        td_set = set(train_dates)
        vd_set = set(val_dates)
        t_snaps = [s for s in snapshots if s.as_of_date in td_set]
        v_snaps = [s for s in snapshots if s.as_of_date in vd_set]

        # Mini-config for this period (same settings, different dates)
        period_cfg = TuningConfig(
            train_dates=train_dates,
            val_dates=val_dates,
            horizon=config.horizon,
            max_weight_iterations=config.max_weight_iterations,
            weight_floor=config.weight_floor,
            weight_step=config.weight_step,
            min_improvement=config.min_improvement,
            overfit_gap_warn=config.overfit_gap_warn,
            rng_seed=config.rng_seed + split,  # different seed per period
        )

        try:
            best_w, _ = _optimize_weights(t_snaps, train_dates, period_cfg)
            t_spread = _evaluate_weights(t_snaps, train_dates, best_w, config.horizon)
            v_spread = _evaluate_weights(v_snaps, val_dates, best_w, config.horizon)
            gap = (t_spread - v_spread) if t_spread is not None and v_spread is not None else None
        except Exception as exc:
            logger.warning("walk-forward period %d failed: %s", split, exc)
            best_w = _baseline_weights()
            t_spread = v_spread = gap = None

        periods.append(WalkForwardPeriod(
            train_dates=train_dates,
            val_dates=val_dates,
            tuned_weights=best_w,
            train_spread=t_spread,
            val_spread=v_spread,
            gap=gap,
        ))

    return periods


def _weight_stability(periods: list[WalkForwardPeriod]) -> dict[str, float]:
    """Compute std of each weight component across walk-forward periods.

    Keys are "profile/dimension" e.g. "compounder/moat".
    A std > 0.10 indicates instability (no stable pattern in data).
    """
    if len(periods) < 2:
        return {}

    # Collect per-period weight values
    by_key: dict[str, list[float]] = {}
    for p in periods:
        for profile, w in p.tuned_weights.items():
            for dim, val in w.items():
                key = f"{profile}/{dim}"
                by_key.setdefault(key, []).append(val)

    return {
        k: float(np.std(vals))
        for k, vals in by_key.items()
        if len(vals) >= 2
    }


# ── Noise check ────────────────────────────────────────────────────────────

def _run_noise_check(
    train_snapshots: list[SnapshotResult],
    config: TuningConfig,
) -> tuple[bool, str]:
    """Shuffle return assignments and verify optimiser finds no improvement.

    For each of n_shuffles shuffles:
      - Randomly reassign net_return_1y values among all training snapshots
        (preserving the return distribution, destroying score-return correlation).
      - Run optimisation on the shuffled data.
      - Record whether it claims improvement > min_improvement.

    PASSES (returns True) if ≤ 1 shuffle out of n shows apparent improvement.
    FAILS (returns False) if ≥ 2 shuffles show apparent improvement.

    Rationale: with random returns, any improvement is pure noise from the small
    grid search.  Occasionally one shuffle will show improvement by chance (Type I
    error); requiring 2+ failures is robust to this.
    """
    rng = np.random.default_rng(config.rng_seed + 999)
    ret_field = f"net_return_{config.horizon}"
    n = config.noise_check_n_shuffles
    apparent_improvements = 0

    baseline = _baseline_weights()
    baseline_spread = _evaluate_weights(
        train_snapshots, config.train_dates, baseline, config.horizon
    )
    if baseline_spread is None:
        return True, "Noise check skipped — insufficient training data for spread computation."

    for i in range(n):
        # Collect all non-None returns and shuffle them
        returns = [getattr(s, ret_field) for s in train_snapshots
                   if getattr(s, ret_field) is not None]
        if len(returns) < 4:
            break
        shuffled_returns = list(rng.permutation(returns))
        ri = 0
        shuffled_snaps: list[SnapshotResult] = []
        for s in train_snapshots:
            if getattr(s, ret_field) is not None:
                shuffled_snaps.append(s.model_copy(update={ret_field: shuffled_returns[ri]}))
                ri += 1
            else:
                shuffled_snaps.append(s)

        shuffled_baseline = _evaluate_weights(
            shuffled_snaps, config.train_dates, baseline, config.horizon
        )

        best_w, _ = _optimize_weights(shuffled_snaps, config.train_dates, config)
        shuffled_tuned = _evaluate_weights(
            shuffled_snaps, config.train_dates, best_w, config.horizon
        )

        if (shuffled_baseline is not None and shuffled_tuned is not None
                and (shuffled_tuned - shuffled_baseline) > config.min_improvement):
            apparent_improvements += 1
            logger.debug("noise check shuffle %d: apparent improvement %.3f%%", i,
                         (shuffled_tuned - shuffled_baseline) * 100)

    passed = apparent_improvements <= 1

    note = (
        f"Noise check: {n} shuffles, {apparent_improvements} showed apparent "
        f"improvement > {config.min_improvement:.1%}.  "
        + ("PASSED — tuner did not find edge in pure noise." if passed
           else f"FAILED — {apparent_improvements}/{n} shuffles showed apparent edge.  "
                "This indicates the optimiser is exploiting noise; tuned weights are suspect.")
    )

    return passed, note


# ── Reporting ──────────────────────────────────────────────────────────────

def _stability_note(stability: dict[str, float], periods: list[WalkForwardPeriod]) -> str:
    if not stability:
        return "Insufficient walk-forward periods to assess weight stability."

    unstable = [(k, v) for k, v in stability.items() if v > 0.10]
    if not unstable:
        return (
            f"Weights appear stable across {len(periods)} walk-forward periods "
            f"(max std={max(stability.values()):.3f}).  "
            "Stability is necessary but not sufficient evidence of a real pattern."
        )
    top = sorted(unstable, key=lambda x: -x[1])[:3]
    top_str = ", ".join(f"{k}={v:.3f}" for k, v in top)
    return (
        f"{len(unstable)} weight components show high variability across "
        f"{len(periods)} walk-forward periods (std > 0.10): {top_str}.  "
        "Unstable weights indicate there is NO stable pattern in the data — "
        "any apparent tuned improvement is likely period-specific noise."
    )


def _generate_conclusion(
    train_improvement: float | None,
    val_improvement: float | None,
    train_val_gap: float | None,
    overfit_flagged: bool,
    stability: dict[str, float],
    noise_passed: bool | None,
    config: TuningConfig,
) -> str:
    lines: list[str] = []

    # Noise check verdict first — if it failed, nothing else matters
    if noise_passed is False:
        lines.append(
            "⛔ NOISE CHECK FAILED: the optimiser found apparent improvement on shuffled "
            "(randomised) returns.  The tuned weights are exploiting noise, NOT a real edge.  "
            "Do not use these weights.  Reduce max_weight_iterations or increase min_improvement."
        )
        return "  ".join(lines)

    # Training result
    if train_improvement is None:
        lines.append("INCONCLUSIVE — insufficient training data for spread computation.")
        return "  ".join(lines)

    if train_improvement < config.min_improvement:
        lines.append(
            f"NO MEANINGFUL IMPROVEMENT on training window "
            f"(gain={train_improvement * 100:+.2f}%, threshold={config.min_improvement:.1%}).  "
            "Default weights from score/config.py are retained — no tuning needed."
        )
    else:
        lines.append(
            f"Training window: tuned weights improved Q1-Q5 spread by "
            f"{train_improvement * 100:+.2f}% over baseline."
        )

    # Overfit check
    if overfit_flagged:
        lines.append(
            f"⚠ OVERFIT WARNING: train-val gap = {(train_val_gap or 0) * 100:+.2f}% "
            f"(threshold {config.overfit_gap_warn:.1%}).  "
            "The tuned weights performed much better on training than validation — "
            "they likely captured period-specific noise.  Treat with scepticism."
        )
    elif train_val_gap is not None:
        if val_improvement is not None and val_improvement > 0:
            lines.append(
                f"Validation window: improvement held out-of-sample "
                f"(+{val_improvement * 100:.2f}%).  Train-val gap={train_val_gap * 100:+.2f}%.  "
                "Encouraging, but remember: small sample — this could be luck."
            )
        else:
            lines.append(
                f"Validation window: improvement did NOT hold out-of-sample "
                f"(val improvement={( val_improvement or 0) * 100:+.2f}%).  "
                "Train-val gap={train_val_gap * 100:+.2f}%.  "
                "Default weights may be more robust than tuned weights."
            )

    # Stability
    unstable_count = sum(1 for v in stability.values() if v > 0.10)
    if unstable_count > 0:
        lines.append(
            f"Walk-forward stability: {unstable_count} weight components swing >0.10 "
            "across periods — no stable pattern detected in the data."
        )
    elif stability:
        lines.append("Walk-forward weights are stable across periods.")

    # Standing caveat
    lines.append(
        "Reality check: with a small universe (~45 tickers) and short history "
        "(9 annual dates), any apparent edge is statistically weak.  "
        "Section 9's factor foundation is where you would get a sample large "
        "enough to mean something.  These results are calibration context, not strategy."
    )

    return "  ".join(lines)


# ── Weight utilities ───────────────────────────────────────────────────────

def _baseline_weights() -> WeightMap:
    """Return a deep copy of the current CATEGORY_WEIGHTS from config."""
    return {profile: dict(w) for profile, w in CATEGORY_WEIGHTS.items()}


def _flat_weights() -> WeightProfile:
    return {"moat": 0.25, "health": 0.25, "valuation": 0.25, "management": 0.25}


def _weight_combinations(
    floor: float,
    step: float,
    max_n: int,
    seed: int = 42,
) -> list[WeightProfile]:
    """Generate up to *max_n* weight combinations for 4 dimensions summing to 1.0.

    Uses integer grid arithmetic to avoid floating-point accumulation errors.
    Components are at multiples of *step*, each ≥ *floor*.

    Example: floor=0.10, step=0.05 → 455 possible combinations.
    With max_n=50, returns a random sample of 50.
    """
    steps_total = round(1.0 / step)      # e.g. 20 for step=0.05
    steps_floor = round(floor / step)    # e.g. 2 for floor=0.10

    combos: list[WeightProfile] = []
    for a in range(steps_floor, steps_total - 3 * steps_floor + 1):
        for b in range(steps_floor, steps_total - 2 * steps_floor - a + 1):
            for c in range(steps_floor, steps_total - steps_floor - a - b + 1):
                d = steps_total - a - b - c
                if d >= steps_floor:
                    combos.append({
                        "moat":       round(a * step, 4),
                        "health":     round(b * step, 4),
                        "valuation":  round(c * step, 4),
                        "management": round(d * step, 4),
                    })

    if len(combos) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combos), max_n, replace=False)
        combos = [combos[int(i)] for i in sorted(idx)]

    return combos
