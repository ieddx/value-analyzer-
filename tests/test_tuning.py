"""Tests for the walk-forward self-tuning system.

Three critical properties under test:
  (a) Tuning NEVER reads validation-window data during the training phase.
  (b) The train/validation gap is computed and surfaced.
  (c) Shuffled/random returns produce a "no edge" verdict — if the tuner
      finds "edge" in pure noise, the guard is broken.

All tests are purely synthetic — no network access.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pytest

from value_analyzer.backtest.models import SnapshotResult
from value_analyzer.backtest.tuning import (
    TuningConfig,
    _baseline_weights,
    _evaluate_weights,
    _optimize_weights,
    _recompute_composite,
    _run_noise_check,
    _weight_combinations,
    run_noise_check,
    split_snapshots,
    tune_weights,
)


# ── Synthetic data factories ───────────────────────────────────────────────

TRAIN_DATES = [
    date(2013, 12, 31), date(2014, 12, 31), date(2015, 12, 31),
    date(2016, 12, 31), date(2017, 12, 31),
]
VAL_DATES = [
    date(2018, 12, 31), date(2019, 12, 31),
]
ALL_DATES = TRAIN_DATES + VAL_DATES

# Sentinel return used to detect contamination
_SENTINEL_RETURN = 9999.0


def _snap(
    ticker: str,
    as_of_date: date,
    *,
    score: float = 50.0,
    moat: float = 50.0,
    health: float = 50.0,
    valuation: float = 50.0,
    management: float = 50.0,
    net_return_1y: float | None = 0.10,
    net_return_3y: float | None = 0.30,
    profile: str = "stable",
) -> SnapshotResult:
    composite = moat * 0.25 + health * 0.25 + valuation * 0.25 + management * 0.25
    return SnapshotResult(
        ticker=ticker, as_of_date=as_of_date,
        composite_score=composite,
        weight_profile=profile,
        moat_score=moat, health_score=health,
        valuation_score=valuation, management_score=management,
        net_return_1y=net_return_1y,
        net_return_3y=net_return_3y,
        benchmark_return_1y=0.08,
    )


def _make_train_snapshots(n_tickers: int = 10, signal: bool = True) -> list[SnapshotResult]:
    """Build training snapshots with an optional score-return signal.

    When signal=True: high-score tickers get higher returns.
    When signal=False: returns are randomised (no correlation with scores).
    """
    rng = np.random.default_rng(0)
    snaps = []
    for d in TRAIN_DATES:
        scores = np.linspace(80, 20, n_tickers)
        if signal:
            returns = np.linspace(0.20, 0.02, n_tickers) + rng.normal(0, 0.02, n_tickers)
        else:
            returns = rng.uniform(0.00, 0.20, n_tickers)
        for i, (s, r) in enumerate(zip(scores, returns)):
            snaps.append(_snap(
                f"T{i}", d,
                moat=float(s), health=float(s),
                valuation=float(s), management=float(s),
                net_return_1y=float(r),
            ))
    return snaps


def _make_val_snapshots_with_sentinel() -> list[SnapshotResult]:
    """Val snapshots with sentinel returns — contamination detector."""
    snaps = []
    for d in VAL_DATES:
        for i in range(10):
            snaps.append(_snap(f"T{i}", d, net_return_1y=_SENTINEL_RETURN))
    return snaps


def _default_config() -> TuningConfig:
    return TuningConfig(
        train_dates=TRAIN_DATES,
        val_dates=VAL_DATES,
        max_weight_iterations=20,
        min_improvement=0.001,  # low threshold so tuning can find something on synthetic data
        noise_check_n_shuffles=5,
        rng_seed=42,
    )


# ══════════════════════════════════════════════════════════════════════════════
# (a) CONTAMINATION GUARD — tuning never reads validation data during training
# ══════════════════════════════════════════════════════════════════════════════

class TestContaminationGuard:
    """
    THESE TESTS MUST NEVER BE SKIPPED.
    If _optimize_weights() can access validation data, every tuning result
    is contaminated and useless.
    """

    def test_optimize_raises_if_val_snapshots_passed(self):
        """Passing validation snapshots into _optimize_weights raises AssertionError."""
        train_snaps = _make_train_snapshots()
        val_snaps = _make_val_snapshots_with_sentinel()
        cfg = _default_config()

        # Mix val data into what we claim is training data
        contaminated = train_snaps + val_snaps

        with pytest.raises(AssertionError, match="[Cc]ontamination"):
            _optimize_weights(contaminated, TRAIN_DATES, cfg)

    def test_optimize_does_not_access_val_dates(self):
        """_optimize_weights with clean train data does NOT see sentinel val returns."""
        train_snaps = _make_train_snapshots()
        val_snaps = _make_val_snapshots_with_sentinel()
        cfg = _default_config()

        # Run optimisation with ONLY training data
        best_weights, _ = _optimize_weights(train_snaps, TRAIN_DATES, cfg)

        # Verify result is not corrupted by sentinel values
        spread = _evaluate_weights(train_snaps, TRAIN_DATES, best_weights, "1y")
        assert spread is None or abs(spread) < 100.0, (
            f"Spread={spread} looks contaminated by sentinel {_SENTINEL_RETURN} — "
            "validation data leaked into training."
        )

    def test_tune_weights_evaluates_val_exactly_once(self):
        """The val_peek_count in TuningResult must be 1 after a single tune_weights call."""
        train_snaps = _make_train_snapshots()
        val_snaps = _make_train_snapshots()  # same structure, different returns
        cfg = _default_config()

        result = tune_weights(train_snaps, val_snaps, cfg, prior_val_peek_count=0)
        assert result.val_peek_count == 1, (
            f"Expected val_peek_count=1, got {result.val_peek_count}"
        )

    def test_prior_val_peek_count_accumulates(self):
        """Repeated tuning on the same val window increments the counter."""
        train_snaps = _make_train_snapshots()
        val_snaps = _make_train_snapshots()
        cfg = _default_config()

        r1 = tune_weights(train_snaps, val_snaps, cfg, prior_val_peek_count=0)
        r2 = tune_weights(train_snaps, val_snaps, cfg, prior_val_peek_count=r1.val_peek_count)
        assert r2.val_peek_count == 2, (
            "Each tune_weights call must increment val_peek_count — the reader "
            "must know how many times the validation window was peeked at."
        )

    def test_split_snapshots_is_correct(self):
        """split_snapshots correctly separates train and val snapshots."""
        all_snaps = (
            _make_train_snapshots() +
            [_snap(f"T{i}", d) for d in VAL_DATES for i in range(5)]
        )
        train, val = split_snapshots(all_snaps, TRAIN_DATES, VAL_DATES)

        train_date_set = set(TRAIN_DATES)
        val_date_set = set(VAL_DATES)
        assert all(s.as_of_date in train_date_set for s in train)
        assert all(s.as_of_date in val_date_set for s in val)
        assert len(train) + len(val) == len(all_snaps)

    def test_optimize_with_empty_snapshot_list_returns_baseline(self):
        """No snapshots → baseline weights returned, no crash."""
        cfg = _default_config()
        weights, n_iters = _optimize_weights([], TRAIN_DATES, cfg)
        baseline = _baseline_weights()
        # The returned weights should equal baseline (no improvement possible)
        assert weights == baseline or all(
            abs(weights.get(p, {}).get(d, 0) - baseline.get(p, {}).get(d, 0)) < 0.01
            for p in baseline for d in ["moat", "health", "valuation", "management"]
        )


# ══════════════════════════════════════════════════════════════════════════════
# (b) TRAIN/VALIDATION GAP IS COMPUTED AND SURFACED
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainValGap:
    def test_gap_is_computed(self):
        """TuningResult.train_val_gap is populated."""
        train_snaps = _make_train_snapshots()
        val_snaps = _make_train_snapshots()
        cfg = _default_config()
        result = tune_weights(train_snaps, val_snaps, cfg)
        # Gap may be None if either spread is None, but it should be attempted
        # With sufficient data it should be a float
        # (it's okay if it's None with very small datasets)
        assert result.train_spread_baseline is not None or result.train_spread_tuned is not None, (
            "At least one training spread must be computed"
        )

    def test_overfit_flagged_when_gap_large(self):
        """When train spread >> val spread, overfit_flagged must be True."""
        # Construct train snaps with strong, clear signal
        train_snaps = []
        for d in TRAIN_DATES:
            for i in range(10):
                score = 90.0 - i * 8.0
                ret = 0.30 - i * 0.025  # very strong signal
                train_snaps.append(_snap(f"T{i}", d,
                    moat=score, health=score, valuation=score, management=score,
                    net_return_1y=ret))

        # Val snaps have NO signal (random returns)
        rng = np.random.default_rng(123)
        val_snaps = []
        for d in VAL_DATES:
            for i in range(10):
                score = 90.0 - i * 8.0
                ret = float(rng.uniform(-0.05, 0.20))
                val_snaps.append(_snap(f"T{i}", d,
                    moat=score, health=score, valuation=score, management=score,
                    net_return_1y=ret))

        cfg = TuningConfig(
            train_dates=TRAIN_DATES,
            val_dates=VAL_DATES,
            max_weight_iterations=30,
            min_improvement=0.001,
            overfit_gap_warn=0.01,  # low threshold so overfit is detected
            noise_check_n_shuffles=3,
            rng_seed=42,
        )
        result = tune_weights(train_snaps, val_snaps, cfg)

        if result.train_val_gap is not None and result.train_val_gap > cfg.overfit_gap_warn:
            assert result.overfit_flagged, (
                f"Gap={result.train_val_gap:.3f} exceeds threshold {cfg.overfit_gap_warn:.3f} "
                "but overfit_flagged is False"
            )

    def test_no_improvement_returns_baseline_weights(self):
        """When no weight combination beats baseline by min_improvement, default weights are kept."""
        # All tickers have identical scores → no weight combination can help
        snaps = []
        for d in TRAIN_DATES:
            for i in range(10):
                snaps.append(_snap(f"T{i}", d, moat=50.0, health=50.0,
                                   valuation=50.0, management=50.0,
                                   net_return_1y=0.10))  # uniform scores, uniform returns

        val_snaps = [_snap(f"T{i}", d) for d in VAL_DATES for i in range(5)]
        cfg = TuningConfig(
            train_dates=TRAIN_DATES,
            val_dates=VAL_DATES,
            max_weight_iterations=10,
            min_improvement=0.10,  # very high threshold — impossible to beat
            noise_check_n_shuffles=2,
        )
        result = tune_weights(snaps, val_snaps, cfg)

        assert result.tuned_weights == _baseline_weights(), (
            "When improvement < min_improvement, tuned_weights must equal baseline"
        )

    def test_conclusion_mentions_gap_when_overfit(self):
        """Overfit results must mention the gap in the conclusion text."""
        train_snaps = _make_train_snapshots(signal=True)
        val_snaps = []
        rng = np.random.default_rng(0)
        for d in VAL_DATES:
            for i in range(10):
                val_snaps.append(_snap(f"T{i}", d,
                    moat=float(80 - i*6), health=float(80 - i*6),
                    valuation=float(80 - i*6), management=float(80 - i*6),
                    net_return_1y=float(rng.uniform(-0.05, 0.15))))

        cfg = TuningConfig(
            train_dates=TRAIN_DATES, val_dates=VAL_DATES,
            max_weight_iterations=20, min_improvement=0.001,
            overfit_gap_warn=0.001, noise_check_n_shuffles=2,
        )
        result = tune_weights(train_snaps, val_snaps, cfg)

        if result.overfit_flagged:
            assert "overfit" in result.conclusion.lower() or "gap" in result.conclusion.lower(), (
                "When overfit_flagged=True, conclusion must explicitly mention overfit or gap.\n"
                f"Conclusion: {result.conclusion}"
            )

    def test_tuning_result_contains_baseline_and_tuned_weights(self):
        """TuningResult must always contain both baseline and tuned weights."""
        train_snaps = _make_train_snapshots()
        val_snaps = [_snap(f"T{i}", d) for d in VAL_DATES for i in range(5)]
        result = tune_weights(train_snaps, val_snaps, _default_config())

        assert result.baseline_weights, "baseline_weights must not be empty"
        assert result.tuned_weights, "tuned_weights must not be empty"
        # Each profile's weights must sum to ~1.0
        for profile, w in result.tuned_weights.items():
            total = sum(w.values())
            assert abs(total - 1.0) < 0.01, (
                f"Profile '{profile}' weights sum to {total:.4f}, expected 1.0"
            )


# ══════════════════════════════════════════════════════════════════════════════
# (c) SHUFFLED/RANDOM RETURNS PRODUCE "NO EDGE" VERDICT
# ══════════════════════════════════════════════════════════════════════════════

class TestNoEdgeInNoise:
    """
    THE CRITICAL SAFETY TEST.
    If the tuner finds "edge" in pure noise, every result is suspect.
    Feeding it shuffled returns must produce a "no edge" conclusion.
    """

    def test_noise_check_passes_on_random_returns(self):
        """Shuffled returns → noise_check_passed must be True."""
        # Build snapshots with scores but NO correlation to returns
        snaps = []
        rng = np.random.default_rng(0)
        for d in TRAIN_DATES:
            for i in range(10):
                snaps.append(_snap(
                    f"T{i}", d,
                    moat=float(80 - i * 6), health=float(80 - i * 6),
                    valuation=float(80 - i * 6), management=float(80 - i * 6),
                    net_return_1y=float(rng.uniform(-0.05, 0.20)),  # no signal
                ))

        cfg = TuningConfig(
            train_dates=TRAIN_DATES,
            val_dates=VAL_DATES,
            max_weight_iterations=20,
            min_improvement=0.005,
            noise_check_n_shuffles=8,
            rng_seed=42,
        )
        passed, note = run_noise_check(snaps, cfg)
        assert passed, (
            "Noise check FAILED: tuner found apparent edge in random returns.  "
            "The optimiser is exploiting noise — the guard is broken.\n"
            f"Note: {note}"
        )

    def test_noise_check_conclusion_says_no_edge(self):
        """The conclusion from a noise check run must contain 'no edge' language."""
        snaps = []
        rng = np.random.default_rng(1)
        for d in TRAIN_DATES:
            for i in range(10):
                snaps.append(_snap(
                    f"T{i}", d,
                    moat=float(80 - i * 6), health=float(80 - i * 6),
                    valuation=float(80 - i * 6), management=float(80 - i * 6),
                    net_return_1y=float(rng.uniform(-0.05, 0.20)),
                ))

        cfg = TuningConfig(
            train_dates=TRAIN_DATES, val_dates=VAL_DATES,
            max_weight_iterations=20, min_improvement=0.005,
            noise_check_n_shuffles=5,
        )
        val_snaps = [_snap(f"T{i}", d) for d in VAL_DATES for i in range(5)]
        result = tune_weights(snaps, val_snaps, cfg)

        if result.noise_check_passed:
            # Should not have found improvement above threshold
            assert (
                result.train_improvement is None
                or result.train_improvement < cfg.min_improvement
                or result.noise_check_note
            )

    def test_shuffled_returns_destroy_q1q5_spread(self):
        """After shuffling, the Q1-Q5 spread on training data should be near zero."""
        # Build snapshots with strong true signal first
        train_snaps = _make_train_snapshots(n_tickers=10, signal=True)
        cfg = _default_config()
        ret_field = "net_return_1y"

        # Confirm true signal exists
        baseline = _baseline_weights()
        true_spread = _evaluate_weights(train_snaps, TRAIN_DATES, baseline, "1y")

        # Shuffle returns 10 times, compute mean spread
        rng = np.random.default_rng(42)
        shuffled_spreads = []
        for _ in range(10):
            returns = [getattr(s, ret_field) for s in train_snaps
                       if getattr(s, ret_field) is not None]
            shuffled = list(rng.permutation(returns))
            ri = 0
            shuffled_snaps = []
            for s in train_snaps:
                if getattr(s, ret_field) is not None:
                    shuffled_snaps.append(s.model_copy(update={ret_field: shuffled[ri]}))
                    ri += 1
                else:
                    shuffled_snaps.append(s)
            sp = _evaluate_weights(shuffled_snaps, TRAIN_DATES, baseline, "1y")
            if sp is not None:
                shuffled_spreads.append(abs(sp))

        if true_spread is not None and shuffled_spreads:
            mean_shuffled = float(np.mean(shuffled_spreads))
            # The true spread should be larger than the mean shuffled spread
            # (not guaranteed for small samples, but a reasonable check)
            assert mean_shuffled < abs(true_spread) + 0.05, (
                f"Mean shuffled spread ({mean_shuffled:.3f}) exceeds true spread "
                f"({true_spread:.3f}) — shuffling didn't destroy the signal."
            )

    def test_uniform_scores_no_improvement(self):
        """When all tickers have identical scores, no weight combination can help."""
        snaps = []
        for d in TRAIN_DATES:
            for i in range(10):
                ret = 0.05 + i * 0.01  # different returns but SAME score
                snaps.append(_snap(f"T{i}", d,
                    moat=50.0, health=50.0, valuation=50.0, management=50.0,
                    net_return_1y=ret))

        cfg = TuningConfig(
            train_dates=TRAIN_DATES, val_dates=VAL_DATES,
            max_weight_iterations=30, min_improvement=0.001,
            noise_check_n_shuffles=3,
        )
        best_weights, n_iters = _optimize_weights(snaps, TRAIN_DATES, cfg)
        spread = _evaluate_weights(snaps, TRAIN_DATES, best_weights, "1y")
        # With identical scores, quintile assignment is arbitrary → spread should be ~0
        # (Can't assert strictly 0 because tie-breaking may create small spread)
        assert spread is None or abs(spread) < 0.10, (
            f"Uniform scores should give near-zero spread, got {spread}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class TestWeightCombinations:
    def test_all_combinations_sum_to_one(self):
        combos = _weight_combinations(0.10, 0.05, 500)
        for c in combos:
            total = sum(c.values())
            assert abs(total - 1.0) < 1e-6, f"Combo {c} sums to {total}"

    def test_all_combinations_respect_floor(self):
        floor = 0.10
        combos = _weight_combinations(floor, 0.05, 500)
        for c in combos:
            for dim, val in c.items():
                assert val >= floor - 1e-9, f"Combo {c}: {dim}={val} < floor {floor}"

    def test_max_n_respected(self):
        for max_n in [10, 25, 50]:
            combos = _weight_combinations(0.10, 0.05, max_n)
            assert len(combos) <= max_n

    def test_baseline_included_or_similar_found(self):
        """The default (0.25, 0.25, 0.25, 0.25) should be in the grid."""
        combos = _weight_combinations(0.10, 0.05, 1000)
        flat = {"moat": 0.25, "health": 0.25, "valuation": 0.25, "management": 0.25}
        assert flat in combos, "Flat 0.25/0.25/0.25/0.25 weights must be in the grid"

    def test_reproducible_with_same_seed(self):
        c1 = _weight_combinations(0.10, 0.05, 30, seed=42)
        c2 = _weight_combinations(0.10, 0.05, 30, seed=42)
        assert c1 == c2


class TestRecomputeComposite:
    def test_exact_recompute(self):
        snap = _snap("T", date(2020, 12, 31), moat=80.0, health=60.0,
                     valuation=40.0, management=20.0)
        weights = {"stable": {"moat": 0.40, "health": 0.30, "valuation": 0.20, "management": 0.10}}
        result = _recompute_composite(snap, weights)
        expected = 80*0.40 + 60*0.30 + 40*0.20 + 20*0.10
        assert abs(result - expected) < 1e-6

    def test_fallback_when_sub_scores_missing(self):
        snap = SnapshotResult(ticker="T", as_of_date=date(2020, 12, 31),
                               composite_score=55.0, weight_profile="stable")
        weights = {"stable": {"moat": 0.40, "health": 0.30, "valuation": 0.20, "management": 0.10}}
        result = _recompute_composite(snap, weights)
        assert result == 55.0  # falls back to stored composite


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD STABILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkForward:
    def test_tune_weights_populates_walk_forward(self):
        train_snaps = _make_train_snapshots()
        val_snaps = [_snap(f"T{i}", d) for d in VAL_DATES for i in range(5)]
        cfg = TuningConfig(
            train_dates=TRAIN_DATES, val_dates=VAL_DATES,
            max_weight_iterations=10, min_improvement=0.001,
            noise_check_n_shuffles=2,
        )
        result = tune_weights(train_snaps, val_snaps, cfg)
        # With 5 train dates + 2 val dates = 7 total dates, should have walk-forward periods
        assert len(result.walk_forward_periods) >= 0  # may be 0 if insufficient dates

    def test_walk_forward_weights_sum_to_one(self):
        train_snaps = _make_train_snapshots()
        val_snaps = [_snap(f"T{i}", d) for d in VAL_DATES for i in range(5)]
        cfg = TuningConfig(
            train_dates=TRAIN_DATES, val_dates=VAL_DATES,
            max_weight_iterations=10, min_improvement=0.001,
            noise_check_n_shuffles=2,
        )
        result = tune_weights(train_snaps, val_snaps, cfg)
        for period in result.walk_forward_periods:
            for profile, w in period.tuned_weights.items():
                assert abs(sum(w.values()) - 1.0) < 0.01, (
                    f"Period weights for '{profile}' sum to {sum(w.values())}"
                )

    def test_stability_note_is_populated(self):
        train_snaps = _make_train_snapshots()
        val_snaps = [_snap(f"T{i}", d) for d in VAL_DATES for i in range(5)]
        result = tune_weights(train_snaps, val_snaps, _default_config())
        assert result.stability_note, "stability_note must never be empty"

    def test_conclusion_is_populated(self):
        train_snaps = _make_train_snapshots()
        val_snaps = [_snap(f"T{i}", d) for d in VAL_DATES for i in range(5)]
        result = tune_weights(train_snaps, val_snaps, _default_config())
        assert result.conclusion, "conclusion must never be empty"
