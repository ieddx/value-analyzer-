"""Unit tests for the point-in-time guard.  No network access required."""

from datetime import date

import pandas as pd
import pytest

from value_analyzer.data.point_in_time import as_of, assert_no_lookahead


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_fundamentals(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["concept", "period_end", "filed", "value"])
    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df


def _make_prices(dates: list[str], values: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates, name="date")
    return pd.DataFrame({"close": values}, index=idx)


# ── Fundamentals tests ─────────────────────────────────────────────────────

class TestFundamentalsAsOf:
    def test_filters_future_filings(self):
        df = _make_fundamentals(
            [
                {"concept": "revenue", "period_end": "2020-12-31", "filed": "2021-02-10", "value": 100},
                {"concept": "revenue", "period_end": "2021-12-31", "filed": "2022-02-08", "value": 110},
                {"concept": "revenue", "period_end": "2022-12-31", "filed": "2023-02-01", "value": 120},
            ]
        )
        result = as_of(df, date(2022, 6, 1))
        assert len(result) == 2
        assert result["filed"].max() <= pd.Timestamp("2022-06-01")

    def test_never_returns_filings_after_cutoff(self):
        """The critical lookahead-bias test."""
        df = _make_fundamentals(
            [
                {"concept": "net_income", "period_end": "2019-12-31", "filed": "2020-02-01", "value": 50},
                {"concept": "net_income", "period_end": "2020-12-31", "filed": "2021-02-15", "value": 60},
                {"concept": "net_income", "period_end": "2021-12-31", "filed": "2022-03-01", "value": 70},
            ]
        )
        cutoff = date(2021, 1, 1)
        result = as_of(df, cutoff)
        assert (result["filed"] <= pd.Timestamp(cutoff)).all(), (
            "as_of() returned rows filed after the cutoff — lookahead bias!"
        )

    def test_exact_cutoff_date_is_included(self):
        """A filing dated exactly on the cutoff must be included."""
        df = _make_fundamentals(
            [{"concept": "revenue", "period_end": "2022-12-31", "filed": "2023-02-14", "value": 100}]
        )
        result = as_of(df, date(2023, 2, 14))
        assert len(result) == 1

    def test_day_after_cutoff_is_excluded(self):
        df = _make_fundamentals(
            [{"concept": "revenue", "period_end": "2022-12-31", "filed": "2023-02-14", "value": 100}]
        )
        result = as_of(df, date(2023, 2, 13))
        assert len(result) == 0

    def test_dedup_keeps_latest_amendment(self):
        """When two filings cover the same period, keep the later one (amended)."""
        df = _make_fundamentals(
            [
                # Original 10-K
                {"concept": "revenue", "period_end": "2022-12-31", "filed": "2023-02-10", "value": 100},
                # Amended 10-K/A filed later
                {"concept": "revenue", "period_end": "2022-12-31", "filed": "2023-04-15", "value": 102},
            ]
        )
        result = as_of(df, date(2023, 12, 31))
        # Only one row per (concept, period_end) — the amended value
        assert len(result) == 1
        assert result.iloc[0]["value"] == 102.0

    def test_dedup_only_within_cutoff(self):
        """If the amendment was filed after the cutoff, the original must be used."""
        df = _make_fundamentals(
            [
                {"concept": "revenue", "period_end": "2022-12-31", "filed": "2023-02-10", "value": 100},
                {"concept": "revenue", "period_end": "2022-12-31", "filed": "2023-04-15", "value": 102},
            ]
        )
        # Cutoff is before the amendment
        result = as_of(df, date(2023, 3, 1))
        assert len(result) == 1
        assert result.iloc[0]["value"] == 100.0

    def test_multiple_concepts_not_collapsed(self):
        """Different concepts with the same period_end must NOT be deduplicated together."""
        df = _make_fundamentals(
            [
                {"concept": "revenue", "period_end": "2022-12-31", "filed": "2023-02-01", "value": 500},
                {"concept": "net_income", "period_end": "2022-12-31", "filed": "2023-02-01", "value": 80},
            ]
        )
        result = as_of(df, date(2023, 12, 31))
        assert len(result) == 2

    def test_empty_df_returns_empty(self):
        df = _make_fundamentals([])
        result = as_of(df, date(2023, 1, 1))
        assert result.empty

    def test_no_data_before_cutoff_returns_empty(self):
        df = _make_fundamentals(
            [{"concept": "revenue", "period_end": "2023-12-31", "filed": "2024-02-01", "value": 100}]
        )
        result = as_of(df, date(2020, 1, 1))
        assert result.empty


# ── Price DataFrame tests ──────────────────────────────────────────────────

class TestPricesAsOf:
    def test_filters_future_prices(self):
        df = _make_prices(
            ["2023-01-03", "2023-01-04", "2023-01-05", "2023-06-01"],
            [100.0, 101.0, 102.0, 150.0],
        )
        result = as_of(df, date(2023, 1, 5))
        assert len(result) == 3
        assert result.index.max() <= pd.Timestamp("2023-01-05")

    def test_never_returns_prices_after_cutoff(self):
        """The critical test for price data."""
        df = _make_prices(
            ["2022-01-03", "2023-06-15", "2024-01-02"],
            [90.0, 120.0, 145.0],
        )
        cutoff = date(2023, 1, 1)
        result = as_of(df, cutoff)
        assert (result.index <= pd.Timestamp(cutoff)).all(), (
            "as_of() returned price rows after the cutoff — lookahead bias!"
        )

    def test_empty_price_df(self):
        df = _make_prices([], [])
        result = as_of(df, date(2023, 1, 1))
        assert result.empty


# ── assert_no_lookahead ────────────────────────────────────────────────────

class TestAssertNoLookahead:
    def test_clean_df_passes(self):
        df = _make_fundamentals(
            [{"concept": "revenue", "period_end": "2022-12-31", "filed": "2023-01-15", "value": 100}]
        )
        assert_no_lookahead(df, date(2023, 6, 1))  # must not raise

    def test_future_filing_raises(self):
        df = _make_fundamentals(
            [{"concept": "revenue", "period_end": "2024-12-31", "filed": "2025-02-01", "value": 999}]
        )
        with pytest.raises(AssertionError, match="Lookahead"):
            assert_no_lookahead(df, date(2024, 6, 1))

    def test_empty_df_passes(self):
        assert_no_lookahead(_make_fundamentals([]), date(2023, 1, 1))
