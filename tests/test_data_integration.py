"""Integration tests for the data layer — hit real APIs (results are cached).

Run once with internet access; subsequent runs use the disk cache and are fast.
Mark with -m integration to run selectively:  pytest -m integration
"""

from datetime import date

import pandas as pd
import pytest

from value_analyzer.data import (
    as_of,
    assert_no_lookahead,
    fetch_fundamentals,
    fetch_prices,
    lookup_cik,
)

pytestmark = pytest.mark.integration

# ── Known CIK values (verified against SEC) ────────────────────────────────
KNOWN_CIKS = {
    "KO": "0000021344",    # The Coca-Cola Company
    "AAPL": "0000320193",  # Apple Inc.
}


# ── CIK lookup ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ticker,expected_cik", list(KNOWN_CIKS.items()))
def test_lookup_cik(ticker, expected_cik):
    cik = lookup_cik(ticker)
    assert cik == expected_cik, f"Expected CIK {expected_cik} for {ticker}, got {cik}"


def test_lookup_cik_unknown_ticker():
    cik = lookup_cik("ZZZZNOTREAL")
    assert cik is None


# ── Prices ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ticker", ["KO", "AAPL"])
def test_fetch_prices_shape(ticker):
    df = fetch_prices(ticker)
    assert not df.empty, f"fetch_prices({ticker!r}) returned empty DataFrame"
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
    assert isinstance(df.index, pd.DatetimeIndex)
    # 10 years of trading days ≈ 2500 rows; accept anything over 1000
    assert len(df) > 1000, f"Expected >1000 rows for {ticker}, got {len(df)}"


@pytest.mark.parametrize("ticker", ["KO", "AAPL"])
def test_prices_as_of_no_lookahead(ticker):
    """as_of() must never return prices dated after the cutoff."""
    df = fetch_prices(ticker)
    cutoff = date(2022, 6, 30)
    result = as_of(df, cutoff)

    assert not result.empty
    assert (result.index <= pd.Timestamp(cutoff)).all(), (
        f"Lookahead bias in prices for {ticker}: found rows after {cutoff}"
    )
    assert_no_lookahead(result, cutoff)


def test_prices_delisted_ticker():
    """Unknown tickers should return an empty DataFrame, not raise."""
    df = fetch_prices("ZZZZNOTREAL")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ── Fundamentals ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ticker", ["KO", "AAPL"])
def test_fetch_fundamentals_schema(ticker):
    df = fetch_fundamentals(ticker)
    assert not df.empty, f"fetch_fundamentals({ticker!r}) returned empty DataFrame"

    required_cols = {"concept", "period_end", "filed", "value", "form", "source"}
    assert required_cols <= set(df.columns), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )
    assert df["filed"].notna().all(), "Some rows have NaT filed date"
    assert df["value"].notna().any(), "No valid values found"


@pytest.mark.parametrize("ticker", ["KO", "AAPL"])
def test_fundamentals_has_key_concepts(ticker):
    df = fetch_fundamentals(ticker)
    concepts = set(df["concept"].unique())
    for required in ("revenue", "net_income", "total_assets", "operating_cf"):
        assert required in concepts, (
            f"Concept {required!r} missing for {ticker}. Found: {sorted(concepts)}"
        )


@pytest.mark.parametrize("ticker", ["KO", "AAPL"])
@pytest.mark.parametrize("cutoff", [date(2018, 1, 1), date(2020, 6, 30), date(2022, 12, 31)])
def test_fundamentals_as_of_no_lookahead(ticker, cutoff):
    """The cardinal rule: as_of() must NEVER return a filing dated after cutoff."""
    df = fetch_fundamentals(ticker)
    result = as_of(df, cutoff)

    if result.empty:
        pytest.skip(f"No data before {cutoff} for {ticker}")

    violations = result[result["filed"] > pd.Timestamp(cutoff)]
    assert violations.empty, (
        f"LOOKAHEAD BIAS for {ticker} as_of {cutoff}: "
        f"{len(violations)} row(s) with filed dates after cutoff.\n"
        f"Worst offender: filed={violations['filed'].max().date()}"
    )
    # Double-check using the guard utility
    assert_no_lookahead(result, cutoff)


@pytest.mark.parametrize("ticker", ["KO", "AAPL"])
def test_fundamentals_annual_and_quarterly_present(ticker):
    df = fetch_fundamentals(ticker)
    forms = set(df["form"].str.replace("/A", "", regex=False).unique())
    assert "10-K" in forms, f"No annual (10-K) data for {ticker}"
    assert "10-Q" in forms, f"No quarterly (10-Q) data for {ticker}"


@pytest.mark.parametrize("ticker", ["KO", "AAPL"])
def test_fundamentals_edgar_source_preferred(ticker):
    """EDGAR-sourced rows should dominate (fallback to yfinance is last resort)."""
    df = fetch_fundamentals(ticker)
    if "source" not in df.columns:
        pytest.skip("source column not present")
    pct_edgar = (df["source"] == "edgar").mean()
    assert pct_edgar > 0.5, (
        f"Less than 50% of rows are EDGAR-sourced for {ticker} ({pct_edgar:.0%}). "
        "Check if the EDGAR fetch is working."
    )


@pytest.mark.parametrize("ticker", ["KO", "AAPL"])
def test_fundamentals_values_are_plausible(ticker):
    """Revenue for major companies should be in the billions of USD."""
    df = fetch_fundamentals(ticker)
    annual_rev = df[
        (df["concept"] == "revenue") & df["form"].isin({"10-K", "10-K/A"})
    ]
    assert not annual_rev.empty, f"No annual revenue for {ticker}"
    max_rev = annual_rev["value"].max()
    assert max_rev > 1e9, (
        f"Annual revenue for {ticker} seems too small: {max_rev:,.0f}. "
        "Check unit scaling."
    )
