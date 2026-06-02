"""Point-in-time guard — the lookahead-bias firewall.

Every DataFrame that crosses a layer boundary must be filtered through
``as_of`` before being used in any calculation.  No code outside the
``data`` layer should ever receive rows with a ``filed`` date in the
future relative to the analysis date.

Rules (from CLAUDE.md):
- Financial statements use the *filing date*, not the fiscal-period end.
- Price data is cut off at market close on ``as_of``.
- Violations corrupt every backtest result — treat as a critical bug.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


def as_of(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Return only the rows of *df* that were available as of *cutoff*.

    Two modes, detected automatically:

    **Price DataFrames** (DatetimeIndex, no ``filed`` column):
        Keep rows where ``index <= cutoff``.

    **Fundamentals DataFrames** (has ``filed`` column):
        1. Keep rows where ``filed <= cutoff``.
        2. Deduplicate: for each ``(concept, period_end)`` pair, keep the
           *latest-filed* row within the cutoff window.  This correctly
           handles amended filings — an investor on *cutoff* would see the
           most recent amendment, not the original.

    Parameters
    ----------
    df:
        DataFrame produced by ``fetch_prices`` or ``fetch_fundamentals``.
    cutoff:
        The analysis date.  No data filed/published after this date will
        appear in the result.

    Returns
    -------
    pd.DataFrame
        A filtered (and, for fundamentals, deduplicated) copy of *df*.
    """
    cutoff_ts = pd.Timestamp(cutoff)

    if "filed" not in df.columns:
        # Price DataFrame — the index IS the availability date.
        result = df[df.index <= cutoff_ts].copy()
        logger.debug("as_of prices(%s): %d/%d rows", cutoff, len(result), len(df))
        return result

    # Fundamentals DataFrame — filter by filing date first.
    result = df[df["filed"] <= cutoff_ts].copy()

    # Deduplicate to latest amendment per (concept, period_end).
    dedup_cols = [c for c in ("concept", "period_end") if c in result.columns]
    if dedup_cols and not result.empty:
        result = (
            result
            .sort_values("filed")
            .groupby(dedup_cols, sort=False, as_index=False)
            .last()
        )

    logger.debug(
        "as_of fundamentals(%s): %d rows (from %d raw)", cutoff, len(result), len(df)
    )
    return result


def assert_no_lookahead(df: pd.DataFrame, cutoff: date) -> None:
    """Raise AssertionError if *df* contains any row filed after *cutoff*.

    Use in tests and assertions inside the data layer.
    """
    if df.empty:
        return
    cutoff_ts = pd.Timestamp(cutoff)

    if "filed" in df.columns:
        bad = df[df["filed"] > cutoff_ts]
        if not bad.empty:
            worst = bad["filed"].max()
            raise AssertionError(
                f"Lookahead bias detected: {len(bad)} row(s) filed after "
                f"cutoff {cutoff}. Latest offending filed date: {worst.date()}"
            )
    else:
        bad = df[df.index > cutoff_ts]
        if not bad.empty:
            raise AssertionError(
                f"Lookahead bias detected: {len(bad)} price row(s) dated "
                f"after cutoff {cutoff}."
            )
