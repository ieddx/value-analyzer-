"""Fetch and normalise financial statement data.

Primary source: SEC EDGAR companyfacts API (free, official, has filing dates).
Fallback: yfinance financials (filing dates estimated from period-end + lag).

Every row in the returned DataFrame carries a ``filed`` column — the date the
data was publicly available — making point-in-time filtering via ``as_of``
possible.  The ``filed`` column is the *only* thing that anchors data to
reality; never use ``period_end`` as the availability date.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

from . import cache

logger = logging.getLogger(__name__)

# ── SEC EDGAR ──────────────────────────────────────────────────────────────
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_HEADERS = {
    # SEC EDGAR Fair Access Policy requires a descriptive User-Agent with contact info.
    "User-Agent": "value-analyzer/0.1 personal-educational-use contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

_ANNUAL_FORMS = {"10-K", "10-K/A"}
_QUARTERLY_FORMS = {"10-Q", "10-Q/A"}
_ALL_FORMS = _ANNUAL_FORMS | _QUARTERLY_FORMS

# ── Concept map ────────────────────────────────────────────────────────────
# Maps a friendly name to ordered list of GAAP taxonomy concepts (first hit wins).
CONCEPT_MAP: dict[str, list[str]] = {
    # Income statement
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenuesNetOfInterestExpense",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "interest_expense": ["InterestExpense", "InterestAndDebtExpense"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    # Balance sheet
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "short_term_debt": ["ShortTermBorrowings", "NotesPayableCurrent"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "goodwill": ["Goodwill"],
    "intangibles": [
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet",
    ],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    # Cash flow statement
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities"],
    "investing_cf": ["NetCashProvidedByUsedInInvestingActivities"],
    "financing_cf": ["NetCashProvidedByUsedInFinancingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "depreciation": [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "Depreciation",
    ],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    # Share data
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}

# Which unit to prefer for each concept (first match in the EDGAR units dict wins).
_UNIT_PREF: dict[str, list[str]] = {
    "shares_outstanding": ["shares"],
    "shares_diluted": ["shares"],
    "eps_basic": ["USD/shares"],
    "eps_diluted": ["USD/shares"],
}
_DEFAULT_UNITS = ["USD"]

# ── Schema ─────────────────────────────────────────────────────────────────
SCHEMA: dict[str, str] = {
    "concept": "str",       # friendly name from CONCEPT_MAP
    "gaap_name": "str",     # actual GAAP taxonomy concept used
    "period_end": "datetime64[ns]",
    "period_start": "datetime64[ns]",
    "value": "float64",
    "filed": "datetime64[ns]",   # ← point-in-time anchor
    "form": "str",          # 10-K, 10-Q, 10-K/A, 10-Q/A
    "unit": "str",          # USD, shares, USD/shares
    "fiscal_period": "str", # FY, Q1, Q2, Q3, Q4
    "accession": "str",
    "source": "str",        # "edgar" or "yfinance"
}


# ── CIK lookup ─────────────────────────────────────────────────────────────

def lookup_cik(ticker: str, *, refresh: bool = False) -> Optional[str]:
    """Return the zero-padded 10-digit SEC CIK for *ticker*, or None."""
    key = "sec_company_tickers"
    data: dict | None = None if refresh else cache.load_json(key)  # type: ignore[assignment]

    if data is None:
        logger.info("fetching SEC company tickers list")
        try:
            resp = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            cache.save_json(data, key)
        except Exception as exc:
            logger.error("failed to fetch SEC tickers list: %s", exc)
            return None

    upper = ticker.upper()
    for entry in data.values():  # type: ignore[union-attr]
        if str(entry.get("ticker", "")).upper() == upper:
            return str(entry["cik_str"]).zfill(10)

    logger.warning("CIK not found for %s", ticker)
    return None


# ── EDGAR facts ────────────────────────────────────────────────────────────

def fetch_edgar_facts(cik: str, *, refresh: bool = False) -> Optional[dict]:
    """Fetch raw EDGAR companyfacts JSON for *cik*. Returns None on failure."""
    key = f"edgar_facts_{cik}"
    data: dict | None = None if refresh else cache.load_json(key)  # type: ignore[assignment]

    if data is None:
        url = _FACTS_URL.format(cik=cik)
        logger.info("fetching EDGAR facts for CIK %s", cik)
        try:
            time.sleep(0.15)  # SEC rate limit: ≤10 req/s
            resp = requests.get(url, headers=_HEADERS, timeout=45)
            resp.raise_for_status()
            data = resp.json()
            cache.save_json(data, key)
        except Exception as exc:
            logger.error("EDGAR facts fetch failed for CIK %s: %s", cik, exc)
            return None

    return data


def _extract_concept(facts: dict, concept: str, gaap_options: list[str]) -> pd.DataFrame:
    """Pull all 10-K / 10-Q rows for one friendly concept from raw EDGAR facts."""
    us_gaap: dict = facts.get("facts", {}).get("us-gaap", {})
    preferred_units = _UNIT_PREF.get(concept, _DEFAULT_UNITS)

    for gaap_name in gaap_options:
        entry = us_gaap.get(gaap_name)
        if not entry:
            continue

        for unit_label in preferred_units:
            records: list[dict] = entry.get("units", {}).get(unit_label, [])
            if not records:
                continue

            rows = []
            for r in records:
                if r.get("form", "") not in _ALL_FORMS:
                    continue
                if not r.get("filed") or not r.get("end"):
                    continue
                rows.append(
                    {
                        "concept": concept,
                        "gaap_name": gaap_name,
                        "period_end": pd.Timestamp(r["end"]),
                        "period_start": pd.Timestamp(r["start"]) if r.get("start") else pd.NaT,
                        "value": float(r["val"]),
                        "filed": pd.Timestamp(r["filed"]),
                        "form": r["form"],
                        "unit": unit_label,
                        "fiscal_period": r.get("fp", ""),
                        "accession": r.get("accn", ""),
                        "source": "edgar",
                    }
                )

            if rows:
                return pd.DataFrame(rows)

    return pd.DataFrame(columns=list(SCHEMA.keys()))


def _from_edgar(cik: str, *, refresh: bool = False) -> pd.DataFrame:
    facts = fetch_edgar_facts(cik, refresh=refresh)
    if not facts:
        return pd.DataFrame(columns=list(SCHEMA.keys()))

    pieces = [
        _extract_concept(facts, concept, gaap_options)
        for concept, gaap_options in CONCEPT_MAP.items()
    ]
    non_empty = [p for p in pieces if not p.empty]
    if not non_empty:
        return pd.DataFrame(columns=list(SCHEMA.keys()))

    return pd.concat(non_empty, ignore_index=True)


# ── yfinance fallback ──────────────────────────────────────────────────────
# yfinance does not expose filing dates, so we estimate them conservatively.
# Annual 10-K: large filers must file within 60 days of fiscal year end.
# Quarterly 10-Q: large filers must file within 40 days; we use 45 for safety.
# These estimates may be too early for smaller filers — always prefer EDGAR.

_YF_INCOME: dict[str, str] = {
    "Total Revenue": "revenue",
    "Gross Profit": "gross_profit",
    "Operating Income": "operating_income",
    "Net Income": "net_income",
    "Research And Development": "rd_expense",
    "Interest Expense": "interest_expense",
    "Income Tax Expense": "income_tax",
    "Basic EPS": "eps_basic",
    "Diluted EPS": "eps_diluted",
}
_YF_BALANCE: dict[str, str] = {
    "Total Assets": "total_assets",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Stockholders Equity": "equity",
    "Cash And Cash Equivalents": "cash",
    "Long Term Debt": "long_term_debt",
    "Inventory": "inventory",
    "Accounts Receivable": "receivables",
    "Current Assets": "current_assets",
    "Current Liabilities": "current_liabilities",
    "Goodwill": "goodwill",
    "Retained Earnings": "retained_earnings",
    "Net PPE": "ppe_net",
}
_YF_CASHFLOW: dict[str, str] = {
    "Operating Cash Flow": "operating_cf",
    "Investing Cash Flow": "investing_cf",
    "Financing Cash Flow": "financing_cf",
    "Capital Expenditure": "capex",
    "Depreciation And Amortization": "depreciation",
}


def _yf_stmt_rows(
    stmt: pd.DataFrame | None,
    field_map: dict[str, str],
    form: str,
    lag_days: int,
) -> list[dict]:
    if stmt is None or stmt.empty:
        return []
    rows = []
    for col in stmt.columns:
        period_end = pd.Timestamp(col)
        filed = period_end + pd.Timedelta(days=lag_days)
        for field, concept in field_map.items():
            if field not in stmt.index:
                continue
            val = stmt.loc[field, col]
            if pd.isna(val):
                continue
            rows.append(
                {
                    "concept": concept,
                    "gaap_name": f"yfinance:{field}",
                    "period_end": period_end,
                    "period_start": pd.NaT,
                    "value": float(val),
                    "filed": filed,
                    "form": form,
                    "unit": "USD",
                    "fiscal_period": "FY" if "K" in form else "",
                    "accession": "",
                    "source": "yfinance",
                }
            )
    return rows


def _from_yfinance(ticker: str) -> pd.DataFrame:
    import yfinance as yf  # local import keeps EDGAR path fast

    logger.info("fetching fundamentals for %s via yfinance (fallback)", ticker)
    try:
        t = yf.Ticker(ticker)

        # yfinance renamed attributes in 0.2.x; try both.
        income_a = _yf_get(t, "income_stmt", "financials")
        income_q = _yf_get(t, "quarterly_income_stmt", "quarterly_financials")
        balance_a = _yf_get(t, "balance_sheet", "balance_sheet")
        balance_q = _yf_get(t, "quarterly_balance_sheet", "quarterly_balance_sheet")
        cf_a = _yf_get(t, "cash_flow", "cashflow")
        cf_q = _yf_get(t, "quarterly_cash_flow", "quarterly_cashflow")

        rows: list[dict] = []
        rows += _yf_stmt_rows(income_a, _YF_INCOME, "10-K", 60)
        rows += _yf_stmt_rows(income_q, _YF_INCOME, "10-Q", 45)
        rows += _yf_stmt_rows(balance_a, _YF_BALANCE, "10-K", 60)
        rows += _yf_stmt_rows(balance_q, _YF_BALANCE, "10-Q", 45)
        rows += _yf_stmt_rows(cf_a, _YF_CASHFLOW, "10-K", 60)
        rows += _yf_stmt_rows(cf_q, _YF_CASHFLOW, "10-Q", 45)

        if not rows:
            return pd.DataFrame(columns=list(SCHEMA.keys()))
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.error("yfinance fundamentals failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=list(SCHEMA.keys()))


def _yf_get(obj: object, *attrs: str) -> pd.DataFrame | None:
    """Try attribute names in order, return first non-None/non-empty result."""
    for attr in attrs:
        val = getattr(obj, attr, None)
        if val is not None and not (hasattr(val, "empty") and val.empty):
            return val  # type: ignore[return-value]
    return None


# ── Public API ─────────────────────────────────────────────────────────────

def fetch_fundamentals(ticker: str, *, refresh: bool = False) -> pd.DataFrame:
    """Return a tidy DataFrame of financial statement data for *ticker*.

    Columns: see ``SCHEMA``.  The ``filed`` column is the SEC filing date
    (or estimated date for yfinance-sourced rows) — it is the point-in-time
    anchor for lookahead-bias prevention.

    Results are cached to disk.  Pass ``refresh=True`` to force re-fetch.
    """
    key = f"fundamentals_{ticker.upper()}"
    if not refresh:
        cached = cache.load_df(key)
        if cached is not None:
            return cached

    cik = lookup_cik(ticker)
    df = _from_edgar(cik, refresh=refresh) if cik else pd.DataFrame()

    if df.empty:
        logger.warning("EDGAR data empty for %s — falling back to yfinance", ticker)
        df = _from_yfinance(ticker)

    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"])
        df["period_start"] = pd.to_datetime(df["period_start"])
        df["filed"] = pd.to_datetime(df["filed"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values(["concept", "period_end", "filed"]).reset_index(drop=True)
        cache.save_df(df, key)

    return df
