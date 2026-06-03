"""Fetch 13F-HR holdings from SEC EDGAR.

Used offline to refresh the peer registry seed list — not called in the
live scoring pipeline.

Usage
-----
    from value_analyzer.peers.filings import fetch_13f_holdings
    from datetime import date

    # Berkshire CIK = 1067983, Markel CIK = 1096343
    holdings = fetch_13f_holdings(1067983, as_of=date(2024, 12, 31))
    # returns [(nameOfIssuer, cusip), ...]
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date

import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "value-analyzer/0.1 personal-educational-use contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accno}/"
_INDEX_JSON_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accno}/{acc_orig}-index.json"

# CIKs of investors whose 13F filings define the reference universe
INVESTOR_CIKS: dict[str, int] = {
    "Berkshire Hathaway": 1067983,
    "Markel Corporation": 1096343,
}

# Best-effort CUSIP→ticker map for common holdings. Populated from known 13F data.
# When a CUSIP is not listed here the holding is skipped at ticker-resolution time.
CUSIP_TO_TICKER: dict[str, str] = {
    "035229905": "AAPL",
    "025816109": "AXP",
    "191216100": "KO",
    "608190104": "MCO",
    "166764100": "CVX",
    "674599105": "OXY",
    "060505104": "BAC",
    "064058100": "BK",
    "23311P100": "DVA",
    "02079K107": "GOOG",
    "437076102": "HD",
    "654106103": "NKE",
    "254687106": "DIS",
    "478160104": "JNJ",
    "742718109": "PG",
    "931142103": "WMT",
    "92826C839": "V",
    "57636Q104": "MA",
    "670346105": "NUE",
    "149123101": "CAT",
    "30231G102": "XOM",
}


def fetch_13f_holdings(cik: int, as_of: date) -> list[tuple[str, str]]:
    """Return list of (nameOfIssuer, cusip) from the most recent 13F-HR before *as_of*.

    Returns an empty list if no filing is found or parsing fails.
    """
    acc = _latest_13f_accession(cik, as_of)
    if acc is None:
        logger.warning("No 13F-HR found for CIK %d before %s", cik, as_of)
        return []
    infotable_url = _find_infotable_url(cik, acc)
    if infotable_url is None:
        logger.warning("Infotable XML not found for CIK %d accession %s", cik, acc)
        return []
    return _parse_infotable(infotable_url)


def holdings_to_tickers(
    holdings: list[tuple[str, str]],
    cusip_map: dict[str, str] | None = None,
) -> list[str]:
    """Resolve (name, cusip) pairs to ticker symbols using *cusip_map*.

    Unknown CUSIPs are silently skipped.  Pass a custom *cusip_map* to
    supplement or override the built-in table.
    """
    mapping = {**CUSIP_TO_TICKER, **(cusip_map or {})}
    seen: set[str] = set()
    result: list[str] = []
    for _name, cusip in holdings:
        ticker = mapping.get(cusip)
        if ticker and ticker not in seen:
            result.append(ticker)
            seen.add(ticker)
    return result


# ── Private helpers ────────────────────────────────────────────────────────

def _latest_13f_accession(cik: int, cutoff: date) -> str | None:
    url = _SUBMISSIONS_URL.format(cik=cik)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("submissions fetch failed for CIK %d: %s", cik, exc)
        return None

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accnos = recent.get("accessionNumber", [])

    best_acc: str | None = None
    best_date: date | None = None
    for form, filing_date_str, acc in zip(forms, dates, accnos):
        if form not in ("13F-HR", "13F-HR/A"):
            continue
        try:
            filing_date = date.fromisoformat(filing_date_str)
        except ValueError:
            continue
        if filing_date > cutoff:
            continue
        if best_date is None or filing_date > best_date:
            best_date = filing_date
            best_acc = acc

    return best_acc


def _find_infotable_url(cik: int, accession: str) -> str | None:
    acc_nodash = accession.replace("-", "")
    index_url = _INDEX_JSON_URL.format(cik=cik, accno=acc_nodash, acc_orig=accession)
    try:
        time.sleep(0.15)
        resp = requests.get(index_url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("directory", {}).get("item", [])
    except Exception as exc:
        logger.warning("filing index fetch failed: %s", exc)
        return None

    base = _ARCHIVE_BASE.format(cik=cik, accno=acc_nodash)
    for item in items:
        name: str = item.get("name", "")
        item_type: str = item.get("type", "")
        if ("infotable" in name.lower() and name.lower().endswith(".xml")) or \
                item_type.upper() == "INFORMATION TABLE":
            return base + name

    return None


def _parse_infotable(url: str) -> list[tuple[str, str]]:
    try:
        time.sleep(0.15)
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("infotable fetch failed: %s", exc)
        return []

    holdings: list[tuple[str, str]] = []
    try:
        xml_str = resp.text
        # Strip namespace declarations so ElementTree can parse without prefixes
        xml_str = re.sub(r'\s+xmlns[^=]*="[^"]*"', "", xml_str)
        xml_str = re.sub(r"<([a-z][a-z0-9]*):([A-Za-z])", r"<\2", xml_str)
        xml_str = re.sub(r"</([a-z][a-z0-9]*):([A-Za-z])", r"</\2", xml_str)
        root = ET.fromstring(xml_str)
        for info in root.iter("infoTable"):
            name = (info.findtext("nameOfIssuer") or "").strip()
            cusip = (info.findtext("cusip") or "").strip()
            if name and cusip:
                holdings.append((name, cusip))
    except ET.ParseError as exc:
        logger.warning("infotable XML parse error: %s", exc)

    return holdings
